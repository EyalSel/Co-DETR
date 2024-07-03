# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings
from itertools import product
from pathlib import Path

import cv2
import mmcv
import torch
from copied_functions import (FractioningSchema, FrameFraction,
                              OfflineWaymoSensorV1_1, scenario_to_path)
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel.data_container import DataContainer
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from torch.utils.data import Dataset
from tqdm import tqdm

from mmdet.datasets import build_dataloader, replace_ImageToTensor
from mmdet.datasets.coco import CocoDataset
from mmdet.models import build_detector
from mmdet.utils import (build_dp, compat_cfg, get_device, replace_cfg_vals,
                         setup_multi_processes, update_data_root)
from projects import *

resolution = (1280, 1920)


class EqualFractions(FractioningSchema):
    """
    Cuts frame into equal pieces (ratio can be a tuple for h and w), and
    supports adding margins to reduce object cropping (margin can be specified
    for h and w dimensions).
    """

    def __init__(self, ratio, margin, with_full_frame=False):
        if isinstance(ratio, tuple):
            assert len(ratio) == 2, ratio
        else:
            ratio = (ratio, ratio)
        self.h_ratio, self.w_ratio = ratio
        if isinstance(margin, tuple):
            assert len(margin) == 2, margin
        else:
            margin = (margin, margin)
        self.h_margin, self.w_margin = margin
        self.with_full_frame = with_full_frame

    def get_split_specs(self, h, w):
        result = []
        for i, j in product(range(self.h_ratio), range(self.w_ratio)):
            ymin = np.clip(int(h / self.h_ratio) * i - self.h_margin, 0, h)
            ymax = np.clip(
                int(h / self.h_ratio) * (i + 1) + self.h_margin, 0, h)
            xmin = np.clip(int(w / self.w_ratio) * j - self.w_margin, 0, w)
            xmax = np.clip(
                int(w / self.w_ratio) * (j + 1) + self.w_margin, 0, w)
            result.append(
                FrameFraction(int(ymin), int(xmin), int(ymax), int(xmax)))
        if self.with_full_frame:
            # Preserves the aspect ratio of the full frame while reducing the
            # area to the size of the crop.
            side_ratio = np.sqrt(self.h_ratio * self.w_ratio)
            (new_h,
             new_w) = (None,
                       None) if side_ratio == 1 else (int(h / side_ratio),
                                                      int(w / side_ratio))
            result.append(FrameFraction(0, 0, h, w, new_h, new_w))
        return result


def convert_codetr_result_to_edet_format(frame_preds):
    frame_preds = np.concatenate([
        np.concatenate([
            np.full(len(boxes), -1).reshape(-1, 1), boxes,
            np.full(len(boxes), i).reshape(-1, 1)
        ],
                       axis=1) for i, boxes in enumerate(frame_preds)
    ],
                                 axis=0)
    frame_preds = frame_preds[np.argsort(frame_preds[:, 5])[::-1]][:100]
    frame_preds = frame_preds[:, [0, 2, 1, 4, 3, 5, 6]]
    return frame_preds


def run_split_specs_fn(run_fn, split_specs):

    def fn(inp):
        all_results = []
        for split_spec in split_specs:
            # get data from metadata DataContainer
            data = inp["img_metas"][0].data
            img = inp["img"]
            assert len(img) == 1, len(img)
            img = img[0]
            assert img.shape[0] == 1, img.shape
            img = img[0]
            _, og_h, og_w = img.shape
            # crop img
            img = img[:, split_spec.ymin:split_spec.ymax,
                      split_spec.xmin:split_spec.xmax]
            if split_spec.height is not None:
                # resize imgs
                img, w_scale, h_scale = mmcv.imresize(img.numpy().transpose(
                    [1, 2, 0]), (split_spec.width, split_spec.height),
                                                      return_scale=True,
                                                      interpolation='bilinear',
                                                      backend='cv2')
                img = torch.tensor(img.transpose([2, 0, 1]))
                # Reshape
                scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                        dtype=np.float32)
                data[0][0]["scale_factor"] = scale_factor
            data[0][0]["img_shape"] = (img.shape[1], img.shape[2],
                                       img.shape[0])
            # turn metadata back into DataContainer
            d = DataContainer(data, cpu_only=True)
            new_inp = {"img": [img[np.newaxis, :]], "img_metas": [d]}
            result = run_fn(new_inp)
            assert len(result) == 1, len(result)
            result = result[0]
            margin_dist = 3
            # xmin, ymin, xmax, ymax
            for i in range(len(result)):
                result[i][:, 0] += split_spec.xmin
                result[i][:, 1] += split_spec.ymin
                result[i][:, 2] += split_spec.xmin
                result[i][:, 3] += split_spec.ymin
                if split_spec.ymin > 0:
                    result[i] = result[i][result[i][:, 1] > split_spec.ymin +
                                          margin_dist]
                if split_spec.ymax < og_h:
                    result[i] = result[i][result[i][:, 3] < split_spec.ymax -
                                          margin_dist]
                if split_spec.xmin > 0:
                    result[i] = result[i][result[i][:, 0] > split_spec.xmin +
                                          margin_dist]
                if split_spec.xmax < og_w:
                    result[i] = result[i][result[i][:, 2] < split_spec.xmax -
                                          margin_dist]
            all_results.append(result)
        all_preds = []
        for class_id, all_class_preds in enumerate(zip(*all_results)):
            if len(all_class_preds) > 1:
                all_class_preds = np.concatenate(all_class_preds, axis=0)
            else:
                all_class_preds = all_class_preds[0]
            all_preds.append(
                np.concatenate([
                    np.full([len(all_class_preds), 1], -1),
                    all_class_preds,
                    np.full([len(all_class_preds), 1], class_id),
                ],
                               axis=1))
            # breakpoint()
        all_preds = np.concatenate(all_preds, axis=0)
        # breakpoint()
        xmin_ymin_wh_boxes = np.concatenate([
            all_preds[:, [1, 2]], all_preds[:, [3]] - all_preds[:, [1]],
            all_preds[:, [4]] - all_preds[:, [2]]
        ],
                                            axis=1)

        indices = cv2.dnn.NMSBoxes(xmin_ymin_wh_boxes, all_preds[:, 5], 0.3,
                                   0.5)

        all_preds = all_preds[indices]
        all_preds = all_preds[:, [0, 2, 1, 4, 3, 5, 6]]
        all_preds = all_preds[np.argsort(all_preds[:, 5])[::-1]][:100]

        # The commented code below converts the results into a format that can
        # be saved as an image. If this is uncommented, the uncomment
        # show_preds at the bottom of main().

        # x = [[] for _ in range(80)]
        # for entry in all_preds:
        #     x[int(entry[6])].append(entry[[2, 1, 4, 3, 5]])
        # x = [np.array(y) for y in x]
        # x = [y.reshape(0, 5) if len(y) == 0 else y for y in x]
        # x = [x]
        # return x
        return all_preds

    return fn


class WaymoDataset(Dataset):

    CLASSES = CocoDataset.CLASSES
    PALETTE = CocoDataset.PALETTE

    def __init__(self, scenario_path) -> None:
        self.scenario_path = Path(scenario_path)
        self.scenario_name = self.scenario_path.parent.name + "-" + self.scenario_path.stem
        self.reader = OfflineWaymoSensorV1_1(scenario_path)

    def __getitem__(self, index):
        d = {
            'filename': self.scenario_name + f"-{index}.jpg",
            'ori_filename': self.scenario_name + f"-{index}.jpg",
            'ori_shape': (*resolution, 3),
            'img_shape': (*resolution, 3),
            'pad_shape': (*resolution, 3),
            'scale_factor': np.array([1, 1, 1, 1], dtype=np.float32),
            'flip': False,
            'flip_direction': None,
            'img_norm_cfg': {
                'mean': np.array([123.675, 116.28, 103.53], dtype=np.float32),
                'std': np.array([58.395, 57.12, 57.375], dtype=np.float32),
                'to_rgb': True
            },
            'batch_input_shape': resolution
        }
        img = self.reader.get_frame(index)["center_camera_feed"]
        # Code used to unit test frame sharding (with an image of one dog or
        # two dogs)
        # from PIL import Image
        # # img = np.array(Image.open("dog.jpg"))
        # img = np.array(Image.open("two-dogs-1x.jpg"))
        # img = img[:, :, ::-1]
        img = img.transpose([2, 0, 1])
        # bgr to rgb
        img = img[::-1, :, :]
        img = img.astype(np.float32)
        img = (img - np.array([123.675, 116.28, 103.53]).reshape([-1, 1, 1])
              ) / np.array([58.395, 57.12, 57.375]).reshape([-1, 1, 1])
        img = img.astype(np.float32)
        d = DataContainer(d, cpu_only=True)
        return {"img": [img], "img_metas": [d]}

    def __len__(self):
        return self.reader.total_num_frames()


def show_preds(data, result, out_dir, model, PALETTE, show, show_score_thr):
    from mmcv.image import tensor2imgs
    batch_size = len(result)
    if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
        img_tensor = data['img'][0]
    else:
        img_tensor = data['img'][0].data[0]
    img_metas = data['img_metas'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    assert len(imgs) == len(img_metas)

    for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]

        ori_h, ori_w = img_meta['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

        if out_dir:
            out_file = osp.join(out_dir, img_meta['ori_filename'])
        else:
            out_file = None

        model.module.show_result(img_show,
                                 result[i],
                                 bbox_color=PALETTE,
                                 text_color=PALETTE,
                                 mask_color=PALETTE,
                                 show=show,
                                 out_file=out_file,
                                 score_thr=show_score_thr)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--gpu-id',
                        type=int,
                        default=0,
                        help='id of gpu to use '
                        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir',
                        help='directory where painted images will be saved')
    parser.add_argument('--show-score-thr',
                        type=float,
                        default=0.3,
                        help='score threshold (default: 0.3)')
    parser.add_argument('--gpu-collect',
                        action='store_true',
                        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument('--scenarios',
                        nargs='+',
                        help='Waymo scenarios to process')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    BATCHSIZE = 1
    test_dataloader_default_args = dict(samples_per_gpu=BATCHSIZE,
                                        workers_per_gpu=2,
                                        dist=distributed,
                                        shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {}),
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        # model.CLASSES = dataset.CLASSES
        model.CLASSES = CocoDataset.CLASSES

    # TODO: replace data_loader with a random image dataloader

    assert not distributed, "This script should run on a single GPU"
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    model.eval()

    # UNIT TESTS
    # baseline
    # split_specs = EqualFractions((1, 1), (0, 0), False)
    # split_specs = split_specs.get_split_specs(1280, 1920)
    # resize only
    # split_specs = [FrameFraction(0, 0, 1280, 1920, 320, 480)]
    # crop
    # split_specs = [
    #     # FrameFraction(0, 0, 1280, 1920, None, None),
    #     FrameFraction(0, 0, 1280, 960+60, None, None),
    #     FrameFraction(0, 960-60, 1280, 1920, None, None)
    # ]
    # crop + scale
    sharding_schema = "full_frame"
    split_specs = {
        "full_frame": EqualFractions((1, 1), (0, 0), False),
        "quarters": EqualFractions((2, 2), (30, 30), False),
        "16ths": EqualFractions((4, 4), (30, 30), True)
    }[sharding_schema]
    split_specs = split_specs.get_split_specs(1280, 1920)

    run_model_fn = run_split_specs_fn(
        lambda x: model(return_loss=False, rescale=True, **x), split_specs)

    # for data in tqdm(data_loader):
    #     with torch.no_grad():
    #         result = model(return_loss=False, rescale=True, **data)
    #         if args.show or args.show_dir:
    #             show_preds(data, result, args.show_dir, model, dataset.PALETTE,
    #                        args.show, args.show_score_thr)
    #     for frame_preds in result:
    #         frame_preds = convert_codetr_result_to_edet_format(frame_preds)
    #         all_preds.append(frame_preds)

    # build the dataloader
    # dataset = build_dataset(cfg.data.test)

    base_path = Path("waymo-co_detr-predictions")
    base_path.mkdir(exist_ok=True)
    model_name = Path(args.config).stem
    for scenario in args.scenarios:
        fn = f"preds--{scenario}__{model_name}"
        if (base_path / (fn + ".npy")).exists():
            print("Skipping", scenario)
            continue
        # scenario = "training_0003-S12"
        print(scenario)
        dataset = WaymoDataset(
            Path("../ad-config-search") / scenario_to_path(scenario, "waymo"))
        data_loader = build_dataloader(dataset, **test_loader_cfg)
        all_preds = []

        for data in tqdm(data_loader):
            with torch.no_grad():
                frame_preds = run_model_fn(data)
                # show_preds(data, frame_preds, args.show_dir, model,
                #            dataset.PALETTE, args.show, args.show_score_thr)
                all_preds.append(frame_preds)

        all_preds = np.array(all_preds)
        np.save(str(base_path / fn), all_preds)


if __name__ == '__main__':
    main()
