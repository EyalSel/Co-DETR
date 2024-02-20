# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import pickle
import time
import warnings
from pathlib import Path

import mmcv
import torch
import torchvision
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel.data_container import DataContainer
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from torch.utils.data import Dataset
from tqdm import tqdm

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.datasets.coco import CocoDataset
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
from projects import *

resolution = (1280, 1920)

class OfflineWaymoSensorV1_1():
    """
    Same as version 1 except _child_index_dir is run offline first
    """

    def __init__(self, data_path):
        with open(data_path, 'rb') as handle:
            dicts = pickle.load(handle)
        self.all_data = dicts

    def total_num_frames(self):
        return len(self.all_data)

    def get_frame(self, frame_index):
        return self.all_data[frame_index]


class WaymoDataset(Dataset):
    
    CLASSES = CocoDataset.CLASSES
    PALETTE  = CocoDataset.PALETTE

    def __init__(self, scenario_path) -> None:
        self.scenario_path = Path(scenario_path)
        self.scenario_name = self.scenario_path.parent.name + "-" + self.scenario_path.stem
        self.reader = OfflineWaymoSensorV1_1(scenario_path)

    def __getitem__(self, index):
        d = {'filename': self.scenario_name+f"-{index}.jpg",
             'ori_filename': self.scenario_name+f"-{index}.jpg",
             'ori_shape': (*resolution, 3),
             'img_shape': (*resolution, 3),
             'pad_shape': (*resolution, 3),
             'scale_factor': np.array([1, 1, 1, 1], dtype=np.float32),
             'flip': False,
             'flip_direction': None,
             'img_norm_cfg': {'mean': np.array([123.675, 116.28, 103.53], dtype=np.float32),
                              'std': np.array([58.395, 57.12, 57.375], dtype=np.float32), 'to_rgb': True},
             'batch_input_shape': resolution}
        img = self.reader.get_frame(index)["center_camera_feed"]
        img = img.transpose([2, 0, 1])
        # bgr to rgb
        img = img[::-1, :, :]
        img = img.astype(np.float32)
        img = (img - np.array([123.675, 116.28, 103.53]).reshape([-1, 1, 1])) / np.array([58.395, 57.12, 57.375]).reshape([-1, 1, 1])
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

        model.module.show_result(
            img_show,
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
    parser.add_argument(
        '--gpu-id',
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
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
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
    parser.add_argument(
        '--launcher',
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
    test_dataloader_default_args = dict(
        samples_per_gpu=BATCHSIZE, workers_per_gpu=2, dist=distributed, shuffle=False)

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

    # build the dataloader
    # dataset = build_dataset(cfg.data.test)
    dataset = WaymoDataset("../ad-config-search/waymo_pl/training00_05/training_0003/S12.pl")
    data_loader = build_dataloader(dataset, **test_loader_cfg)

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
        model.CLASSES = dataset.CLASSES

    # TODO: replace data_loader with a random image dataloader

    assert not distributed, "This script should run on a single GPU"
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    model.eval()


    all_preds = []
    for data in tqdm(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            if args.show or args.show_dir:
                show_preds(data, result, args.show_dir, model, dataset.PALETTE,
                           args.show, args.show_score_thr)
        for frame_preds in result:
            frame_preds = np.concatenate([
                np.concatenate([np.full(len(boxes), -1).reshape(-1, 1),
                                boxes,
                                np.full(len(boxes), i).reshape(-1, 1)], axis=1)
                for i, boxes in enumerate(frame_preds)], axis=0)
            frame_preds = frame_preds[np.argsort(frame_preds[:, 4])[::-1]][:100]
            all_preds.append(frame_preds)
    all_preds = np.array(all_preds)
    all_preds = all_preds[:, [0, 2, 1, 4, 3, 5, 6]]
    np.save("scenario", all_preds)

if __name__ == '__main__':
    main()
