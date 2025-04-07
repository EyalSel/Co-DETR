# Copyright (c) OpenMMLab. All rights reserved.
#!/usr/bin/env python3
"""
CodeTR Latency Profiler

This script profiles the inference latency of CodeTR models with
different configurations. It measures the time taken for inference across
multiple iterations and reports p10, p50, and p90 latency statistics in
milliseconds.

The script uses MMDetection to load models and perform inference. It first 
performs warmup iterations to ensure the model is fully loaded and optimized, 
then measures latency over a specified number of iterations.

Results are printed to the console with p10, p50, and p90 latency statistics.

Usage:
    python profile_codetr_latency.py --model=co_dino_5scale_vit_large
    --batchsize=1 --width=960 --height=640

Required flags:
    --model: CodeTR model to profile (e.g., co_dino_5scale_vit_large)
    --batchsize: Batch size for inference
    --width: Input width
    --height: Input height

Optional flags:
    --output_dir: Directory to save the latency profile (default:
    ./latency_profiles)
    --warmup_iters: Number of warmup iterations (default: 10)
    --measure_iters: Number of measurement iterations (default: 10)
"""
import argparse
import json
import os.path as osp
import time
from pathlib import Path

import mmcv
import numpy as np
import torch
import torchvision
from absl import app, flags
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel.data_container import DataContainer
from mmcv.runner import (get_dist_info, load_checkpoint, wrap_fp16_model)
from torchvision.datasets.fakedata import FakeData
from tqdm import tqdm

from mmdet.datasets import (build_dataloader, replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_dp, compat_cfg, get_device, replace_cfg_vals,
                         setup_multi_processes, update_data_root)
from projects import *

# Dictionary mapping model names to their config and checkpoint paths
with open('model_registry.json', 'r') as f:
    MODEL_CONFIGS = json.load(f)

# Set up flags
FLAGS = flags.FLAGS
flags.DEFINE_enum("model", None, list(MODEL_CONFIGS.keys()),
                  "CodeTR model to profile")
flags.DEFINE_integer("batchsize", None, "Batch size for inference")
flags.DEFINE_integer("width", None, "Input width")
flags.DEFINE_integer("height", None, "Input height")
flags.DEFINE_string("output_dir", "./latency_profiles",
                    "Directory to save the latency profile")
flags.DEFINE_integer("warmup_iters", 100, "Number of warmup iterations")
flags.DEFINE_integer("measure_iters", 200, "Number of measurement iterations")

flags.mark_flag_as_required("model")
flags.mark_flag_as_required("batchsize")
flags.mark_flag_as_required("width")
flags.mark_flag_as_required("height")


def get_nvidia_gpu_name():
    import subprocess
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"])
    return output.decode("utf-8").strip().split('\n')


class RandomImages(FakeData):
    CLASSES = [1, 2, 3]

    def __getitem__(self, index: int):
        img, _ = super().__getitem__(index)
        img = torchvision.transforms.functional.pil_to_tensor(img)
        img = img.type(torch.float32)
        resolution = (self.height, self.width)
        d = {
            'filename': 'fake_image',
            'ori_filename': 'fake_img',
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
        d = DataContainer(d, cpu_only=True)
        return {"img": [img], "img_metas": [d]}

    def __init__(self, size, image_size, height, width):
        self.height = height
        self.width = width
        super().__init__(size, image_size)


def setup_model(cfg, checkpoint, fuse_conv_bn_flag, dataset_classes):
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    if fuse_conv_bn_flag:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset_classes

    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    model.eval()
    return model


def get_args():
    args = argparse.Namespace(
        config=None,  # will be set by the absl flags
        checkpoint=None,  # will be set by the absl flags
        work_dir=None,
        out=None,
        fuse_conv_bn=False,
        gpu_ids=None,
        gpu_id=0,
        format_only=False,
        eval=['bbox'],
        show=False,
        show_dir=None,
        show_score_thr=0.3,
        gpu_collect=False,
        tmpdir=None,
        cfg_options=None,
        options=None,
        eval_options=None,
        launcher='none',
        local_rank=0)

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    return args


def get_config(args):
    # Get config
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

    return cfg


def profile_model_latency(model_name,
                          batchsize,
                          width,
                          height,
                          warmup_iters=100,
                          measure_iters=200):
    """
    Profile the latency of a CodeTR model with specified parameters.
    
    Args:
        model_name: CodeTR model name
        batchsize: Batch size for inference
        width: Input width
        height: Input height
        warmup_iters: Number of warmup iterations
        measure_iters: Number of measurement iterations
    
    Returns:
        Dictionary with p10, p50, and p90 latency statistics in milliseconds
    """
    gpu_name = get_nvidia_gpu_name()[0]
    print(
        f"Profiling {model_name} with batchsize={batchsize}, width={width}, height={height}, gpu={gpu_name}"
    )

    # Get model config and checkpoint
    model_config = MODEL_CONFIGS[model_name]

    # Update args with model-specific config and checkpoint
    args = get_args()
    args.config = model_config["config"]
    args.checkpoint = model_config["checkpoint"]

    cfg = get_config(args)

    # init distributed env first, since logger depends on the dist info.
    assert args.launcher == 'none', "This script should run on a single GPU"
    distributed = False

    test_dataloader_default_args = dict(samples_per_gpu=batchsize,
                                        workers_per_gpu=2,
                                        dist=distributed,
                                        shuffle=False)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = RandomImages(1000, (3, height, width), height, width)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    model = setup_model(cfg, args.checkpoint, args.fuse_conv_bn,
                        dataset.CLASSES)

    # warmup
    print(f"Running {warmup_iters} warmup iterations...")
    for i, data in tqdm(enumerate(data_loader), desc="Warmup"):
        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **data)
        if i == warmup_iters - 1:
            break

    # measurement
    print(f"Running {measure_iters} measurement iterations...")
    lats = []
    for i, data in tqdm(enumerate(data_loader), desc="Measurement"):
        start = time.time()
        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **data)
        lat = time.time() - start
        lats.append(lat * 1000)  # Convert to milliseconds
        if i == measure_iters - 1:
            break

    # Calculate statistics
    p10 = np.percentile(lats, 10)
    p50 = np.percentile(lats, 50)
    p90 = np.percentile(lats, 90)

    return {
        "model": model_name,
        "batchsize": batchsize,
        "width": width,
        "height": height,
        "latency_ms": {
            "p10": float(p10),
            "p50": float(p50),
            "p90": float(p90)
        },
        "num_iterations": measure_iters,
        "gpu_name": gpu_name
    }


def main(_):
    # Create output directory if it doesn't exist
    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / (
        f"model_latency--model={FLAGS.model}__batchsize={FLAGS.batchsize}__"
        f"resolution={FLAGS.width}x{FLAGS.height}.json")

    if output_file.exists():
        print(
            f"Results already exist for {FLAGS.model} with batchsize={FLAGS.batchsize} and resolution={FLAGS.width}x{FLAGS.height}"
        )
        return

    # Profile the model
    results = profile_model_latency(model_name=FLAGS.model,
                                    batchsize=FLAGS.batchsize,
                                    width=FLAGS.width,
                                    height=FLAGS.height,
                                    warmup_iters=FLAGS.warmup_iters,
                                    measure_iters=FLAGS.measure_iters)

    # Save results to a JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")
    print(f"Latency statistics (ms): p10={results['latency_ms']['p10']:.2f}, "
          f"p50={results['latency_ms']['p50']:.2f}, "
          f"p90={results['latency_ms']['p90']:.2f}")


if __name__ == '__main__':
    app.run(main)
