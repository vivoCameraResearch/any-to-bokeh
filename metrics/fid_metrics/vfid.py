
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from rich.progress import track

from z_train.metrics.fid_metrics import (
    ImageDataset,
    ImageSequenceDataset,
    VideoDataset,
    build_inception,
    build_inception3d,
    build_resnet3d,
    calculate_fid,
    is_image_dir_path,
    is_video_path,
)

DATA_CONFIG = dict(
    dataset=dict(
        sequence_length=10,
        resize_shape=[224, 224],
    ),
    batch_size=4,
    num_workers=16,
)


def build_loaders(type, paths, cfg):
    dls = []
    for path in paths:
        bs = cfg.get('batch_size')
        dataset_cfgs = cfg.get('dataset')
        if is_video_path(path):
            if type == 'fid':
                if dataset_cfgs:
                    dataset_cfgs = dict(dataset_cfgs)
                    dataset_cfgs['sequence_length'] = bs
                else:
                    dataset_cfgs = {'sequence_length': bs}
                bs = 1
            C = VideoDataset
        elif is_image_dir_path(path):
            C = ImageDataset if type == 'fid' else ImageSequenceDataset
        else:
            raise NotImplementedError

        dataset = C(path, **dataset_cfgs) if dataset_cfgs else C(path)
        dl = torch.utils.data.DataLoader(dataset, bs, shuffle=True, num_workers=cfg.get('num_workers'))
        dls.append(dl)
    return dls


def build_model(type, cfg, modeltype, sample_duration):
    if type == 'fid':
        return build_inception(cfg.dims)
    elif type == 'fvd':
        if modeltype == 'i3d':
            return build_inception3d(cfg.get('path'))
        elif modeltype == 'resnext':
            return build_resnet3d(cfg.get('path'), sample_duration=sample_duration)
    else:
        raise NotImplementedError


def vfid(
    gt_path: str,
    pred_path: str,
    ckpt_path: str,
    device: str = "cuda",
):
    vifd_results = {}
    for model_type in ['i3d', 'resnext']:
        dls = build_loaders('fvd', [f'{gt_path}/*', f'{pred_path}/*'], DATA_CONFIG)
        model = build_model('fvd', {
            "path": os.path.join(ckpt_path, {'i3d': 'i3d.pt', 'resnext': 'resnext-101.pth'}[model_type]),
            "modeltype": model_type,
        }, modeltype=model_type, sample_duration=DATA_CONFIG["dataset"]["sequence_length"]).to(device).eval()
        
        feats = [[], []]
        for i, dl in enumerate(dls):
            seq = range(len(dl))
            dl = iter(dl)
            for _ in track(seq, description=f'{type}_{i}'):
                x = next(dl).to(device)
                x = x * 2 - 1
                with torch.no_grad():
                    pred = model.extract_features(x)
                    if model_type == 'i3d':
                        pred = pred.squeeze(3).squeeze(3).mean(2)
                    elif model_type == 'resnext':
                        pass
                feats[i].append(pred.cpu().numpy())
            feats[i] = np.concatenate(feats[i], axis=0)
        fid = calculate_fid(*feats)
        
        vifd_results[model_type] = fid
    return vifd_results