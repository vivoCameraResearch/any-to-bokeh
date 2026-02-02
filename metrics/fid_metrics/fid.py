"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from .inception import InceptionV3
from .inception3d import InceptionI3d
from .resnext3d import resnet101


def build_inception(dims):
    assert dims in list(InceptionV3.BLOCK_INDEX_BY_DIM)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    return model


# def build_inception3d(path):
#     return torch.jit.load(path)


def build_inception3d(path):
    model = InceptionI3d(400, in_channels=3)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(path))
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    return model


def build_resnet3d(path, sample_duration=16):
    model = resnet101(
        num_classes=400,
        shortcut_type="B",
        sample_size=224,
        sample_duration=sample_duration,
        last_fc=False)
    model_sd = torch.load(path, map_location='cpu')
    model_sd_new = {}
    for k, v in model_sd['state_dict'].items():
        model_sd_new[k.replace('module.', '')] = v

    missing_keys, unexpected_keys = model.load_state_dict(model_sd_new)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    return model


def calculate_act_statistics(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(np.dot(sigma1, sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(
            f'fid calculation produces singular product, ',
            'adding {eps} to diagonal of cov estimates',
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    return np.dot(diff, diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def calculate_fid(act1, act2):
    m1, s1 = calculate_act_statistics(act1)
    m2, s2 = calculate_act_statistics(act2)
    return calculate_frechet_distance(m1, s1, m2, s2)


def postprocess_i2d_pred(pred):
    pred = pred[0]
    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    return pred.squeeze()
