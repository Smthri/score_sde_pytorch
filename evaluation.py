# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for computing FID/Inception scores."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy import linalg
import os


INCEPTION_DEFAULT_IMAGE_SIZE = 299


class InceptionV3(nn.Module):
  """Inception v3 model for feature extraction."""
  def __init__(self, normalize_input=False):
    super(InceptionV3, self).__init__()
    self.normalize_input = normalize_input
    inception = inception_v3(pretrained=True, transform_input=False)
    
    # Extract the layers we need
    self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
    self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
    self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
    self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
    self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
    self.Mixed_5b = inception.Mixed_5b
    self.Mixed_5c = inception.Mixed_5c
    self.Mixed_5d = inception.Mixed_5d
    self.Mixed_6a = inception.Mixed_6a
    self.Mixed_6b = inception.Mixed_6b
    self.Mixed_6c = inception.Mixed_6c
    self.Mixed_6d = inception.Mixed_6d
    self.Mixed_6e = inception.Mixed_6e
    self.Mixed_7a = inception.Mixed_7a
    self.Mixed_7b = inception.Mixed_7b
    self.Mixed_7c = inception.Mixed_7c
    
    # For logits
    self.fc = inception.fc
    
  def forward(self, x):
    if self.normalize_input:
      x = (x - 0.5) * 2.0  # Normalize to [-1, 1]
    
    # Inception v3 forward pass
    x = self.Conv2d_1a_3x3(x)
    x = self.Conv2d_2a_3x3(x)
    x = self.Conv2d_2b_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = self.Conv2d_3b_1x1(x)
    x = self.Conv2d_4a_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = self.Mixed_5b(x)
    x = self.Mixed_5c(x)
    x = self.Mixed_5d(x)
    x = self.Mixed_6a(x)
    x = self.Mixed_6b(x)
    x = self.Mixed_6c(x)
    x = self.Mixed_6d(x)
    x = self.Mixed_6e(x)
    x = self.Mixed_7a(x)
    x = self.Mixed_7b(x)
    x = self.Mixed_7c(x)
    
    # Pool3: adaptive average pooling
    pool3 = F.adaptive_avg_pool2d(x, (1, 1))
    pool3 = pool3.view(pool3.size(0), -1)
    
    # Logits
    logits = self.fc(pool3)
    
    return pool3, logits


def get_inception_model(inceptionv3=False, device='cuda'):
  """Get Inception model for feature extraction."""
  model = InceptionV3(normalize_input=True)
  model.eval()
  model = model.to(device)
  return model


def load_dataset_stats(config):
  """Load the pre-computed dataset statistics."""
  if config.data.dataset == 'CIFAR10':
    filename = 'assets/stats/cifar10_stats.npz'
  elif config.data.dataset == 'CELEBA':
    filename = 'assets/stats/celeba_stats.npz'
  elif config.data.dataset == 'LSUN':
    filename = f'assets/stats/lsun_{config.data.category}_{config.data.image_size}_stats.npz'
  else:
    raise ValueError(f'Dataset {config.data.dataset} stats not found.')

  with open(filename, 'rb') as fin:
    stats = np.load(fin)
    return stats


def run_inception_jit(inputs, inception_model, num_batches=1, inceptionv3=False, device='cuda'):
  """Running the inception network. Assuming input is within [0, 255]."""
  # Convert numpy array to torch tensor if needed
  if isinstance(inputs, np.ndarray):
    inputs = torch.from_numpy(inputs).float()
  
  # Normalize to [-1, 1] for Inception
  if not inceptionv3:
    inputs = (inputs - 127.5) / 127.5
  else:
    inputs = inputs / 255.0
    inputs = (inputs - 0.5) * 2.0  # Normalize to [-1, 1]

  # Move to device and ensure correct shape [B, C, H, W]
  if inputs.dim() == 4 and inputs.shape[-1] == 3:
    # Convert from [B, H, W, C] to [B, C, H, W]
    inputs = inputs.permute(0, 3, 1, 2)
  
  inputs = inputs.to(device)
  
  # Resize to Inception input size
  if inputs.shape[2] != INCEPTION_DEFAULT_IMAGE_SIZE or inputs.shape[3] != INCEPTION_DEFAULT_IMAGE_SIZE:
    inputs = F.interpolate(inputs, size=(INCEPTION_DEFAULT_IMAGE_SIZE, INCEPTION_DEFAULT_IMAGE_SIZE), 
                          mode='bilinear', align_corners=False)

  with torch.no_grad():
    pool3, logits = inception_model(inputs)
    
    pool3 = pool3.cpu().numpy()
    if not inceptionv3:
      logits = logits.cpu().numpy()
    else:
      logits = None

  return {
    'pool_3': pool3,
    'logits': logits
  }


def run_inception_distributed(input_tensor, inception_model, num_batches=1, inceptionv3=False, device='cuda'):
  """Run inception network on input tensor.
  
  Args:
    input_tensor: The input images as numpy array. Assumed to be within [0, 255].
    inception_model: The inception network model.
    num_batches: The number of batches used for dividing the input.
    inceptionv3: If `True`, use InceptionV3, otherwise use InceptionV1-style.
    device: Device to run the model on.

  Returns:
    A dictionary with key `pool_3` and `logits`, representing the pool_3 and
      logits of the inception network respectively.
  """
  # Convert to numpy if torch tensor
  if isinstance(input_tensor, torch.Tensor):
    input_tensor = input_tensor.cpu().numpy()
  
  # Process in batches to avoid memory issues
  batch_size = 50  # Process 50 images at a time
  all_pool3 = []
  all_logits = []
  
  for i in range(0, len(input_tensor), batch_size):
    batch = input_tensor[i:i+batch_size]
    result = run_inception_jit(batch, inception_model, num_batches=num_batches, 
                               inceptionv3=inceptionv3, device=device)
    all_pool3.append(result['pool_3'])
    if result['logits'] is not None:
      all_logits.append(result['logits'])

  return {
    'pool_3': np.concatenate(all_pool3, axis=0),
    'logits': np.concatenate(all_logits, axis=0) if all_logits else None
  }


def compute_inception_score(logits, splits=10):
  """Compute Inception Score from logits."""
  scores = []
  for i in range(splits):
    part = logits[i * (len(logits) // splits): (i + 1) * (len(logits) // splits)]
    py = np.mean(part, axis=0)
    scores.append(np.exp(np.mean([np.sum(p * np.log(p / py + 1e-10)) for p in part])))
  return np.mean(scores)


def compute_fid(real_features, fake_features):
  """Compute Frechet Inception Distance."""
  mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
  mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
  
  ssdiff = np.sum((mu1 - mu2) ** 2.0)
  covmean = linalg.sqrtm(sigma1.dot(sigma2))
  
  if np.iscomplexobj(covmean):
    covmean = covmean.real
  
  fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
  return fid


def compute_kid(real_features, fake_features, max_subset_size=1000):
  """Compute Kernel Inception Distance."""
  n_samples_real = len(real_features)
  n_samples_fake = len(fake_features)
  
  n_subsets = max(1, n_samples_real // max_subset_size)
  n = min(n_samples_real, n_samples_fake, max_subset_size)
  
  t = 0
  for _ in range(n_subsets):
    x = real_features[np.random.choice(n_samples_real, n, replace=False)]
    y = fake_features[np.random.choice(n_samples_fake, n, replace=False)]
    
    # Polynomial kernel of degree 3
    kxx = (np.dot(x, x.T) / n + 1) ** 3
    kyy = (np.dot(y, y.T) / n + 1) ** 3
    kxy = (np.dot(x, y.T) / n + 1) ** 3
    
    t += np.mean(kxx) + np.mean(kyy) - 2 * np.mean(kxy)
  
  kid = t / n_subsets
  return kid
