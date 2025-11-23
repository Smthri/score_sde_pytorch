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

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
import logging
from datetime import datetime


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


class CustomDataset(Dataset):
  """Wrapper dataset that applies transforms and returns dict format."""
  def __init__(self, dataset, transform=None, uniform_dequantization=False, random_flip=False):
    self.dataset = dataset
    self.transform = transform
    self.uniform_dequantization = uniform_dequantization
    self.random_flip = random_flip

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    if isinstance(item, tuple):
      img, label = item
    else:
      img = item
      label = None

    if self.transform:
      img = self.transform(img)

    # Convert to tensor if PIL Image
    if isinstance(img, Image.Image):
      img = transforms.ToTensor()(img)

    # Ensure image is in [0, 1] range
    if img.max() > 1.0:
      img = img / 255.0

    # Random flip (before converting to NHWC)
    if self.random_flip and torch.rand(1) < 0.5:
      img = torch.flip(img, dims=[2])  # Flip horizontally (width dimension)

    # Uniform dequantization
    if self.uniform_dequantization:
      img = (torch.rand_like(img) + img * 255.) / 256.

    # Convert from [C, H, W] to [H, W, C] format to match original code expectations
    img = img.permute(1, 2, 0)

    return {'image': img, 'label': label}


def get_dataset(config, uniform_dequantization=False, evaluation=False):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing dataset: {config.data.dataset}")
  logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dataset parameters: uniform_dequantization={uniform_dequantization}, evaluation={evaluation}")
  
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
  logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Batch size: {batch_size}")

  # Create dataset builders for each dataset.
  if config.data.dataset == 'CIFAR10':
    data_root = './data'
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading CIFAR10 dataset from {data_root}")
    transform = transforms.Compose([
      transforms.Resize((config.data.image_size, config.data.image_size)),
      transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.CIFAR10(
      root='./data', train=True, download=True, transform=transform)
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CIFAR10 train dataset loaded: {len(train_dataset)} samples")
    eval_dataset = torchvision.datasets.CIFAR10(
      root='./data', train=False, download=True, transform=transform)
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CIFAR10 eval dataset loaded: {len(eval_dataset)} samples")
    dataset_builder = None

  elif config.data.dataset == 'SVHN':
    data_root = './data'
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading SVHN dataset from {data_root}")
    transform = transforms.Compose([
      transforms.Resize((config.data.image_size, config.data.image_size)),
      transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.SVHN(
      root='./data', split='train', download=True, transform=transform)
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SVHN train dataset loaded: {len(train_dataset)} samples")
    eval_dataset = torchvision.datasets.SVHN(
      root='./data', split='test', download=True, transform=transform)
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SVHN eval dataset loaded: {len(eval_dataset)} samples")
    dataset_builder = None

  elif config.data.dataset == 'CELEBA':
    data_root = './data'
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading CelebA dataset from {data_root}")
    # CelebA requires center crop and resize
    transform = transforms.Compose([
      transforms.CenterCrop(140),
      transforms.Resize((config.data.image_size, config.data.image_size)),
      transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.CelebA(
      root='./data', split='train', download=True, transform=transform)
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CelebA train dataset loaded: {len(train_dataset)} samples")
    eval_dataset = torchvision.datasets.CelebA(
      root='./data', split='valid', download=True, transform=transform)
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CelebA eval dataset loaded: {len(eval_dataset)} samples")
    dataset_builder = None

  elif config.data.dataset == 'LSUN':
    data_root = './data'
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading LSUN dataset from {data_root}, category: {config.data.category}")
    # LSUN dataset handling
    if config.data.image_size == 128:
      transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.CenterCrop(config.data.image_size),
        transforms.ToTensor(),
      ])
    else:
      transform = transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.ToTensor(),
      ])
    
    try:
      train_dataset = torchvision.datasets.LSUN(
        root='./data', classes=[config.data.category], transform=transform)
      logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] LSUN train dataset loaded: {len(train_dataset)} samples")
      eval_dataset = train_dataset  # LSUN doesn't have a standard test split
      logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] LSUN eval dataset loaded: {len(eval_dataset)} samples (using train split)")
    except Exception as e:
      logging.error(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to load LSUN dataset: {e}")
      raise NotImplementedError(
        f'LSUN dataset {config.data.category} not available. Error: {e}')

    dataset_builder = None

  elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
    # For FFHQ/CelebAHQ, we expect TFRecord files
    # This is a placeholder - users may need to implement custom dataset loader
    raise NotImplementedError(
      f'Dataset {config.data.dataset} requires custom TFRecord loader. '
      f'Please implement a custom Dataset class for {config.data.tfrecords_path}')

  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')

  # Wrap datasets with custom transforms
  train_ds = CustomDataset(
    train_dataset,
    uniform_dequantization=uniform_dequantization,
    random_flip=config.data.random_flip if not evaluation else False)
  eval_ds = CustomDataset(
    eval_dataset,
    uniform_dequantization=uniform_dequantization,
    random_flip=False)

  # Create data loaders
  num_workers = 4
  logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating data loaders with {num_workers} workers")
  train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=not evaluation,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True)
  
  eval_loader = DataLoader(
    eval_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True)

  num_train_batches = len(train_loader)
  num_eval_batches = len(eval_loader)
  logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Data loaders created:")
  logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Train batches: {num_train_batches} (shuffle={not evaluation})")
  logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Eval batches: {num_eval_batches}")

  return train_loader, eval_loader, dataset_builder
