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

"""Training and evaluation"""

import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import sys
from datetime import datetime

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
  # Configure logging to output to both stdout and file
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  
  # Remove existing handlers to avoid duplicates
  logger.handlers = []
  
  # Create formatter with timestamp
  formatter = logging.Formatter('%(message)s')
  
  # Handler for stdout
  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setFormatter(formatter)
  logger.addHandler(stdout_handler)
  
  if FLAGS.mode == "train":
    # Create the working directory
    os.makedirs(FLAGS.workdir, exist_ok=True)
    # Set logger so that it outputs to both console and file
    log_file = os.path.join(FLAGS.workdir, 'stdout.txt')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting training mode")
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Config file: {FLAGS.config}")
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Work directory: {FLAGS.workdir}")
    # Run the training pipeline
    run_lib.train(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == "eval":
    # Create the working directory if it doesn't exist
    os.makedirs(FLAGS.workdir, exist_ok=True)
    log_file = os.path.join(FLAGS.workdir, 'eval_stdout.txt')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting evaluation mode")
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Config file: {FLAGS.config}")
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Work directory: {FLAGS.workdir}")
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Eval folder: {FLAGS.eval_folder}")
    # Run the evaluation pipeline
    run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
