# coding=utf-8
# Copyright 2018 Google LLC & Hwalsuk Lee.
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

"""Evaluation for GAN tasks."""
from __future__ import absolute_import
from __future__ import division

import csv
import os

from compare_gan.src import fid_score as fid_score_lib
from compare_gan.src import gan_lib
from compare_gan.src import params

import numpy as np
import tensorflow as tf

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("num_test_examples", 10000,
                     "Number of test examples to evaluate on.")
FLAGS = flags.FLAGS


# This stores statistics for the real data and should be computed only
# once for the whole execution.
MU_REAL, SIGMA_REAL = None, None

# Special value returned when fake image generated by GAN has nans.
NAN_DETECTED = 31337.0

# Special value returned when FID code returned exception.
FID_CODE_FAILED = 4242.0

# If the given param was not specified in the model, use this default.
# This is mostly for COLAB, which tries to automatically infer the type
# of the column
DEFAULT_VALUES = {
    "weight_clipping": -1.0,
    "y_dim": -1,
    "lambda": -1.0,
    "disc_iters": -1,
    "beta1": -1.0,
    "gamma": -1.0,
}

# Inception batch size.
INCEPTION_BATCH = 50


def GetAllTrainingParams():
  all_params = set()
  supported_gans = ["GAN", "GAN_MINMAX", "WGAN", "WGAN_GP",
                    "DRAGAN", "LSGAN", "VAE", "BEGAN"]
  for gan_type in supported_gans:
    for dataset in ["mnist", "fashion-mnist", "cifar10", "celeba"]:
      p = params.GetParameters(gan_type, dataset, "wide")
      all_params.update(p.keys())
  logging.info("All training parameter exported: %s", sorted(all_params))
  return sorted(all_params)


# Fake images are already re-scaled to [0, 255] range.
def GetInceptionScore(fake_images, inception_graph):
  num_images = fake_images.shape[0]
  assert num_images % INCEPTION_BATCH == 0

  with tf.Graph().as_default():
    images = tf.constant(fake_images)
    inception_score_op = fid_score_lib.inception_score_fn(
        images, num_batches=num_images // INCEPTION_BATCH,
        inception_graph=inception_graph)
    with tf.train.MonitoredSession() as sess:
      inception_score = sess.run(inception_score_op)
      return inception_score


# Images must have the same resolution and pixels must be in 0..255 range.
def ComputeTFGanFIDScore(fake_images, real_images, inception_graph):
  """Compute FID score using TF.Gan library."""
  assert fake_images.shape == real_images.shape
  with tf.Graph().as_default():
    fake_images_batch, real_images_batch = tf.train.batch(
        [tf.convert_to_tensor(fake_images, dtype=tf.float32),
         tf.convert_to_tensor(real_images, dtype=tf.float32)],
        enqueue_many=True,
        batch_size=INCEPTION_BATCH)
    eval_fn = fid_score_lib.get_fid_function(
        gen_image_tensor=fake_images_batch,
        eval_image_tensor=real_images_batch,
        num_eval_images=real_images.shape[0],
        image_range="0_255",
        inception_graph=inception_graph)
    with tf.train.MonitoredTrainingSession() as sess:
      fid_score = eval_fn(sess)
  return fid_score


def RunCheckpointEval(checkpoint_path, task_workdir, options, inception_graph):
  """Evaluate model at given checkpoint_path."""

  # Make sure that the same latent variables are used for each evaluation.
  np.random.seed(42)

  checkpoint_dir = os.path.join(task_workdir, "checkpoint")
  result_dir = os.path.join(task_workdir, "result")
  gan_log_dir = os.path.join(task_workdir, "logs")

  gan_type = options["gan_type"]

  supported_gans = ["GAN", "GAN_MINMAX", "WGAN", "WGAN_GP",
                    "DRAGAN", "LSGAN", "VAE", "BEGAN"]
  if gan_type not in supported_gans:
    raise ValueError("Gan type %s is not supported." % gan_type)
  dataset = options["dataset"]

  dataset_content = gan_lib.load_dataset(dataset, split_name="test")
  num_test_examples = FLAGS.num_test_examples

  if num_test_examples % INCEPTION_BATCH != 0:
    logging.info("Padding number of examples to fit inception batch.")
    num_test_examples -= num_test_examples % INCEPTION_BATCH

  # Get real images from the dataset. In the case of a 1-channel
  # dataset (like mnist) convert it to 3 channels.

  data_x = []
  with tf.Graph().as_default():
    get_next = dataset_content.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
      for _ in range(num_test_examples):
        data_x.append(sess.run(get_next[0]))

  real_images = np.array(data_x)
  if real_images.shape[0] != num_test_examples:
    raise ValueError("Not enough examples in the dataset.")

  if real_images.shape[3] == 1:
    real_images = np.tile(real_images, [1, 1, 1, 3])
  real_images *= 255.0
  logging.info("Real data processed.")

  # Get Fake images from the generator.
  samples = []
  logging.info("Running eval on checkpoint path: %s", checkpoint_path)
  with tf.Graph().as_default():
    with tf.Session() as sess:
      gan = gan_lib.create_gan(
          gan_type=gan_type,
          dataset=dataset,
          sess=sess,
          dataset_content=dataset_content,
          options=options,
          checkpoint_dir=checkpoint_dir,
          result_dir=result_dir,
          gan_log_dir=gan_log_dir)

      gan.build_model(is_training=False)

      tf.global_variables_initializer().run()
      saver = tf.train.Saver()
      saver.restore(sess, checkpoint_path)

      # Make sure we have >= examples as in the test set.
      num_batches = int(np.ceil(num_test_examples / gan.batch_size))
      for _ in range(num_batches):
        z_sample = np.random.uniform(-1, 1, size=(gan.batch_size, gan.z_dim))
        feed_dict = {gan.z: z_sample}
        x = sess.run(gan.fake_images, feed_dict=feed_dict)
        # If NaNs were generated, ignore this checkpoint and assign a very high
        # FID score which we handle specially later.
        while np.isnan(x).any():
          logging.error("Detected NaN in fake_images! Returning NaN.")
          return NAN_DETECTED, NAN_DETECTED
        samples.append(x)

  fake_images = np.concatenate(samples, axis=0)
  # Adjust the number of fake images to the number of images in the test set.
  fake_images = fake_images[:num_test_examples, :, :, :]
  # In case we use a 1-channel dataset (like mnist) - convert it to 3 channel.
  if fake_images.shape[3] == 1:
    fake_images = np.tile(fake_images, [1, 1, 1, 3])
  fake_images *= 255.0
  logging.info("Fake data processed, computing inception score.")
  inception_score = GetInceptionScore(fake_images, inception_graph)
  logging.info("Inception score computed: %.3f", inception_score)

  assert fake_images.shape == real_images.shape

  fid_score = ComputeTFGanFIDScore(fake_images, real_images, inception_graph)

  logging.info("Frechet Inception Distance for checkpoint %s is %.3f",
               checkpoint_path, fid_score)
  return inception_score, fid_score


def RunTaskEval(options, task_workdir, inception_graph, out_file="scores.csv"):
  """Evaluates all checkpoints for the given task."""
  # If the output file doesn't exist, create it.
  csv_header = [
      "checkpoint_path", "model", "dataset",
      "tf_seed", "inception_score", "fid_score", "sample_id"]
  train_params = GetAllTrainingParams()
  csv_header.extend(train_params)

  scores_path = os.path.join(task_workdir, out_file)
  if not tf.gfile.Exists(scores_path):
    with tf.gfile.Open(scores_path, "w") as f:
      writer = csv.writer(f)
      writer.writerow(csv_header)

  # Get the list of records that were already computed, to not re-do them.
  finished_checkpoints = set()
  with tf.gfile.Open(scores_path, "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header.
    for row in reader:
      finished_checkpoints.add(row[0])

  # Compute all records not done yet.
  with tf.gfile.Open(scores_path, "a") as f:
    writer = csv.writer(f)
    checkpoint_dir = os.path.join(task_workdir, "checkpoint")
    # Fetch checkpoint to eval.
    checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)

    all_checkpoint_paths = checkpoint_state.all_model_checkpoint_paths
    for checkpoint_path in all_checkpoint_paths:
      if checkpoint_path in finished_checkpoints:
        logging.info("Skipping already computed path: %s", checkpoint_path)
        continue

      # Write the FID score and all training params.
      inception_score, fid_score = RunCheckpointEval(
          checkpoint_path, task_workdir, options, inception_graph)
      logging.info("Fid score: %f", fid_score)
      tf_seed = str(options.get("tf_seed", -1))
      sample_id = str(options.get("sample_id", -1))
      output_row = [
          checkpoint_path, options["gan_type"], options["dataset"],
          tf_seed, "%.3f" % inception_score, "%.3f" % fid_score, sample_id]
      for param in train_params:
        if param in options:
          output_row.append(options[param])
        else:
          output_row.append(str(DEFAULT_VALUES[param]))
      writer.writerow(output_row)

      f.flush()