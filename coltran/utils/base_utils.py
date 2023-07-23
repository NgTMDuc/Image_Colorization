from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf


def nats_to_bits(nats):
  return nats / np.log(2)


def act_to_func(act):
  cond_act_map = {
      'relu': tf.nn.relu,
      'sigmoid': tf.math.sigmoid,
      'tanh': tf.math.tanh,
      'identity': lambda x: x}
  return cond_act_map[act]


def roll_channels_to_batch(tensor):
  # Switch from [B, H, W, C, D] to [B, C, H, W, D]
  return tf.transpose(tensor, perm=[0, 3, 1, 2, 4])


def roll_channels_from_batch(tensor):
  # Switch from [B, C, H, W, D] to [B, H, W, C, D]
  return tf.transpose(tensor, perm=[0, 2, 3, 1, 4])


def image_to_hist(image, num_symbols):

  _, height, width, channels = image.shape
  image = tf.one_hot(image, depth=num_symbols)
  image = tf.reshape(image, shape=[-1, height*width, channels, num_symbols])
  # average spatially.
  image = tf.reduce_mean(image, axis=1)

  # smooth
  eps = 1e-8
  image = (image + eps) / (1 + num_symbols*eps)
  return image


def get_bw_and_color(inputs, colorspace):

  if colorspace == 'rgb':
    grayscale = tf.image.rgb_to_grayscale(inputs)
  elif colorspace == 'ycbcr':
    inputs = rgb_to_ycbcr(inputs)
    grayscale, inputs = inputs[Ellipsis, :1], inputs[Ellipsis, 1:]
  return grayscale, inputs


def rgb_to_ycbcr(rgb):
  """Map from RGB to YCbCr colorspace."""
  rgb = tf.cast(rgb, dtype=tf.float32)
  r, g, b = tf.unstack(rgb, axis=-1)
  y = r * 0.299 + g * 0.587 + b * 0.114
  cb = r * -0.1687 - g * 0.3313 + b * 0.5
  cr = r * 0.5 - g * 0.4187 - b * 0.0813
  cb += 128.0
  cr += 128.0

  ycbcr = tf.stack((y, cb, cr), axis=-1)
  ycbcr = tf.clip_by_value(ycbcr, 0, 255)
  ycbcr = tf.cast(ycbcr, dtype=tf.int32)
  return ycbcr


def ycbcr_to_rgb(ycbcr):
  """Map from YCbCr to Colorspace."""
  ycbcr = tf.cast(ycbcr, dtype=tf.float32)
  y, cb, cr = tf.unstack(ycbcr, axis=-1)

  cb -= 128.0
  cr -= 128.0

  r = y * 1. + cb * 0. + cr * 1.402
  g = y * 1. - cb * 0.34414 - cr * 0.71414
  b = y * 1. + cb * 1.772 + cr * 0.

  rgb = tf.stack((r, g, b), axis=-1)
  rgb = tf.clip_by_value(rgb, 0, 255)
  rgb = tf.cast(rgb, dtype=tf.int32)
  return rgb


def convert_bits(x, n_bits_out=8, n_bits_in=8):
  """Quantize / dequantize from n_bits_in to n_bits_out."""
  if n_bits_in == n_bits_out:
    return x
  x = tf.cast(x, dtype=tf.float32)
  x = x / 2**(n_bits_in - n_bits_out)
  x = tf.cast(x, dtype=tf.int32)
  return x


def get_patch(upscaled, window, normalize=True):
  """Extract patch of size from upscaled.shape[1]//window from upscaled."""
  upscaled = tf.cast(upscaled, dtype=tf.float32)

  # pool + quantize + normalize
  patch = tf.nn.avg_pool2d(
      upscaled, ksize=window, strides=window, padding='VALID')

  if normalize:
    patch = tf.cast(patch, dtype=tf.float32)
    patch /= 256.0
  else:
    patch = tf.cast(patch, dtype=tf.int32)
  return patch


def labels_to_bins(labels, num_symbols_per_channel):

  labels = tf.cast(labels, dtype=tf.float32)
  channel_hash = [num_symbols_per_channel**2, num_symbols_per_channel, 1.0]
  channel_hash = tf.constant(channel_hash)
  labels = labels * channel_hash

  labels = tf.reduce_sum(labels, axis=-1)
  labels = tf.cast(labels, dtype=tf.int32)
  return labels


def bins_to_labels(bins, num_symbols_per_channel):
 
  labels = []
  factor = int(num_symbols_per_channel**2)

  for _ in range(3):
    channel = tf.math.floordiv(bins, factor)
    labels.append(channel)

    bins = tf.math.floormod(bins, factor)
    factor = factor // num_symbols_per_channel
  return tf.stack(labels, axis=-1)

