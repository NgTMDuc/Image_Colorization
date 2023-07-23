
import tensorflow as tf


def change_resolution(image, res, method='area'):
  image = tf.image.resize(image, method=method, antialias=True,
                          size=(res, res))
  image = tf.cast(tf.round(image), dtype=tf.int32)
  return image


def downsample_and_upsample(x, train, downsample_res, upsample_res, method):
  """Downsample and upsample."""
  keys = ['targets']
  if train and 'targets_slice' in x.keys():
    keys += ['targets_slice']

  for key in keys:
    inputs = x[key]
    # Conditional low resolution input.
    x_down = change_resolution(inputs, res=downsample_res, method=method)
    x['%s_%d' % (key, downsample_res)] = x_down

    # We upsample here instead of in the model code because some upsampling
    # methods are not TPU friendly.
    x_up = change_resolution(x_down, res=upsample_res, method=method)
    x['%s_%d_up_back' % (key, downsample_res)] = x_up
  return x


def random_channel_slice(x):
  random_channel = tf.random.uniform(
      shape=[], minval=0, maxval=3, dtype=tf.int32)
  targets = x['targets']
  res = targets.shape[1]
  image_slice = targets[Ellipsis, random_channel: random_channel+1]
  image_slice.set_shape([res, res, 1])
  x['targets_slice'] = image_slice
  x['channel_index'] = random_channel
  return x

