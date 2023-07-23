from ml_collections import ConfigDict


def get_config():
  """Experiment configuration."""
  config = ConfigDict()

  # Data.
  config.dataset = 'imagenet'
  config.downsample = True
  config.downsample_res = 64
  config.resolution = [256, 256]
  config.random_channel = True

  # Training.
  config.batch_size = 24
  config.max_train_steps = 300000
  config.save_checkpoint_secs = 900
  config.num_epochs = -1
  config.polyak_decay = 0.999
  config.eval_num_examples = 20000
  config.eval_batch_size = 16
  config.eval_checkpoint_wait_secs = -1

  config.optimizer = ConfigDict()
  config.optimizer.type = 'rmsprop'
  config.optimizer.learning_rate = 3e-4

  # Model.
  config.model = ConfigDict()
  config.model.hidden_size = 512
  config.model.ff_size = 512
  config.model.num_heads = 4
  config.model.num_encoder_layers = 4
  config.model.resolution = [64, 64]
  config.model.name = 'color_upsampler'

  config.sample = ConfigDict()
  config.sample.gen_data_dir = ''
  config.sample.log_dir = 'samples_sweep'
  config.sample.batch_size = 1
  config.sample.mode = 'argmax'
  config.sample.num_samples = 1
  config.sample.num_outputs = 1
  config.sample.skip_batches = 0
  config.sample.gen_file = 'gen0'

  return config
