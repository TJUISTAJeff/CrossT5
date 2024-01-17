from configparser import ConfigParser
import sys, os

sys.path.append('..')


# import models

class Configurable(object):
    def __init__(self, config_file, extra_args=[]):
        config = ConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
        self._config = config
        config.write(open(self.config_file, 'w'))
        print('Loaded config file sucessfully.')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

    def check_dirs(self):
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.isdir(self.generation_dir):
            os.makedirs(self.generation_dir)

    @property
    def pretrained_embeddings_file(self):
        return self._config.get('Data', 'pretrained_embeddings_file')

    @property
    def data_dir(self):
        return self._config.get('Data', 'data_dir')

    @property
    def middle_res_dir(self):
        return self._config.get('Data','middle_res_dir')
        return
    @property
    def train_file(self):
        return self._config.get('Data', 'train_file')

    @property
    def dev_file(self):
        return self._config.get('Data', 'dev_file')

    @property
    def test_file(self):
        return self._config.get('Data', 'test_file')

    # @property
    # def min_occur_count(self):
    #     return self._config.getint('Data', 'min_occur_count')

    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @property
    def config_file(self):
        return self._config.get('Save', 'config_file')

    @property
    def generation_dir(self):
        return self._config.get('Save', 'generation_dir')

    # @property
    # def save_model_path(self):
    #     return self._config.get('Save', 'save_model_path')

    # @property
    # def save_vocab_path(self):
    #     return self._config.get('Save', 'save_vocab_path')

    # @property
    # def load_dir(self):
    #     return self._config.get('Save', 'load_dir')

    # @property
    # def load_model_path(self):
    #     return self._config.get('Save', 'load_model_path')

    # @property
    # def load_vocab_path(self):
    #     return self._config.get('Save', 'load_vocab_path')
    #
    # @property
    # def lstm_layers(self):
    #     return self._config.getint('Network', 'lstm_layers')
    #
    @property
    def word_dims(self):
        return self._config.getint('Network', 'word_dims')

    @property
    def context_max_len(self):
        return self._config.getint('Network', 'context_max_len')

    @property
    def query_max_len(self):
        return self._config.getint('Network', 'query_max_len')

    @property
    def char_seq_max_len(self):
        return self._config.getint('Network', 'char_seq_max_len')

    @property
    def encoder_layers(self):
        return self._config.getint('Network', 'encoder_layers')

    @property
    def decoder_layers(self):
        return self._config.getint('Network', 'decoder_layers')

    @property
    def char_vocab_size(self):
        return self._config.getint('Network', 'char_vocab_size')

    @property
    def num_heads(self):
        return self._config.getint('Network', 'num_heads')

    @property
    def input_dropout(self):
        return self._config.getfloat('Network', 'input_dropout')

    @property
    def train_batch_size(self):
        return self._config.getint('Run', 'train_batch_size')

    @property
    def train_iters(self):
        return self._config.getint('Run', 'train_iters')

    @property
    def accumulation_steps(self):
        return self._config.getint('Run', 'accumulation_steps')

    #
    # @property
    # def dropout_emb(self):
    #     return self._config.getfloat('Network', 'dropout_emb')
    #
    # @property
    # def lstm_hiddens(self):
    #     return self._config.getint('Network', 'lstm_hiddens')
    #
    # @property
    # def dropout_lstm_input(self):
    #     return self._config.getfloat('Network', 'dropout_lstm_input')
    #
    # @property
    # def dropout_lstm_hidden(self):
    #     return self._config.getfloat('Network', 'dropout_lstm_hidden')

    @property
    def learning_rate(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def decay(self):
        return self._config.getfloat('Optimizer', 'decay')

    @property
    def decay_steps(self):
        return self._config.getint('Optimizer', 'decay_steps')

    @property
    def beta_1(self):
        return self._config.getfloat('Optimizer', 'beta_1')

    @property
    def beta_2(self):
        return self._config.getfloat('Optimizer', 'beta_2')

    @property
    def epsilon(self):
        return self._config.getfloat('Optimizer', 'epsilon')

    @property
    def clip(self):
        return self._config.getfloat('Optimizer', 'clip')

    @property
    def test_batch_size(self):
        return self._config.getint('Run', 'test_batch_size')

    @property
    def validate_every(self):
        return self._config.getint('Run', 'validate_every')

    @property
    def save_after(self):
        return self._config.getint('Run', 'save_after')

    @property
    def update_every(self):
        return self._config.getint('Run', 'update_every')

    @property
    def save_every(self):
        return self._config.getint('Run','save_every')

    @property
    def use_cosine(self):
        return self._config.getint('Network', 'use_cosine') == 1

    @property
    def val_batch_size(self):
        return self._config.getint('Run', 'val_batch_size')

    @property
    def beam_size(self):
        return self._config.getint('Run', 'beam_size')

    @property
    def beam_alpha(self):
        return self._config.getfloat('Beam', 'alpha')

    @property
    def beam_beta(self):
        return self._config.getfloat('Beam', 'beta')


