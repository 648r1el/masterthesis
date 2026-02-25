import argparse
import os

from params.basic_params import BasicParams
from params.train_test_params import TrainTestParams


class LoadParams:

    def __init__(self, experiment_name: str, epochs: int, class_num: int, checkpoints_dir: str = './checkpoints',
                 use_gpus: bool = False):
        self._param = argparse.Namespace()
        self._INT_PARAMS = {
            'batch_size',
            'class_num',
            'conv_k_size',
            'decay_step_size',
            'epoch_count',
            'epoch_num_decay',
            'epoch_num_p1',
            'epoch_num_p2',
            'epoch_num_p3',
            'filter_num',
            'latent_space_dim',
            'num_threads',
            'print_freq',
            'save_epoch_freq',
            'seed'
        }
        self._FLOAT_PARAMS = {
            'beta1',
            'dropout_p',
            'init_gain',
            'k_embed',
            'k_kl',
            'leaky_slope',
            'lr',
            'weight_decay'
        }
        self._BOOL_PARAMS = {
            'continue_train',
            'detail',
            'detect_na',
            'deterministic',
            'isTest',
            'isTrain',
            'not_stratified',
            'save_latent_space',
            'save_model',
            'set_pin_memory',
            'use_feature_lists',
            'use_sample_list'
        }
        self._checkpoints_dir = checkpoints_dir
        self._load_cmd_params(experiment_name, epochs, class_num, use_gpus)

    def _load_cmd_params(self, experiment_name: str, epochs: int, num_classes: int, use_gpus: bool):
        d = {}
        # variable which indicates whether the first lines, which contain unnecessary information and are named header,
        # are passed. The header ends with the Running parameters line which starts with a '-'. Each row before this line is
        # skipped
        header_skipped = False
        # read cmds
        with open(os.path.join(self._checkpoints_dir, experiment_name, 'cmd_parameters.txt')) as f:
            while line := f.readline():
                if line.startswith('-'):
                    header_skipped = not header_skipped
                    continue
                if header_skipped:
                    argument = line.strip().replace(' ', '').split(':')
                    key = argument[0]
                    value = argument[1].split('\t[')[0]
                    if key in self._INT_PARAMS:
                        value = int(value)
                    if key in self._FLOAT_PARAMS:
                        value = float(value)
                    if key in self._BOOL_PARAMS:
                        value = bool(value)
                    d[key] = value
            # adjust the attributes whose values differ from what is written in the file
            d['continue_train'] = True
            d['gpu_ids'] = [int(gpu_id) for gpu_id in d['gpu_ids'].split(',') if int(gpu_id) >= 0 and use_gpus]
            d['epoch_to_load'] = str(epochs)
            d['experiment_to_load'] = experiment_name
            d['class_num'] = num_classes if d['class_num'] == 0 else d['class_num']
            d['ch_separate'] = False
            d['checkpoints_dir'] = self._checkpoints_dir
            # add arguments
            d['epoch_num'] = d['epoch_num_p1'] + d['epoch_num_p2'] + d['epoch_num_p3']
            self._param.__dict__.update(d)

    def set_omics_dims(self, omics_dfs: list):
        self._param.omics_dims = [len(omics_df) for omics_df in omics_dfs]
        self._param.omics_num = len(omics_dfs)

    def get_params(self):
        return self._param