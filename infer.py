import argparse
import os
import torch

from everything_at_once import data_loader as module_data
from everything_at_once import model as module_arch

from everything_at_once.metric import RetrievalMetric
from everything_at_once.utils.util import state_dict_data_parallel_fix

from parse_config import ConfigParser

W2V_PATH = '.data/GoogleNews-vectors-negative300.bin'
FEATURES_PATH = './data/msrvtt/resnet/msrvtt_jsfusion_test.pkl'
MODEL_PATH = './pretrained_models/everything_at_once_tva/latest_model.pth'


def infer(inf_in, model_path, config_path, clip_text=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if clip_text:
        import clip
        clip_text_model, _ = clip.load("ViT-B/32", device=device)
        clip_text_model.eval()
    else:
        clip_text_model = None

    model =
    pass


def arg_infer(inf_in, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model architecture
    if config['trainer'].get("use_clip_text_model", False):
        import clip
        clip_text_model, _ = clip.load("ViT-B/32", device=device)
        clip_text_model.eval()
    else:
        clip_text_model = None

    #! initialize needs to be changed to remove args
    # kwargs: config file
    model = config.initialize('arch', module_arch)

    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)

    metrics = [RetrievalMetric(met) for met in config['metrics']]


def main():

    # TODO: Remove argparse for API method inferencing, create config dict manually from input instead
    args.add_argument('-r', '--resume', required=True, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-c', '--config', default=True, type=str,
                      help='config file path (default: None)')

    # n_gpu = torch.cuda.device_count()

    # taking in inference text
    inf_in_path = './data/test/text_queries/'
    inf_in = os.path.join(inf_in_path, 'query_1.txt')

    dataset_ch = 'msrvtt'  # 'msrvtt' 'youcook'

    # custom cli options to modify configuration from default values given in json file.
    config = ConfigParser(args, test=True)
    args = args.parse_args()
    ``
    # TODO: Switch to infer from arg_infer
    # infer(inf_in, dataset_ch='') # , clip_text=False
    arg_infer(inf_in, config)


if __name__ == '__main__':
    main()
