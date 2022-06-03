import argparse
import collections
import json
import os
import pandas as pd

import tqdm
import pickle
from gensim.models.keyedvectors import KeyedVectors
import torch

from everything_at_once.dataset.utils import _tokenize_text, create_text_features
from everything_at_once.model.utils.utils import sim_matrix
from everything_at_once.trainer.clip_utils import _apply_clip_text_model

from everything_at_once import data_loader as module_data
from everything_at_once import model as module_arch

from everything_at_once.metric import RetrievalMetric
from everything_at_once.utils.util import state_dict_data_parallel_fix
from everything_at_once.trainer.utils import _move_to_device, average_embeddings, format_dataloader_output

from parse_config import ConfigParser

W2V_PATH = '.data/GoogleNews-vectors-negative300.bin'
FEATURES_PATH = './data/msrvtt/resnet/msrvtt_jsfusion_test.pkl'
MODEL_PATH = './pretrained_models/everything_at_once_tva/latest_model.pth'


def compute_embed_arr(model, dl, device, metrics, clip_text_model=None):
    '''If embed_arr does not already exist, compute embeddings for entire dataset.'''
    torch.cuda.empty_cache()

    meta_arr = []
    ids_arr = []
    embed_arr = collections.defaultdict(lambda: [])
    mod_shapes = {}
    first_run_flag = True

    with torch.no_grad():
        # complete dataset, can also use a split, .pkl format
        for data in tqdm.tqdm(dl):

            # if first_run_flag:
            #     for k, v in data.items():
            #         if k == 'meta':
            #             continue
            #         mod_shapes[k] = v.shape
            #     first_run_flag = False

            if first_run_flag:
                for k, v in data.items():
                    if k == 'meta':
                        continue
                    if isinstance(v, list):
                        mod_shapes[k] = (len(v), )
                    else:
                        mod_shapes[k] = v.shape
                first_run_flag = False

            data = format_dataloader_output(data)

            meta_arr.append(data['meta'])
            ids_arr.extend(data['meta']['ids'])

            # TODO: Link ids with embed_arr

            if clip_text_model is not None:
                data = _apply_clip_text_model(clip_text_model, data, device)

            for field in ['text', 'text_mask', 'video', 'video_mask', 'audio', 'audio_mask', 'audio_STFT_nframes', 'caption', 'image']:
                if field in data:
                    data[field] = _move_to_device(data[field], device)
            embeds = model(data, force_cross_modal=True)
            for name, embed in embeds.items():
                if '_embed' in name:
                    embed_arr[name].append(embed.cpu())

            # loss computation removed for inference based functions

            del data, embeds

    # nested_metrics = {}

    for name, embed in embed_arr.items():
        embed_arr[name] = torch.cat(embed, dim=0)
        # print(embed_arr[name].shape)

    # needed for 'cut_clips: true' ablation
    embed_arr = average_embeddings(ids_arr, embed_arr, verbose=True)

    json.dump(mod_shapes, open('./results/mod_shapes.json', 'w'))

    return embed_arr


def process_single_text(query, we, max_words=20, we_dim=300):
    # tokenize and get text features
    words = _tokenize_text(query)
    text, text_mask, raw_text = create_text_features(
        words, max_words, we, we_dim)

    return text, text_mask, raw_text


def embed_single_text(text, text_mask, raw_text, model, device):
    '''send in a text in the format of the eav model, pad for video and audio modalities,
    get text embeddings in the form of multimodal embeddings with single filled embedding'''

    text_data = {'text': text, 'text_mask': text_mask, 'raw_text': raw_text}
    mod_shapes = json.load(open('./results/mod_shapes.json'))
    for k, v in mod_shapes.items():
        if k not in text_data.keys():
            text_data[k] = torch.zeros(v)

    # text_data = {'text': text, 'text_mask': text_mask, 'raw_text': raw_text, 'video': None, 'video_mask': None,
    #              'audio': None, 'audio_mask': None, 'audio_STFT_nframes': None, 'unroll_clips': None, 'meta':
    #                  {'paths': 'inf_path', 'ids': 'inf', 'dataset': 'MSRVTT'}}
    # torch.empty()

    text_data['text'] = _move_to_device(text_data['text'], device)
    text_data['text_mask'] = _move_to_device(text_data['text_mask'], device)

    text_embed = model(text_data)  # force_cross_modal=False

    text_embed = torch.cat(text_embed, dim=0)

    text_embed = average_embeddings(['inf_text'], text_embed, verbose=True)
    return text_embed


def similarity_computation(text_embed, embed_arr):
    # requires computation of entire dataset, can do out of distribution inference after that

    sims = {}
    #  text_embed: embed1: inference embed of text
    for embed_name, embed in embed_arr.items():
        # if text is second embed name or already computed, skip
        if 'text' in embed_name or f't2{embed_name}' in sims:
            continue

        # get tensor similarity matrix, convert to numpy
        sims[f't2{embed_name}'] = sim_matrix(
            text_embed, embed).detach().cpu().numpy()

    return sims


def run_inference_arg(inf_in, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model architecture
    if config['trainer'].get("use_clip_text_model", False):
        import clip
        clip_text_model, _ = clip.load("ViT-B/32", device=device)
        clip_text_model.eval()
    else:
        clip_text_model = None

    model = config.initialize('arch', module_arch)

    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)

    metrics = [RetrievalMetric(met) for met in config['metrics']]

    # load torch pretrained model, config params
    checkpoint = torch.load(config.resume, map_location=device)
    epoch = checkpoint['epoch']

    # load state_dict, then fix comparing with newly created model state_dict
    state_dict = checkpoint['state_dict']
    new_state_dict = state_dict_data_parallel_fix(
        state_dict, model.state_dict())
    model.load_state_dict(new_state_dict, strict=True)

    # prepare model for testing
    model = model.to(device)
    model.eval()  # eval mode pt

    # inf_in
    # metrics, vid_id, co_sim
    embed_arr = compute_embed_arr(model, data_loader, device, metrics,
                                  clip_text_model=clip_text_model)

    # word2vec path from config
    # word2vec_path = dataset_kwargs.pop('word2vec_path')
    we = data_loader.we
    # word2vec_path = '.data/GoogleNews-vectors-negative300.bin'
    # if we is None:
    #     we = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    # else:
    #     print('Using loaded we')

    text, text_mask, raw_text = process_single_text(inf_in, we)
    text_embed = embed_single_text(text, text_mask, raw_text, model, device)

    inf_sims = similarity_computation(text_embed, embed_arr)

    if not os.path.exists('results/'):
        os.mkdir('results/')
    curr_no = sorted(os.listdir('.output/'))[-1].split('out')[1].split('.')[0]
    pd.DataFrame(inf_sims).to_csv(f"test_inf_out{curr_no}.csv")

    # print top 5 most similar

    return inf_sims

    # eval code
    # nested_metrics, val_loss, val_loss_detailed = eval(model, data_loader, device, metrics,
    #                                                 loss_func=None,
    #                                                 clip_text_model=clip_text_model)

    # short_verbose(epoch=epoch, dl_nested_metrics=nested_metrics,
    #               dataset_name=data_loader.dataset_name)
    # for metric in metrics:
    #     metric_name = metric.__name__
    #     res = nested_metrics[metric_name]
    #     verbose(epoch=epoch, metrics=res, name="", mode=metric_name)


def main():

    args = argparse.ArgumentParser(description='PyTorch Template')
    # TODO: Remove argparse for API method inferencing, create config dict manually from input instead
    args.add_argument('-r', '--resume', required=True, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-c', '--config', default=True, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    # n_gpu = torch.cuda.device_count()

    # taking in inference text
    inf_in_path = './data/test/text_queries/'
    inf_in = os.path.join(inf_in_path, 'query_1.txt')
    inf_in = "The quick brown fox jumps over the lazy dog"

    dataset_ch = 'msrvtt'  # 'msrvtt' 'youcook'

    # custom cli options to modify configuration from default values given in json file.
    config = ConfigParser(args, test=True)
    args = args.parse_args()

    # TODO: Switch to infer from arg_infer
    # infer(inf_in, dataset_ch='') # , clip_text=False
    run_inference_arg(inf_in, config)


if __name__ == '__main__':
    main()


# # goes in trainer.py with eval()
# def infer(inf_in, model, dl, device, metrics, loss_func=None, clip_text_model=None):
#     """Run single input inference using loaded model.

#     Args:
#         inf_in (str): Path to inference input file.
#         model (nn.Module): Loaded model to run inference on.
#         dl (torch.utils.data.DataLoader): DataLoader for inference data.
#         device (torch.device): Device to run inference on.
#         metrics (list): List of metrics to run inference with.
#         loss_func (nn.Module): Loss function to run inference with.
#         clip_text_model (nn.Module): Clip text model to run inference with.
#     """
#     torch.cuda.empty_cache()

#     # total_val_loss = 0
#     # total_val_loss_detailed = collections.defaultdict(lambda: 0)
#     # meta_arr = []  # maybe keep for top-n results?
#     ids_arr = []
#     # embed_arr = collections.defaultdict(lambda: [])

#     #
#     with torch.no_grad():

#         # TODO: Structure
#         # ? data loading manually? single instance/point dataloading
#         #! create text features: dataset.utils
#         # ^ follow msrvtt_dataset
#         # ^ we -> word2vec from path
#         # caption = self.data[idx]['eval_caption']
#         # idx?
#         # returns dict {'video': video, ... 'text': text}

#         #! format dataloader output after this

#         #! data to_device

#         #! get embeds by using model() on data
#         # ? cross_modal? what happens if false
#         # seemed to accomodate for single modality

#         # _embed in embed.name?

#         # loss
#         # loss, loss_info = loss_func(embed)

#         # embed array, can make an array of a single embed instead?
#         # dont need to average embeddings, or run it on single

#         # sims (cosine similarity)
#         # text_embed2video_embed/audio_embed
#         # -> t2v/a
#         # sim matrix for single w all in dataset?
#         # ? how does sim matrix work

#         # metrics
#         # ? metric(sims, size=1?) what does this do
#         # try doing this manually for 1 embed input

#         # format data
#         # TODO: Change format_dataloader_output for single instance inference
#         data = format_data_item(inf_in)

#         if clip_text_model is not None:
#             data = _apply_clip_text_model(clip_text_model, data, device)

#         # considiering only text input
#         #! need to send in the entire dataloader?
#         data = _move_to_device(data, device)

#         embeds = model(data)


# def infer_load(inf_in, model_path, config_path, clip_text=False):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     if clip_text:
#         import clip
#         clip_text_model, _ = clip.load("ViT-B/32", device=device)
#         clip_text_model.eval()
#     else:
#         clip_text_model = None

#     # model =
#     pass
