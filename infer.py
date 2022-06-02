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


# goes in trainer.py with eval()
def infer(inf_in, model, dl, device, metrics, loss_func=None, clip_text_model=None):
    """Run single input inference using loaded model.

    Args:
        inf_in (str): Path to inference input file.
        model (nn.Module): Loaded model to run inference on.
        dl (torch.utils.data.DataLoader): DataLoader for inference data.
        device (torch.device): Device to run inference on.
        metrics (list): List of metrics to run inference with.
        loss_func (nn.Module): Loss function to run inference with.
        clip_text_model (nn.Module): Clip text model to run inference with.
    """
    torch.cuda.empty_cache()

    # total_val_loss = 0
    # total_val_loss_detailed = collections.defaultdict(lambda: 0)
    # meta_arr = []  # maybe keep for top-n results?
    ids_arr = []
    # embed_arr = collections.defaultdict(lambda: [])
    
    # 
    with torch.no_grad():
        
        # TODO: Structure
        #? data loading manually? single instance/point dataloading
        #! create text features: dataset.utils
        # ^ follow msrvtt_dataset
        # ^ we -> word2vec from path
        # caption = self.data[idx]['eval_caption']
        # idx?
        # returns dict {'video': video, ... 'text': text}
        
        #! format dataloader output after this
        
        #! data to_device
        
        #! get embeds by using model() on data
        #? cross_modal? what happens if false
        # seemed to accomodate for single modality
        
        # _embed in embed.name?
        
        # loss
        # loss, loss_info = loss_func(embed)
        
        # embed array, can make an array of a single embed instead?
        # dont need to average embeddings, or run it on single
        
        # sims (cosine similarity)
        # text_embed2video_embed/audio_embed
        # -> t2v/a
        # sim matrix for single w all in dataset?
        #? how does sim matrix work
        
        # metrics
        #? metric(sims, size=1?) what does this do
        # try doing this manually for 1 embed input
        
        
        # format data
        # TODO: Change format_dataloader_output for single instance inference
        data = format_data_item(inf_in)
        
        if clip_text_model is not None:
                data = _apply_clip_text_model(clip_text_model, data, device)

        # considiering only text input
        #! need to send in the entire dataloader?
        data = _move_to_device(data, device)
        
        embeds = model(data)
        
        


def infer_load(inf_in, model_path, config_path, clip_text=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if clip_text:
        import clip
        clip_text_model, _ = clip.load("ViT-B/32", device=device)
        clip_text_model.eval()
    else:
        clip_text_model = None

    model = 
    pass


def arg_infer_load(inf_in, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model architecture
    if config['trainer'].get("use_clip_text_model", False):
        import clip
        clip_text_model, _ = clip.load("ViT-B/32", device=device)
        clip_text_model.eval()
    else:
        clip_text_model = None

    #! initialize needs to be changed to remove args OR keep and use config dict
    # kwargs: config file
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

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    model = model.to(device)
    model.eval()  # eval mode pt

    metrics, vid_id, co_sim = infer(inf_in, model, data_loader, device, metrics,
                                    loss_func=None, clip_text_model=clip_text_model)

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

    # TODO: Switch to infer from arg_infer
    # infer(inf_in, dataset_ch='') # , clip_text=False
    arg_infer_load(inf_in, config)


if __name__ == '__main__':
    main()
