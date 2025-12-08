import os
import numpy as np
import torch
import torchvision
from .get_transform import get_transf
from .get_logger import logger
from IBA.estimator import TorchWelfordEstimator


def parse_iba_op(name, args):
    if args.network == 'dino':
        if name == 'blocks':
            return 'model.blocks.{}'
        return 'model.blocks.{}.' + name
    elif 'swin' in args.network:
        swin_index = args.swin_index.split(',')
        return 'layers.' + str(swin_index[0]) + '.blocks.' + str(swin_index[1]) + '.' + name
    elif 'resnet' in args.network:
        return 'layer{}'
    elif 'vgg' in args.network:
        return 'features.{}'
    elif name == 'blocks':
        return 'blocks.{}'
    elif name == 'features':
        return 'features.{}'
    elif name == 'layer':
        return 'layer{}'
    else:
        return 'blocks.{}.' + name 
    

def iba_layers(args, start_key='blocks'):
    parsed_ops = [parse_iba_op(opname, args) for opname in args.op_name]
    args.target_layer = []
    
    parsed_ops = sorted(parsed_ops)
    
    for parsed_op in parsed_ops:
        args.target_layer += [parsed_op.format(i) for i in range(args.start_target, args.end_target)]
    
    return args


def ifnone(a, b):
    """If a is None return b."""
    if a is None:
        return b
    else:
        return a


def save_estimator(args, target_layers, iba, model):
    model_transform = get_transf(args, model)
    
    logger.info('Transformation for data estimation: {}'.format(model_transform))
    
    if 'train' in args.data_path:
        trainset = torchvision.datasets.ImageFolder(
            # args.data_path.replace('val', 'train'),
            os.path.join(args.data_path, 'train'),
            transform = model_transform
        )
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8, drop_last=False)
        loader_ = trainloader
        n_samples = len(loader_.dataset)
    else:
        valset = torchvision.datasets.ImageFolder(
            os.path.join(args.data_path, 'val'),
            transform=model_transform)
        valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True, num_workers=8, drop_last=False)
        loader_ = valloader
        n_samples = len(loader_.dataset)
    iba.estimate(model, loader_, device=args.rank, n_samples=n_samples)
    os.makedirs('{}/{}'.format(args.iba_path, args.network), exist_ok=True)
    logger.info('Estimations: saved at {}/{}'.format(args.iba_path, args.network))

    for i, target_layer in enumerate(target_layers):
        torch.save(iba.layer_estimator[target_layer].cpu(), f'{args.iba_path}/{args.network}/layer_{target_layer}.torch')
        iba.layer_estimator[target_layer].cuda(f'cuda:{args.rank}')
        iba.readout_estimators = iba.layer_estimator
        
    return iba

def load_estimator(args, target_layers, iba):
    for i, target_layer in enumerate(target_layers):

        iba.layer_estimator[target_layer] = torch.load(f'{args.iba_path}/{args.network}/layer_{target_layer}.torch', map_location=f'cuda:{args.rank}', weights_only=False).cuda(f'cuda:{args.rank}')
        iba.layer_estimator[target_layer].device = f'cuda:{args.rank}'
    iba._build()
    
    return iba


def get_module_by_name(model, target):

    target_modules = []
    layer_estimator = dict()
    # target_layer = []
    
    for module in model.named_modules():
        if module[0] in target:
            target_modules.append((module[0], module[1]))
            # Addressing definition bug
            layer_estimator[module[0]] = TorchWelfordEstimator()
            # target_layer.append(module[0])
    return target_modules, layer_estimator#, target_layer


def postprocess_heatmap(heatmap):
    heatmap = heatmap[1:, :].nansum(dim=-1)
    heatmap = heatmap.view(1, 1, int(np.sqrt(heatmap.size(-1))), int(np.sqrt(heatmap.size(-1))))
    
    return heatmap


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)    

