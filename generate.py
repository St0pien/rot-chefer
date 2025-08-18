import cv2
import torch
import matplotlib.cm as cm
import numpy as np
from utils.imagenet_labelmap import get_labelmap

import timm

import os
from PIL import Image
from tqdm import tqdm
from parsing import parsing as org_parsing
from utils.vis_utils import batch_normalize_map
from utils.imagenet_misc import prediction_imagenet_a, prediction_imagenet_r

from utils.get_model import get_model
from attributions import get_attribution
from utils.get_logger import logger

def parsing():
    parser = org_parsing()
    parser.add_argument('--cut-off', '--cut_off', type=int, default=-1, help='Cut off the number of images to process')
    parser.add_argument('--log-interval', '--log_interval', type=int, default=10, help='Interval for logging progress')
    parser.add_argument('--save-path', '--save_path', type=str, default='results/iba', help='Path to save the results')
    parser.add_argument('--img-size', type=int, default=224)
    
    return parser.parse_args()
    

def loadImage(imagePath, imageSize): 
    # Load image and convert to RGB
    # Preprocess is done by timm transformation function
    image = Image.open(imagePath).convert('RGB')
    return image

def saveMapWithColorMap(filename, map, image):
    cmap = cm.jet_r(map)[..., :3] * 255.0
    map = (cmap.astype(float) + image.astype(float) * 255) / 2
    if not cv2.imwrite(filename, np.uint8(map)):
        print('sibar')

def get_image_list(args):
    label_map_file = open(os.path.join(args.data_path, 'imagenet_labelmap.txt'), 'r')
    lines = label_map_file.readlines()
    key_map = dict()
    for line in lines:
        cls_str, _, cls_name = line.replace('\n', '').split(' ')
        key_map[cls_name] = cls_str
    original_labelmap = get_labelmap()
    holder = dict()
    for k, v in original_labelmap.items():
        holder[k] = v.split(',')[0]

    image_map = dict()
    for i in range(1000):
        image_map[i] = os.listdir(os.path.join(args.data_path, 'val', key_map[holder[i]]))
    
    return image_map, key_map, holder

def computeAndSaveMaps(args):
    model, args.input_size = get_model(args.network, args)
    model.eval()
    model = model.cuda()
    
    cam = get_attribution(args.xai, model, args)
    image_map, key_map, label_map = get_image_list(args)

    original_model, args.input_size = get_model(args.network, args)
    original_model.eval()
    original_model = original_model.cuda()
    
    if args.cut_off < 0:
        args.cut_off = 1000
    else:
        logger.info('[Setting] Cut off: {}'.format(args.cut_off))
        
    pbar = tqdm(range(args.cut_off), total=args.cut_off)
    
    config = timm.data.resolve_data_config({}, model=model)
    model_transform = timm.data.transforms_factory.create_transform(**config)
    image_transform = timm.data.transforms_factory.create_transform(**config)
    args.input_size = config['input_size'][1]
    image_transform.transforms = image_transform.transforms[:-2]
    logger.info("[Setting] Image Transform: {}".format(image_transform))
    
    ces = 0; ci = 0
    scores = {'pos': [], 'neg': []}
    
    logger.info('[Setting] Data path: {}'.format(args.data_path))
    logger.info("Model Transform: {}".format(model_transform))
    logger.info('[Setting] Save path: {}'.format(args.save_path))
    
    for idx in pbar:
        
        if idx >= args.cut_off:
            logger.info('Done at {}'.format(args.cut_off))
            break
        
        cls = key_map[label_map[idx]]
        savedir = '{}_{}'.format(key_map[label_map[idx]], label_map[idx])
        
        for c, file in enumerate(image_map[idx]):
            result = 'T'
            os.makedirs(os.path.join(args.save_path, savedir), exist_ok=True)

            image = loadImage(os.path.join(args.data_path, 'val', cls, file), imageSize=args.input_size)
            rawImage = image.copy()
            image = model_transform(image).unsqueeze(dim=0).cuda()
            
            orig_pred = torch.nn.functional.softmax(original_model(image))
            if 'imagenet-a' in args.data_path:
                orig_pred = prediction_imagenet_a(orig_pred)
            elif 'imagenet-r' in args.data_path:
                orig_pred = prediction_imagenet_r(orig_pred)
                
            preds = orig_pred.argmax(dim=1)
            preds = preds.item()
        
            if 'iba' in args.xai:
                saliencyMap = cam(image, class_idx=idx, return_type='acc', wonorm=False, image_size=(args.input_size, args.input_size))
            else:
                saliencyMap = cam(image, class_idx=idx, return_type=None, wonorm=False, image_size=(args.input_size, args.input_size))
                
            preds = orig_pred.argmax(dim=1).item()
            if len(saliencyMap.shape) == 3:
                saliencyMap = saliencyMap.unsqueeze(0)
            elif len(saliencyMap.shape) == 2:
                saliencyMap = saliencyMap.unsqueeze(0).unsqueeze(0)
            saliencyMap = batch_normalize_map(saliencyMap, image_size=(args.input_size, ) * 2 if saliencyMap.shape[-1] != args.input_size else None)
            
            if preds != idx:
                result = 'F'

            rawImage = np.array(image_transform(rawImage)) / 255
            name, ext = file.split('.')
            saliencyMap = saliencyMap.detach().cpu().squeeze(0).squeeze(0)
            saveMapWithColorMap(os.path.join(args.save_path, savedir, '{}_{}_{}_{:.4f}.{}'.format(name, result, preds, orig_pred[0, idx].item(), ext)), saliencyMap, rawImage)

            ci += 1

            pbar.set_description('{} / {}'.format(c, len(image_map[idx])))


def get_label_list(args):
    label_map_file = open('data/imagenet/imagenet_labelmap.txt', 'r')
    # label_map_file = open(os.path.join(args.data_path, 'imagenet_labelmap.txt'), 'r')
    
    lines = label_map_file.readlines()
    key_map = dict()
    for line in lines:
        cls_str, _, cls_name = line.replace('\n', '').split(' ')
        key_map[cls_str] = cls_name
    original_labelmap = get_labelmap()
    holder = dict()
    for k, v in original_labelmap.items():
        holder[v.split(',')[0]] = k

    image_map = dict()
    
    return image_map, key_map, holder


if __name__ == '__main__':
    args = parsing()
    computeAndSaveMaps(args)