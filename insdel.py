# This is a code for evaluating insertion and deletion metrics on images.
# We implement the code based on the Timm library and PyTorch.
import os

import numpy as np

import timm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from kornia.filters.gaussian import gaussian_blur2d
from tqdm import tqdm

from utils.get_model import get_model
from attributions import get_attribution

from parsing import parsing as org_parsing
from utils.get_logger import logger
from utils.iba_utils import fix_random_seeds


__all__ = ['CausalMetric', 'auc']

import warnings
warnings.filterwarnings("ignore")


HW = 224 * 224


def parsing():
    parser = org_parsing()
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training and evaluation')    
    parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--pre-norm', '--pre_norm', action='store_true', help='Whether to normalize images before insertion/deletion')
    parser.add_argument('--cut-off', '--cut_off', type=int, default=0, help='Cut-off value for saliency maps')
    return parser.parse_args()

    
def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class CausalMetric(object):
    def __init__(self, model, mode, step, substrate_fn, norm_fn):
        """Create deletion/insertion metric instance.
        Args:
            model(nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model.eval().cuda()
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        self.norm_fn = norm_fn

    def evaluate(self, img, pred, mask, idx=None, cls_idx=None, b=16, save_to=None, orig_score=None):
        """Run metric on one image-saliency pair.
        Args:
            img (Tensor): normalized image tensor.
            mask (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step.
            save_to (str): directory to save every step plots to.
        Return:
            scores (nd.array): Array containing scores at every step.
        """
        if cls_idx is None:
            cls_idx = pred
        HW = img.size(2) * img.size(3)
        n_steps = (HW + self.step - 1) // self.step
        if self.mode == 'del':
            start = img.clone()
            finish = self.substrate_fn(img)
        elif self.mode == 'ins':
            start = self.substrate_fn(img)
            finish = img.clone()
        scores = np.empty((b, n_steps + 1), dtype='float32')
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(mask.reshape(b, HW), axis=1), axis=-1)

        for i in range(n_steps + 1):
            ### Normalize the image after manipulation to avoid artifacts
            ### Note that the order of results is not changed
            logit = self.model(self.norm_fn(start.cuda()))
            
            score = F.softmax(logit, dim=-1)[torch.arange(0, b), cls_idx]
            if b > 1:
                score = score.squeeze(dim=-1)
            
            for j in range(b):
                scores[j, i] = score[j]
            
            coords = []
            for j in range(b):
                coords = salient_order[j, self.step * i: self.step * (i + 1)]
                start.cpu().numpy().reshape(b, 3, HW)[j, :, coords] = \
                    finish.cpu().numpy().reshape(b, 3, HW)[j, :, coords]

        aucs = np.empty(b, dtype='float32')
        for i in range(b):
            aucs[i] = auc(scores[i].reshape(-1))
    
        return aucs


def main():
    fix_random_seeds()
    args = parsing()
    
    val_dir = os.path.join(args.data_path, 'val')
    batch_size = args.batch_size
    num_workers = args.workers

    model, input_size = get_model(args.network, args)
    model.eval()
    model = model.cuda()

    config = timm.data.resolve_data_config({}, model=model)
    model_transform = timm.data.transforms_factory.create_transform(**config)
    if args.pre_norm:
        norm_fn = lambda x: x
    else:
        norm_fn = model_transform.transforms[-1]
        model_transform.transforms = model_transform.transforms[:-1]

    # Get dataset loader
    Datasets = datasets.ImageFolder(val_dir,
                                    model_transform)
    val_loader = DataLoader(
        Datasets,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Function that blurs input image
    blur = lambda x: gaussian_blur2d(x, kernel_size=(51, 51), sigma=(50., 50.))
    
    # Evaluate a batch of explanations
    insertion = CausalMetric(model, 'ins', int(input_size * 8), substrate_fn=blur, norm_fn=norm_fn)
    deletion = CausalMetric(model, 'del', int(input_size * 8), substrate_fn=torch.zeros_like, norm_fn=norm_fn)

    scores = {'del': [], 'ins': []}
    cam = get_attribution(args.xai, model, args)

    progress = tqdm(val_loader, desc='Explaining Images')
    
    for i, (images, target) in enumerate(progress):
        
        if args.cut_off and i >= args.cut_off:
            logger.info('Cut-off at {} images'.format(args.cut_off))
            break
        
        target = target.cuda()
        images = images.cuda()
        norm_images = norm_fn(images.clone())
        saliency_maps = cam(norm_images, class_idx=target)
        
        # resize saliency maps to match image size
        saliency_maps = F.interpolate(saliency_maps, size=(input_size, input_size), mode='bilinear', align_corners=False)
        
        saliency_maps = saliency_maps.data.cpu().numpy()

        del_score = deletion.evaluate(img=images.detach().cpu(), pred=target, mask=saliency_maps, idx=i, b=batch_size)
        ins_score = insertion.evaluate(img=images.detach().cpu(), pred=target, mask=saliency_maps, idx=i, b=batch_size)

        for delscore in del_score:
            scores['del'].append(delscore)
        for insscore in ins_score:
            scores['ins'].append(insscore)
            
        progress.set_postfix({'Deletion': np.mean(scores['del']), 'Insertion': np.mean(scores['ins'])})
            
    logger.info('----------------------------------------------------------------')
    logger.info('Final:\nDeletion - {:.5f}\nInsertion - {:.5f}'.format(np.mean(scores['del']), np.mean(scores['ins'])))


if __name__ == '__main__':
    main()