
import timm
from .get_logger import logger

def get_mean_std(name):
    
    if 'vit' in name or 'deit' in name:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    return mean, std


def get_transf(args, model, return_norm=False):
    
    config = timm.data.resolve_data_config({}, model=model)
    logger.info('Data Augmentation - timm default mode')
    model_transform = timm.data.transforms_factory.create_transform(**config)
    if return_norm:
        norm_fn = model_transform.transforms[-1]
        model_transform.transforms = model_transform.transforms[:-1]
        return model_transform, norm_fn
    return model_transform
