from utils.imagenet_r import imagenet_r_mask
from utils.imagenet_a import indices_in_1k

def prediction_imagenet_a(output):
    output = output[:, indices_in_1k]
    return output

def prediction_imagenet_r(output):
    output = output[:, imagenet_r_mask]
    return output
