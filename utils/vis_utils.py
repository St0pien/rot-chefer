import torch.nn.functional as F

def batch_normalize_map(x, relu=False, image_size=None, score=None, interpolation_mode='bilinear'):
        x = x.detach()
        if relu:
            x = F.relu(x)
        if image_size is not None:
            if interpolation_mode == 'nearest':
                x = F.interpolate(x, size=image_size, mode=interpolation_mode)
            else:
                x = F.interpolate(x, size=image_size, mode=interpolation_mode, align_corners=False)
        x_min = x.min(dim=-1)[0]
        x_min = x_min.min(dim=-1)[0].view(x.size(0), 1, 1, 1)
        x_max = x.max(dim=-1)[0]
        x_max = x_max.max(dim=-1)[0].view(x.size(0), 1, 1, 1)
        x = (x - x_min) / (x_max - x_min)

        return x
