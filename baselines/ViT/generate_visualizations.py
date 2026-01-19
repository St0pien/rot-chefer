import os
from tqdm import tqdm
import h5py

import argparse

# Import saliency methods and models
from misc_functions import *

from ViT_explanation_generator import Baselines, LRP, RotChefer
from ViT_new import vit_base_patch16_224
from ViT_LRP import vit_base_patch16_224 as vit_LRP
from ViT_orig_LRP import vit_base_patch16_224 as vit_orig_LRP

from torchvision.datasets import ImageNet


def normalize(tensor,
              mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def is_zero_attribution(arr, eps=1e-8):
    return np.all(np.abs(arr) < eps)


def compute_saliency_and_save(args):
    results_path = os.path.join(args.method_dir, "results.hdf5")

    with h5py.File(results_path, "a") as f:
        # --------------------------------------------------
        # Create or load datasets
        # --------------------------------------------------
        if "vis" in f:
            data_cam = f["vis"]
            data_image = f["image"]
            data_target = f["target"]

            start_idx = data_cam.shape[0]
            while start_idx > 0:
                last_cam = data_cam[start_idx - 1]
                if is_zero_attribution(last_cam):
                    start_idx -= 1
                else:
                    break

            # Shrink datasets if needed
            if start_idx < data_cam.shape[0]:
                data_cam.resize(start_idx, axis=0)
                data_image.resize(start_idx, axis=0)
                data_target.resize(start_idx, axis=0)
        else:
            data_cam = f.create_dataset(
                "vis",
                shape=(0, 1, 224, 224),
                maxshape=(None, 1, 224, 224),
                dtype=np.float32,
                compression="gzip",
            )
            data_image = f.create_dataset(
                "image",
                shape=(0, 3, 224, 224),
                maxshape=(None, 3, 224, 224),
                dtype=np.float32,
                compression="gzip",
            )
            data_target = f.create_dataset(
                "target",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int32,
                compression="gzip",
            )
            start_idx = 0

        # --------------------------------------------------
        # Iterate loader, skipping already-processed samples
        # --------------------------------------------------
        seen_samples = 0

        for data, target in tqdm(sample_loader):
            batch_size = data.shape[0]

            # Skip batches that are already fully written
            if seen_samples + batch_size <= start_idx:
                seen_samples += batch_size
                continue

            # Handle partially-written batch
            if seen_samples < start_idx:
                offset = start_idx - seen_samples
                data = data[offset:]
                target = target[offset:]
                batch_size = data.shape[0]

            seen_samples += batch_size

            # --------------------------------------------------
            # Resize datasets
            # --------------------------------------------------
            new_size = data_cam.shape[0] + batch_size
            data_cam.resize(new_size, axis=0)
            data_image.resize(new_size, axis=0)
            data_target.resize(new_size, axis=0)

            # --------------------------------------------------
            # Save image & target
            # --------------------------------------------------
            data_image[-batch_size:] = data.cpu().numpy()
            data_target[-batch_size:] = target.cpu().numpy()

            # --------------------------------------------------
            # Compute saliency
            # --------------------------------------------------
            target = target.to(device)
            data = normalize(data).to(device)
            data.requires_grad_()

            index = target if args.vis_class == "target" else None

            if args.method == "rollout":
                Res = baselines.generate_rollout(data, start_layer=1).reshape(batch_size, 1, 14, 14)

            elif args.method == "lrp":
                Res = lrp.generate_LRP(data, start_layer=1, index=index).reshape(batch_size, 1, 14, 14)

            elif args.method == "transformer_attribution":
                Res = lrp.generate_LRP(
                    data, start_layer=1, method="grad", index=index
                ).reshape(batch_size, 1, 14, 14)

            elif args.method == "full_lrp":
                Res = orig_lrp.generate_LRP(
                    data, method="full", index=index
                ).reshape(batch_size, 1, 224, 224)

            elif args.method == "lrp_last_layer":
                Res = orig_lrp.generate_LRP(
                    data,
                    method="last_layer",
                    is_ablation=args.is_ablation,
                    index=index,
                ).reshape(batch_size, 1, 14, 14)

            elif args.method == "attn_last_layer":
                Res = lrp.generate_LRP(
                    data,
                    method="last_layer_attn",
                    is_ablation=args.is_ablation,
                ).reshape(batch_size, 1, 14, 14)

            elif args.method == "attn_gradcam":
                Res = baselines.generate_cam_attn(
                    data, index=index
                ).reshape(batch_size, 1, 14, 14)

            elif args.method == "rot_chefer":
                Res = rot_lrp.generate_LRP(data, is_ablation=args.is_ablation)

            # --------------------------------------------------
            # Post-processing
            # --------------------------------------------------
            if args.method not in {"full_lrp", "input_grads", "rot_chefer"}:
                Res = torch.nn.functional.interpolate(
                    Res, scale_factor=16, mode="bilinear"
                ).cuda()

            Res = (Res - Res.min()) / (Res.max() - Res.min() + 1e-8)

            data_cam[-batch_size:] = Res.detach().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a segmentation')
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')
    parser.add_argument('--method', type=str,
                        default='grad_rollout',
                        choices=['rollout', 'lrp', 'transformer_attribution', 'full_lrp', 'lrp_last_layer',
                                 'attn_last_layer', 'attn_gradcam', 'rot_chefer'],
                        help='')
    parser.add_argument('--lmd', type=float,
                        default=10,
                        help='')
    parser.add_argument('--vis-class', type=str,
                        default='top',
                        choices=['top', 'target', 'index'],
                        help='')
    parser.add_argument('--class-id', type=int,
                        default=0,
                        help='')
    parser.add_argument('--cls-agn', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-ia', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-fx', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-fgx', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-m', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-reg', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--is-ablation', type=bool,
                        default=False,
                        help='')
    parser.add_argument('--imagenet-validation-path', type=str,
                        required=True,
                        help='')
    args = parser.parse_args()

    # PATH variables
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.makedirs(os.path.join(PATH, 'visualizations'), exist_ok=True)

    try:
        os.remove(os.path.join(PATH, 'visualizations/{}/{}/results.hdf5'.format(args.method,
                                                                                args.vis_class)))
    except OSError:
        pass


    os.makedirs(os.path.join(PATH, 'visualizations/{}'.format(args.method)), exist_ok=True)
    if args.vis_class == 'index':
        os.makedirs(os.path.join(PATH, 'visualizations/{}/{}_{}'.format(args.method,
                                                                        args.vis_class,
                                                                        args.class_id)), exist_ok=True)
        args.method_dir = os.path.join(PATH, 'visualizations/{}/{}_{}'.format(args.method,
                                                                              args.vis_class,
                                                                              args.class_id))
    else:
        ablation_fold = 'ablation' if args.is_ablation else 'not_ablation'
        os.makedirs(os.path.join(PATH, 'visualizations/{}/{}/{}'.format(args.method,
                                                                     args.vis_class, ablation_fold)), exist_ok=True)
        args.method_dir = os.path.join(PATH, 'visualizations/{}/{}/{}'.format(args.method,
                                                                           args.vis_class, ablation_fold))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # Model
    model = vit_base_patch16_224(pretrained=True).cuda()
    baselines = Baselines(model)

    # LRP
    model_LRP = vit_LRP(pretrained=True).cuda()
    model_LRP.eval()
    lrp = LRP(model_LRP)
    rot_lrp = RotChefer(model_LRP, n_samples=64, batch_size=16)

    # orig LRP
    model_orig_LRP = vit_orig_LRP(pretrained=True).cuda()
    model_orig_LRP.eval()
    orig_lrp = LRP(model_orig_LRP)

    # Dataset loader for sample images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    imagenet_ds = ImageNet(args.imagenet_validation_path, split='val', transform=transform)
    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    compute_saliency_and_save(args)
