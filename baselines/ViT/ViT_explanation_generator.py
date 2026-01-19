import argparse
import torch
import numpy as np
from numpy import *
from torchvision.transforms.functional import affine
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import kornia


# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = (
        torch.eye(num_tokens)
        .expand(batch_size, num_tokens, num_tokens)
        .to(all_layer_matrices[0].device)
    )
    all_layer_matrices = [
        all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))
    ]
    matrices_aug = [
        all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
        for i in range(len(all_layer_matrices))
    ]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer + 1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention


class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(
        self,
        input,
        index=None,
        method="transformer_attribution",
        is_ablation=False,
        start_layer=0,
    ):
        output = self.model(input)
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        return self.model.relprop(
            torch.tensor(one_hot_vector).to(input.device),
            method=method,
            is_ablation=is_ablation,
            start_layer=start_layer,
            **kwargs
        )

    def generate_LRP_parallel(
        self,
        input,
        index=None,
        method="transformer_attribution",
        is_ablation=False,
        start_layer=0,
    ):
        output = self.model(input)  # [B, C]

        if index is None:
            index = output.argmax(dim=-1)

        selected = output.gather(1, index.unsqueeze(1))  # [B, 1]

        self.model.zero_grad()

        torch.autograd.backward(selected, grad_tensors=torch.ones_like(selected))

        R = torch.zeros_like(output)
        R.scatter_(1, index.unsqueeze(1), 1.0)

        return self.model.relprop(
            R,
            method=method,
            is_ablation=is_ablation,
            start_layer=start_layer,
            alpha=1,
        )


class RotChefer_nodiff:
    def __init__(
        self,
        model,
        n_samples=50,
        batch_size=16,
        angle_range: tuple = (-90.0, 90.0),
        translation_offset=5,
    ):
        self.model = model
        self.model.eval()
        self.start_angle = angle_range[0]
        self.end_angle = angle_range[1]
        self.transation_offset = translation_offset
        self.n_samples = n_samples
        self.batch_size = batch_size

    def perturbe_input(self, input: torch.Tensor):
        images = [input]
        rev_angles = [0]
        rev_translations = [(0, 0)]
        angles = np.linspace(self.start_angle, self.end_angle, self.n_samples - 1)
        for i in range(self.n_samples - 1):
            x, y = np.random.randint(
                -self.transation_offset, self.transation_offset, size=2
            )
            angle = angles[i]
            images.append(
                affine(input.clone(), angle=angle, translate=(x, y), scale=1, shear=0)
            )
            rev_angles.append(-angle)
            rev_translations.append((-x, -y))

        return images, rev_angles, rev_translations

    def generate_LRP(
        self,
        input,
        index=None,
        is_ablation=False,
        start_layer=0,
    ):
        perturbed_data, angles, translations = self.perturbe_input(input)
        perturbed_dataset = TensorDataset(torch.cat(perturbed_data))

        perturbed_dataloader = DataLoader(perturbed_dataset, batch_size=self.batch_size)

        full_cam = torch.zeros(1, input.shape[2], input.shape[3]).to(input.device)

        if index is None:
            output = self.model(input)
            index = np.argmax(output.detach().cpu())

        i = 0
        for batch in perturbed_dataloader:
            batch = torch.cat(batch)
            batch.detach_()
            output = self.model(batch)  # [B, C]

            b_index = torch.full((batch.shape[0], 1), index).to(input.device)
            selected = output.gather(1, b_index)

            self.model.zero_grad()

            torch.autograd.backward(selected, grad_tensors=torch.ones_like(selected))

            R = torch.zeros_like(output)
            R.scatter_(1, b_index, 1.0)

            batch_cam = self.model.relprop(
                R,
                method="full_transformer_attribution",
                is_ablation=is_ablation,
                start_layer=start_layer,
                alpha=1,
            ).sum(dim=1)
            batch_cam.detach_()
            with torch.no_grad():
                inv_cams = []
                for cam in batch_cam:
                    inv = affine(
                        cam.unsqueeze(0),
                        angle=angles[i],
                        translate=translations[i],
                        scale=1,
                        shear=0,
                    )
                    inv_cams.append(inv)
                    i += 1
                full_cam += torch.cat(inv_cams).sum(dim=0).unsqueeze(0)

        return full_cam.detach()


class RotChefer:
    def __init__(
        self, model, n_samples=50, batch_size=16, angle_range: tuple = (-90.0, 90.0)
    ):
        self.model = model
        self.model.eval()
        self.start_angle = angle_range[0]
        self.end_angle = angle_range[1]
        self.n_samples = n_samples
        self.batch_size = batch_size

    def perturbe_input(self, input: torch.Tensor):
        angles = np.linspace(self.start_angle, self.end_angle, self.n_samples - 1)
        images = input.clone().repeat(self.n_samples - 1, 1, 1, 1)
        images = torch.cat([input, images])
        rotated_images = kornia.geometry.rotate(
            images, torch.tensor([0, *angles], device=input.device, dtype=torch.float32)
        )

        return images, rotated_images

    def generate_LRP(
        self,
        input,
        index=None,
        is_ablation=False,
        start_layer=0,
    ):
        if index is None:
            output = self.model(input)
            index = np.argmax(output.detach().cpu())

        angles = torch.linspace(
            self.start_angle,
            self.end_angle,
            self.n_samples - 1,
            device=input.device,
            dtype=torch.float32,
        )

        angles = torch.cat(
            [torch.zeros(1, device=input.device, dtype=torch.float32), angles]
        )
        images = input.clone().repeat(self.n_samples, 1, 1, 1)

        dataset = TensorDataset(images, angles)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        full_cam = torch.zeros(1, input.shape[2], input.shape[3]).to(input.device)

        for imgs, b_angles in dataloader:
            with torch.no_grad():
                rotated_imgs = kornia.geometry.rotate(imgs, b_angles)

            output = self.model(rotated_imgs)

            b_index = torch.full((imgs.shape[0], 1), index).to(input.device)
            selected = output.gather(1, b_index)

            self.model.zero_grad()

            torch.autograd.backward(selected, grad_tensors=torch.ones_like(selected))

            R = torch.zeros_like(output)
            R.scatter_(1, b_index, 1.0)

            batch_cam = self.model.relprop(
                R,
                method="full_transformer_attribution",
                is_ablation=is_ablation,
                start_layer=start_layer,
                alpha=1,
            )

            imgs.requires_grad_(True)
            z = kornia.geometry.rotate(imgs, b_angles)
            
            grad = torch.autograd.grad(
                outputs=z,
                inputs=imgs,
                grad_outputs=batch_cam,
                retain_graph=False,
                create_graph=False,
            )[0]

            with torch.no_grad():
                pre_rot_cam = imgs.abs() * grad

                full_cam += pre_rot_cam.sum(dim=(0,1)).unsqueeze(0).detach()

        # return full_list
        return (full_cam - full_cam.min()) / (full_cam.max() - full_cam.min())


class Baselines:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_cam_attn(self, input, index=None):
        output = self.model(input.cuda(), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        #################### attn
        grad = self.model.blocks[-1].attn.get_attn_gradients()
        cam = self.model.blocks[-1].attn.get_attention_map()
        cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam
        #################### attn

    def generate_rollout(self, input, start_layer=0):
        self.model(input)
        blocks = self.model.blocks
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attn.get_attention_map()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        rollout = compute_rollout_attention(
            all_layer_attentions, start_layer=start_layer
        )
        return rollout[:, 0, 1:]
