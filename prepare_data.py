# import os
# import random
# from tqdm import tqdm
# from datasets import load_dataset


# # -------------------------
# # Get ID → WordNet class mapping
# # -------------------------
# def load_class_index():
#     # load the dataset builder info (metadata)
#     builder = load_dataset("imagenet-1k", split="validation", streaming=True)
#     # id2label is in builder.info.features["label"].names
#     # Example: ["n01440764", "n01443537", ...]
#     wnids = builder.features["label"].names

#     # Map id -> wnid (0 -> n01440764, etc.)
#     return {i: wnid for i, wnid in enumerate(wnids)}


# # -------------------------
# # Reservoir Sampling
# # -------------------------
# def reservoir_sample(stream, k=6000, seed=42):
#     random.seed(seed)
#     sample = []
    
#     for i, item in enumerate(tqdm(stream, desc="Streaming ImageNet-1k")):
#         if i < k:
#             sample.append(item)
#         else:
#             j = random.randint(0, i)
#             if j < k:
#                 sample[j] = item
    
#     return sample


# # -------------------------
# # Save images to ImageFolder structure
# # -------------------------
# def save_sample(sample, class_index, out_dir="imagenet/val"):
#     for i, item in enumerate(tqdm(sample, desc="Saving images")):
#         img = item["image"]
#         label = item["label"]

#         wnid = class_index[label]   # e.g. "n01440764"
#         class_dir = os.path.join(out_dir, wnid)
#         os.makedirs(class_dir, exist_ok=True)

#         path = os.path.join(class_dir, f"{i:06d}.jpeg")
#         img.save(path, format="JPEG")


# # -------------------------
# # Main
# # -------------------------
# def main():
#     # Stream dataset
#     ds = load_dataset("imagenet-1k", split="validation", streaming=True)

#     # Get correct mapping from dataset itself
#     class_index = load_class_index()

#     # Sample 6000 random images
#     sample = reservoir_sample(ds, k=6000)

#     # Save in torchvision-compatible structure
#     save_sample(sample, class_index)

#     print("Done! Images saved in ./imagenet/val/")


# if __name__ == "__main__":
#     main()


import os
import random
from tqdm import tqdm
from datasets import load_dataset, load_dataset_builder


# -------------------------
# Load WordNet IDs (imagenet class IDs)
# -------------------------
def load_class_index():
    # Load dataset builder to access metadata
    builder = load_dataset_builder("imagenet-1k")
    wnids = builder.info.features["label"].names  # ["n01440764", "n01443537", ...]
    return {i: wnid for i, wnid in enumerate(wnids)}


# -------------------------
# Reservoir Sampling
# -------------------------
def reservoir_sample(stream, k=6000, seed=42):
    random.seed(seed)
    sample = []

    for i, item in enumerate(tqdm(stream, desc="Streaming ImageNet-1k")):
        if i < k:
            sample.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                sample[j] = item

    return sample


# -------------------------
# Save images with ImageNet IDs as folder names
# -------------------------
def save_sample(sample, class_index, out_dir="imagenet/val"):
    for i, item in enumerate(tqdm(sample, desc="Saving images")):
        img = item["image"]
        label = item["label"]

        wnid = class_index[label]  # e.g. "n01440764"
        class_dir = os.path.join(out_dir, wnid)
        os.makedirs(class_dir, exist_ok=True)

        path = os.path.join(class_dir, f"{i:06d}.jpeg")
        img.save(path, format="JPEG")


# -------------------------
# Main
# -------------------------
def main():
    # Stream validation split
    ds = load_dataset("imagenet-1k", split="validation", streaming=True)

    # Map label indices → ImageNet class IDs (wnids)
    class_index = load_class_index()

    # Sample 6000 images
    sample = reservoir_sample(ds, k=6000)

    # Save in torchvision-compatible ImageFolder structure
    save_sample(sample, class_index)

    print("Done! Images saved in ./imagenet/val/")


if __name__ == "__main__":
    main()
