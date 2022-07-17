import os
import random
from functools import partial

import albumentations as A
import cv2
import jittor as jt
import numpy as np
from jittor import transform
from jittor.dataset import Dataset
from PIL import Image

from utils import ToVar, convert_to_negetive_one_positive_one, to_onehot


def train_val_split(
    image_root='/nas/landscape/train_val/images', 
    ratio=0.9,
    seed=42, 
):
    assert os.path.isdir(image_root)
    image_names =  os.listdir(image_root)
    random.seed(seed)
    random.shuffle(image_names)
    train_length = int(ratio * len(image_names))
    print(f"There are total {len(image_names)} images, ", 
          f"use {train_length} images for training and ", 
          f"{len(image_names) - train_length} images for validation.")
    train_images = image_names[:train_length]
    val_images = image_names[train_length :]
    with open('./assets/train.txt', 'w') as file:
        for i in train_images:
            file.write(i + '\n')
    with open('./assets/val.txt', 'w') as file:
        for i in val_images:
            file.write(i + '\n')
    print('save train and val image names in train.txt and val.txt')
    return train_images, val_images


class VQDataset(Dataset):
    def __init__(
        self,
        image_root='/data/landscape/train_val/images', 
        image_names=None,
        transform=None,
    ):
        super().__init__()
        self.image_root = image_root
        self.image_names = image_names
        self.transform = transform
        
        self.set_attrs(total_len=len(self.image_names))
    
    def __getitem__(self, idx):
        p_img = os.path.join(self.image_root, self.image_names[idx])
        image = cv2.imread(p_img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        name = os.path.splitext(self.image_names[idx])[0] # remove ext (i.e. .png, .jpg)
        if image.shape[-1] ==  3:
            image = image.permute(2, 0, 1)
        return image, name
    

def get_vq_dataloader( 
    image_root='/nas/landscape/train_val/images', 
    train_val_split_ratio=0.9,
    train_val_split_seed=42,
    batch_size=6, 
    num_workers=2, 
    image_size=256,
    crop_ratio=1.5,
):
    """
    First rescale an image so that minimum side is equal to {crop_ratio * image_size}, 
    keeping the aspect ratio of the initial image, then random crop (image_size, image_size)
    image from the resized image.
    
    e.g. input image is 768x1024, crop_ratio is 1.5, image_size is 256, then we resize the 
    image so that its short size (i.e. height: 768) is resized to 256 * 1.5 = 386, the 
    corresponding width is 512, then we random crop 256x256 images from 384x512.

    TODO: try different crop strategies and data augmentations
    """
    assert crop_ratio >= 1.0

    train_images, val_images = train_val_split(
        image_root=image_root, 
        ratio=train_val_split_ratio, 
        seed=train_val_split_seed, 
    )
    
    train_transform = A.Compose([ 
        A.SmallestMaxSize(max_size=int(crop_ratio * image_size)),
        A.RandomCrop(width=image_size, height=image_size),
        A.HorizontalFlip(p=0.5),
        A.Lambda(image=convert_to_negetive_one_positive_one),
        ToVar(),
    ])
    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=image_size),
        A.RandomCrop(width=image_size, height=image_size),
        A.Lambda(image=convert_to_negetive_one_positive_one),
        ToVar(),
    ])

    train_loader = VQDataset(
        image_root, train_images, train_transform,
    ).set_attrs(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = VQDataset(
        image_root, val_images, val_transform
    ).set_attrs(
        batch_size=batch_size,
        shuffle=True, # random sample some images for validating
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


class LDMDataset(Dataset):
    def __init__(
            self, 
            image_root='/nas/landscape/train_val/images',
            segmentation_root='/nas/landscape/train_val/labels',
            image_names=None,
            transform=None,
    ):
        super().__init__()
        self.image_root = image_root
        self.segmentation_root = segmentation_root
        self.image_names = image_names
        self.transform = transform

        self.set_attrs(total_len=len(self.image_names))

    def __getitem__(self, idx):
        path_img = os.path.join(self.image_root, self.image_names[idx])
        path_seg = os.path.join(self.segmentation_root, 
                                self.image_names[idx].replace('.jpg', '.png'))
        image = Image.open(path_img).convert('RGB')
        image = np.asarray(image)
        seg = Image.open(path_seg)
        seg = np.asarray(seg)
        if self.transform is not None:
            image, seg = self.transform(image=image, mask=seg).values()
        name = os.path.splitext(self.image_names[idx])[0] # remove ext (i.e. .png, .jpg)
        return image, seg, name


# TODO: add size support
def get_ldm_dataloader(
    image_root='/nas/landscape/train_val/images',
    segmentation_root='/nas/landscape/train_val/labels',
    train_val_split_ratio=0.9,
    train_val_split_seed=42,
    batch_size=6,
    num_workers=2,
    n_labels=29,
    image_size=256,
    crop_ratio=1.4, # if fine tune, this can be smaller, such as 1.2
):
    assert crop_ratio >= 1.0
    
    train_images, val_images = train_val_split(
        image_root=image_root, 
        ratio=train_val_split_ratio, 
        seed=train_val_split_seed, 
    )
    height, width = image_size 

    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=int(crop_ratio * height)),
        A.RandomCrop(width=width, height=height),
        A.HorizontalFlip(p=0.5),
        A.Lambda(image=convert_to_negetive_one_positive_one,
                 mask=partial(to_onehot, n_labels=n_labels)),
        ToVar(),
    ])
    val_transform = A.Compose([ # use 384x512 directly?
        A.SmallestMaxSize(max_size=height),
        A.RandomCrop(width=width, height=height),
        A.Lambda(image=convert_to_negetive_one_positive_one,
                 mask=partial(to_onehot, n_labels=n_labels)),
        ToVar(),
    ])

    train_dataloader = LDMDataset(
        image_root=image_root,
        segmentation_root=segmentation_root,
        image_names=train_images,
        transform=train_transform,
    ).set_attrs(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_dataloader = LDMDataset(
        image_root=image_root,
        segmentation_root=segmentation_root,
        image_names=val_images,
        transform=val_transform,
    ).set_attrs(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True, # set to True to avoid some bug
    )
    return train_dataloader, val_dataloader


class InferenceDataset(Dataset):
    """ use for inference on val(test) dataset, resize (768x1024) to (384x512)
    """
    def __init__(
        self,
        n_labels=29,
        segmentation_root='/data/landscape/test/labels', 
    ):
        super().__init__()
        self.segmentation_root = segmentation_root
        self.n_labels = n_labels
        
        self.seg_path = os.listdir(self.segmentation_root)

        # dataset is small, load into memory directly
        self.segmentations = []
        self.names = []
        for path in self.seg_path:
            name = os.path.splitext(path)[0] # remove .jpg in image path
            seg_path = os.path.join(self.segmentation_root, path)
            seg = Image.open(seg_path)
            self.segmentations.append(seg)
            self.names.append(name)
        self.set_attrs(total_len=len(self.names))
        
        self.resize = transform.Compose([
            transform.Resize((384, 512)),
            transform.ToTensor(),
        ])
    
    def __getitem__(self, idx):
        name = self.names[idx]
        seg = self.segmentations[idx]
        seg = jt.array(self.resize(seg)).long()

        seg = jt.init.eye(self.n_labels)[seg]
        seg = seg.permute(0, 3, 1, 2).squeeze(0) 
        return (seg, name)
