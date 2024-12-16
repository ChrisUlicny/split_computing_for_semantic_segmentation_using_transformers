import random

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import albumentations as A
from datasets import *



def get_augmentations(img_height, img_width):
    train_transform = A.Compose(
        [
            A.Resize(height=img_height, width=img_width),
            A.Rotate(limit=10, p=0.2),
            A.HorizontalFlip(p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.6),
                A.ColorJitter(p=0.4),
            ], p=0.3),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=img_height, width=img_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    return train_transform, val_transform



def get_augmentations_new(img_height, img_width):
    train_transform = A.Compose(
        [
            # A.Resize(height=img_height, width=img_width),
            A.RandomResizedCrop(height=img_height, width=img_width, scale=(0.5, 2.0)),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=img_height, width=img_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    return train_transform, val_transform


def get_loaders_camvid(
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
        class_dir="classes.npy",
        train_img_dir="../CamVid/train_images/",
        train_mask_dir="../CamVid/train_labels/",
        val_img_dir="../CamVid/val_images/",
        val_mask_dir="../CamVid/val_labels/",
        # train_img_dir="../CamVid/train_images_one/",
        # train_mask_dir="../CamVid/train_labels_one/",
        # val_img_dir="../CamVid/train_images_one/",
        # val_mask_dir="../CamVid/train_labels_one/",
        test_img_dir="../CamVid/test_images/",
        test_mask_dir="../CamVid/test_labels/"
        ):

    train_dataset = CamVidDataset(classes_dir=class_dir, image_dir=train_img_dir, mask_dir=train_mask_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dataset = CamVidDataset(classes_dir=class_dir, image_dir=val_img_dir, mask_dir=val_mask_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_dataset = CamVidDataset(classes_dir=class_dir, image_dir=test_img_dir, mask_dir=test_mask_dir,
                                 transform=val_transform)
    test_loader = DataLoader(test_dataset, 1, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    color_map = get_colormap_camvid()
    return train_loader, val_loader, test_loader, color_map


def get_loaders_coco(
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
        dataset_dir="cocodataset/"):
    train_dataset = CocoDataset(dataset_dir, 'train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dataset = CocoDataset(dataset_dir, 'val', transform=val_transform)
    val_loader = DataLoader(val_dataset, 1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    color_map = generate_color_map(81)
    prediction_folder = "prediction_coco/"
    return train_loader, val_loader, color_map, prediction_folder


def get_loaders_cityscapes(
        batch_size,
        train_transform,
        val_transform,
        num_workers=0,
        pin_memory=True):
    train_dataset = CityscapesDataset(split='train', transforms=train_transform)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dataset = CityscapesDataset(split='val', transforms=val_transform)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_dataset = CityscapesDataset(split='val', transforms=val_transform)
    test_loader = DataLoader(test_dataset, 1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    color_map = get_colormap_cityscapes()
    return train_loader, val_loader, test_loader, color_map


def get_camvid(transform, class_dir="classes.npy", test_img_dir="CamVid/test_images/", test_mask_dir="CamVid/test_labels/"):
    test_dataset = CamVidDataset(classes_dir=class_dir, image_dir=test_img_dir, mask_dir=test_mask_dir,
                                 transform=transform)
    color_map = get_colormap_camvid()
    return test_dataset, color_map


def get_colormap_camvid():
    color_map = {
        1: [64, 128, 64],
        2: [192, 0, 128],
        3: [0, 128, 192],
        4: [0, 128, 64],
        5: [128, 0, 0],
        6: [64, 0, 128],
        7: [64, 0, 192],
        8: [192, 128, 64],
        9: [192, 192, 128],
        10: [64, 64, 128],
        11: [128, 0, 192],
        12: [192, 0, 64],
        13: [128, 128, 64],
        14: [192, 0, 192],
        15: [128, 64, 64],
        16: [64, 192, 128],
        17: [64, 64, 0],
        18: [128, 64, 128],
        19: [128, 128, 192],
        20: [0, 0, 192],
        21: [192, 128, 128],
        22: [128, 128, 128],
        23: [64, 128, 192],
        24: [0, 0, 64],
        25: [0, 64, 64],
        26: [192, 64, 128],
        27: [128, 128, 0],
        28: [192, 128, 192],
        29: [64, 0, 64],
        30: [192, 192, 0],
        31: [0, 0, 0],
        32: [64, 192, 0]
    }

    return color_map


def get_colormap_camvid_12():
    color_map = {
        0: [128, 128, 128],
        1: [128, 0, 0],
        2: [192, 192, 128],
        3: [128, 64, 128],
        4: [60, 40, 222],
        5: [128, 128, 0],
        6: [192, 128, 128],
        7: [64, 64, 128],
        8: [64, 0, 128],
        9: [64, 64, 0],
        10: [0, 128, 192],
        11: [0, 0, 0]
    }

    return color_map


def get_colormap_cityscapes():

    color_map = [[0, 0, 0],
                  [128, 64, 128],
                  [244, 35, 232],
                  [70, 70, 70],
                  [102, 102, 156],
                  [190, 153, 153],
                  [153, 153, 153],
                  [250, 170, 30],
                  [220, 220, 0],
                  [107, 142, 35],
                  [152, 251, 152],
                  [0, 130, 180],
                  [220, 20, 60],
                  [255, 0, 0],
                  [0, 0, 142],
                  [0, 0, 70],
                  [0, 60, 100],
                  [0, 80, 100],
                  [0, 0, 230],
                  [119, 11, 32]
                 ]
    return dict(zip(range(0, len(color_map)), color_map))


def generate_color_map(num_classes):
    color_dict = {}
    for i in range(num_classes):
        color_dict[i] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color_dict
