import os
import random
from typing import Any, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from pycocotools.coco import COCO
from torchvision.datasets import Cityscapes

### For visualizing the outputs ###
import matplotlib.pyplot as plt


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = listdir_no_hidden(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        # np array for PIL lib
        # probably RGB by default
        image = np.array(Image.open(img_path).convert("RGB"))
        # convert to grayscale is L
        # 0.0, 255.0
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # mask preprocessing
        # this is because of sigmoid as activation
        mask[mask == 255.0] = 1.0

        # data augmentations
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask


class CamVidDataset(Dataset):

    def __init__(self, classes_dir, image_dir, mask_dir, transform=None):
        # array of RGB values of K classes -> K x 3
        self.classes = np.load(classes_dir, allow_pickle=True)
        self.image_dir = image_dir
        self.label_dir = mask_dir
        self.transform = transform
        self.images = listdir_no_hidden(image_dir)
        self.mapping = {
            1: 1,  # Archway -> Building
            2: 2,  # Bicyclist
            3: 1,  # Bridge -> Building
            4: 1,  # Building
            5: 3,  # Car
            7: 0,  # Pedestrian
            8: 4,  # Pole
            9: 5,  # Fence
            10: 6,  # LaneMkgsDriv -> Road
            11: 6,  # LaneMkgsNonDriv -> Road
            12: 7,  # MiscText -> SignSymbol
            14: 3,  # OtherMoving -> Car
            15: 8,  # ParkingBlock -> SideWalk
            16: 0,  # Pedestrian
            17: 6,  # Road
            18: 8,  # RoadShoulder -> Sidewalk
            19: 8,  # Sidewalk
            20: 7,  # SignSymbol
            21: 9,  # Sky
            22: 3,  # SUVPickupTruck -> Car
            23: 4,  # TrafficCone -> Pole
            24: 7,  # TrafficLight -> SignSymbol
            26: 10,  # Tree
            27: 3,  # Truck_Bus -> Car
            28: 1,  # Tunnel -> Building
            29: 10,  # VegetationMisc -> Tree
            30: 11,  # Unlabeled
            31: 1  # Wall -> Building
        }


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index, user_image_name=None):

        if user_image_name is None:
            img_name = self.images[index]
        else:
            img_name = user_image_name

        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".png", "_L.png"))
        # np array for PIL lib
        # probably RGB by default
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(label_path))
        mask = self.one_hot(mask)
        # remap_func = np.vectorize(lambda x: self.mapping.get(x, 11))
        # mask = self.rgb_to_labels(mask)
        # mask = remap_func(mask)
        # mask = self.labels_to_one_hot(mask)

        # data augmentations
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask

    def one_hot(self, image):
        # output is one-hot encoded tensor -> M x N x K
        output_shape = (image.shape[0], image.shape[1], self.classes.shape[0])
        output = np.zeros(output_shape)
        for cls in range(self.classes.shape[0]):
            label = np.nanmin(self.classes[cls] == image, axis=2)
            output[:, :, cls] = label
        return output

    def rgb_to_labels(self, mask):
        output_shape = (mask.shape[0], mask.shape[1])
        output = np.zeros(output_shape)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                # where all vales are equal
                output[i][j] = np.where((self.classes == mask[i][j]).all(axis=1))[0][0]
        return output

    def labels_to_one_hot(self, mask):
        output_shape = (mask.shape[0], mask.shape[1], 12)
        output = np.zeros(output_shape)
        for cls in range(12):
            label = np.where(cls == mask, 1, 0)
            output[:, :, cls] = label
        return output

    def get_image_by_name(self, image_name):
        if image_name == "":
            image_name = None
        index = random.randint(0, 100)
        return self.__getitem__(index, image_name)


class CustomSplitDataset(Dataset):

    def __init__(self, samples, augment):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        features = self.samples[index]
        if self.augment:
            features = self.add_gaussian_noise(features, std=0.05)
        # features = self.normalize(features)

        return features, features

    def add_gaussian_noise(self, tensor, mean=0.0, std=0.1):
        noise = torch.randn(tensor.size()) * std + mean
        return tensor + noise

    def normalize(self, tensor):
        return (tensor - tensor.mean()) / tensor.std()



class CocoDataset(Dataset):
    def __init__(self, root, type, transform=None):
        self.image_dir = '{}images/{}2017'.format(root, type)
        self.transform = transform
        annFile = '{}/annotations/instances_{}2017.json'.format(root, type)

        # Initialize the COCO api for instance annotations
        self.coco = COCO(annFile)
        # Load the categories in a variable
        self.catIDs = self.coco.getCatIds()
        # print(catIDs)
        self.cats = self.coco.loadCats(self.catIDs)
        # nms = [cat['name'] for cat in self.cats]
        # print(len(nms), 'COCO categories: \n{}\n'.format(' '.join(nms)))

        # nms = set([cat['supercategory'] for cat in self.cats])
        # print(len(nms), 'COCO supercategories: \n{}'.format(' '.join(nms)))
        # print(self.cats)
        # Load all images
        imgIds = self.coco.getImgIds()
        all_images = self.coco.loadImgs(imgIds)
        # filter out the repeated images
        self.images = []
        for i in range(len(all_images)):
            if all_images[i] not in self.images:
                self.images.append(all_images[i])
        # print(self.dataset_size)
        # img = self.images[11]
        # I = io.imread(img['coco_url'])
        # plt.axis('off')
        # plt.imshow(I)
        # plt.show()
        # catIds = self.coco.getCatIds(catNms=['person', 'laptop'])
        # ann_ids = self.coco.getAnnIds(imgIds=img['id'])
        # anns = self.coco.loadAnns(ann_ids)
        # plt.axis('off')
        # plt.imshow(I)
        # self.coco.showAnns(anns)
        # plt.show()
        # self.mask = self.generate_seg_mask(img, anns)
        # self.mask = self.generate_one_hot_mask(img, anns, catIDs)

    def __len__(self):
        return len(self.images)

    def generate_seg_mask(self, img, anns):
        mask = np.zeros((img['height'], img['width']))
        for i in range(0, len(anns)):
            pixel_value = anns[i]["category_id"]
            # pixel_value = i + 1
            print(pixel_value)
            # takes maximum of mask vs prev mask
            mask = np.maximum(self.coco.annToMask(anns[i]) * pixel_value, mask)
        plt.axis('off')
        plt.imshow(mask)
        plt.show()
        return mask


    def generate_one_hot_mask(self, img, anns):
        num_classes = len(self.catIDs) + 1
        mask = np.zeros((img['height'], img['width'], num_classes))
        # binary_masks = []
        for i in range(0, len(anns)):
            category_id = anns[i]["category_id"]
            # print(category_id)
            category_index = self.catIDs.index(category_id)
            # print(category_index)
            binary_mask = self.coco.annToMask(anns[i])
            # binary_masks.append((binary_mask, category_index))
            # for binary_mask, category_index in binary_masks:
            mask[:, :, category_index] = np.maximum(mask[:, :, category_index], binary_mask)
            # plt.imshow(binary_mask)
            # plt.show()
            # mask[:, :, category_index] = binary_mask
        # Identify the background pixels
        # background = np.logical_not(np.sum(mask, axis=-1, keepdims=True)).astype(mask.dtype)
        # mask = np.concatenate([mask, background], axis=-1)
        background = (mask.sum(axis=2) == 0)
        mask[background, -1] = 1
        # print(mask.shape)
        return mask

    def __getitem__(self, index):
        image_metadata = self.images[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_metadata['id'])
        anns = self.coco.loadAnns(ann_ids)
        mask = self.generate_one_hot_mask(image_metadata, anns)
        path = image_metadata['file_name']
        image = None
        try:
            image = np.array(Image.open(os.path.join(self.image_dir, path)).convert('RGB'))
        except FileNotFoundError as e:
            print('Img was not found:', image_metadata['file_name'])
            if index + 1 < len(self.images):
                # Increment index and try again with the next image
                return self.__getitem__(index + 1)
            else:
                return None
        # print('Image shape:', image.shape)
        # print('Mask shape:', mask.shape)

        # data augmentations
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask


class CityscapesDataset(Cityscapes):
    def __init__(self, root='../CityScapes/', split='val', transforms=None):
        super().__init__(root=root, split=split, target_type='semantic', transform=transforms)
        # self.dataset = Cityscapes('CityScapes/', split=split)
        self.ignore_index = 255
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [self.ignore_index, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
                            'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
                            'train', 'motorcycle', 'bicycle']
        self.n_classes = len(self.valid_classes)
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Change to current working directory
        os.chdir(os.path.dirname(__file__))
        try:
            image = Image.open(self.images[index]).convert('RGB')

            targets: Any = []
            for i, t in enumerate(self.target_type):
                if t == 'polygon':
                    target = self._load_json(self.targets[index][i])
                else:
                    target = Image.open(self.targets[index][i])
                targets.append(target)
            target = tuple(targets) if len(targets) > 1 else targets[0]
            target = np.asarray(targets[0], dtype=np.uint8)
            target = target.copy()
            target = self.encode_segmap(target)
            target = self.one_hot_segmap(target)
            # target = self.color_segmap(target)

            if self.transforms is not None:
                transformed = self.transform(image=np.array(image), mask=np.array(target))
            return transformed['image'], transformed['mask']

        except FileNotFoundError:
            # Move to the next index if file is not found
            index = (index + 1) % len(self.images)


    '''
        creates a map with edited class labels
        input: mask = mask before label correction
    '''
    def encode_segmap(self, mask):
        # remove unwanted classes and rectify the labels of wanted classes
        for void_class in self.void_classes:
            mask[mask == void_class] = self.ignore_index
        for valid_class in self.valid_classes:
            mask[mask == valid_class] = self.class_map[valid_class]
        return mask

    # def color_segmap(self, temp):
    #     # convert gray scale to color
    #     # temp = temp.numpy()
    #     r = temp.copy()
    #     g = temp.copy()
    #     b = temp.copy()
    #     for l in range(0, self.n_classes):
    #         r[temp == l] = self.label_colours[l][0]
    #         g[temp == l] = self.label_colours[l][1]
    #         b[temp == l] = self.label_colours[l][2]
    #
    #     rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    #     rgb[:, :, 0] = r / 255.0
    #     rgb[:, :, 1] = g / 255.0
    #     rgb[:, :, 2] = b / 255.0
    #     return rgb

    def one_hot_segmap(self, mask):
        output_shape = (mask.shape[0], mask.shape[1], self.n_classes)
        output = np.zeros(output_shape, dtype=np.float32)
        for cls in range(self.n_classes):
            label = (cls == mask).astype(np.float32)
            output[:, :, cls] = label
        return output


def listdir_no_hidden(path):
    files = []
    for file in os.listdir(path):
        if not file.startswith('.'):
            files.append(file)
    return files



