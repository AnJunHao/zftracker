# Declaration: Part of the code in this file is written by genrative AI including Copilot, ChatGPT and GPT-4.

import random
import math

import albumentations as A
from albumentations.pytorch import ToTensorV2
from dask import compute, delayed
from scipy import signal
from torch.utils.data import Dataset, IterableDataset
import torch
import numpy as np
from icecream import ic
from matplotlib import pyplot as plt # For debugging
import sklearn
from shapely.geometry import Point

from .util.tqdm import TQDM as tqdm
from .preprocess.evaluate_annotation_v2 import unify_midsection, create_fish

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

import dask
dask.config.set(scheduler='processes')  # overwrite default with multiprocessing scheduler

def get_mean_distances(coords, head_midsec_distance=32):

    head_midsec_distancs = []
    head_tail_distances = []
    midsec_tail_distances = []

    for coords in coords:

        heads = np.array([coords[i] for i in range(0, 28, 4) if sum(coords[i]) != 0])
        midsec = np.array([coords[i] for i in range(1, 28, 4) if sum(coords[i]) != 0])
        tails = np.array([coords[i] for i in range(2, 28, 4) if sum(coords[i]) != 0])

        min_length = min(len(heads), len(midsec), len(tails))
        heads = heads[:min_length]
        midsec = midsec[:min_length]
        tails = tails[:min_length]

        unified_midsec = [unify_midsection(h, m, distance=head_midsec_distance)
                            for h, m in zip(heads, midsec)]
        unify_midsec = np.array(unified_midsec)

        head_midsec_distance = np.mean([np.linalg.norm(dif) for dif in heads - unify_midsec])
        head_tail_distance = np.mean([np.linalg.norm(dif) for dif in heads - tails])
        midsec_tail_distance = np.mean([np.linalg.norm(dif) for dif in unify_midsec - tails])

        head_midsec_distancs.append(head_midsec_distance)
        head_tail_distances.append(head_tail_distance)
        midsec_tail_distances.append(midsec_tail_distance)
    
    return {('head', 'tail'): np.mean(head_tail_distances),
            ('midsec', 'tail'): np.mean(midsec_tail_distances),
            ('head', 'midsec'): np.mean(head_midsec_distancs)}

def get_mean_distances_DEPRECATED(coords):

    head_midsec_distancs = []
    head_tail_distances = []
    midsec_tail_distances = []

    for coords in coords:
        heads = np.array([coords[i] for i in range(0, 28, 4) if sum(coords[i]) != 0])
        midsec = np.array([coords[i] for i in range(1, 28, 4) if sum(coords[i]) != 0])
        tails = np.array([coords[i] for i in range(2, 28, 4) if sum(coords[i]) != 0])
        min_length = min(len(heads), len(midsec), len(tails))
        heads = heads[:min_length]
        midsec = midsec[:min_length]
        tails = tails[:min_length]
        head_midsec_distance = np.mean([np.linalg.norm(dif) for dif in heads - midsec])
        head_tail_distance = np.mean([np.linalg.norm(dif) for dif in heads - tails])
        midsec_tail_distance = np.mean([np.linalg.norm(dif) for dif in midsec - tails])
        head_midsec_distancs.append(head_midsec_distance)
        head_tail_distances.append(head_tail_distance)
        midsec_tail_distances.append(midsec_tail_distance)

    # We don't return the mean distance of head_midsec as it is handled separately.
    return {('head', 'tail'): np.mean(head_tail_distances),
            ('midsec', 'tail'): np.mean(midsec_tail_distances)} 

def randint_from_float(a: float, b: float):
    """
    Returns a random integer between a and b,
    this will take the float part into consideration.
    Args:
        a (float): The lower bound of the range.
        b (float): The upper bound of the range.
    Returns:
        int: A random integer between a and b.
    """

    if a > b:
        raise Exception('a must be smaller than b')
    elif math.floor(a) == math.floor(b):
        if random.random() > (a + b) / 2 % 1:
            return math.floor(a)
        else:
            return math.ceil(a)
    if a % 1 == 0:
        lower_bond = a
    elif random.random() > a % 1:
        lower_bond = math.floor(a)
    else:
        lower_bond = math.ceil(a)

    if b % 1 == 0:
        upper_bond = b
    elif random.random() > b % 1:
        upper_bond = math.floor(b)
    else:
        upper_bond = math.ceil(b)
    
    return random.randint(lower_bond, upper_bond)

def get_rand_aug(n: float, m: float):
    """
    Returns a customized implementation of the RandAugment data augmentation technique, using albumentations library.
    The number of augmentation operations to apply (N) is randomly sampled from a uniform distribution between n/2 and n.
    Args:
        n (float): Number of augmentation operations to apply.
        m (float): Magnitude of each operation on a scale of 0 to 20.
    Returns:
        albumentations.Compose: A sequence of N augmentation operations, randomly sampled from a larger set of options, to be applied to input images.
    """
    m = m / 20

    n = min(randint_from_float(n/2, n), 20)

    if n != 0 and m != 0:

        all_transforms = [
            # 01 Blur Speed: 950 images / sec
            A.GaussianBlur(blur_limit=(1, round(1 + (8 * m) // 2 * 2)), p=1),
            # 02 Noise Speed: 1000 images / sec
            A.MultiplicativeNoise(multiplier=(1 - 0.9 * m, 1 + 0.9 * m), p=1),
            # 03 Brightness & Contrast Speed: 1100 images / sec
            A.RandomBrightnessContrast(
                brightness_limit=(-0.6 * m, 0.6 * m),
                contrast_limit=(-0.6 * m, 0.6 * m),
                p=1,
            ),
            # 04 Solarize Speed: 1300 images / sec
            A.Solarize(threshold=(255 - 255 * m, 255), p=1),
            # 05 Posterize Speed: 1300 images / sec
            A.Posterize(num_bits=(round(8 - 4 * m), 8), p=1),
            # 06 RandomToneCurve Speed: 650 images / sec
            A.RandomToneCurve(scale=0.6 * m, p=1),
            # 07 Gamma Speed: 1300 images / sec
            A.RandomGamma(gamma_limit=(100 - 60 * m, 100 + 60 * m), p=1),
            # 08 Downscale Speed: 1250 images / sec
            A.Downscale(scale_min=0.99 - 0.49 * m, scale_max=0.99, p=1),
            # 09 Dropout Speed: 1250 images / sec
            A.CoarseDropout(
                max_holes=round(1 + 99 * m),
                min_holes=1,
                max_height=0.1 * m,
                min_height=0.001,
                max_width=0.1 * m,
                min_width=0.001,
                p=1,
            ),
            # 10~16 Geometric Speed: 250 images / sec
            A.Affine(
                p=1,
                scale={"x": (1 - 0.4 * m, 1 + 0.4 * m), "y": (1, 1)},
                translate_percent=None,
                rotate=None,
                shear=None,
            ),
            A.Affine(
                p=1,
                scale={"x": (1, 1), "y": (1 - 0.4 * m, 1 + 0.4 * m)},
                translate_percent=None,
                rotate=None,
                shear=None,
            ),
            A.Affine(
                p=1,
                scale=None,
                translate_percent={"x": (-0.4 * m, 0.4 * m), "y": (0, 0)},
                rotate=None,
                shear=None,
            ),
            A.Affine(
                p=1,
                scale=None,
                translate_percent={"x": (0, 0), "y": (-0.4 * m, 0.4 * m)},
                rotate=None,
                shear=None,
            ),
            A.Affine(
                p=1,
                scale=None,
                translate_percent=None,
                rotate=(-45 * m, 45 * m),
                shear=None,
            ),
            A.Affine(
                p=1,
                scale=None,
                translate_percent=None,
                rotate=None,
                shear={"x": (-45 * m, 45 * m), "y": (0, 0)},
            ),
            A.Affine(
                p=1,
                scale=None,
                translate_percent=None,
                rotate=None,
                shear={"x": (0, 0), "y": (-45 * m, 45 * m)},
            ),
            # 17 Convolutional
            A.OneOf(
                [
                    A.Emboss(
                        alpha=(0, m), strength=1, p=1
                    ),  # Emboss Speed: 350 images / sec
                    A.Sharpen(alpha=(0, m), lightness=1, p=1),
                ]
            ),  # Sharpen Speed: 300 images / sec
            # 18 RGB Speed: 300 images / sec
            A.RGBShift(
                r_shift_limit=(-150 * m, 150 * m),
                g_shift_limit=(-150 * m, 150 * m),
                b_shift_limit=(-150 * m, 150 * m),
                p=1,
            ),
            # 19 Equalize Speed: 250 images / sec
            A.Equalize(p=1),
            # 20 HSV Speed: 150 images / sec
            A.HueSaturationValue(
                hue_shift_limit=(-60 * m, 60 * m),
                sat_shift_limit=(-60 * m, 60 * m),
                val_shift_limit=(-60 * m, 60 * m),
                p=1,
            ),
        ]
    
        random_transforms = random.sample(all_transforms, n)

    else:

        random_transforms = []

    constant_transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ToFloat(),
        ToTensorV2(),
    ]

    chosen_transforms = random_transforms + constant_transforms

    return A.Compose(
        chosen_transforms,
        keypoint_params=A.KeypointParams(
            format="xy", remove_invisible=False, label_fields=['fish_id']),
    )

def get_rand_aug_keep_center(n: float, m: float):
    """
    Returns a customized implementation of the RandAugment data augmentation technique, using albumentations library.
    The number of augmentation operations to apply (N) is randomly sampled from a uniform distribution between n/2 and n.
    This version of RandAugment will keep the center of the image unchanged. And is intended to be used for classification tasks.
    Args:
        n (float): Number of augmentation operations to apply.
        m (float): Magnitude of each operation on a scale of 0 to 20.
    Returns:
        albumentations.Compose: A sequence of N augmentation operations, randomly sampled from a larger set of options, to be applied to input images.
    """
    m = m / 16

    n = min(randint_from_float(n/2, n), 16)

    if n != 0 and m != 0:

        all_transforms = [
            # 01 Blur Speed: 950 images / sec
            A.GaussianBlur(blur_limit=(1, round(1 + (8 * m) // 2 * 2)), p=1),
            # 02 Noise Speed: 1000 images / sec
            A.MultiplicativeNoise(multiplier=(1 - 0.9 * m, 1 + 0.9 * m), p=1),
            # 03 Brightness & Contrast Speed: 1100 images / sec
            A.RandomBrightnessContrast(
                brightness_limit=(-0.6 * m, 0.6 * m),
                contrast_limit=(-0.6 * m, 0.6 * m),
                p=1,
            ),
            # 04 Solarize Speed: 1300 images / sec
            A.Solarize(threshold=(255 - 255 * m, 255), p=1),
            # 05 Posterize Speed: 1300 images / sec
            A.Posterize(num_bits=(round(8 - 4 * m), 8), p=1),
            # 06 RandomToneCurve Speed: 650 images / sec
            A.RandomToneCurve(scale=0.6 * m, p=1),
            # 07 Gamma Speed: 1300 images / sec
            A.RandomGamma(gamma_limit=(100 - 60 * m, 100 + 60 * m), p=1),
            # 08 Downscale Speed: 1250 images / sec
            A.Downscale(scale_min=0.99 - 0.49 * m, scale_max=0.99, p=1),
            # 09 Dropout Speed: 1250 images / sec
            A.CoarseDropout(
                max_holes=round(1 + 99 * m),
                min_holes=1,
                max_height=0.1 * m,
                min_height=0.001,
                max_width=0.1 * m,
                min_width=0.001,
                p=1,
            ),
            # 10~12 Geometric Speed: 250 images / sec
            A.Affine(
                p=1,
                scale={"x": (1 - 0.4 * m, 1 + 0.4 * m), "y": (1, 1)},
                translate_percent=None,
                rotate=None,
                shear=None,
            ),
            A.Affine(
                p=1,
                scale={"x": (1, 1), "y": (1 - 0.4 * m, 1 + 0.4 * m)},
                translate_percent=None,
                rotate=None,
                shear=None,
            ),
            A.Affine(
                p=1,
                scale=None,
                translate_percent=None,
                rotate=(-45 * m, 45 * m),
                shear=None,
            ),
            # 13 Convolutional
            A.OneOf(
                [
                    A.Emboss(
                        alpha=(0, m), strength=1, p=1
                    ),  # Emboss Speed: 350 images / sec
                    A.Sharpen(alpha=(0, m), lightness=1, p=1),
                ]
            ),  # Sharpen Speed: 300 images / sec
            # 14 RGB Speed: 300 images / sec
            A.RGBShift(
                r_shift_limit=(-150 * m, 150 * m),
                g_shift_limit=(-150 * m, 150 * m),
                b_shift_limit=(-150 * m, 150 * m),
                p=1,
            ),
            # 15 Equalize Speed: 250 images / sec
            A.Equalize(p=1),
            # 16 HSV Speed: 150 images / sec
            A.HueSaturationValue(
                hue_shift_limit=(-60 * m, 60 * m),
                sat_shift_limit=(-60 * m, 60 * m),
                val_shift_limit=(-60 * m, 60 * m),
                p=1,
            ),
        ]
    
        random_transforms = random.sample(all_transforms, n)

    else:

        random_transforms = []

    constant_transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ToFloat(),
        ToTensorV2(),
    ]

    chosen_transforms = random_transforms + constant_transforms

    return A.Compose(
        chosen_transforms
    )

class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = []
        self.y = []
        for pos_seq, pos in tqdm(zip(x, y), total=len(x)):
            pos -= pos_seq[0] # Normalize the positions
            pos_seq = pos_seq[1:] - pos_seq[0] # Normalize the positions
            velocity = pos - pos_seq[-1] # The pred target is the velocity from the last position
            y_dict = {'h': torch.tensor(velocity[0:2], dtype=torch.float32), # Predicting Head
                      'm': torch.tensor(velocity[2:4], dtype=torch.float32),
                      't': torch.tensor(velocity[4:6], dtype=torch.float32),
                      'hm': torch.tensor(velocity[0:4], dtype=torch.float32), # Predicting Head and Midsec
                      'mt': torch.tensor(velocity[2:6], dtype=torch.float32),
                      'ht': torch.tensor(velocity[[0, 1, 4, 5]], dtype=torch.float32),
                      'hmt': torch.tensor(velocity, dtype=torch.float32)}
            x_dict = {'seq': torch.tensor(pos_seq.flatten(), dtype=torch.float32),
                      'h': torch.tensor(pos[2:6], dtype=torch.float32), # To predict head, we need midsec and tail
                      'm': torch.tensor(pos[[0, 1, 4, 5]], dtype=torch.float32),
                      't': torch.tensor(pos[0:4], dtype=torch.float32),
                        'hm': torch.tensor(pos[4:6], dtype=torch.float32), # To predict head and midsec, we need tail
                        'mt': torch.tensor(pos[0:2], dtype=torch.float32),
                        'ht': torch.tensor(pos[2:4], dtype=torch.float32),
                        'hmt': torch.tensor([], dtype=torch.float32)}
            self.x.append(x_dict)
            self.y.append(y_dict)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class TestDataset(Dataset):

    def __init__(self, images):
        # The images is with shape (n, height, width, channel)
        # we need to convert it to (n, channel, height, width)
        self.transforms = A.Compose([A.ToFloat(), ToTensorV2()])
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transforms(image=self.images[idx])['image']
    
class FrameIterableDataset(IterableDataset):
    def __init__(self, frame_generator):
        super().__init__()
        self.frame_generator = frame_generator
        self.transform = A.Compose([A.ToFloat(), ToTensorV2()])

    def __iter__(self):
        for frame in self.frame_generator:
            frame = self.transform(image=frame)['image']
            yield frame

class KeypointsHeatmapDataset(Dataset):
    """
    A dataset that generates the input image and the corresponding heatmap for each image and keypoints.
    Important Note:
    1. The shape of images, heatmaps or other included arrays (e.g. offset maps) should be (height, width, channel) or (channel, height, width).
    2. We use (x, y) coordinates for the keypoints, where x is the horizontal coordinate and y is the vertical coordinate.
    """

    _kwarg = tuple()

    def __init__(
        self, images, keypoints, heatmap_sigma, heatmap_shape, inference=False, **kwarg
    ):
        """
        Initializes the dataset.
        Note: This will not generate the input images and the corresponding heatmaps if the dataset is not in inference mode.
        Args:
            images (list): A list of input images.
            keypoints (list): A list of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            heatmap_sigma (int): The standard deviation of the Gaussian distribution to be used for generating the heatmap.
            heatmap_shape (tuple): A tuple of (height, width) values for the heatmap.
            inference (bool): Whether the dataset is in inference mode.
        """
        count = 0
        for key in kwarg:
            if key not in self._kwarg:
                raise Exception(
                    f"Received unexpected keyword argument: ('{key}'). Accepted keyword arguments include {self._kwarg}")
            else:
                count += 1
        if count != len(self._kwarg):
            raise Exception(
                f"Missing required keyword argument(s). Required keyword argument(s) include: {self._kwarg}")
        self.images = images
        self.keypoints = keypoints
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_shape = heatmap_shape
        self.inference = inference
        self.kwarg = kwarg
        self._is_initialized = False
        self.gaussian_kernel = self.get_gaussian_kernel(self.heatmap_sigma)
        if self.inference:
            self.sequential_initialize(rangaug_param=None, include_coords=True)
            self.report()

    def report(self):
        """
        Prints the length of the dataset, and the shape of the input and output.
        """
        if self._is_initialized:
            print(f'Length of dataset: {len(self.data)}')
            print(f'Shape of X: {self.data[0][0].shape}')
            print(f'Shape of Y: {self.data[0][1].shape}')
            print(f'Shape of Y coords: {self.data[0][2].shape}')
        else:
            raise Exception("Dataset not initialized.")

    def sequential_initialize(self, rangaug_param, include_coords=False, **kwarg):
        """
        Initializes the dataset by generating the input images and the corresponding heatmaps for each image without using parallelization.
        Args:
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
            include_coords (bool): Whether to include the coordinates of the keypoints in the output.
        """
        if self.inference and self._is_initialized:
            raise Exception(
                "Dataset is in inference mode and has been initialized.")
        self.data = []
        for img, kps in tqdm(zip(self.images, self.keypoints), total=len(self.images)):
            self.data.append(self.generate_xy(img, kps, rangaug_param, self.gaussian_kernel,
                             self.heatmap_shape, include_coords=include_coords, **self.kwarg))
        self._is_initialized = True

    def parallel_initialize(self, rangaug_param, batch_size, include_coords=False):
        """
        Initializes the dataset by generating the input images and the corresponding heatmaps for each image using parallelization.
        Args:
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
            batch_size (int): The batch size to be used for parallelization.
            include_coords (bool): Whether to include the coordinates of the keypoints in the output.
        """
        if self.inference and self._is_initialized:
            raise Exception(
                "Dataset is in inference mode and has been initialized.")
        # Create batches
        image_batches = [
            self.images[i: i + batch_size]
            for i in range(0, len(self.images), batch_size)
        ]
        kps_batches = [
            self.keypoints[i: i + batch_size]
            for i in range(0, len(self.keypoints), batch_size)
        ]

        # Create delayed tasks for each batch
        tasks = [
            delayed(self.batch_generate_xy)(
                img_batch, kps_batch, rangaug_param, self.gaussian_kernel, self.heatmap_shape, include_coords, **self.kwarg
            )
            for img_batch, kps_batch in zip(image_batches, kps_batches)
        ]

        # Compute all tasks, then flatten the result into a single list
        self.data = [item for sublist in compute(*tasks) for item in sublist]
        self._is_initialized = True

    def batch_generate_xy(
        self, images, keypoints, rangaug_param, gaussian_kernel, heatmap_shape, include_coords, **kwarg
    ):
        """
        Generates the input image and the corresponding heatmap for the given image and keypoints.
        Args:
            images (list): A list of input images.
            keypoints (list): A list of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
            gaussian_kernel (np.ndarray): A 2D Gaussian kernel array.
            heatmap_shape (tuple): A tuple of (height, width) values for the heatmap.
        Returns:
            A list of tuples of (image, heatmap) where image is a torch.Tensor of shape (3, height, width) and heatmap is a torch.Tensor of shape (1, height, width).
            If include_coords is True, then the tuples will also contain the coordinates of the keypoints.
        """
        return [
            self.generate_xy(img, kps, rangaug_param, gaussian_kernel,
                             heatmap_shape, include_coords, **kwarg)
            for img, kps in zip(images, keypoints)
        ]

    def generate_xy(
        self, image, keypoints, rangaug_param, gaussian_kernel, heatmap_shape, include_coords, **kwarg
    ):
        """
        Generates the input image and the corresponding heatmap for the given image and keypoints.
        Args:
            image (np.ndarray): The input image.
            keypoints (list): A list of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
            gaussian_kernel (np.ndarray): A 2D Gaussian kernel array.
            heatmap_shape (tuple): A tuple of (height, width) values for the heatmap.
            include_coords (bool): Whether to include the coordinates of the keypoints in the output.
        Returns:
            A tuple of (image, heatmap) where image is a torch.Tensor of shape (3, height, width) and heatmap is a torch.Tensor of shape (1, height, width).
            If include_coords is True, then the tuple will also contain the coordinates of the keypoints.
        """
        # Apply RandAugment
        aug_result = self.apply_randaugment(image, keypoints, rangaug_param)

        # Create heatmap
        ratio_x = image.shape[1] / heatmap_shape[1] # Image shape is (height, width, channel)
        ratio_y = image.shape[0] / heatmap_shape[0]
        resized_keypoints = self.resize_keypoints(
            aug_result["keypoints"], ratio_x, ratio_y)
        heatmap = self.generate_map(
            heatmap_shape, resized_keypoints, gaussian_kernel)

        if include_coords:
            return aug_result["image"], heatmap, resized_keypoints
        else:
            return aug_result["image"], heatmap

    def resize_keypoints(self, keypoints, ratio_x, ratio_y):
        """
        Resizes the keypoints to match the size of the heatmap.
        Args:
            keypoints (list or dict): A list of keypoints, where each keypoint is a tuple of (x, y) coordinates.
                                      Or a dict of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            ratio_x (float): The ratio of the width of the heatmap to the width of the original image.
            ratio_y (float): The ratio of the height of the heatmap to the height of the original image.
        Returns:
            A list of resized keypoints, where each keypoint is a tuple of (x, y) coordinates.
        """
        if isinstance(keypoints, dict):
            return {key: (x / ratio_x, y / ratio_y) for key, (x, y) in keypoints.items()}
        return np.array([(x / ratio_x, y / ratio_y) for x, y in keypoints])

    def apply_randaugment(self, image, keypoints, rangaug_param, num_keypoints_per_fish=2):
        """
        Applies RandAugment to the given image and keypoints.
        If rangaug_param is None, then no RandAugment will be applied.
        Instead, the image will be converted to torch.Tensor and returned together with the keypoints.
        Args:
            image (np.ndarray): The input image.
            keypoints (list): A list of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
            num_keypoints_per_fish (int): The number of keypoints per fish.
        Returns:
            A dictionary containing the augmented image and keypoints.
        """
        if rangaug_param is None:
            transforms = A.Compose(
                [A.ToFloat(), ToTensorV2()],
                keypoint_params=A.KeypointParams(format="xy"),
            )
            return transforms(image=image, keypoints=keypoints,
                            fish_id=[(i // num_keypoints_per_fish + i % num_keypoints_per_fish * 100)
                                    for i in range(len(keypoints))])
        else:
            return get_rand_aug(*rangaug_param)(image=image, keypoints=keypoints,
                                                fish_id=[(i // num_keypoints_per_fish + i % num_keypoints_per_fish * 100)
                                                         for i in range(len(keypoints))])

    def generate_map(self, heatmap_shape, keypoints, gaussian_kernel):
        """
        Generates a heatmap for the given keypoints.
        Args:
            heatmap_shape (tuple): A tuple of (height, width) values for the heatmap.
            keypoints (list): A list of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            gaussian_kernel (np.ndarray): A 2D Gaussian kernel array.
        Returns:
            A torch.Tensor of shape (1, height, width) containing the heatmap.
        """
        heatmap = np.zeros(heatmap_shape, dtype=np.float32)
        for x, y in keypoints:
            # Ensure the keypoints are within the heatmap boundaries
            if (
                x >= 0 and y >= 0 and y < heatmap.shape[0] and x < heatmap.shape[1]
            ):  # x is the horizontal coordinate and y is the vertical coordinate
                self.place_kernel(heatmap, gaussian_kernel, round(x), round(y))
        return torch.tensor(heatmap).unsqueeze(0)

    def get_gaussian_kernel(self, sigma):
        """
        Returns a 2D Gaussian kernel array with the given standard deviation (sigma).
        Args:
            sigma (int): The standard deviation of the Gaussian distribution to be used.
        Returns:
            A numpy array of shape (ceil(8 * sigma + 1), ceil(8 * sigma + 1)) containing the 2D Gaussian kernel.
        """
        gk1d = signal.gaussian(math.ceil(8 * sigma + 1), std=sigma).reshape(
            math.ceil(8 * sigma + 1), 1
        )
        gk2d = np.outer(gk1d, gk1d)
        return gk2d

    def place_kernel(self, heatmap, kernel, x, y, abs_maximum=False):
        """
        Places the given kernel onto the heatmap at the given coordinates.
        Args:
            heatmap (np.ndarray): The heatmap to place the kernel onto.
            kernel (np.ndarray): The kernel to be placed onto the heatmap.
            x (int): The x-coordinate of the center of the kernel. (Second dimension of the heatmap array)
            y (int): The y-coordinate of the center of the kernel. (First dimension of the heatmap array)
            abs_maximum (bool): If True, the value with larger absolute value will be chosen; otherwise, the value with larger value will be chosen.
        """
        
        # Calculate the range of coordinates for placing the kernel
        start_y = max(0, y - kernel.shape[0] // 2)
        end_y = min(heatmap.shape[0], y + kernel.shape[0] // 2)
        start_x = max(0, x - kernel.shape[1] // 2)
        end_x = min(heatmap.shape[1], x + kernel.shape[1] // 2)
        
        # Calculate the range of coordinates for the kernel
        kernel_start_y = max(0, kernel.shape[0] // 2 - (y - start_y))
        kernel_end_y = min(kernel.shape[0], kernel.shape[0] // 2 + (end_y - y))
        kernel_start_x = max(0, kernel.shape[1] // 2 - (x - start_x))
        kernel_end_x = min(kernel.shape[1], kernel.shape[1] // 2 + (end_x - x))
        
        # Place the kernel onto the heatmap
        if not abs_maximum:
            heatmap[start_y:end_y, start_x:end_x] = np.maximum(
                heatmap[start_y:end_y, start_x:end_x],
                kernel[kernel_start_y:kernel_end_y, kernel_start_x:kernel_end_x],
            )
        else:
            abs_heatmap = np.abs(heatmap[start_y:end_y, start_x:end_x])
            abs_kernel = np.abs(kernel[kernel_start_y:kernel_end_y, kernel_start_x:kernel_end_x])

            # Use np.where to choose the value from heatmap or kernel based on the condition
            heatmap[start_y:end_y, start_x:end_x] = np.where(
                abs_heatmap >= abs_kernel,
                heatmap[start_y:end_y, start_x:end_x],
                kernel[kernel_start_y:kernel_end_y, kernel_start_x:kernel_end_x]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns the input image and the corresponding heatmap for the given index.
        Args:
            idx (int): The index of the image and heatmap to be returned.
        Returns:
            A tuple of (image, heatmap) where image is a torch.Tensor of shape (3, height, width) and heatmap is a torch.Tensor of shape (1, height, width).
            Note: In inference mode, the returned tuple will also contain the coordinates of the keypoints.
        """
        if not self._is_initialized:
            raise Exception("Dataset not initialized.")
        else:
            return self.data[idx]


class DoubleKeypointsHeatmapDataset(KeypointsHeatmapDataset):

    _kwarg = ('distance', )

    def generate_xy(
        self, image, keypoints, rangaug_param, gaussian_kernel, heatmap_shape, include_coords, distance, **kwarg
    ):
        """
        Generates the input image and the corresponding heatmap for the given image and keypoints.
        Args:
            image (np.ndarray): The input image.
            keypoints (list): A list of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
            gaussian_kernel (np.ndarray): A 2D Gaussian kernel array.
            heatmap_shape (tuple): A tuple of (height, width) values for the heatmap.
            distance (int): The distance between the two keypoints.
            include_coords (bool): Whether to include the coordinates of the keypoints in the output.
        Returns:
            A tuple of (image, heatmap) where image is a torch.Tensor of shape (3, height, width) and heatmap is a torch.Tensor of shape (2, height, width).
            If include_coords is True, then the tuple will also contain the coordinates of the keypoints.
        """
        # Apply RandAugment
        aug_result = self.apply_randaugment(image, keypoints, rangaug_param)

        # Normalize the distance between two keypoints
        keypoints_head, keypoints_middle = self.normalize_keypoints_from_augment_result(
            aug_result, distance)

        # Create heatmap
        ratio_x = image.shape[1] / heatmap_shape[1]
        ratio_y = image.shape[0] / heatmap_shape[0]
        resized_keypoints_head = self.resize_keypoints(
            keypoints_head, ratio_x, ratio_y)
        resized_keypoints_middle = self.resize_keypoints(
            keypoints_middle, ratio_x, ratio_y)
        map_head = self.generate_map(
            heatmap_shape, resized_keypoints_head, gaussian_kernel)
        map_middle = self.generate_map(
            heatmap_shape, resized_keypoints_middle, gaussian_kernel)

        # Cat heatmaps along a new dimension to create a 3D tensor
        heatmaps = torch.cat((map_head, map_middle), dim=0)

        if include_coords:
            return aug_result["image"], heatmaps, np.array([resized_keypoints_head, resized_keypoints_middle])
        else:
            return aug_result["image"], heatmaps

    def normalize_keypoints_from_augment_result(self, augment_result, distance, return_dict=False):
        """
        Normalizes the distance between two keypoints.
        Args:
            augment_result (dict): A dictionary containing the augmented image and keypoints.
            distance (int): The distance between the two keypoints.
        Returns:
            A tuple of (keypoints_head, keypoints_middle) where keypoints_head is a list of keypoints at the head of the fish,
            and keypoints_middle is a list of keypoints at the middle of the fish.
        """
        if return_dict:
            keypoints_head = dict()
            keypoints_middle = dict()
        else:
            keypoints_head = []
            keypoints_middle = []
        for index, id in enumerate(augment_result['fish_id']):
            if id // 100 == 0:
                if return_dict:
                    keypoints_head[id] = augment_result['keypoints'][index]
                else:
                    keypoints_head.append(augment_result['keypoints'][index])
            elif id // 100 == 1:
                if id % 100 in augment_result['fish_id']:
                    head_index = augment_result['fish_id'].index(id % 100)
                    if return_dict:
                        keypoints_middle[id % 100] = self.normalize_distance(augment_result['keypoints'][head_index],
                                                                        augment_result['keypoints'][index],
                                                                        distance)
                    else:
                        keypoints_middle.append(self.normalize_distance(augment_result['keypoints'][head_index],
                                                                        augment_result['keypoints'][index],
                                                                        distance))
                else:
                    pass
                    # raise Exception('Keypoint at tail failed to pair with keypoint at head')
            else: # Third Keypoint (tail)
                pass
            
        return keypoints_head, keypoints_middle

    def normalize_distance(self, x, y, new_distance):
        """
        Normalizes the distance between two keypoints.
        Args:
            x (tuple): A tuple of (x, y) coordinates for the first keypoint.
            y (tuple): A tuple of (x, y) coordinates for the second keypoint.
            new_distance (int): The distance between the two keypoints.
        Returns:
            A tuple of (x, y) coordinates for the second keypoint.
        """
        # Convert to numpy arrays for easier calculations
        x = np.array(x)
        y = np.array(y)

        # Calculate the vector from x to y
        v = y - x

        # Normalize the vector
        v_norm = v / np.linalg.norm(v)

        # Multiply the normalized vector by the new distance
        y_norm = x + v_norm * new_distance

        return y_norm


class DoubleKeypointsHeatmapLocalOffsetDataset(DoubleKeypointsHeatmapDataset):

    def generate_xy(self, image, keypoints, rangaug_param, gaussian_kernel, heatmap_shape, include_coords, distance, **kwarg):
        """
        Generates the input image, the corresponding heatmap and offset map for the given image and keypoints.
        Args:
            image (np.ndarray): The input image.
            keypoints (list): A list of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
            gaussian_kernel (np.ndarray): A 2D Gaussian kernel array.
            heatmap_shape (tuple): A tuple of (height, width) values for the heatmap.
            distance (int): The distance between the two keypoints.
            include_coords (bool): Whether to include the coordinates of the keypoints in the output.
        Returns:
            A tuple of (image, maps) where 'image' is a torch.Tensor of shape (3, height, width),
            'maps' is a torch.Tensor of shape (6, height, width) arranged in the order of:
            heatmap_head, local_offset_head_y, local_offset_head_x, heatmap_middle, local_offset_middle_y, local_offset_middle_x.
        """
        return super().generate_xy(image, keypoints, rangaug_param, gaussian_kernel, heatmap_shape, include_coords, distance, **kwarg)

    def generate_map(self, heatmap_shape, keypoints, gaussian_kernel):
        """
        Generates a heatmap and xy offset maps for the given keypoints.
        Args:
            heatmap_shape (tuple): A tuple of (height, width) values for the heatmap.
            keypoints (list): A list of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            gaussian_kernel (np.ndarray): A 2D Gaussian kernel array.
        Returns:
            A torch.Tensor of shape (3, height, width) containing the heatmap and offset map.
        """
        heatmap = np.zeros(heatmap_shape, dtype=np.float32)
        local_offset = np.zeros((2, ) + heatmap_shape, dtype=np.float32)
        for x, y in keypoints:
            # Ensure the keypoints are within the heatmap boundaries
            if (
                y >= 0 and x >= 0 and
                y < heatmap.shape[0] and x < heatmap.shape[1]
            ):
                self.place_kernel(heatmap, gaussian_kernel, round(x), round(y))
                self.place_local_offset(local_offset, x, y)
        # Cat the (h x w) heatmap and the (2 x h x w) local offset
        return torch.cat((torch.tensor(heatmap).unsqueeze(0), torch.tensor(local_offset)), dim=0)

    def place_local_offset(self, array, x, y):
        """
        Places a local offset onto the array at the given coordinates.
        Note: The first channel of array is the y-coordinate and the second channel is the x-coordinate.
        Args:
            array (np.ndarray): The array with shape (2, height, width) to place the local offset onto.
            x (float): The x-coordinate of the center of the local offset. (Third dimension of the array)
            y (float): The y-coordinate of the center of the local offset. (Second dimension of the array)
        """
        # Calculate the range of coordinates for placing the local offset
        start_y = max(0, math.floor(y))
        end_y = min(array.shape[1] - 1, math.floor(y) + 1)
        start_x = max(0, math.floor(x))
        end_x = min(array.shape[2] - 1, math.floor(x) + 1)

        if start_y <= end_y and start_x <= end_x:
            self.place_2x2_towards_grond_truth(array, x, y, start_x, end_x, start_y, end_y)

    def place_2x2_towards_grond_truth(self, array, gt_x, gt_y, start_x, end_x, start_y, end_y, normalize_factor=1):

        # Calculate the difference of the x-coordinate and the y-coordinate between the ground truth and the 2x2 location
        if start_y == end_y and start_x == end_x:
            x_range = 1
            y_range = 1
            y_offset = np.array([[gt_y - start_y]]) / normalize_factor
            x_offset = np.array([[gt_x - start_x]]) / normalize_factor
        elif start_y == end_y:
            x_range = 2
            y_range = 1
            y_offset = np.array([[gt_y - start_y, gt_y - start_y]]) / normalize_factor
            x_offset = np.array([[gt_x - start_x, gt_x - end_x]]) / normalize_factor
        elif start_x == end_x:
            x_range = 1
            y_range = 2
            y_offset = np.array([[gt_y - start_y], [gt_y - end_y]]) / normalize_factor
            x_offset = np.array([[gt_x - start_x], [gt_x - start_x]]) / normalize_factor
        else:
            x_range = 2
            y_range = 2
            y_offset = np.array([[gt_y - start_y, gt_y - start_y],
                                [gt_y - end_y, gt_y - end_y]]) / normalize_factor
            x_offset = np.array([[gt_x - start_x, gt_x - end_x],
                                [gt_x - start_x, gt_x - end_x]]) / normalize_factor

        # Check if there are non-zero values in the original array at the 2x2 location
        for i in range(y_range):
            for j in range(x_range):
                if array[0, start_y+i, start_x+j] != 0 or array[1, start_y+i, start_x+j] != 0:
                    if array[0, start_y+i, start_x+j] ** 2 + array[1, start_y+i, start_x+j] ** 2 < y_offset[i, j] ** 2 + x_offset[i, j] ** 2:
                        y_offset[i, j] = array[0, start_y+i, start_x+j]
                        x_offset[i, j] = array[1, start_y+i, start_x+j]

        # Place the 2x2 difference of the y-coordinate onto the array
        array[0, start_y:end_y+1, start_x:end_x+1] = y_offset

        # Place the 2x2 difference of the x-coordinate onto the array
        array[1, start_y:end_y+1, start_x:end_x+1] = x_offset

class DoubleKeypointsHeatmapLocalOffsetDoublecheckDataset(DoubleKeypointsHeatmapLocalOffsetDataset):

    def generate_xy(self, image, keypoints, rangaug_param, gaussian_kernel, heatmap_shape, include_coords, distance, **kwarg):
        """
        Generates the input image and the corresponding heatmap for the given image and keypoints.
        Args:
            image (np.ndarray): The input image.
            keypoints (list): A list of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
            gaussian_kernel (np.ndarray): A 2D Gaussian kernel array.
            heatmap_shape (tuple): A tuple of (height, width) values for the heatmap.
            distance (int): The distance between the two keypoints.
            include_coords (bool): Whether to include the coordinates of the keypoints in the output.
        Returns:
            A tuple of (image, heatmap) where image is a torch.Tensor of shape (3, height, width) and heatmap is a torch.Tensor of shape (2, height, width).
            If include_coords is True, then the tuple will also contain the coordinates of the keypoints.
        """
        # Apply RandAugment
        aug_result = self.apply_randaugment(image, keypoints, rangaug_param)

        # Normalize the distance between two keypoints
        keypoints_head, keypoints_middle = self.normalize_keypoints_from_augment_result(
            aug_result, distance, return_dict=True)

        # Create heatmap
        ratio_x = image.shape[1] / heatmap_shape[1]
        ratio_y = image.shape[0] / heatmap_shape[0]
        double_check_normalization_factor = distance / math.sqrt(ratio_x * ratio_y) # After normalization, the value of double checker will be typically within (-1, 1)
        resized_keypoints_head = self.resize_keypoints(
            keypoints_head, ratio_x, ratio_y)
        resized_keypoints_middle = self.resize_keypoints(
            keypoints_middle, ratio_x, ratio_y)
        map_head = self.generate_map(
            heatmap_shape, resized_keypoints_head, gaussian_kernel, resized_keypoints_middle, double_check_normalization_factor) # This also contains the double-check map
        map_middle = self.generate_map(
            heatmap_shape, resized_keypoints_middle, gaussian_kernel, resized_keypoints_head, double_check_normalization_factor)

        # Cat heatmaps along a new dimension to create a 3D tensor
        heatmaps = torch.cat((map_head, map_middle), dim=0)

        if include_coords:
            return aug_result["image"], heatmaps, np.array([list(resized_keypoints_head.values()),
                                                            list(resized_keypoints_middle.values())])
        else:
            return aug_result["image"], heatmaps
        
    def generate_map(self, heatmap_shape, keypoints, gaussian_kernel, double_check_target, double_check_normalization_factor):
        """
        Generates a heatmap, xy offset maps and xy double-check maps for the given keypoints.
        Args:
            heatmap_shape (tuple): A tuple of (height, width) values for the heatmap.
            keypoints (dict): A dictionary of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            gaussian_kernel (np.ndarray): A 2D Gaussian kernel array.
            double_check_target (dict): A dictionary containing the target coordinates for the double-check map.
            double_check_normalization_factor (float): The normalization factor for the double-check map.
        Returns:
            A torch.Tensor of shape (5, height, width) containing the heatmap and offset map.
        """
        heatmap_and_offset_map = super().generate_map(heatmap_shape, list(keypoints.values()), gaussian_kernel)

        double_check_map = np.zeros((2, ) + heatmap_shape, dtype=np.float32)
        for key in keypoints:
            if key in double_check_target:
                self.place_double_check_map(double_check_map, *keypoints[key], *double_check_target[key], double_check_normalization_factor)

        return torch.cat((heatmap_and_offset_map, torch.tensor(double_check_map)), dim=0)
    
    def place_double_check_map(self, array, from_x, from_y, to_x, to_y, double_check_normalization_factor):

        # Calculate the range of coordinates for placing the local offset
        start_y = max(0, math.floor(from_y))
        end_y = min(array.shape[1] - 1, math.floor(from_y) + 1)
        start_x = max(0, math.floor(from_x))
        end_x = min(array.shape[2] - 1, math.floor(from_x) + 1)

        if start_y <= end_y and start_x <= end_x:
            self.place_2x2_towards_grond_truth(array, to_x, to_y, start_x, end_x, start_y, end_y, double_check_normalization_factor)

class TripleKeypointsHeatmapLocalOffsetDoublecheckClassifyDataset(DoubleKeypointsHeatmapLocalOffsetDoublecheckDataset):

    _kwarg = ('head_midsec_distance', 'mean_distances')

    def dict_to_sorted_list(self, dictionary, num_fish):
        """
        Converts a dictionary to a sorted list of values.
        Args:
            dictionary (dict): A dictionary to be converted.
        Returns:
            A list of values sorted in the order of the keys.
        """
        return [dictionary.get(key, (0, 0)) for key in range(num_fish)]

    def generate_xy(self, image, keypoints, rangaug_param, gaussian_kernel, heatmap_shape, include_coords, head_midsec_distance, mean_distances, **kwarg):
        """
        Generates the input image, the corresponding heatmap and classification label for the given image and keypoints.
        """
        # Extract coords from 'keypoints'. The 'keypoints' is a list values sorted in this order:
        # [ [head_x, head_y], [midsection_x, midsection_y], [tail_x, tail_y], [classificaiton_label_a, classification_label_b],
        # repeat for the next fish... ]
        aug_result = self.apply_randaugment(
            image, [keypoints[i]
                    for i in range(len(keypoints))
                    if i % 4 != 3 and sum(keypoints[i]) != 0],
            rangaug_param, num_keypoints_per_fish=3)

        # Normalize the distance between head and midsection.
        keypoints_head, keypoints_middle = self.normalize_keypoints_from_augment_result(aug_result, head_midsec_distance, return_dict=True)

        # Get tail keypoints and classification labels
        keypoints_tail = {id-200: aug_result['keypoints'][index] for index, id in enumerate(aug_result['fish_id']) if id // 100 == 2}
        classification_labels = {index // 4: keypoints[index] for index in range(3, len(keypoints), 4)}

        # Create heatmap
        ratio_x = image.shape[1] / heatmap_shape[1]
        ratio_y = image.shape[0] / heatmap_shape[0]
        head_midsec_double_check_normalization_factor = head_midsec_distance / math.sqrt(ratio_x * ratio_y) # After normalization, the value of double checker will be typically within (-1, 1)
        head_tail_double_check_normalization_factor = mean_distances[('head', 'tail')] / math.sqrt(ratio_x * ratio_y)
        midsec_tail_double_check_normalization_factor = mean_distances[('midsec', 'tail')] / math.sqrt(ratio_x * ratio_y)
        keypoints_head = self.resize_keypoints(
            keypoints_head, ratio_x, ratio_y)
        keypoints_middle = self.resize_keypoints(
            keypoints_middle, ratio_x, ratio_y)
        keypoints_tail = self.resize_keypoints(
            keypoints_tail, ratio_x, ratio_y)
        map_head = self.generate_map(
            heatmap_shape, keypoints_head, gaussian_kernel,
            keypoints_middle, keypoints_tail, classification_labels,
            head_midsec_double_check_normalization_factor,
            head_tail_double_check_normalization_factor) # This also contains the offset map and double-check map
        map_middle = self.generate_map(
            heatmap_shape, keypoints_middle, gaussian_kernel,
            keypoints_head, keypoints_tail, classification_labels,
            head_midsec_double_check_normalization_factor,
            midsec_tail_double_check_normalization_factor)
        map_tail = self.generate_map(
            heatmap_shape, keypoints_tail, gaussian_kernel,
            keypoints_head, keypoints_middle, classification_labels,
            head_tail_double_check_normalization_factor,
            midsec_tail_double_check_normalization_factor)

        # Cat heatmaps along a new dimension to create a 3D tensor
        heatmaps = torch.cat((map_head, map_middle, map_tail), dim=0)

        if include_coords:
            num_fish = len(classification_labels.values())
            return aug_result["image"], heatmaps, np.array(self.dict_to_sorted_list(keypoints_head, num_fish) +
                                                            self.dict_to_sorted_list(keypoints_middle, num_fish) +
                                                            self.dict_to_sorted_list(keypoints_tail, num_fish) +
                                                            self.dict_to_sorted_list(classification_labels, num_fish))
        else:
            return aug_result["image"], heatmaps
        
    def generate_map(self, heatmap_shape, keypoints, gaussian_kernel, double_check_target_a, double_check_target_b, classification_labels,
                     double_check_normalization_factor_a, double_check_normalization_factor_b):
        """
        Generates a heatmap, xy offset maps and xy double-check maps for the given keypoints.
        Args:
            heatmap_shape (tuple): A tuple of (height, width) values for the heatmap.
            keypoints (list): A list of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            gaussian_kernel (np.ndarray): A 2D Gaussian kernel array.
            double_check_target_a (dict): A dictionary containing the target coordinates for the double-check map.
            double_check_target_b (dict): A dictionary containing the target coordinates for the double-check map.
            classification_labels (dict): A dictionary containing the classification labels.
            double_check_normalization_factor_a (float): The normalization factor for the double-check map.
            double_check_normalization_factor_b (float): The normalization factor for the double-check map.
        Returns:
            A torch.Tensor of shape (9, height, width) containing:
            Dimension 0: heatmap of the keypoints
            Dimension 1&2: local offset of the keypoints
            Dimension 3&4: double-check map of the keypoints targeting double_check_target_a
            Dimension 5&6: double-check map of the keypoints targeting double_check_target_b
            Dimension 7&8: classification map of the keypoints
        """
        heatmap_and_offset_map = super(DoubleKeypointsHeatmapLocalOffsetDoublecheckDataset, self).generate_map(
            heatmap_shape, list(keypoints.values()), gaussian_kernel)
        
        double_check_map_a = np.zeros((2, ) + heatmap_shape, dtype=np.float32)
        for key in keypoints:
            if key in double_check_target_a:
                self.place_double_check_map(double_check_map_a, *keypoints[key], *double_check_target_a[key], double_check_normalization_factor_a)
        
        double_check_map_b = np.zeros((2, ) + heatmap_shape, dtype=np.float32)
        for key in keypoints:
            if key in double_check_target_b:
                self.place_double_check_map(double_check_map_b, *keypoints[key], *double_check_target_b[key], double_check_normalization_factor_b)
        
        classification_map = self.get_classification_map_v2(
            heatmap_shape, keypoints, classification_labels, gaussian_kernel)
        
        return torch.cat((heatmap_and_offset_map, torch.tensor(double_check_map_a), torch.tensor(double_check_map_b), torch.tensor(classification_map)), dim=0)
    
    def get_classification_map_v1(self, heatmap_shape, keypoints, classification_labels, gaussian_kernel):
        classification_map = np.full((2, ) + heatmap_shape, -1, dtype=np.float32)
        for key, (x, y) in keypoints.items():
            # Calculate the range of coordinates for placing the labels
            start_y = max(0, math.floor(y))
            end_y = min(classification_map.shape[1] - 1, math.floor(y) + 1)
            start_x = max(0, math.floor(x))
            end_x = min(classification_map.shape[2] - 1, math.floor(x) + 1)
            classification_map[0, start_y:end_y+1, start_x:end_x+1] = classification_labels[key][0]
            classification_map[1, start_y:end_y+1, start_x:end_x+1] = classification_labels[key][1]
        return classification_map
    
    def get_classification_map_v2(self,
                                  heatmap_shape,
                                  keypoints,
                                  classification_labels,
                                  gaussian_kernel):
        classification_map = np.zeros((2, ) + heatmap_shape, dtype=np.float32)
        for key, (x, y) in keypoints.items():
            # Ensure the keypoints are within the heatmap boundaries
            if (
                y >= 0 and x >= 0 and
                y < heatmap_shape[0] and x < heatmap_shape[1]
            ):
                if classification_labels[key][0] != 0:
                    self.place_kernel(classification_map[0],
                                      gaussian_kernel * classification_labels[key][0],
                                      round(x), round(y),
                                      abs_maximum=True)
                if classification_labels[key][1] != 0:
                    self.place_kernel(classification_map[1],
                                      gaussian_kernel * classification_labels[key][1],
                                      round(x), round(y),
                                      abs_maximum=True)
        return classification_map
            
    def report(self):
        print(f'Length of dataset: {len(self.data)}')
        print(f'Shape of X: {self.data[0][0].shape}')
        print(f'Shape of Y: {self.data[0][1].shape}')
        print(f'Shape of Y coords: {self.data[0][2].shape}')

class MemorySavingTripleKeypointsDataset(TripleKeypointsHeatmapLocalOffsetDoublecheckClassifyDataset):

    _kwarg = tuple()

    def __init__(
        self,
        images,
        keypoints,
        heatmap_sigma,
        heatmap_shape,
        head_midsec_distance,
        mean_distances,
        inference=False,
        initialization_batch=None,
        parallel=False,
        parallel_batch_size=None,
    ):
        """
        Initializes the dataset.
        Note: This will not generate the input images and the corresponding heatmaps if the dataset is not in inference mode.
        Args:
            images (list): A list of input images.
            keypoints (list): A list of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            heatmap_sigma (int): The standard deviation of the Gaussian distribution to be used for generating the heatmap.
            heatmap_shape (tuple): A tuple of (height, width) values for the heatmap.
            head_midsec_distance (int): The distance between the head and midsection keypoints.
            mean_distances (dict): A dictionary containing the mean distances between the keypoints.
            inference (bool): Whether the dataset is in inference mode.
        """
        count = 0
        self.images = images
        self.keypoints = keypoints
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_shape = heatmap_shape
        self.inference = inference
        self.kwarg = {'head_midsec_distance': head_midsec_distance,
                      'mean_distances': mean_distances}
        self.initialization_batch = initialization_batch
        self._is_initialized = False
        self.gaussian_kernel = self.get_gaussian_kernel(self.heatmap_sigma)
        self.parallel = parallel

        if self.inference:
            if self.parallel:
                self.parallel_initialize(rangaug_param=None, batch_size=parallel_batch_size, include_coords=True)
            else:
                self.sequential_initialize(rangaug_param=None, include_coords=True)
            self.report()
        
        if self.__len__() % self.initialization_batch != 0 and not self.inference:
            raise Exception(f'The length of the dataset {self.__len__()} must be divisible by initialization_batch {self.initialization_batch}')
        self.unshuffled_indices = list(range(self.__len__()))
    
    def sequential_initialize(self, rangaug_param, include_coords=False, **kwarg):

        self.rangaug_param = rangaug_param
        self.include_coords = include_coords

        if self.parallel:
            raise Exception('Parallelization is enabled. Use parallel_initialize() instead.')
        if self.inference:
            if self.initialization_batch is None:
                self.initialization_batch = len(self.images)
            self.sequential_batch_initialize(rangaug_param, include_coords, **kwarg)
            self.batch_index = 0
            self.index = 0
            self._is_initialized = True
        else:
            # shuffle the indices
            self.images, self.keypoints, self.unshuffled_indices = (
                sklearn.utils.shuffle(self.images, self.keypoints, self.unshuffled_indices)
            )
            self.sequential_batch_initialize(self.rangaug_param,
                                             self.include_coords,
                                             batch_index=0,
                                             **kwarg)
            self._is_initialized = True
    
    def sequential_batch_initialize(self, rangaug_param, include_coords=False, batch_index=0, **kwarg):
        
        self.data = [] # del(self.data)
        self.index = batch_index * self.initialization_batch
        self.batch_index = batch_index
        for img, kps in tqdm(zip(self.images[self.index:self.index+self.initialization_batch],
                                 self.keypoints[self.index:self.index+self.initialization_batch]),
                             total=self.initialization_batch):
            self.data.append(self.generate_xy(img, kps, rangaug_param, self.gaussian_kernel,
                                              self.heatmap_shape, include_coords=include_coords, **self.kwarg))
    
    def parallel_initialize(self,
                            rangaug_param,
                            batch_size,
                            include_coords=False,
                            **kwarg):
        
        self.rangaug_param = rangaug_param
        self.include_coords = include_coords
        
        if not self.parallel:
            raise Exception('Parallelization is disabled. Use sequential_initialize() instead.')
        
        if self.inference and self.initialization_batch is None:
            self.initialization_batch = len(self.images)

        if self.initialization_batch % batch_size != 0:
            raise Exception(f'The length of initialization_batch {self.initialization_batch} must be divisible by batch_size {batch_size}')
        else:
            self.parallel_batch_size = batch_size

        if self.inference:
            self.parallel_batch_initialize(rangaug_param, include_coords,
                                           batch_index=0, **kwarg)
            self.batch_index = 0
            self.index = 0
            self._is_initialized = True
        else:
            # shuffle the indices
            self.images, self.keypoints, self.unshuffled_indices = (
                sklearn.utils.shuffle(self.images, self.keypoints, self.unshuffled_indices)
            )
            self.parallel_batch_initialize(self.rangaug_param,
                                           self.include_coords,
                                           batch_index=0,
                                           **kwarg)
            self._is_initialized = True

    def parallel_batch_initialize(self,
                                  rangaug_param,
                                  include_coords=False,
                                  batch_index=0,
                                  **kwarg):
        
        self.data = [] # del(self.data)
        self.index = batch_index * self.initialization_batch
        self.batch_index = batch_index
        # Create batches
        image_batches = [
            self.images[i: i + self.parallel_batch_size]
            for i in range(self.index,
                           self.index + self.initialization_batch,
                           self.parallel_batch_size)
        ]
        kps_batches = [
            self.keypoints[i: i + self.parallel_batch_size]
            for i in range(self.index,
                           self.index + self.initialization_batch,
                           self.parallel_batch_size)
        ]

        # Create delayed tasks for each batch
        tasks = [
            delayed(self.batch_generate_xy)(
                img_batch, kps_batch, rangaug_param, self.gaussian_kernel, self.heatmap_shape, include_coords, **self.kwarg
            )
            for img_batch, kps_batch in zip(image_batches, kps_batches)
        ]

        # Compute all tasks, then flatten the result into a single list
        self.data = [item for sublist in compute(*tasks) for item in sublist]

    def __getitem__(self, idx):
        if not self._is_initialized:
            raise Exception("Dataset not initialized.")
        expected_batch_index = idx // self.initialization_batch
        if expected_batch_index == self.batch_index:
            return self.data[idx % self.initialization_batch]
        elif expected_batch_index == self.batch_index + 1:
            if idx >= self.__len__():
                raise IndexError(f'Index {idx} is larger than the length of the dataset {self.__len__()}')
            if self.parallel:
                self.parallel_batch_initialize(
                    self.rangaug_param, self.include_coords, expected_batch_index)
            else:
                self.sequential_batch_initialize(
                    self.rangaug_param, self.include_coords, expected_batch_index)
            return self.data[idx % self.initialization_batch]
        elif self.inference and expected_batch_index == 0 and self.batch_index == self.__len__() // self.initialization_batch - 1:
            if self.parallel:
                self.parallel_batch_initialize(
                    self.rangaug_param, self.include_coords, 0)
            else:
                self.sequential_batch_initialize(
                    self.rangaug_param, self.include_coords, 0)
            return self.data[idx % self.initialization_batch]
        else:
            raise Exception(f'Requested index {idx} is at batch index {expected_batch_index}. '
                            f'This is not sequential to current batch_index {self.batch_index}')
        
    def __len__(self):
        return len(self.images)
    
class TripleKeypointsHeatmapLocalOffsetDoublecheckClassifyDatasetV2(MemorySavingTripleKeypointsDataset):

    def generate_map(self,
                     heatmap_shape: tuple,
                     keypoints: dict,
                     gaussian_kernel: np.ndarray,
                     double_check_target_a: dict,
                     double_check_target_b: dict,
                     classification_labels: dict,
                     double_check_normalization_factor_a: float,
                     double_check_normalization_factor_b: float):
        """
        Generates a heatmap, xy offset maps and xy double-check maps for the given keypoints.
        Args:
            heatmap_shape (tuple): A tuple of (height, width) values for the heatmap.
            keypoints (dict): A dictionary of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            gaussian_kernel (np.ndarray): A 2D Gaussian kernel array.
            double_check_target_a (dict): A dictionary containing the target coordinates for the double-check map.
            double_check_target_b (dict): A dictionary containing the target coordinates for the double-check map.
            classification_labels (dict): A dictionary containing the classification labels.
            double_check_normalization_factor_a (float): The normalization factor for the double-check map.
            double_check_normalization_factor_b (float): The normalization factor for the double-check map.
        Returns:
            A torch.Tensor of shape (10, height, width) containing:
            Dimension 0: heatmap of the keypoints classified as 'not covering'
            Dimension 1: heatmap of the keypoints classified as 'covering'
            Dimension 2: heatmap of the keypoints classified as 'not covered'
            Dimension 3: heatmap of the keypoints classified as 'covered'
            Dimension 4&5: local offset of the keypoints
            Dimension 6&7: double-check map of the keypoints targeting double_check_target_a
            Dimension 8&9: double-check map of the keypoints targeting double_check_target_b
        """
        classified_heatmap = np.zeros((4, ) + heatmap_shape, dtype=np.float32)
        offset_map = np.zeros((2, ) + heatmap_shape, dtype=np.float32)
        for key, (x, y) in keypoints.items():
            # Ensure the keypoints are within the heatmap boundaries
            if (
                y >= 0 and x >= 0 and
                y < heatmap_shape[0] and x < heatmap_shape[1]
            ):
                if classification_labels[key][0] == 0:
                    self.place_kernel(classified_heatmap[0], gaussian_kernel, round(x), round(y))
                else:
                    self.place_kernel(classified_heatmap[1], gaussian_kernel, round(x), round(y))
                if classification_labels[key][1] == 0:
                    self.place_kernel(classified_heatmap[2], gaussian_kernel, round(x), round(y))
                else:
                    self.place_kernel(classified_heatmap[3], gaussian_kernel, round(x), round(y))
                self.place_local_offset(offset_map, x, y)
        
        double_check_map_a = np.zeros((2, ) + heatmap_shape, dtype=np.float32)
        for key in keypoints:
            if key in double_check_target_a:
                self.place_double_check_map(double_check_map_a, *keypoints[key], *double_check_target_a[key], double_check_normalization_factor_a)
        
        double_check_map_b = np.zeros((2, ) + heatmap_shape, dtype=np.float32)
        for key in keypoints:
            if key in double_check_target_b:
                self.place_double_check_map(double_check_map_b, *keypoints[key], *double_check_target_b[key], double_check_normalization_factor_b)
        
        return torch.cat((torch.tensor(classified_heatmap),
                          torch.tensor(offset_map),
                          torch.tensor(double_check_map_a),
                          torch.tensor(double_check_map_b)), dim=0)

class TripleKeypointsHeatmapLocalOffsetDoublecheckDataset(MemorySavingTripleKeypointsDataset):

    def generate_map(self,
                     heatmap_shape: tuple,
                     keypoints: dict,
                     gaussian_kernel: np.ndarray,
                     double_check_target_a: dict,
                     double_check_target_b: dict,
                     classification_labels: dict,
                     double_check_normalization_factor_a: float,
                     double_check_normalization_factor_b: float):
        """
        Generates a heatmap, xy offset maps and xy double-check maps for the given keypoints.
        Args:
            heatmap_shape (tuple): A tuple of (height, width) values for the heatmap.
            keypoints (dict): A dictionary of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            gaussian_kernel (np.ndarray): A 2D Gaussian kernel array.
            double_check_target_a (dict): A dictionary containing the target coordinates for the double-check map.
            double_check_target_b (dict): A dictionary containing the target coordinates for the double-check map.
            classification_labels (dict): This is not used in this function.
            double_check_normalization_factor_a (float): The normalization factor for the double-check map.
            double_check_normalization_factor_b (float): The normalization factor for the double-check map.
        Returns:
            A torch.Tensor of shape (7, height, width) containing:
            Dimension 0: heatmap of the keypoints
            Dimension 1&2: local offset of the keypoints
            Dimension 3&4: double-check map of the keypoints targeting double_check_target_a
            Dimension 5&6: double-check map of the keypoints targeting double_check_target_b
        """
        heatmap_and_offset_map = super(DoubleKeypointsHeatmapLocalOffsetDoublecheckDataset, self).generate_map(
            heatmap_shape, list(keypoints.values()), gaussian_kernel)
        
        double_check_map_a = np.zeros((2, ) + heatmap_shape, dtype=np.float32)
        for key in keypoints:
            if key in double_check_target_a:
                self.place_double_check_map(double_check_map_a, *keypoints[key], *double_check_target_a[key], double_check_normalization_factor_a)
        
        double_check_map_b = np.zeros((2, ) + heatmap_shape, dtype=np.float32)
        for key in keypoints:
            if key in double_check_target_b:
                self.place_double_check_map(double_check_map_b, *keypoints[key], *double_check_target_b[key], double_check_normalization_factor_b)
        
        return torch.cat((torch.tensor(heatmap_and_offset_map),
                          torch.tensor(double_check_map_a),
                          torch.tensor(double_check_map_b)), dim=0)
    
class TripleKeypointsHeatmapLocalOffsetDoublecheckDataset4X4(TripleKeypointsHeatmapLocalOffsetDoublecheckDataset):
    """
    This will create larger supervising areas for the double-check maps.
    The size is increased from 2x2 to 4x4.
    """

    def place_double_check_map(self, array, from_x, from_y, to_x, to_y, double_check_normalization_factor):
        
        super().place_double_check_map(array, from_x-1, from_y-1, to_x, to_y, double_check_normalization_factor)
        super().place_double_check_map(array, from_x-1, from_y+1, to_x, to_y, double_check_normalization_factor)
        super().place_double_check_map(array, from_x+1, from_y-1, to_x, to_y, double_check_normalization_factor)
        super().place_double_check_map(array, from_x+1, from_y+1, to_x, to_y, double_check_normalization_factor)

class LocalClassifyDataset(Dataset):
    """
    A dataset that generates the input image, coords and classification label for the given image and keypoints.
    Note that the input labels are in this format:
    [ [head_x, head_y], [midsection_x, midsection_y], [tail_x, tail_y], # The first fish is the covering fish
    [head_x, head_y], [midsection_x, midsection_y], [tail_x, tail_y] ] # The second fish is the covered fish
    """

    def __init__(
        self, images, keypoints, head_midsec_distance=8, inference=False, **kwarg
    ):
        """
        Initializes the dataset.
        Note: This will not generate the input images and the corresponding heatmaps if the dataset is not in inference mode.
        Args:
            images (list): A list of input images.
            keypoints (list): A list of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            inference (bool): Whether the dataset is in inference mode.
        """
        self.images = images
        self.keypoints = keypoints
        self.head_midsec_distance = head_midsec_distance
        self.inference = inference
        self.kwarg = kwarg
        self._is_initialized = False
        if self.inference:
            self.sequential_initialize(rangaug_param=None)
            self.report()

    def report(self):
        """
        Prints the length of the dataset, and the shape of the input and output.
        """
        if self._is_initialized:
            print(f'Length of dataset: {len(self.data)}')
            print(f'Shape of X image: {self.data[0][0].shape}')
            print(f'Shape of X vector: {self.data[0][1].shape}')
            print(f'Shape of Y: {self.data[0][2].shape}')
        else:
            raise Exception("Dataset not initialized.")

    def sequential_initialize(self, rangaug_param):
        """
        Initializes the dataset by generating the input images and the corresponding heatmaps for each image without using parallelization.
        Args:
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
            include_coords (bool): Whether to include the coordinates of the keypoints in the output.
        """
        if self.inference and self._is_initialized:
            raise Exception(
                "Dataset is in inference mode and has been initialized.")
        self.data = []
        for img, kps in tqdm(zip(self.images, self.keypoints), total=len(self.images)):
            self.data.append(self.generate_xy(img, kps, rangaug_param, self.head_midsec_distance, **self.kwarg))
        self._is_initialized = True

    def parallel_initialize(self, rangaug_param, batch_size):
        """
        Initializes the dataset by generating the input images and the corresponding heatmaps for each image using parallelization.
        Args:
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
            batch_size (int): The batch size to be used for parallelization.
            include_coords (bool): Whether to include the coordinates of the keypoints in the output.
        """
        if self.inference and self._is_initialized:
            raise Exception(
                "Dataset is in inference mode and has been initialized.")
        # Create batches
        image_batches = [
            self.images[i: i + batch_size]
            for i in range(0, len(self.images), batch_size)
        ]
        kps_batches = [
            self.keypoints[i: i + batch_size]
            for i in range(0, len(self.keypoints), batch_size)
        ]

        # Create delayed tasks for each batch
        tasks = [
            delayed(self.batch_generate_xy)(
                img_batch, kps_batch, rangaug_param, self.head_midsec_distance, **self.kwarg
            )
            for img_batch, kps_batch in zip(image_batches, kps_batches)
        ]

        # Compute all tasks, then flatten the result into a single list
        self.data = [item for sublist in compute(*tasks) for item in sublist]
        self._is_initialized = True

    def batch_generate_xy(
        self, images, keypoints, rangaug_param, head_midsec_distance, **kwarg
    ):
        """
        Generates the input image and the corresponding heatmap for the given image and keypoints.
        Args:
            images (list): A list of input images.
            keypoints (list): A list of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
            head_midsec_distance (int): The distance between the head and midsection keypoints.
        Returns:
            A list of tuples of (image, vector, label) where image is a torch.Tensor of shape (3, height, width) and vector is a torch.Tensor of length 12
            If include_coords is True, then the tuples will also contain the coordinates of the keypoints.
        """
        return [
            self.generate_xy(img, kps, rangaug_param, head_midsec_distance, **kwarg)
            for img, kps in zip(images, keypoints)
        ]

    def generate_xy(
        self, image, keypoints, rangaug_param, head_midsec_distance, **kwarg
    ):
        """
        Generates the input image and the corresponding heatmap for the given image and keypoints.
        Args:
            image (np.ndarray): The input image.
            keypoints (list): A list of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
            head_midsec_distance (int): The distance between the head and midsection keypoints.
        Returns:
            A tuple of (image, vector, label) where image is a torch.Tensor of shape (3, height, width) and vector is a torch.Tensor of length 12
        """
        # Apply RandAugment
        aug_result = self.apply_randaugment(image,
                                            keypoints,
                                            rangaug_param,
                                            num_keypoints_per_fish=3)
        
        sorted_keypoints = []

        for i in (0, 100, 200, 1, 101, 201):
            if i in aug_result['fish_id']:
                sorted_keypoints.append(aug_result['keypoints'][aug_result['fish_id'].index(i)])
            else:
                sorted_keypoints.append((0, 0))

        if sum(sorted_keypoints[0]) != 0 and sum(sorted_keypoints[1]) != 0:
            sorted_keypoints[1] = self.normalize_distance(sorted_keypoints[0], sorted_keypoints[1],
                                                          new_distance=head_midsec_distance)
        if sum(sorted_keypoints[3]) != 0 and sum(sorted_keypoints[4]) != 0:
            sorted_keypoints[4] = self.normalize_distance(sorted_keypoints[3], sorted_keypoints[4],
                                                          new_distance=head_midsec_distance)

        # Generate a random label (0 or 1)
        label = torch.randint(0, 2, (1,), dtype=torch.float32)
        if label:
            keypoints = sorted_keypoints
        else:
            keypoints = sorted_keypoints[3:] + sorted_keypoints[:3]
        # Flatten the keypoints
        keypoints = [coord for kp in keypoints for coord in kp]
        
        return aug_result["image"], torch.tensor(keypoints, dtype=torch.float32), label


    def resize_keypoints(self, keypoints, ratio_x, ratio_y):
        """
        Resizes the keypoints to match the size of the heatmap.
        Args:
            keypoints (list or dict): A list of keypoints, where each keypoint is a tuple of (x, y) coordinates.
                                      Or a dict of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            ratio_x (float): The ratio of the width of the heatmap to the width of the original image.
            ratio_y (float): The ratio of the height of the heatmap to the height of the original image.
        Returns:
            A list of resized keypoints, where each keypoint is a tuple of (x, y) coordinates.
        """
        if isinstance(keypoints, dict):
            return {key: (x / ratio_x, y / ratio_y) for key, (x, y) in keypoints.items()}
        return np.array([(x / ratio_x, y / ratio_y) for x, y in keypoints])
    
    def normalize_distance(self, x, y, new_distance):
        """
        Normalizes the distance between two keypoints.
        Args:
            x (tuple): A tuple of (x, y) coordinates for the first keypoint.
            y (tuple): A tuple of (x, y) coordinates for the second keypoint.
            new_distance (int): The distance between the two keypoints.
        Returns:
            A tuple of (x, y) coordinates for the second keypoint.
        """
        # Convert to numpy arrays for easier calculations
        x = np.array(x)
        y = np.array(y)

        # Calculate the vector from x to y
        v = y - x

        # Normalize the vector
        v_norm = v / np.linalg.norm(v)

        # Multiply the normalized vector by the new distance
        y_norm = x + v_norm * new_distance

        return y_norm

    def apply_randaugment(self, image, keypoints, rangaug_param, num_keypoints_per_fish=2):
        """
        Applies RandAugment to the given image and keypoints.
        If rangaug_param is None, then no RandAugment will be applied.
        Instead, the image will be converted to torch.Tensor and returned together with the keypoints.
        Args:
            image (np.ndarray): The input image.
            keypoints (list): A list of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
            num_keypoints_per_fish (int): The number of keypoints per fish.
        Returns:
            A dictionary containing the augmented image and keypoints.
        """
        if rangaug_param is None:
            transforms = A.Compose(
                [A.ToFloat(), ToTensorV2()],
                keypoint_params=A.KeypointParams(format="xy"),
            )
            return transforms(image=image, keypoints=keypoints,
                            fish_id=[(i // num_keypoints_per_fish + i % num_keypoints_per_fish * 100)
                                    for i in range(len(keypoints))])
        else:
            return get_rand_aug(*rangaug_param)(image=image, keypoints=keypoints,
                                                fish_id=[(i // num_keypoints_per_fish + i % num_keypoints_per_fish * 100)
                                                         for i in range(len(keypoints))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns the input image and the corresponding heatmap for the given index.
        Args:
            idx (int): The index of the image and heatmap to be returned.
        Returns:
            A tuple of (image, heatmap) where image is a torch.Tensor of shape (3, height, width) and heatmap is a torch.Tensor of shape (1, height, width).
            Note: In inference mode, the returned tuple will also contain the coordinates of the keypoints.
        """
        if not self._is_initialized:
            raise Exception("Dataset not initialized.")
        else:
            return self.data[idx]

class LocalClassifyDatasetV2(DoubleKeypointsHeatmapDataset):

    _kwarg = ('head_midsec_distance', )

    def __init__(
            self, images, keypoints, heatmap_sigma, heatmap_shape=None, inference=False, **kwarg
    ):
        super().__init__(images, keypoints, heatmap_sigma, None, False, **kwarg)
        self.heatmap_shape = self.images[0].shape[:2]
        self.inference = inference
        if self.inference:
            self.sequential_initialize(rangaug_param=None)
            self.report()

    def report(self):
        print(f'Length of dataset: {len(self.data)}')
        print(f'Shape of X: {self.data[0][0].shape}')
        print(f'Shape of Y: {self.data[0][1].shape}')

    def generate_xy(
        self, image, keypoints, rangaug_param, gaussian_kernel, heatmap_shape, include_coords, head_midsec_distance, **kwarg
    ):
        """
        Generates the input image and the corresponding heatmap for the given image and keypoints.
        Args:
            image (np.ndarray): The input image.
            keypoints (list): A list of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
            gaussian_kernel (np.ndarray): A 2D Gaussian kernel array.
            heatmap_shape (tuple): A tuple of (height, width) values for the heatmap.
            head_midsec_distance (int): The distance between the two keypoints.
            include_coords (bool): Whether to include the coordinates of the keypoints in the output.
        Returns:
        """

        # Apply RandAugment
        aug_result = self.apply_randaugment(image,
                                            keypoints,
                                            rangaug_param,
                                            num_keypoints_per_fish=3)
        
        sorted_keypoints = []

        for i in (0, 100, 200, 1, 101, 201):
            if i in aug_result['fish_id']:
                sorted_keypoints.append(aug_result['keypoints'][aug_result['fish_id'].index(i)])
            else:
                sorted_keypoints.append((0, 0))

        if sum(sorted_keypoints[0]) != 0 and sum(sorted_keypoints[1]) != 0:
            sorted_keypoints[1] = self.normalize_distance(sorted_keypoints[0], sorted_keypoints[1],
                                                          new_distance=head_midsec_distance)
        if sum(sorted_keypoints[3]) != 0 and sum(sorted_keypoints[4]) != 0:
            sorted_keypoints[4] = self.normalize_distance(sorted_keypoints[3], sorted_keypoints[4],
                                                          new_distance=head_midsec_distance)

        # Generate a random label (0 or 1)
        label = torch.randint(0, 2, (1,), dtype=torch.float32)
        if label:
            keypoints = sorted_keypoints
        else:
            keypoints = sorted_keypoints[3:] + sorted_keypoints[:3]

        # Create heatmap
        heatmap_0 = self.generate_map(heatmap_shape, keypoints[:3], gaussian_kernel)
        heatmap_1 = self.generate_map(heatmap_shape, keypoints[3:], gaussian_kernel)

        x = torch.cat((aug_result['image'], heatmap_0, heatmap_1), dim=0)

        return x, label

class LocalClassifyDatasetV3(LocalClassifyDataset):

    def generate_xy(self, image, keypoints, rangaug_param, head_midsec_distance, attention_radius):

        # Apply RandAugment
        aug_result = self.apply_randaugment(image,
                                            keypoints[:3],
                                            rangaug_param,
                                            num_keypoints_per_fish=3)
        
        sorted_keypoints = []

        for i in (0, 100, 200):
            if i in aug_result['fish_id']:
                sorted_keypoints.append(aug_result['keypoints'][aug_result['fish_id'].index(i)])
            else:
                return None
            
        heatmap_shape = image.shape[:2]
            
        fish = create_fish(*sorted_keypoints, unify_distance=head_midsec_distance, numsegments=16)
        heatmap = np.zeros(heatmap_shape, dtype=np.float32)
        bounding_box = fish.bounds # (x_min, y_min, x_max, y_max)

        start_row = max(0, round(bounding_box[1])-attention_radius)
        end_row = min(round(bounding_box[3])+attention_radius, heatmap_shape[0]) # 0 or 1? Check this

        base_start_col = max(0, round(bounding_box[0])-attention_radius)
        base_end_col = min(round(bounding_box[2])+attention_radius, heatmap_shape[1])

        top_left_dist = fish.distance(Point(base_start_col, start_row)) - attention_radius
        top_right_dist = fish.distance(Point(base_end_col - 1, start_row)) - attention_radius
        bottom_left_dist = fish.distance(Point(base_start_col, end_row - 1)) - attention_radius
        bottom_right_dist = fish.distance(Point(base_end_col - 1, end_row - 1)) - attention_radius

        row = start_row

        while row < end_row:
            
            if row < start_row + top_left_dist:
                current_start_col = base_start_col + int(top_left_dist) - (row - start_row)
            elif row > end_row - 1 - bottom_left_dist:
                current_start_col = base_start_col + int(bottom_left_dist) - (end_row - 1 - row)
            else:
                current_start_col = base_start_col
            if row < start_row + top_right_dist:
                current_end_col = base_end_col - int(top_right_dist) + (row - start_row)
            elif row > end_row - 1 - bottom_right_dist:
                current_end_col = base_end_col - int(bottom_right_dist) + (end_row - 1 - row)
            else:
                current_end_col = base_end_col

            col = current_start_col
            while col < current_end_col:
                dist = fish.distance(Point(col, row))
                if dist < attention_radius:
                    heatmap[row, col] = 1 - dist / attention_radius
                    col += 1
                else:
                    col += max(int(dist - attention_radius), 1)
            row += 1

        label = torch.tensor(keypoints[3][1], dtype=torch.float32)
        masked_image = aug_result['image'] * heatmap

        return masked_image, label
    
class LocalClassifyTestDataset(LocalClassifyDataset):

    def generate_xy(self, image, keypoints, rangaug_param, head_midsec_distance, attention_radius):

        # Apply RandAugment
        transforms = A.Compose(
            [A.ToFloat(), ToTensorV2()])
        aug_result = transforms(image=image)
            
        heatmap_shape = image.shape[:2]
            
        fish = create_fish(keypoints[:2], keypoints[2:4], keypoints[4:], unify_distance=head_midsec_distance, numsegments=16)
        heatmap = np.zeros(heatmap_shape, dtype=np.float32)
        bounding_box = fish.bounds # (x_min, y_min, x_max, y_max)

        start_row = max(0, round(bounding_box[1])-attention_radius)
        end_row = min(round(bounding_box[3])+attention_radius, heatmap_shape[0]) # 0 or 1? Check this

        base_start_col = max(0, round(bounding_box[0])-attention_radius)
        base_end_col = min(round(bounding_box[2])+attention_radius, heatmap_shape[1])

        top_left_dist = fish.distance(Point(base_start_col, start_row)) - attention_radius
        top_right_dist = fish.distance(Point(base_end_col - 1, start_row)) - attention_radius
        bottom_left_dist = fish.distance(Point(base_start_col, end_row - 1)) - attention_radius
        bottom_right_dist = fish.distance(Point(base_end_col - 1, end_row - 1)) - attention_radius

        row = start_row

        while row < end_row:
            
            if row < start_row + top_left_dist:
                current_start_col = base_start_col + int(top_left_dist) - (row - start_row)
            elif row > end_row - 1 - bottom_left_dist:
                current_start_col = base_start_col + int(bottom_left_dist) - (end_row - 1 - row)
            else:
                current_start_col = base_start_col
            if row < start_row + top_right_dist:
                current_end_col = base_end_col - int(top_right_dist) + (row - start_row)
            elif row > end_row - 1 - bottom_right_dist:
                current_end_col = base_end_col - int(bottom_right_dist) + (end_row - 1 - row)
            else:
                current_end_col = base_end_col

            col = current_start_col
            while col < current_end_col:
                dist = fish.distance(Point(col, row))
                if dist < attention_radius:
                    heatmap[row, col] = 1 - dist / attention_radius
                    col += 1
                else:
                    col += max(int(dist - attention_radius), 1)
            row += 1

        masked_image = aug_result['image'] * heatmap

        return masked_image
    
class LocalClassifyDatasetV4Preloader(LocalClassifyDataset):

    def generate_xy(self, image, keypoints, rangaug_param, head_midsec_distance):

        heatmap_shape = image.shape[:2]
        fish = create_fish(*keypoints[:3], unify_distance=head_midsec_distance, numsegments=16)
        heatmap = np.zeros(heatmap_shape, dtype=np.float32)

        for row in range(heatmap_shape[0]):
            for col in range(heatmap_shape[1]):
                dist = fish.distance(Point(col, row))
                heatmap[row, col] = -dist
        
        label = np.array(keypoints[3][1], dtype=np.float32)

        return image, heatmap, label
    
class LocalClassifyDatasetV3Simple(Dataset):

    def __init__(
        self, images, labels, inference=False, **kwarg
    ):
        """
        Initializes the dataset.
        Note: This will not generate the input images and the corresponding heatmaps if the dataset is not in inference mode.
        Args:
            images (np.ndarray): A list of input images.
            labels (np.ndarray): A list of labels.
            inference (bool): Whether the dataset is in inference mode.
        """
        self.images = images
        self.labels = labels
        self.inference = inference
        self.kwarg = kwarg
        self._is_initialized = False
        if self.inference:
            self.sequential_initialize(rangaug_param=None)
            self.report()

    def report(self):
        """
        Prints the length of the dataset, and the shape of the input and output.
        """
        if self._is_initialized:
            print(f'Length of dataset: {len(self.data)}')
            print(f'Shape of X: {self.data[0][0].shape}')
            print(f'Shape of Y: {self.data[0][1].shape}')
        else:
            raise Exception("Dataset not initialized.")
    
    def sequential_initialize(self, rangaug_param):
        """
        Initializes the dataset by augmenting the input images and the corresponding labels without using parallelization.
        Args:
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
        """
        if self.inference and self._is_initialized:
            raise Exception(
                "Dataset is in inference mode and has been initialized.")
        self.data = []
        for img, label in tqdm(zip(self.images, self.labels), total=len(self.images)):
            self.data.append(self.generate_xy(img, label, rangaug_param, **self.kwarg))
        self._is_initialized = True

    def apply_randaugment(self, image, rangaug_param):
        """
        Applies RandAugment to the given image.
        If rangaug_param is None, then no RandAugment will be applied.
        Instead, the image will be converted to torch.Tensor and returned.
        Args:
            image (np.ndarray): The input image.
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
        Returns:
            A dictionary containing the augmented image.
        """
        if rangaug_param is None:
            transforms = A.Compose(
                [A.ToFloat(), ToTensorV2()]
            )
            return transforms(image=image)
        else:
            return get_rand_aug_keep_center(*rangaug_param)(image=image)

    def generate_xy(self, image, label, rangaug_param, **kwarg):
        """
        Generates the input image and the corresponding heatmap for the given image and keypoints.
        Args:
            image (np.ndarray): The input image.
            label (int): The label.
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
        Returns:
        """
        # Apply RandAugment
        aug_result = self.apply_randaugment(image, rangaug_param)
        
        return aug_result["image"], torch.tensor([label], dtype=torch.float32)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns the input image and the corresponding heatmap for the given index.
        Args:
            idx (int): The index of the image and heatmap to be returned.
        Returns:
            A tuple of (image, heatmap) where image is a torch.Tensor of shape (3, height, width) and heatmap is a torch.Tensor of shape (1, height, width).
            Note: In inference mode, the returned tuple will also contain the coordinates of the keypoints.
        """
        if not self._is_initialized:
            raise Exception("Dataset not initialized.")
        else:
            return self.data[idx]
        
    def parallel_initialize(self, rangaug_param, batch_size):
        """
        Initializes the dataset by augmenting the input images and the corresponding labels using parallelization.
        Args:
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
            batch_size (int): The batch size to be used for parallelization.
        """
        if self.inference and self._is_initialized:
            raise Exception(
                "Dataset is in inference mode and has been initialized.")
        # Create batches
        image_batches = [
            self.images[i: i + batch_size]
            for i in range(0, len(self.images), batch_size)
        ]
        label_batches = [
            self.labels[i: i + batch_size]
            for i in range(0, len(self.labels), batch_size)
        ]

        # Create delayed tasks for each batch
        tasks = [
            delayed(self.batch_generate_xy)(
                img_batch, label_batch, rangaug_param
            )
            for img_batch, label_batch in zip(image_batches, label_batches)
        ]

        # Compute all tasks, then flatten the result into a single list
        self.data = [item for sublist in compute(*tasks) for item in sublist]
        self._is_initialized = True

    def batch_generate_xy(self, images, labels, rangaug_param):
        """
        Generates the input image and the corresponding heatmap for the given image and keypoints.
        Args:
            images (list): A list of input images.
            labels (list): A list of labels.
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
        Returns:
            A list of tuples of (image, label) where image is a torch.Tensor of shape (3, height, width) and label is a torch.Tensor of shape (1).
        """
        return [
            self.generate_xy(img, label, rangaug_param)
            for img, label in zip(images, labels)
        ]
    
class LocalClassifyDatasetV4(Dataset):

    def __init__(
        self, images, masks, labels, inference=False, **kwarg
    ):
        """
        Initializes the dataset.
        Note: This will not generate the input images and the corresponding heatmaps if the dataset is not in inference mode.
        Args:
            images (np.ndarray): A list of input images.
            masks (np.ndarray): A list of masks.
            labels (np.ndarray): A list of labels.
            inference (bool): Whether the dataset is in inference mode.
        """
        self.images = images
        self.masks = masks
        self.labels = labels
        self.inference = inference
        self.kwarg = kwarg
        self._is_initialized = False

    def report(self):
        """
        Prints the length of the dataset, and the shape of the input and output.
        """
        if self._is_initialized:
            print(f'Length of dataset: {len(self.data)}')
            print(f'Shape of X: {self.data[0][0].shape}')
            print(f'Shape of X mask: {self.data[0][1].shape}')
            print(f'Shape of Y: {self.data[0][2].shape}')
        else:
            raise Exception("Dataset not initialized.")
    
    def sequential_initialize(self, rangaug_param, attention_radius):
        """
        Initializes the dataset by augmenting the input images and the corresponding labels without using parallelization.
        Args:
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
        """
        if self.inference and rangaug_param is not None:
            raise Exception(
                "Dataset is in inference mode but RandAugment is enabled.")
        self.data = []
        for img, mask, label in tqdm(zip(self.images, self.masks, self.labels), total=len(self.images)):
            self.data.append(self.generate_xy(img, mask, label, rangaug_param, attention_radius, **self.kwarg))
        self._is_initialized = True

    def apply_randaugment(self, image, mask, rangaug_param):
        """
        Applies RandAugment to the given image.
        If rangaug_param is None, then no RandAugment will be applied.
        Instead, the image will be converted to torch.Tensor and returned.
        Args:
            image (np.ndarray): The input image.
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
        Returns:
            A dictionary containing the augmented image.
        """
        if rangaug_param is None:
            transforms = A.Compose(
                [A.ToFloat(), ToTensorV2()]
            )
            return transforms(image=image, mask=mask)
        else:
            return get_rand_aug_keep_center(*rangaug_param)(image=image, mask=mask)

    def generate_xy(self, image, mask, label, rangaug_param, attention_radius, **kwarg):
        """
        Generates the input image and the corresponding heatmap for the given image and keypoints.
        Args:
            image (np.ndarray): The input image.
            label (int): The label.
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
        Returns:
        """
        # Adjust mask according to attention radius
        adjusted_mask = mask + attention_radius
        adjusted_mask[adjusted_mask < 0] = 0
        adjusted_mask /= attention_radius

        # Apply RandAugment
        aug_result = self.apply_randaugment(image, adjusted_mask, rangaug_param)

        return aug_result["image"] * aug_result["mask"], torch.tensor([label], dtype=torch.float32)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns the input image and the corresponding heatmap for the given index.
        Args:
            idx (int): The index of the image and heatmap to be returned.
        Returns:
            A tuple of (image, heatmap) where image is a torch.Tensor of shape (3, height, width) and heatmap is a torch.Tensor of shape (1, height, width).
            Note: In inference mode, the returned tuple will also contain the coordinates of the keypoints.
        """
        if not self._is_initialized:
            raise Exception("Dataset not initialized.")
        else:
            return self.data[idx]
        
    def parallel_initialize(self, rangaug_param, attention_radius, batch_size):
        """
        Initializes the dataset by augmenting the input images and the corresponding labels using parallelization.
        Args:
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
            attention_radius (int): The radius of the attention area.
            batch_size (int): The batch size to be used for parallelization.
        """
        if self.inference and rangaug_param is not None:
            raise Exception(
                "Dataset is in inference mode but RandAugment is enabled.")
        # Create batches
        image_batches = [
            self.images[i: i + batch_size]
            for i in range(0, len(self.images), batch_size)
        ]
        mask_batches = [
            self.masks[i: i + batch_size]
            for i in range(0, len(self.masks), batch_size)
        ]
        label_batches = [
            self.labels[i: i + batch_size]
            for i in range(0, len(self.labels), batch_size)
        ]

        # Create delayed tasks for each batch
        tasks = [
            delayed(self.batch_generate_xy)(
                img_batch, mask_batch, label_batch, rangaug_param, attention_radius
            )
            for img_batch, mask_batch, label_batch in zip(image_batches, mask_batches, label_batches)
        ]

        # Compute all tasks, then flatten the result into a single list
        self.data = [item for sublist in compute(*tasks) for item in sublist]
        self._is_initialized = True

    def batch_generate_xy(self, images, masks, labels, rangaug_param, attention_radius):
        """
        Generates the input image and the corresponding heatmap for the given image and keypoints.
        Args:
            images (list): A list of input images.
            masks (list): A list of masks.
            labels (list): A list of labels.
            rangaug_param (tuple): A tuple of (n, m) values to be used for RandAugment.
            attention_radius (int): The radius of the attention area.
        Returns:
            A list of tuples of (image, label) where image is a torch.Tensor of shape (3, height, width) and label is a torch.Tensor of shape (1).
        """
        return [
            self.generate_xy(img, mask, label, rangaug_param, attention_radius)
            for img, mask, label in zip(images, masks, labels)
        ]