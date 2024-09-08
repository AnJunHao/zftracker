import os
import json
import numpy as np
from PIL import Image
import cv2
from shapely import geometry
from .evaluate_annotation_v2 import create_fish

from ..util.tqdm import TQDM as tqdm

class JsonToNumpy(object):
    """
    Converts images and annotations from a JSON file into numpy arrays.

    Args:
        image_dir (str): Directory containing the input images.
        anno_file (str, list, or tuple): Path to a JSON annotation file or a list/tuple of paths
                                         to multiple annotation files.
        num_fish (int): Number of fish in each frame.
        num_pt_per_fish (int): Number of points in each fish.
        verbose (bool, optional): If True, prints information about the conversion process.

    """

    def __init__(self, img_dir, anno_dir, num_fish, num_pt_per_fish, verbose=True, read_images=True):
        """
        Initializes the JsonToNumpy object.

        Loads the JSON data from the annotation file and parses it into a dictionary.
        Creates numpy arrays for the annotations and images from the dictionary and image directory.
        Stores the arrays as attributes of the class.

        """
        if isinstance(anno_dir, str):
            # Check if it is a file or a directory
            if os.path.isdir(anno_dir):
                anno_dir = [os.path.join(anno_dir, f)
                             for f in os.listdir(anno_dir)]
                self.dict_data = {}
                self.error = []
                for f in anno_dir:
                    with open(f, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    new_dict_data, new_error = self.parse_json(json_data)
                    self.dict_data.update(new_dict_data)
                    self.error.extend(new_error)
            else:
                with open(anno_dir, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                self.dict_data, self.error = self.parse_json(json_data)
        elif isinstance(anno_dir, (list, tuple)):
            self.dict_data = {}
            self.error = []
            for f in anno_dir:
                with open(f, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                new_dict_data, new_error = self.parse_json(json_data)
                self.dict_data.update(new_dict_data)
                self.error.extend(new_error)
        else:
            raise TypeError(
                f'anno_file should be str or list or tuple, received {type(anno_dir)}')

        self.num_pt_per_fish = num_pt_per_fish

        # Pad the annotation array with zeros to ensure that all frames have the same number of fish
        for k, value in self.dict_data.items():
            while len(value) < num_fish * num_pt_per_fish:
                value.extend([[0, 0]] * num_pt_per_fish)
                self.error.append(('Wrong number of fish or bend', k))
        self.annotation_array = np.array(list(self.dict_data.values()))

        if read_images:
            self.image_array = []

            if verbose:
                iterator = tqdm(self.dict_data)
            else:
                iterator = self.dict_data

            for frame in iterator:

                # Read file from disk using PIL, and convert to np array
                img = Image.open(os.path.join(img_dir, frame))
                self.image_array.append(np.array(img))

                # In case the number of fish in a frame is wrong
                if len(self.dict_data[frame]) != num_fish * num_pt_per_fish:
                    self.error.append(('Wrong number of fish or bend', frame))

            self.image_array = np.array(self.image_array)
        else:
            self.img_dir = img_dir

        if verbose:
            print(f'Number of errors: {len(self.error)}')
            if read_images:
                print(
                    f'Shape of image array: {self.image_array.shape}')
            print(f'Shape of annotation array: {self.annotation_array.shape}')
    
    def get_intersected_fishes(self, numsegments=16, unify_distance=8, max_distance=8, verbose=True, separate=False):
        intersected_fishes = []
        if verbose:
            fp = 0
            tp = 0
        for key, value in tqdm(list(self.dict_data.items()), disable=not verbose):
            for i in range(0, 28, 4):
                for j in range(i+4, 28, 4):
                    if sum(value[i]) == 0 or sum(value[j]) == 0:
                        continue
                    fish_i = create_fish(value[i], value[i+1], value[i+2],
                                        numsegments=numsegments,
                                        unify_distance=unify_distance)
                    fish_j = create_fish(value[j], value[j+1], value[j+2],
                                        numsegments=numsegments,
                                        unify_distance=unify_distance)
                    distance = fish_i.distance(fish_j)
                    if (distance <= max_distance and
                        (value[i+3][0] == value[j+3][1] == 1 or
                        value[i+3][1] == value[j+3][0] == 1)):
                        if verbose:
                            tp += 1
                        if separate:
                            intersected_fishes.append((key, fish_i, value[i:i+4]))
                            intersected_fishes.append((key, fish_j, value[j:j+4]))
                        else:
                            intersected_fishes.append((key,
                                                    geometry.MultiLineString(list(fish_i.geoms)+list(fish_j.geoms)),
                                                    (value[i:i+4], value[j:j+4])))
                    elif distance <= max_distance and verbose:
                        fp += 1
        if verbose:
            print(f'Precision: {tp/(tp+fp)*100:.2f}%')
            print(f'Number of intersected fishes: {len(intersected_fishes)}')
        return intersected_fishes

    def get_crop_centers(self, fishes):
        crop_centers = []
        for i in range(len(fishes)):
            coords = np.array([coord for linestring in fishes[i][1].geoms for coord in linestring.coords])
            crop_centers.append(np.mean([np.max(coords, axis=0),
                                np.min(coords, axis=0)], axis=0))
        return crop_centers
    
    def save_classification_data(self, img_dir, ann_dir, verbose=True, numsegments=16, unify_distance=8, max_distance=8, separate=False):

        cls_images_array, cls_annotations_array = self.get_classification_data(numsegments=numsegments,
                                                                              unify_distance=unify_distance,
                                                                              max_distance=max_distance,
                                                                              verbose=verbose,
                                                                              separate=separate)
        
        np.save(img_dir, cls_images_array)
        np.save(ann_dir, cls_annotations_array)
        if verbose:
            print(f'Images {cls_images_array.shape} saved to "{img_dir}"')
            print(f'Annotations {cls_annotations_array.shape} saved to "{ann_dir}"')

    def get_classification_data(self, numsegments=16, unify_distance=8, max_distance=8, verbose=True, separate=False):

        fishes = self.get_intersected_fishes(numsegments=numsegments,
                                             unify_distance=unify_distance,
                                             max_distance=max_distance,
                                             verbose=verbose,
                                             separate=separate)
        crop_centers = self.get_crop_centers(fishes)
        if separate:
            crop_range = 80
        else:
            crop_range = 128
        cls_images_array = []
        cls_annotations_array = []
        unique_fish = set()
        for fish, center in tqdm(list(zip(fishes, crop_centers)), disable=not verbose):

            if (fish[0], center[0], center[1]) in unique_fish:
                continue
            unique_fish.add((fish[0], center[0], center[1]))

            if fish[2][3][1]  == -1 and separate: # The annotation is missing for the 'covered' class
                continue

            image_index = list(self.dict_data.keys()).index(fish[0])
            image = self.image_array[image_index]
            image_shape = image.shape
            # Assume crop_center is a list of tuples (x, y)
            crop_x, crop_y = center
            crop_x, crop_y = int(crop_x), int(crop_y)

            # Calculate the slicing indices
            top = max(0, crop_y - crop_range)
            bottom = min(image_shape[0], crop_y + crop_range)
            left = max(0, crop_x - crop_range)
            right = min(image_shape[1], crop_x + crop_range)

            # Crop the image
            cropped_image = image[top:bottom, left:right]

            # Calculate padding if necessary
            pad_top = -min(0, crop_y - crop_range)
            pad_bottom = max(0, (crop_y + crop_range) - image_shape[0])
            pad_left = -min(0, crop_x - crop_range)
            pad_right = max(0, (crop_x + crop_range) - image_shape[1])


            # Create padding tuple for np.pad
            padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

            # Pad the image
            cropped_padded_image = np.pad(cropped_image, padding, 'constant', constant_values=0)

            if separate:
                anno = fish[2]
                coords = anno[:3] - np.array([left-pad_left, top-pad_top])
                coords = np.concatenate((coords, [anno[3]]))
                cls_images_array.append(cropped_padded_image)
                cls_annotations_array.append(coords)
            else:
                cls_images_array.append(cropped_padded_image)
                anno_i = fish[2][0][:3] - np.array([left-pad_left, top-pad_top])
                anno_j = fish[2][1][:3] - np.array([left-pad_left, top-pad_top])
                if fish[2][0][3][0] == 1 and fish[2][1][3][1] == 1:
                    cls_annotations_array.append(np.concatenate((anno_i, anno_j)))
                elif fish[2][0][3][1] == 1 and fish[2][1][3][0] == 1:
                    cls_annotations_array.append(np.concatenate((anno_j, anno_i)))
                else:
                    raise ValueError('Annotation error')
        self.cls_images_array = np.array(cls_images_array)
        self.cls_annotations_array = np.array(cls_annotations_array)

        return self.cls_images_array, self.cls_annotations_array
    
    def read_images(self, verbose=True):
        """
        Reads the images from the image directory and stores them as an array.

        Args:
            verbose (bool, optional): If True, prints a message confirming the operation. Default is True.

        """
        self.image_array = []

        if verbose:
            iterator = tqdm(self.dict_data)
        else:
            iterator = self.dict_data

        for frame in iterator:

            # Read file from disk using PIL, and convert to np array
            img = Image.open(os.path.join(self.img_dir, frame))
            self.image_array.append(np.array(img))

            # In case the number of fish in a frame is wrong
            if len(self.dict_data[frame]) != self.num_fish * self.num_pt_per_fish:
                self.error.append(('Wrong number of fish or bend', frame))

        self.image_array = np.array(self.image_array)

        if verbose:
            print(f'Shape of image array: {self.image_array.shape}')

    def visualize_annotation(self, point_index, size=5):
        """
        Visualizes the annotations for a given frame of the dataset.

        Args:
            frame (int or str): Index of the frame to visualize, either as an integer or a string (e.g. 'clip_101_07.jpg').
            point_index (int): Index of the point for each fish.
            size (int, optional): Size of the keypoints in the visualization. Default is 3.
            display (bool, optional): If True, displays the image with keypoints. Default is True.

        Returns:
            numpy.ndarray: Array of the image with keypoints overlaid.

        """

        self.drawn_array = []

        for img, anno in tqdm(zip(self.image_array, self.annotation_array), total=len(self.annotation_array)):
            image_copy = np.copy(img)
            for i in range(point_index, len(anno), self.num_pt_per_fish):
                x, y = anno[i]
                cv2.circle(image_copy, (round(x), round(y)), size, (0, 255, 0), -1)  # Draws a green circle at each keypoint
            self.drawn_array.append(image_copy)

        self.drawn_array = np.array(self.drawn_array)

        return self.drawn_array

    def save(self, img_dir, ann_dir, verbose=True):
        """
        Saves the image and annotation arrays as .npy files to the specified directories.

        Args:
            img_dir (str): Directory to save the image array.
            ann_dir (str): Directory to save the annotation array.
            verbose (bool, optional): If True, prints a message confirming the save operation. Default is True.

        """
        np.save(img_dir, self.image_array)
        np.save(ann_dir, self.annotation_array)
        if verbose:
            print(f'Images saved to "{img_dir}"')
            print(f'Annotations saved to "{ann_dir}"')

    def crop(self, top_px, right_px, bottom_px, left_px, verbose=True):
        """
        Crops the images and annotations by the specified number of pixels from each side.
    
        Args:
            top_px (int): Number of pixels to crop from the top side of the images.
            right_px (int): Number of pixels to crop from the right side of the images.
            bottom_px (int): Number of pixels to crop from the bottom side of the images.
            left_px (int): Number of pixels to crop from the left side of the images.
        """
        if verbose:
            before = self.image_array.shape
        self.image_array = np.array(
            [img[top_px:-bottom_px or None, left_px:-right_px or None] for img in self.image_array])
    
        self.annotation_array = np.array([[[x-left_px, y-top_px] if x-left_px > 0 and y-top_px > 0 else [0, 0]
                                            for x, y in anno] for anno in self.annotation_array])
        if verbose:
            print(f'Before cropping: {before}; After cropping: {self.image_array.shape}')

    def parse_json(self, json_data):
        """
        Parses JSON data to extract labeled points for zebrafish images.

        Args:
            json_data (dict): A dictionary containing JSON data.

        Returns:
            Tuple: A tuple containing two items:
                - A dictionary containing the parsed data, where the keys are filenames and the values are lists
                    of (x,y) coordinates for each labeled point.
                - A list of filenames for which the parsing encountered errors.
        """
        data = {}
        error = []
        if 'info' in json_data:
            key = os.path.basename(json_data['info'])
            labels = []
            for item in json_data['labels']:
                label = [[item['points'][i*2], item['points'][i*2+1]] for i in range(len(item['points'])//2)]
                attr = []
                if item['attr']['是否遮挡其他鱼'] == '否':
                    attr.append(0)
                elif item['attr']['是否遮挡其他鱼'] == '是':
                    attr.append(1)
                elif item['attr']['是否遮挡其他鱼'] == '无法判断':
                    attr.append(-1)
                else:
                    error.append((key, 'Invalid value for "是否遮挡其他鱼"'))
                    continue
                if item['attr']['是否被遮挡'] == '否':
                    attr.append(0)
                elif item['attr']['是否被遮挡'] == '是':
                    attr.append(1)
                elif item['attr']['是否被遮挡'] == '无法判断':
                    attr.append(-1)
                else:
                    error.append((key, 'Invalid value for "是否被遮挡"'))
                    continue
                label.append(attr)
                labels.extend(label)
            data[key] = labels
        else:
            for item in json_data['data']:
                key = os.path.basename(item['info']['info']['url'][0])
                labels = []
                for label in item['labels']:

                    if (label['data']['label'] == 'Zebrafish' and
                        label['data']['drawType'] == 'LINE' and
                            label['data']['group'] == 0):

                        if ('frames' in label['data'] and
                            len(label['data']['frames']) == 1 and
                            label['data']['frames'][0]['frame'] == 0 and
                                label['data']['frames'][0]['outside']):

                            points = label['data']['frames'][0]['points']
                            labels.extend([points[i*2:i*2+2]
                                            for i in range(len(points)//2)])

                        elif ('points' in label['data'] and
                                label['data']['frameIndex'] == 0 and
                                label['data']['keyframe'] and
                                label['data']['outside']):

                            points = label['data']['points']
                            labels.extend([points[i*2:i*2+2]
                                            for i in range(len(points)//2)])

                        else:
                            error.append(key)

                    else:
                        error.append(key)

                data[key] = labels
        return data, error