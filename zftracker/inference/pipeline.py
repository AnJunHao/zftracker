from typing import Any
from .. import model
from .. import preprocess
from .. import dataset
from .. import inference
from .. import visual
from .. import criterion
from time import time
import torch
from ..util.tqdm import TQDM as tqdm
from torch.utils.data import DataLoader
import os
import numpy as np
from icecream import ic
import cv2
from math import ceil
from matplotlib import pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # For precise debugging
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # For reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

class PipelineV1:

    def __init__(self,
                 model_path: list,
                    device: torch.device,
                    file_path_list: list,
                    verbose: bool = True,
                    peak_threshold: float = 0.1,
                    max_detect: int = 20,
                    fps_divisor=1,
                    horizontal_cut: tuple = (145, -155),
                    target_size: tuple = (720, 980),
                    rotate: bool = False,
                    batch_size: int = 20):
        self.model_path = model_path
        self.device = device
        self.file_path_list = file_path_list
        self.num_frames = [preprocess.video_total_frames(file_path)
                           for file_path in file_path_list]
        self.peak_threshold = peak_threshold
        self.max_detect = max_detect
        self.fps_divisor = fps_divisor
        self.horizontal_cut = horizontal_cut
        self.target_size = target_size
        self.rotate = rotate
        self.batch_size = batch_size
        
        if verbose:
            print("Loading model...")
            start_time = time()
        self.load_model()
        if verbose:
            print("Model loaded in {:.2f} seconds.".format(time() - start_time))

        self.called = False
        
    
    def __call__(self, verbose: bool = True):

        if self.called:
            return self.traj_pool

        self.coords = []
        self.confs = []
        self.match_types = []

        for i, file_path in enumerate(tqdm(self.file_path_list)):
            if verbose:
                print('-' * 16)
                print(f"{i + 1}/{len(self.file_path_list)}")
                print('-' * 16)
                epoch_start_time = time()
                print("Loading video: {}".format(os.path.basename(file_path)))
                start_time = time()
            images_array = preprocess.get_frames_from_clip(
                file_path, fps_divisor=self.fps_divisor, rotate=self.rotate, verbose=verbose)[
                    :, :, self.horizontal_cut[0]:self.horizontal_cut[1], :]
            
            # Check if the shape of input images matches the target shape
            if images_array.shape[1:3] != self.target_size:
                # Resize the frames to the target shape
                images_array = preprocess.resize_frames(images_array, self.target_size)
            
            test_dataset = dataset.TestDataset(images_array)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            if verbose:
                print("Video loaded in {:.2f} seconds.".format(time() - start_time))
                print("Running inference...")
                start_time = time()
            model_outputs = self.forward_pass(test_loader, verbose=verbose)
            if verbose:
                print("Inference completed in {:.2f} seconds.".format(time() - start_time))
                print("Matching keypoints...")
                start_time = time()
            coords, conf_and_type = inference.keypoint.CategoryFirstDoublecheckKeypoint(
                    peak_threshold=self.peak_threshold,
                    max_detect=self.max_detect)(model_outputs)
            self.coords.extend(coords)
            self.confs.extend([[i[0] for i in l] for l in conf_and_type])
            self.match_types.extend([[i[1].split('_')[0]
                             for i in l]
                             for l in conf_and_type])
            if verbose:
                print("Keypoints matched in {:.2f} seconds.".format(time() - start_time))
                
            # Release memory.
            del images_array, test_dataset, test_loader, model_outputs

            if verbose:
                print('-' * 16)
                print(f"{i + 1}/{len(self.file_path_list)} completed in {time() - epoch_start_time:.2f} seconds.")
        
        current_time = time()

        tracker = inference.trajectory.Tracker(
            distance_threshold_per_frame=1.5,
            momentum_window=3,
            traj_length_belief=6,
            max_prepend_length=2,
            full_state_length_belief=3,
            overlap_length_threshold=0,
            fragment_pair=True)
        
        self.traj_pool = tracker.track(self.coords, 
                                       self.confs,
                                        self.match_types)
        
        print("Trajectory reconstruction completed in {:.2f} seconds.".format(time() - current_time))

        self.called = True

        return self.traj_pool
    
    def __getitem__(self, frame_index: int) -> dict:
        # Decide which video the frame belongs to.
        video_index = 0
        original_frame_index = frame_index
        while frame_index >= round(self.num_frames[video_index] / self.fps_divisor):
            frame_index -= round(self.num_frames[video_index] / self.fps_divisor)
            video_index += 1

        image = preprocess.get_single_frame_from_clip(
            self.file_path_list[video_index], frame_index*self.fps_divisor, rotate=self.rotate)[
                :, self.horizontal_cut[0]:self.horizontal_cut[1], :]
        
        # Check if the shape of input images matches the target shape
        if image.shape[0:2] != self.target_size:
            # Resize the frames to the target shape
            image = preprocess.resize_frames([image], self.target_size)[0]
        
        test_dataset = dataset.TestDataset(np.array([image]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        model_outputs = self.forward_pass(test_loader, verbose=False)

        if self.called:
            raw_coords = self.coords[original_frame_index]
            raw_image_with_lines = visual.draw_lines_on_image(
                image, raw_coords, display=False)
            coords = []
            for traj in self.traj_pool:
                coords.append(traj[original_frame_index])
            image_with_lines = visual.images.draw_lines_on_image(
                image, coords, display=False)
            return {'image': image,
                    'model_outputs': model_outputs,
                    'raw_coords': raw_coords,
                    'raw_visual': raw_image_with_lines,
                    'coords': coords,
                    'visual': image_with_lines,
                    'warning': "Model output is not deterministic and may be different from the first call."
                            "We keep the coords consistant with the fisrt call."
                            "Thus, there might be a slight discrepancy between the model outputs and the coords."}
        else:
            coords, conf_and_type = inference.keypoint.CategoryFirstDoublecheckKeypoint(
                peak_threshold=self.peak_threshold, max_detect=self.max_detect
                    )(model_outputs)
            image_with_lines = visual.images.draw_lines_on_image(image, coords[0], display=False)
        
            return {'image': image,
                    'model_outputs': model_outputs,
                    'raw_coords': coords,
                    'raw_visual': image_with_lines,
                    'conf_and_type': conf_and_type,
                    'warning': "Model output is not deterministic and may be different from the first call."
                            "We keep the coords consistant with the fisrt call."
                            "Thus, there might be a slight discrepancy between the model outputs and the coords."}
        
    
    def draw_trajectory_on_video(self, output_file):      
        """
        Draws trajectories on a video and saves the result as a new video file.

        Args:
            output_file (str): Path to save the output video file.
        """
        visual.videos.draw_trajectory_on_video(self.file_path_list, self.traj_pool, output_file)

    def forward_pass(self, test_loader, verbose: bool = False):

        self.loaded_model.eval()

        with torch.no_grad():

            all_outputs = []

            if verbose:
                iterator = tqdm(test_loader)
            else:
                iterator = test_loader

            for val_inputs in iterator:

                val_inputs = val_inputs.to(self.device)

                outputs = self.loaded_model(val_inputs)

                all_outputs.append(outputs)

            all_outputs = torch.cat(all_outputs)

        return all_outputs

    def load_model(self):

        self.loaded_model = model.ResnetOffsetDoublecheckDeconv(
                                num_keypoints=2,
                                num_deconv=3,
                                resnet="resnet18",
                                pretrained=False,
                                dropout=0.5,
                                padding=[
                                    (1, 1),
                                    (2, 2),
                                    (1, 1),
                                ],  # These are set to match an exact 1/4 size heatmap.
                                output_padding=[
                                    (0, 0),  # Output padding may cause strange behavior near the edge.
                                    (0, 0),  # Thus, values in the list should better be minimized.
                                    (0, 1),  # The "1" here is necessary to produce an odd output.
                                ],
                            )
        
        msg = self.loaded_model.load_state_dict(torch.load(self.model_path))
        
        self.loaded_model.to(self.device)

        return msg
    
class PipelineV2:

    def __init__(self,
                 model_path: str,
                 file_path_list: list,
                 horizontal_cut: tuple,
                 doublecheck_normalization_factor: dict = {
                     ('head', 'midsec'): 8,
                     ('head', 'tail'): 26.620351900862914,
                     ('midsec', 'tail'): 16.89075517685661}, # The mean distance of Dataset V1.0 (before fix)
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 rotate: bool = False,
                 fps_divisor: int = 1,
                 batch_size: int = 10,
                 doublecheck_version: int = 1, # 1, 2 or 3, version 1 is recommended
                 tracker_version: int = 2): # 1 or 2, version 2 is recommended
        self.model_path = model_path
        self.device = device
        self.file_path_list = file_path_list
        self.horizontal_cut = horizontal_cut
        self.rotate = rotate
        self.fps_divisor = fps_divisor
        self.batch_size = batch_size
        self.called = False

        self.coords_predictor = inference.keypoint.TripleKeypoint(
            num_entities=7,
            max_detect=7,
            peak_threshold=0.5,
            peak_radius=1,
            average_2x2=False,
            doublecheck_normalization_factor=doublecheck_normalization_factor,
            doublecheck_threshold=4,
            head_mid_distance_range=(0.5, 1.5),
            tail_cluster=True,
            doublecheck_version=doublecheck_version)
        
        if tracker_version == 1:
            self.tracker = inference.trajectory.Tracker(
                num_keypoints=7,
                distance_threshold_per_frame=8,
                traj_length_belief=8,
                full_state_length_belief=4, # To counter false positives
                max_prepend_length=2, # 0~2
                fragment_pair=True,
                momentum_window=3, # recommened 3
                overlap_length_threshold=4)
        elif tracker_version == 2:
            self.tracker = inference.trajectory.TrackerV2(
                num_keypoints=7,
                distance_threshold_per_frame=3,
                traj_length_belief=8,
                full_state_length_belief=4,
                momentum_window=3,
                overlap_length_threshold=4,
                max_gap=0,
                tail_weight=1/3)
        else:
            raise ValueError("tracker_version must be 1 or 2.")
        
        self.doublecheck_version = doublecheck_version
        self.tracker_version = tracker_version

        print("Loading model...")
        start_time = time()
        self.load_model()
        print(f"Model loaded to {self.device} in {time() - start_time:.2f} seconds.")
    
    def __call__(self, verbose: bool = False):

        if self.called:
            return self.traj_pool
        
        self.coords = []
        self.confs = []
        self.match_types = []

        for i, file_path in enumerate(tqdm(self.file_path_list)):
            frames = preprocess.video.get_frames_from_clip(
                file_path,
                rotate=self.rotate,
                fps_divisor=self.fps_divisor,
                verbose=verbose,
                horizontal_cut=self.horizontal_cut)
            test_dataset = dataset.TestDataset(frames)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            model_outputs = inference.model.heatmap_regression_inference(self.model, test_loader, self.device, verbose=verbose)
            matched_coords, matched_confs, matched_types = self.coords_predictor(model_outputs)
            matched_coords = [[np.concatenate(fish) for fish in cds]
                  for cds in matched_coords]
            matched_confs = [np.mean(cfs, axis=1) for cfs in matched_confs]
            matched_types = [[tp[np.argmax([len(t) for t in tp])]
                              for tp in tps]
                             for tps in matched_types]
            self.coords.extend(matched_coords)
            self.confs.extend(matched_confs)
            self.match_types.extend(matched_types)
            del frames, test_dataset, test_loader, model_outputs

        self.traj_pool = self.tracker.track(self.coords, self.confs, self.match_types)
        self.called = True

        return self.traj_pool
    
    def __getitem__(self, frame_index: int) -> dict:

        # Decide which video the frame belongs to.
        video_index = 0
        original_frame_index = frame_index
        while frame_index >= ceil(preprocess.video.video_total_frames(self.file_path_list[video_index]) / self.fps_divisor):
            frame_index -= ceil(preprocess.video.video_total_frames(self.file_path_list[video_index]) / self.fps_divisor)
            video_index += 1
        
        image = preprocess.video.get_single_frame_from_clip(
            self.file_path_list[video_index], frame_index*self.fps_divisor, rotate=self.rotate)[
                :, self.horizontal_cut[0]:self.horizontal_cut[1], :]
        
        test_dataset = dataset.TestDataset(np.array([image]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model_outputs = inference.model.heatmap_regression_inference(self.model, test_loader, self.device, verbose=False)

        (matched_coords, matched_confs, matched_types), raw_coords = self.coords_predictor(model_outputs, return_raw_coords=True)
        matched_coords = [[np.concatenate(fish) for fish in cds]
                  for cds in matched_coords]
        matched_confs = [np.mean(cfs, axis=1) for cfs in matched_confs]
        matched_types = [[tp[np.argmax([len(t) for t in tp])]
                          for tp in tps]
                        for tps in matched_types]
        
        # Get the graph and image
        mapping, coords_dict = self.coords_predictor.evaluator.get_mapping_from_pred_coords(raw_coords, 0)
        graph = criterion.graph.generate_full_graph(mapping)
        subgraphs = criterion.graph.get_weakly_connected_components(graph)
        biggest_subgraph = max(subgraphs, key=lambda x: len(x.nodes))

        if self.doublecheck_version == 3:
            graph_pools, unprocessed_graphs = criterion.graph.analyze_full_graph_v3(graph, 7)
            num_missing = 7 - len(graph_pools['original']) - len(graph_pools['denoised']) - len(graph_pools['divided'])
            if num_missing > 0 and len(unprocessed_graphs) > 0:
                graph_pools['clustered'] = self.coords_predictor.evaluator.cluster_awaiting_graph_nodes(unprocessed_graphs, num_missing, coords_dict)
            fishes, types = self.coords_predictor.evaluator.select_fishes_from_graph_pool(graph_pools, coords_dict)

        elif self.doublecheck_version == 2:
            active_graphs, awaiting_graphs, num_missing = criterion.graph.analyze_full_graph_v2(graph, 7)
            if num_missing > 0:
                awaiting_graphs = self.coords_predictor.evaluator.cluster_awaiting_graph_nodes(awaiting_graphs, num_missing, coords_dict)
            graph_pools = {'original': active_graphs,
                            'clustered': awaiting_graphs}
            fishes, types = self.coords_predictor.evaluator.select_fishes_from_graph_pool(graph_pools, coords_dict)

        elif self.doublecheck_version == 1:
            graph_pool = criterion.graph.analyze_full_graph(graph)
            fishes, types = self.coords_predictor.evaluator.select_fishes_from_graph_pool(graph_pool)

        # Draw keypoints on the image
        for fish in fishes:
            for node in fish:
                x, y = coords_dict[node[0]]['coords'][node[1]] * 4
                cv2.circle(image, (round(x), round(y)), 5, (0, 255, 0), -1)
                cv2.putText(image, node[0][0]+str(node[1]), (round(x), round(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, )
        return {'image': test_dataset[0],
                'model_outputs': model_outputs,
                'raw_coords': raw_coords,
                'matched_coords': matched_coords,
                'graph': graph,
                'biggest_subgraph': biggest_subgraph,
                'visual': image,
                'confs': matched_confs,
                'types': matched_types}
    
    def run(self, verbose: bool = False):
        return self(verbose=verbose)
        
    def load_model(self):
        self.model = model.ResnetTripleOffsetDoublecheckDeconv(
            num_deconv=3,
            deconv_channels=256,
            resnet="resnet18",
            pretrained=False,
            dropout=0.5,
            padding=[
                (2, 2), # Increase 1 will reduce output size by 8 pixels.
                (0, 1), # Increase 1 will reduce output size by 4 pixels.
                (0, 0), # Increase 1 will reduce output size by 2 pixels.
            ],  # These are set to match an exact 1/4 size heatmap.
            output_padding=[
                (0, 0),  # Output padding may cause strange behavior near the edge.
                (0, 0),  # Thus, values in the list should better be minimized.
                (0, 0),  # The "1" here is necessary to produce an odd output.
            ],
        )
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(self.device)

    def get_type_string_key_frames(self, str_start_index, string):
        """
        Get the key frames of the string in the string list.
        """
        frames = []
        previous_updated_frame = 0
        duration = 0
        start_frame = 0
        min_duration = 2
        for i, types in enumerate(self.match_types):
            updated = False
            for tps in types:
                if tps[str_start_index:str_start_index+len(string)] == string:
                    updated = True
                    break
            if updated:
                if i >= previous_updated_frame + 10 and duration == 0:
                    duration += 1
                    previous_updated_frame = i
                    start_frame = i
                elif i == previous_updated_frame + 1 and duration > 0:
                    duration += 1
                    previous_updated_frame = i
                elif i > previous_updated_frame + 1 and duration >= min_duration:
                    frames.append(start_frame)
                    duration = 0
                elif i > previous_updated_frame + 1 and duration < min_duration:
                    duration = 0
                    previous_updated_frame = i
        return frames
    
class PipelineV2WithClassification(PipelineV2):
        
    def __init__(self,
                heatmap_model_path: str,
                file_path_list: list,
                horizontal_cut: tuple,
                doublecheck_normalization_factor: dict = {
                    ('head', 'midsec'): 8,
                    ('head', 'tail'): 26.620351900862914,
                    ('midsec', 'tail'): 16.89075517685661}, # The mean distance of Dataset V1.0 (before fix)
                device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                rotate: bool = False,
                fps_divisor: int = 1,
                batch_size: int = 10,
                doublecheck_version: int = 1,
                tracker_version: int = 2,
                classification_model_path: str = None):
        super().__init__(heatmap_model_path, file_path_list, horizontal_cut, doublecheck_normalization_factor, device, rotate, fps_divisor, batch_size, doublecheck_version, tracker_version)
        start_time = time()
        self.classification_model_path = classification_model_path
        self.classification_model = model.ResNetSimpleClassify(resnet='resnet18',
                                    pretrained=True,
                                    dropout=0.5,
                                    dense_units=512,
                                    num_dense_layers=0,
                                    freeze_resnet=True)
        self.classification_model.load_state_dict(torch.load(classification_model_path))
        self.classification_model.to(self.device)
        print(f"Classification model loaded to {self.device} in {time() - start_time:.2f} seconds.")
        
    def classify(self, verbose: bool = True):
        self.overlaps = inference.cover.get_overlaps_from_traj_pool(self.traj_pool, verbose=verbose)

        all_frames = []
        all_keypoints = []

        accumulative_frames = 0
        for i, file_path in enumerate(tqdm(self.file_path_list)):
            total_frames = preprocess.video.video_total_frames(file_path)
            crop_dict = inference.cover.get_crop_dict(
                self.overlaps, self.traj_pool,
                frame_index_minus=accumulative_frames,
                frame_index_multiply=self.fps_divisor,
                rescale_factor=4,
                frame_index_upper_bound=total_frames)
            accumulative_frames += total_frames
            frames, keypoints = preprocess.video.get_cropped_frames_and_keypoints_from_clip(
                file_path, crop_dict, crop_range=80,
                rotate=self.rotate, horizontal_cut=self.horizontal_cut, verbose=verbose)
            all_frames.extend(frames)
            all_keypoints.extend(keypoints)

        self.cls_dataset = dataset.LocalClassifyTestDataset(all_frames, all_keypoints, head_midsec_distance=32, attention_radius=20)
        self.cls_dataset.sequential_initialize(None)
        self.cls_loader = DataLoader(self.cls_dataset, batch_size=self.batch_size, shuffle=False)

        self.cls_result = inference.model.classification_inference(self.classification_model, self.cls_loader, self.device, verbose=verbose)

        return self.group_cls_result(self.cls_result, self.overlaps)

    def group_cls_result(self, cls_result, overlaps):
        result = {}
        accumulative_segments = {}
        current_frame = 0
        not_updated_group = set()
        for index, overlap in enumerate(overlaps):
            if overlap[0] > current_frame:
                current_frame = overlap[0]
                for key in not_updated_group:
                    result[(accumulative_segments[key]['start_frame'], *key)] = accumulative_segments[key]
                    accumulative_segments.pop(key)
                not_updated_group = set(accumulative_segments.keys())
            if overlap[1:] not in accumulative_segments:
                accumulative_segments[overlap[1:]] = {'start_frame': overlap[0],
                                                    'end_frame': overlap[0]+1,
                                                    'start_index': index,
                                                    'end_index': index+1,
                                                    'cls_a': [cls_result[index*2].item()],
                                                    'cls_b': [cls_result[index*2+1].item()]}
            else:
                accumulative_segments[overlap[1:]]['end_frame'] = overlap[0]+1
                accumulative_segments[overlap[1:]]['end_index'] = index+1
                accumulative_segments[overlap[1:]]['cls_a'].append(cls_result[index*2].item())
                accumulative_segments[overlap[1:]]['cls_b'].append(cls_result[index*2+1].item())
                not_updated_group.discard(overlap[1:])

        for segment in accumulative_segments:
            result[(accumulative_segments[segment]['start_frame'], *segment)] = accumulative_segments[segment]

        return result
    
class PipelineV3:

    def __init__(self,
                 heatmap_model_path: str,
                 geometric_model_path: str,
                 file_path: list,
                 horizontal_cut: tuple,
                 doublecheck_normalization_factor: dict = {('head', 'midsec'): 8,
                                                            ('head', 'tail'): 26.620351900862914,
                                                            ('midsec', 'tail'): 19.080064573228345},
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 rotate: bool = False,
                 fps_divisor: int = 2,
                 batch_size: int = 8):
        self.heatmap_model_path = heatmap_model_path
        self.geometric_model_path = geometric_model_path
        self.device = device
        self.file_path = file_path
        self.horizontal_cut = horizontal_cut
        self.rotate = rotate
        self.fps_divisor = fps_divisor
        self.batch_size = batch_size
        self.called = False

        self.coords_predictor = inference.keypoint.TripleKeypoint(
            num_entities=7,
            max_detect=7,
            peak_threshold=0.5,
            peak_radius=1,
            average_2x2=False,
            doublecheck_normalization_factor=doublecheck_normalization_factor,
            doublecheck_threshold=4,
            head_mid_distance_range=(0.5, 1.5),
            tail_cluster=True,
            doublecheck_version=1) # Doesn't matter, as the doublecheck is not used in this version

        print("Loading model...")
        start_time = time()
        self.load_heatmap_model()
        self.load_geometric_model()
        print(f"Model loaded to {self.device} in {time() - start_time:.2f} seconds.")

        predictor = inference.trajectory_v2.GeometricPredictor(
            model=self.geometric_model,
            device=self.device,
            window_length=31)

        geometric_prediction_param = {
            'length': 31,
            'predictor': predictor,
            'weight': {
                'head': 0.25, # Net pred from midsec is generally good
                'midsec': 0.25, # Net pred from head is generally good
                'tail': 0.75, # Net pred from head & midsec is generally bad, we may rely more on geometric prediction
                'hm': {'head': 0.75, # predicting from tail
                    'midsec': 0.75},
                'ht': {'head': 0.25, # predicting from midsec
                    'tail': 0.75},
                'mt': {'midsec': 0.25,
                    'tail': 0.75}
            }
        }
        
        self.tracker = inference.trajectory_v2.TrackerV3(
            num_keypoints=7,
            distance_threshold_per_frame=3, # Tested on FPS=120
            traj_conf_belief=3, # A 3-node prediction has a conf of roughly 0.8, so 3 means a 4 consecutive 3-node predictions
                                    # 2-node prediction has a conf of roughly 0.55
                                    # 1-node prediction has a conf of roughly 0.3
            full_state_length_belief=3,
            momentum_window=3,
            overlap_length_threshold=1,
            tail_weight=1/2,   # Note that in this version, the tail_weight should be better
                            # considered as a factor to normalize the mae (so that the mae
                            # of the tail is comparable to the mae of the head and midsec)
            geometric_prediction_param=geometric_prediction_param,
            doublecheck_threshold=2)
        
    def show_preprocessed_frame(self):
        if isinstance(self.file_path, str):
            frame = preprocess.video.get_single_frame_from_clip(
                self.file_path, 0, rotate=self.rotate)[
                    :, self.horizontal_cut[0]:self.horizontal_cut[1], :]
        elif isinstance(self.file_path, list):
            frame = preprocess.video.get_single_frame_from_clip(
                self.file_path[0], 0, rotate=self.rotate)[
                    :, self.horizontal_cut[0]:self.horizontal_cut[1], :]
        else:
            raise ValueError("file_path must be a string or a list of strings.")
        plt.imshow(frame)
        return frame
    
    def __getitem__(self, frame_index):
        if isinstance(self.file_path, str):
            image = preprocess.video.get_single_frame_from_clip(self.file_path, frame_index*self.fps_divisor, rotate=self.rotate)[
                :, self.horizontal_cut[0]:self.horizontal_cut[1], :]
        elif isinstance(self.file_path, list):
            # Decide which video the frame belongs to.
            video_index = 0
            original_frame_index = frame_index
            while frame_index >= ceil(preprocess.video.video_total_frames(self.file_path[video_index]) / self.fps_divisor):
                frame_index -= ceil(preprocess.video.video_total_frames(self.file_path[video_index]) / self.fps_divisor)
                video_index += 1
            image = preprocess.video.get_single_frame_from_clip_stable(
            self.file_path[video_index], frame_index*self.fps_divisor, rotate=self.rotate)[
                :, self.horizontal_cut[0]:self.horizontal_cut[1], :]
        else:
            raise ValueError("file_path must be a string or a list of strings.")
        
        test_dataset = dataset.TestDataset(np.array([image]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model_outputs = inference.model.heatmap_regression_inference(self.heatmap_model, test_loader, self.device, verbose=False)

        raw_coords = self.coords_predictor(model_outputs, raw_coords_only=True)

        return {'image': image,
                'model_outputs': model_outputs[0],
                'raw_coords': raw_coords}

    def __call__(self, verbose: bool = False):

        if self.called:
            return self.traj_pool
        
        self.coords = []

        if isinstance(self.file_path, str):
            frames = preprocess.video.get_frames_from_clip_generator(
                self.file_path,
                rotate=self.rotate,
                fps_divisor=self.fps_divisor,
                verbose=True,
                horizontal_cut=self.horizontal_cut)
            test_dataset = dataset.FrameIterableDataset(frames)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            model_outputs = inference.model.heatmap_regression_inference_generator(
                self.heatmap_model, test_loader, self.device, verbose=False)
            for outputs in model_outputs:
                self.coords.append(self.coords_predictor(outputs, raw_coords_only=True))
        elif isinstance(self.file_path, list):
            for i, file_path in enumerate(tqdm(self.file_path)):
                frames = preprocess.video.get_frames_from_clip_generator(
                    file_path,
                    rotate=self.rotate,
                    fps_divisor=self.fps_divisor,
                    verbose=True,
                    horizontal_cut=self.horizontal_cut)
                test_dataset = dataset.FrameIterableDataset(frames)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                model_outputs = inference.model.heatmap_regression_inference(self.heatmap_model, test_loader, self.device, verbose=False)
                self.coords.append(self.coords_predictor(model_outputs, raw_coords_only=True))
                del frames, test_dataset, test_loader, model_outputs
        else:
            raise ValueError("file_path must be a string or a list of strings.")

        concatenated_coords = [[[], [], [], []], # Head
                               [[], [], [], []], # Midsec
                               [[], [], [], []]] # Tail
        for cds in self.coords:
            for i, part in enumerate(cds):
                for j, content in enumerate(part):
                    concatenated_coords[i][j].extend(content)

        self.coords = concatenated_coords

        self.traj_pool = self.tracker.track(self.coords, verbose=verbose)

        self.called = True

        return self.traj_pool
    
    def load_heatmap_model(self):
        self.heatmap_model = model.ResnetTripleOffsetDoublecheckDeconv(
            num_deconv=3,
            deconv_channels=256,
            resnet="resnet18",
            pretrained=False,
            dropout=0.5,
            padding=[
                (2, 2), # Increase 1 will reduce output size by 8 pixels.
                (0, 1), # Increase 1 will reduce output size by 4 pixels.
                (0, 0), # Increase 1 will reduce output size by 2 pixels.
            ],  # These are set to match an exact 1/4 size heatmap.
            output_padding=[
                (0, 0),  # Output padding may cause strange behavior near the edge.
                (0, 0),  # Thus, values in the list should better be minimized.
                (0, 0),  # The "1" here is necessary to produce an odd output.
            ],
        )
        self.heatmap_model.load_state_dict(torch.load(self.heatmap_model_path))
        self.heatmap_model.to(self.device)
    
    def load_geometric_model(self):
        self.geometric_model = model.GeometricDense(
            input_size=180,
            num_dense_layers=1,
            dense_units=128,
            dropout=0.5)
        self.geometric_model.load_state_dict(torch.load(self.geometric_model_path))
        self.geometric_model.to(self.device)