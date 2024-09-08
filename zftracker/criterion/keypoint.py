import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from abc import ABC, abstractmethod
import networkx as nx
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_recall_curve
from icecream import ic
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
from .graph import generate_full_graph, analyze_full_graph_v2, analyze_full_graph, analyze_full_graph_v3

from ..util.tqdm import TQDM as tqdm
from ..util.str import format_percentage

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

class KeypointEvaluator(ABC):
    def __init__(self,
                 max_detect: int = 100,
                 peak_threshold: float = 0.5,
                 peak_radius: int = 1,
                 local_distance_threshold: float = 1.5,
                 overlap_as_true_positive: bool = False):
        """
        Args:
            max_detect (int): The maximum number of peaks to detect.
            peak_threshold (float): The minimum intensity of peaks.
            peak_radius (int): The radius of peaks.
            local_distance_threshold (float): The maximum distance between a predicted coordinate and a ground truth coordinate to be considered as a true positive.
            overlap_as_true_positive (bool): Whether to allow two preds to match the same gt.
        """
        self.max_detect = max_detect
        self.peak_threshold = peak_threshold
        self.peak_radius = peak_radius
        self.local_distance_threshold = local_distance_threshold
        self.overlap_as_true_positive = overlap_as_true_positive

    def pair_entities(self, a_to_b_mapping, num_b):
        """
        Given a mapping from entities in A to entities in B, find the maximum
        cardinality matching of A to B.
        Args:
            a_to_b_mapping (dict): A dictionary mapping entities in A to entities in B.
            num_b (int): The number of entities in B.
        Returns:
            pairs (list): A list of pairs of entities from A to B.
        """

        # Create empty bipartite graph
        G = nx.Graph()

        n = len(a_to_b_mapping)

        # Add nodes with the node attribute "bipartite"
        G.add_nodes_from(range(n), bipartite=0)
        G.add_nodes_from(range(n, n+num_b), bipartite=1)

        # Add edges
        for a, b_list in a_to_b_mapping.items():
            for b in b_list:
                G.add_edge(a, b+n)

        G_list = [G.subgraph(c) for c in nx.connected_components(G)]

        # Compute maximum cardinality matching of G
        # It returns a dictionary with a pair for each edge in the matching
        pairs = []
        for sub_G in G_list:
            matching = nx.bipartite.maximum_matching(sub_G)
            pairs += [(a, b-n) for a, b in matching.items() if a < n]

        return pairs

    def calculate_metrics_from_mapping(self,
                                       gt_index_to_pred_candidate_indices: dict,
                                       gt_index_to_pred_distances: dict,
                                       num_pred: int,
                                       return_false_positive: bool = False):
        """
        Args:
            gt_index_to_pred_candidate_indices (dict): Dictionary of indices of the predicted coordinates.
                The key of the dictionary corresponds to the index of the ground truth coordinate.
                The value of the dictionary is a list of indices of possible predicted coordinates (within the local distance threshold).
            gt_index_to_pred_distances (dict): Dictionary of distances between the predicted coordinates and the ground truth coordinates.
                The key of the dictionary corresponds to the index of the ground truth coordinate.
                The value of the dictionary is a list of distances between all predicted coordinates and the ground truth coordinates.
            num_pred (int): Number of predicted coordinates.
            return_false_positive (bool): Whether to return the false positive list.
        Returns:
            tp (int): Number of true positives.
            fp (int): Number of false positives.
            fn (int): Number of false negatives.
            accumulated_mae (float): Accumulated mean average error.
            fp_list (np
        """

        gt_to_pred_pairs = self.pair_entities(gt_index_to_pred_candidate_indices,
                                              num_pred)

        tp = len(gt_to_pred_pairs)
        fn = len(gt_index_to_pred_candidate_indices) - tp
        fp = num_pred - tp

        accumulated_mae = 0
        for gt_i, pred_i in gt_to_pred_pairs:
            accumulated_mae += gt_index_to_pred_distances[gt_i][pred_i]

        if return_false_positive:
            tp_preds = [i[1] for i in gt_to_pred_pairs]
            fp_list = [0 if i in tp_preds else 1 for i in range(num_pred)]
            return {'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'accumulated_mae': accumulated_mae,
                    'fp_list': fp_list}
        else:
            return {'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'accumulated_mae': accumulated_mae}

    def find_local_peaks(self, data, return_confidence=False):
        """
        Args:
            data (torch.Tensor or np.ndarray): The data to find local peaks.
            return_confidence (bool): Whether to return the confidence of the local peaks. Default: ``False``.
        Returns:
            list: List of coordinates of the local peaks, sorted by the confidence in descending order.
            (optional) list: List of confidence of the local peaks.
        """
        return self._find_local_peaks(data, self.peak_threshold, self.peak_radius, self.max_detect, return_confidence)

    def _find_local_peaks(self, data, peak_threshold, peak_radius, max_detect, return_confidence=False):
        """
        Args:
            data (torch.Tensor or np.ndarray): The data to find local peaks.
            peak_threshold (float): The minimum intensity of peaks.
            peak_radius (int): The radius of peaks.
            max_detect (int): The maximum number of peaks to detect.
            return_confidence (bool): Whether to return the confidence of the local peaks. Default: ``False``.
        Returns:
            list: List of coordinates of the local peaks, sorted by the confidence in descending order.
            (optional) list: List of confidence of the local peaks.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if len(data.shape) == 2:
            # Add a new dimension to the data if it is 2D
            data = np.expand_dims(data, axis=0)
        local_peaks = [peak_local_max(heatmap, min_distance=peak_radius, threshold_abs=peak_threshold, num_peaks=max_detect)[
            :, ::-1] for heatmap in data]
        if return_confidence:
            confidences = [[heatmap[coord[1], coord[0]] for coord in coords]
                          for heatmap, coords in zip(data, local_peaks)]
            return local_peaks, confidences
        else:
            return local_peaks

    def calculate_metrics(self, gt_coords, pred_coords, return_false_positive=False):
        """
        Args:
            gt_coords (list): List of ground truth coordinates. Each element is a numpy array of shape (n, 2).
            pred_coords (list): List of predicted coordinates. Each element is a numpy array of shape (n, 2).
        Returns:
            dict: Dictionary of metrics.
        """
        
        return self._calculate_metrics_v2(gt_coords, pred_coords,
                                          self.local_distance_threshold,
                                          return_false_positive)

        return self._calculate_metrics(gt_coords, pred_coords,
                                       self.local_distance_threshold,
                                       overlap_as_true_positive=self.overlap_as_true_positive,
                                       return_false_positive=return_false_positive)
    
    def _calculate_metrics_v2(self,
                              gt_coords: list,
                              pred_coords: list,
                              local_distance_threshold: float,
                              return_false_positive: bool):
        """
        Args:
            gt_coords (list): List of ground truth coordinates. Each element is a numpy array of shape (n, 2).
            pred_coords (list): List of predicted coordinates. Each element is a numpy array of shape (n, 2).
            local_distance_threshold (float): The maximum distance between a predicted coordinate and a ground truth coordinate to be considered as a true positive.
            return_false_positive (bool): Whether to return the false positive list.
        Returns:
            dict: Dictionary of metrics.
        """
        tp = 0
        fp = 0
        fn = 0
        mae = 0

        n_gts = sum(len(gt) for gt in gt_coords)
        n_preds = sum(len(pred) for pred in pred_coords)

        if return_false_positive:
            fp_list = []
        
        for gt_cds, pred_cds in zip(gt_coords, pred_coords):
            # Use Hungarian algorithm to find the optimal matching
            weights = []
            for gt in gt_cds:
                weight = []
                for pred in pred_cds:
                    distances = np.linalg.norm(pred - gt)
                    weight.append(distances)
                weights.append(weight)
            weights = np.array(weights)
            gt_indices, pred_indices = linear_sum_assignment(weights)
            if return_false_positive:
                fp_list.append([1] * len(pred_cds))
            for gt_i, pred_i in zip(gt_indices, pred_indices):
                if weights[gt_i, pred_i] < local_distance_threshold:
                    tp += 1
                    mae += weights[gt_i, pred_i]
                    if return_false_positive:
                        fp_list[-1][pred_i] = 0
                else:
                    fn += 1

        fp = n_preds - tp

        if n_preds == 0 or n_gts == 0:
            precision = False
            recall = False
            mae = False
        else:
            if tp == 0:
                precision = 0
                recall = 0
                mae = 0
            else:
                precision = tp / n_preds
                recall = tp / n_gts
                mae /= tp

        result = {
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0,
            'mae': mae,
            'true_positive': tp,
            'false_positive': fp,
            'false_negative': fn,
            'n_predictions': n_preds,
            'n_ground_truths': n_gts
        }

        if return_false_positive:
            result['false_positive_list'] = fp_list
            
        return result

    def _calculate_metrics(self,
                           gt_coords: list,
                           pred_coords: list,
                           local_distance_threshold: float,
                           overlap_as_true_positive: bool,
                           return_false_positive: bool):
        """
        Args:
            gt_coords (list): List of ground truth coordinates. Each element is a numpy array of shape (n, 2).
            pred_coords (list): List of predicted coordinates. Each element is a numpy array of shape (n, 2).
            local_distance_threshold (float): The maximum distance between a predicted coordinate and a ground truth coordinate to be considered as a true positive.
            overlap_as_true_positive (bool): Whether to allow two preds to match the same gt.
            return_false_positive (bool): Whether to return the false positive list.
        Returns:
            dict: Dictionary of metrics.
        """
        tp = 0
        fp = 0
        fn = 0
        if overlap_as_true_positive:
            overlap = 0
        mae = 0

        n_gts = sum(len(gt) for gt in gt_coords)
        n_preds = sum(len(pred) for pred in pred_coords)

        if return_false_positive:
            fp_list = []

        for gt, pred in zip(gt_coords, pred_coords):

            if len(pred) == 0:
                fn += len(gt)
                continue

            if overlap_as_true_positive:

                unpaired_preds = np.ones(len(pred), dtype=int)

                for kp in gt:
                    distances = np.linalg.norm(pred - kp, axis=1)
                    closest_candidate = np.argmin(distances)
                    if distances[closest_candidate] < local_distance_threshold:
                        tp += 1
                        mae += distances[closest_candidate]
                        if unpaired_preds[closest_candidate] == 0:
                            overlap += 1
                        else:
                            unpaired_preds[closest_candidate] = 0
                    else:
                        fn += 1
                fp += np.sum(unpaired_preds)
                if return_false_positive:
                    fp_list.append(unpaired_preds)
            else:

                gt_index_to_pred_candidate_indices = {}
                gt_index_to_pred_distances = {}

                for gt_i, kp in enumerate(gt):

                    distances = np.linalg.norm(pred - kp, axis=1)
                    gt_index_to_pred_distances[gt_i] = distances

                    # List of indices of the predicted coordinates
                    # In ascending order of distance to the ground truth coordinate
                    # Distances should be within the local distance threshold
                    candidates = np.where(
                        distances < local_distance_threshold)[0]
                    gt_index_to_pred_candidate_indices[gt_i] = candidates

                bpm_metrics = self.calculate_metrics_from_mapping(
                    gt_index_to_pred_candidate_indices,
                    gt_index_to_pred_distances,
                    len(pred),
                    return_false_positive=return_false_positive)

                fp += bpm_metrics['fp']
                tp += bpm_metrics['tp']
                fn += bpm_metrics['fn']
                mae += bpm_metrics['accumulated_mae']

                if return_false_positive:
                    fp_list.append(bpm_metrics['fp_list'])

        if n_preds == 0 or n_gts == 0:
            precision = False
            recall = False
            mae = False
        else:
            if tp == 0:
                precision = 0
                recall = 0
                mae = 0
            else:
                if overlap_as_true_positive:
                    precision = (tp - overlap) / n_preds
                else:
                    precision = tp / n_preds
                recall = tp / n_gts
                mae /= tp

        result = {
            'precision': precision,
            'recall': recall,
            'mae': mae,
            'true_positive': tp,
            'false_positive': fp,
            'false_negative': fn,
            'n_predictions': n_preds,
            'n_ground_truths': n_gts
        }

        if return_false_positive:
            result['false_positive_list'] = fp_list

        if overlap_as_true_positive:
            result['overlap'] = overlap

        return result

    def plot_tradeoff(self, predictions, test_coords, enumerate_range=(11, 90)):

        prev_peak_threshold = self.peak_threshold

        x = []
        y = []

        for i in tqdm(range(*enumerate_range)):
            self.peak_threshold = i / 100
            x.append(self.peak_threshold)
            y.append(
                self.evaluate(
                    predictions,
                    test_coords
                )
            )

        y_precision = [i["precision"] for i in y]
        y_recall = [i["recall"] for i in y]
        y_multiply = [i * j for i, j in zip(y_precision, y_recall)]
        y_average = [(i + j) / 2 for i, j in zip(y_precision, y_recall)]

        plt.plot(x, y_precision, label="precision")
        plt.plot(x, y_recall, label="recall")
        plt.plot(x, y_multiply, label="multiply")
        plt.plot(x, y_average, label="average")

        max_multiply_index = y_multiply.index(max(y_multiply))
        max_average_index = y_average.index(max(y_average))

        for idx, y_values, color, label in [(max_multiply_index, y_multiply, 'r', 'max multiply'),
                                            (max_average_index, y_average, 'b', 'max average')]:
            plt.plot(x[idx], y_values[idx], color+'o', label=label)
            plt.annotate(f'({x[idx]}, {y_values[idx]})',
                         (x[idx], y_values[idx]))

            print(f'{label} at peak_threshold={x[idx]}:')
            print(
                f'Precision / Recall / Average: {100*y_precision[idx]:.2f}% & {100*y_recall[idx]:.2f}% & {100*y_average[idx]:.2f}%')
            print(f"MAE: {y[idx]['mae']:.4f}")

        plt.legend()

        self.peak_threshold = prev_peak_threshold

        return plt

    @abstractmethod
    def evaluate(self, preds, gts, **kwargs):
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def __call__(self, preds, gts, **kwargs):
        return self.evaluate(preds, gts, **kwargs)


class SimpleKeypointEvaluator(KeypointEvaluator):
    def evaluate(self, preds, gts, ground_truths_as_coords=True, apply_sigmoid=False, detailed=False):
        if apply_sigmoid:
            preds = torch.sigmoid(preds)
        pred_coords = self.find_local_peaks(preds)

        if not ground_truths_as_coords:
            gt_coords = self.find_local_peaks(gts)
        else:
            gt_coords = gts

        return_false_positive = detailed

        metrics = self.calculate_metrics(
            gt_coords, pred_coords, return_false_positive)

        if detailed:
            return metrics
        else:
            return {'precision': metrics['precision'], 'recall': metrics['recall'], 'mae': metrics['mae']}

    def __repr__(self) -> str:
        return (f"SimpleKeypointEvaluator(max_detect={self.max_detect}, "
                f"peak_threshold={self.peak_threshold}, "
                f"peak_radius={self.peak_radius}, "
                f"overlap_as_true_positive={self.overlap_as_true_positive}, "
                f"local_distance_threshold={self.local_distance_threshold})")


class OffsetKeypointEvaluator(KeypointEvaluator):
    """
    This evaluator is used to evaluate the performance of the model with offset regression.
    """

    def __init__(self,
                 max_detect: int = 100,
                 peak_threshold: float = 0.5,
                 peak_radius: int = 1,
                 local_distance_threshold: float = 1.5,
                 overlap_as_true_positive: bool = False,
                 average_2x2: bool = False):
        """
        Args:
            max_detect (int): The maximum number of peaks to detect.
            peak_threshold (float): The minimum intensity of peaks.
            peak_radius (int): The radius of peaks.
            local_distance_threshold (float): The maximum distance between a predicted coordinate and a ground truth coordinate to be considered as a true positive.
            average_2x2 (bool): Whether to average the 2x2 grid of offsets.
        """
        super().__init__(max_detect, peak_threshold,
                         peak_radius, local_distance_threshold,
                         overlap_as_true_positive)
        self.average_2x2 = average_2x2

    def __repr__(self) -> str:
        return (f"OffsetKeypointEvaluator(max_detect={self.max_detect}, "
                f"peak_threshold={self.peak_threshold}, "
                f"peak_radius={self.peak_radius}, "
                f"local_distance_threshold={self.local_distance_threshold},"
                f"overlap_as_true_positive={self.overlap_as_true_positive},"
                f"average_2x2={self.average_2x2})")

    def apply_local_offsets(self, pred_coords, preds_local_offsets, average_2x2):
        """
        Args:
            pred_coords (list): List of predicted coordinates. Each element is a numpy array of shape (n, 2).
            preds_local_offsets (torch.Tensor): Predicted local offsets. Shape: (batch_size, 2, vertical_resolution, horizontal_resolution).
            average_2x2 (bool): Whether to average the 2x2 grid of offsets.
        Returns:
            list: List of adjusted coordinates.
        """
        if isinstance(preds_local_offsets, torch.Tensor):
            preds_local_offsets = preds_local_offsets.detach().cpu().numpy()

        adjusted_coords = []

        for pred, offset in zip(pred_coords, preds_local_offsets):

            adjusted_coord = []

            for kp in pred:
                if average_2x2:

                    # Note: To better understand the following code, please keep in mind:
                    # 1. 'kp' is defined as (x, y). In a plt.imshow graph, x denotes the horizontal axis and y denotes the vertical axis.
                    # 2. The shape of 'offset' is (2, vertical_resolution, horizontal_resolution). This is also the shape of the heatmap.
                    # 3. Thus, to access the offset of a specific keypoint, we use 'offset[channel, kp[1], kp[0]]'.
                    # 4. The first channel (index 0) of 'offset' is the y-axis offset, which varies vertically but keeps the same horizontally.
                    # 5. The second channel (index 1) of 'offset' is the x-axis offset, which varies horizontally but keeps the same vertically.
                    # -> Here is a brief description of a typical process:
                    #  | If value in the first channel of 'offset' is positive, it means the keypoint is shifted vertically towards the next y-axis value.
                    #  | In this case, we may want to access the value of the next row (kp[1]+1, which is the 'next' position along y-axis) to get a more accurate offset.
                    #  | For example, if the vertical offset of the current keypoint is 0.2, ideally, we expect the next row to have a vertical offset of -0.8.
                    #  | This is because the keypoint is shifted 0.2 from the current row towards the next row, which is equivalent to shifting 0.8 away from the next row towards the current row.
                    #  | To get the average offset, we need to add 0.5 to the result. For example, the average value is: (0.2 + -0.8) / 2 = -0.3; -0.3 + 0.5 = 0.2 (expected to be more accurate)
                    # -> The same rule applies to other channels and other values.

                    modifier_x = 0.5
                    modifier_y = 0.5

                    if offset[1, kp[1], kp[0]] > 0:

                        x_range = [kp[0], kp[0]+2]

                        if offset[0, kp[1], kp[0]] > 0:

                            y_range = [kp[1], kp[1]+2]

                            # If the point is on boundary, we need to modify the offset accordingly
                            if kp[1] + 1 == offset.shape[1]:
                                modifier_y = 0
                            if kp[0] + 1 == offset.shape[2]:
                                modifier_x = 0

                            adjusted_coord.append(
                                kp + np.array([np.mean(offset[1, y_range[0]:y_range[1], x_range[0]:x_range[1]]) + modifier_x,
                                               np.mean(offset[0, y_range[0]:y_range[1], x_range[0]:x_range[1]]) + modifier_y]))

                        else:

                            y_range = [kp[1]-1, kp[1]+1]

                            # If the point is on boundary, we need to modify the offset accordingly
                            if kp[1] - 1 == -1:
                                y_range[0] = 0
                                modifier_y = 0
                            if kp[0] + 1 == offset.shape[2]:
                                modifier_x = 0

                            adjusted_coord.append(
                                kp + np.array([np.mean(offset[1, y_range[0]:y_range[1], x_range[0]:x_range[1]]) + modifier_x,
                                               np.mean(offset[0, y_range[0]:y_range[1], x_range[0]:x_range[1]]) - modifier_y]))
                    else:

                        x_range = [kp[0]-1, kp[0]+1]

                        if offset[0, kp[1], kp[0]] > 0:

                            y_range = [kp[1], kp[1]+2]

                            # If the point is on boundary, we need to modify the offset accordingly
                            if kp[1] + 1 == offset.shape[1]:
                                modifier_y = 0
                            if kp[0] - 1 == -1:
                                x_range[0] = 0
                                modifier_x = 0

                            adjusted_coord.append(
                                kp + np.array([np.mean(offset[1, y_range[0]:y_range[1], x_range[0]:x_range[1]]) - modifier_x,
                                               np.mean(offset[0, y_range[0]:y_range[1], x_range[0]:x_range[1]]) + modifier_y]))

                        else:

                            y_range = [kp[1]-1, kp[1]+1]

                            # If the point is on boundary, we need to modify the offset accordingly
                            if kp[1] - 1 == -1:
                                y_range[0] = 0
                                modifier_y = 0
                            if kp[0] - 1 == -1:
                                x_range[0] = 0
                                modifier_x = 0

                            adjusted_coord.append(
                                kp + np.array([np.mean(offset[1, y_range[0]:y_range[1], x_range[0]:x_range[1]]) - modifier_x,
                                               np.mean(offset[0, y_range[0]:y_range[1], x_range[0]:x_range[1]]) - modifier_y]))
                else:
                    adjusted_coord.append(
                        kp + np.array([offset[1, kp[1], kp[0]], offset[0, kp[1], kp[0]]]))

            adjusted_coords.append(np.array(adjusted_coord))

        return adjusted_coords

    def evaluate(self, preds, gt_coords, apply_sigmoid=False, detailed=False):
        if apply_sigmoid:
            pred_heatmaps = torch.sigmoid(preds[:, 0])
        else:
            pred_heatmaps = preds[:, 0]
        pred_coords = self.find_local_peaks(pred_heatmaps)

        pred_coords = self.apply_local_offsets(
            pred_coords, preds[:, 1:], average_2x2=self.average_2x2)

        return_false_positive = detailed

        metrics = self.calculate_metrics(gt_coords, pred_coords,
                                         return_false_positive=return_false_positive)

        if detailed:
            return metrics
        else:
            return {'precision': metrics['precision'], 'recall': metrics['recall'], 'mae': metrics['mae']}


class DoublecheckKeypointEvaluator(OffsetKeypointEvaluator):

    def __init__(self,
                 max_detect: int = 100,
                 peak_threshold: float = 0.5,
                 peak_radius: int = 1,
                 local_distance_threshold: float = 1.5,
                 overlap_as_true_positive: bool = False,
                 average_2x2: bool = False,
                 doublecheck_normalization_factor: float = 8.0, # This corresponds to the distance between the head and middle keypoints
                 # When generating the dataset, we manually set the distance between the head and middle keypoints to 32 (orignal image)
                 # Thus, when the image shape is compressed 4 times by the CNN, the distance between the head and middle keypoints becomes 8
                 doublecheck_threshold: float = 1.5,
                 head_mid_distance_range: tuple = (0.8, 1.2)):
        """
        Args:
            max_detect (int): The maximum number of peaks to detect.
            peak_threshold (float): The minimum intensity of peaks.
            peak_radius (int): The radius of peaks.
            local_distance_threshold (float): The maximum distance between a predicted coordinate and
                a ground truth coordinate to be considered as a true positive.
            overlap_as_true_positive (bool): Whether to allow two preds to match the same gt.
            average_2x2 (bool): Whether to average the 2x2 grid of offsets.
            doublecheck_normalization_factor (float): The normalization factor for the doublecheck coordinates.
            doublecheck_threshold (float): The maximum distance between the doublecheck coordinates and
                the offset coordinates to pair head keypoints and middle keypoints.
            head_mid_distance_range (tuple): The range of the distance between the head and middle keypoints.
                If the distance is not within this range, the doublecheck coordinates will not be excluded.
        """
        super().__init__(max_detect, peak_threshold,
                         peak_radius, local_distance_threshold,
                         overlap_as_true_positive, average_2x2)
        self.doublecheck_normalization_factor = doublecheck_normalization_factor
        self.doublecheck_threshold = doublecheck_threshold
        self.head_mid_distance_range = head_mid_distance_range

    def add_prefix_to_dict(self, dictionary, prefix):
        return {prefix + key: value for key, value in dictionary.items()}

    def __repr__(self) -> str:
        return (f"DoublecheckKeypointEvaluator(max_detect={self.max_detect}, "
                f"peak_threshold={self.peak_threshold}, "
                f"peak_radius={self.peak_radius}, "
                f"local_distance_threshold={self.local_distance_threshold}, "
                f"overlap_as_true_positive={self.overlap_as_true_positive}, "
                f"average_2x2={self.average_2x2}, "
                f"doublecheck_normalization_factor={self.doublecheck_normalization_factor}, "
                f"doublecheck_threshold={self.doublecheck_threshold})")

    def get_pred_coords(self, preds, apply_sigmoid=False, return_confidence=False):
        if apply_sigmoid:
            head_pred_heatmaps = torch.sigmoid(preds[:, 0])
            middle_pred_heatmaps = torch.sigmoid(preds[:, 5])
        else:
            head_pred_heatmaps = preds[:, 0]
            middle_pred_heatmaps = preds[:, 5]

        if return_confidence:
            head_peak_pred_coords, head_confidences = self.find_local_peaks(
                head_pred_heatmaps, return_confidence=True)
            middle_peak_pred_coords, middle_confidences = self.find_local_peaks(
                middle_pred_heatmaps, return_confidence=True)
        else:
            head_peak_pred_coords = self.find_local_peaks(head_pred_heatmaps)
            middle_peak_pred_coords = self.find_local_peaks(
                middle_pred_heatmaps)
        
        if return_confidence:
            head_peak_pred_coords, head_confidences = self.exclude_head_mid_distance_range(
                head_peak_pred_coords, preds[:, 3:5], confidences=head_confidences)
            middle_peak_pred_coords, middle_confidences = self.exclude_head_mid_distance_range(
                middle_peak_pred_coords, preds[:, 8:10], confidences=middle_confidences)
        else:
            head_peak_pred_coords = self.exclude_head_mid_distance_range(head_peak_pred_coords, preds[:, 3:5])
            middle_peak_pred_coords = self.exclude_head_mid_distance_range(middle_peak_pred_coords, preds[:, 8:10])

        head_offset_pred_coords = self.apply_local_offsets(
            head_peak_pred_coords, preds[:, 1:3], average_2x2=self.average_2x2)
        middle_offset_pred_coords = self.apply_local_offsets(
            middle_peak_pred_coords, preds[:, 6:8], average_2x2=self.average_2x2)

        head_to_middle_doublecheck_coords = self.apply_local_offsets(
            head_peak_pred_coords, preds[:, 3:5]*self.doublecheck_normalization_factor, average_2x2=False)
        middle_to_head_doublecheck_coords = self.apply_local_offsets(
            middle_peak_pred_coords, preds[:, 8:10]*self.doublecheck_normalization_factor, average_2x2=False)

        if return_confidence:
            return (head_offset_pred_coords, middle_offset_pred_coords,
                    head_to_middle_doublecheck_coords, middle_to_head_doublecheck_coords,
                    head_confidences, middle_confidences)
        else:
            return head_offset_pred_coords, middle_offset_pred_coords, head_to_middle_doublecheck_coords, middle_to_head_doublecheck_coords
        
    def exclude_head_mid_distance_range(self, coords, offset_maps, confidences=None):
        """
        Args:
            coords (list): List of list of coordinates. Each element is a numpy array of shape (n, 2).
            offset_maps (torch.Tensor): Offset maps. Shape: (batch_size, 2, vertical_resolution, horizontal_resolution).
            confidences (list): List of confidence of the predicted coordinates. Each element is a numpy array of shape (n,).
        Returns:
            list: List of coordinates after exclusion.
        """
        if isinstance(offset_maps, torch.Tensor):
            offset_maps = offset_maps.detach().cpu().numpy()

        new_coords = []
        if confidences is not None:
            new_confidences = []
        
        if confidences is not None:
            for cds, offset_map, confs in zip(coords, offset_maps, confidences):
                new_cds = []
                new_confs = []
                for cd, conf in zip(cds, confs):
                    dist = np.linalg.norm([offset_map[0, cd[1], cd[0]], offset_map[1, cd[1], cd[0]]])
                    if self.head_mid_distance_range[0] < dist < self.head_mid_distance_range[1]:
                        new_cds.append(cd)
                        new_confs.append(conf)
                new_coords.append(np.array(new_cds))
                new_confidences.append(np.array(new_confs))
        else:
            for cds, offset_map in zip(coords, offset_maps):
                new_cds = []
                for cd in cds:
                    dist = np.linalg.norm([offset_map[0, cd[1], cd[0]], offset_map[1, cd[1], cd[0]]])
                    if self.head_mid_distance_range[0] < dist < self.head_mid_distance_range[1]:
                        new_cds.append(cd)
                new_coords.append(np.array(new_cds))
        
        if confidences is not None:
            return new_coords, new_confidences
        else:
            return new_coords

    def get_final_coords_from_pred_maps(self, preds, apply_sigmoid=False, process_log=False):

        dual_keypoints_offset_doublecheck_coords_and_confidences = self.get_pred_coords(
            preds, apply_sigmoid=apply_sigmoid, return_confidence=True)

        matched_coords = self.match_coords(
            *dual_keypoints_offset_doublecheck_coords_and_confidences)

        selected_coords = self.select_matched_coords(
            matched_coords, process_log=process_log)

        return selected_coords

    def match_coords(self,
                     head_coords: list,
                     middle_coords: list,
                     head_to_middle_doublecheck_coords: list,
                     middle_to_head_doublecheck_coords: list,
                     head_confidences: list,
                     middle_confidences: list,
                     overlap_match_confidence_method: str = 'unilateral',
                     unilateral_strategy: str = 'doublecheck'):
        """
        Args:
            head_coords (list): List of predicted head coordinates. Each element is a numpy array of shape (n, 2).
            middle_coords (list): List of predicted middle coordinates. Each element is a numpy array of shape (n, 2).
            head_to_middle_doublecheck_coords (list): List of predicted middle coordinates from the head coordinates. Each element is a numpy array of shape (n, 2).
            middle_to_head_doublecheck_coords (list): List of predicted head coordinates from the middle coordinates. Each element is a numpy array of shape (n, 2).
            head_confidences (list): List of confidence of the predicted head coordinates. Each element is a numpy array of shape (n,).
            middle_confidences (list): List of confidence of the predicted middle coordinates. Each element is a numpy array of shape (n,).
            overlap_match_confidence_method (str): The method to calculate the confidence of the overlap match. Default: ``'unilateral'``. Available options: ``'average'``, ``'unilateral'``.
            unilateral_strategy (str): The strategy to generate the final coordinates for unilateral cases. Default: ``'offset'``. Available options: ``'offset'``, ``'doublecheck'``.
        Returns:
            list: List of dictionaries. Each dictionary contains the following keys:
                'perfect_match': List of tuples. Tuples are in the format of (head_coord, middle_coord, confidence).
                'overlap_match': List of tuples. Tuples are in the same format as above.
                'unilateral': List of tuples. Tuples are in the same format as above.
                'isolated': List of tuples. Tuples are in the same format as above.
        """

        all_result = []

        for head_cds, middle_cds, head_to_middle_cds, middle_to_head_cds, head_confs, middle_confs in zip(head_coords, middle_coords,
                                                                                                          head_to_middle_doublecheck_coords, middle_to_head_doublecheck_coords,
                                                                                                          head_confidences, middle_confidences):

            # Create a mapping from head coordinates to middle coordinates and vice versa
            # The mapping is a list of integers, where the index of the list corresponds to the index of the head coordinate.
            # The value of the list is the index of the middle coordinate that is closest to the head coordinate.
            # If the value is -1, it means that there is no middle coordinate that is close to the head coordinate.
            # In this way, we create a one-way mapping from head coordinates to middle coordinates and vice versa.

            map_head_to_middle = []
            map_middle_to_head = []

            for doublecheck_middle_cd in head_to_middle_cds:
                if len(middle_cds) == 0:
                    map_head_to_middle.append(-1)
                else:
                    distances = np.linalg.norm(
                        middle_cds - doublecheck_middle_cd, axis=1)
                    closest_candidate = np.argmin(distances)
                    if distances[closest_candidate] < self.doublecheck_threshold:
                        map_head_to_middle.append(closest_candidate)
                    else:
                        map_head_to_middle.append(-1)  # -1 means no match

            for doublecheck_head_cd in middle_to_head_cds:
                if len(head_cds) == 0:
                    map_middle_to_head.append(-1)
                else:
                    distances = np.linalg.norm(
                        head_cds - doublecheck_head_cd, axis=1)
                    closest_candidate = np.argmin(distances)
                    if distances[closest_candidate] < self.doublecheck_threshold:
                        map_middle_to_head.append(closest_candidate)
                    else:
                        map_middle_to_head.append(-1)

            # There are totally '7' possible cases:
            # 1. Head -> Middle & Middle -> Head (Perfect match)
            # == Middle -> Head & Head -> Middle (Perfect match) *This is the same as case 1
            # 2. Head -> Middle & Middle -> Another Head (Overlap match)
            # 3. Middle -> Head & Head -> Another Middle (Overlap match)
            # 4. Head -> Middle & Middle -> Nothing (Unilateral)
            # 5. Middle -> Head & Head -> Nothing (Unilateral)
            # 6. Head -> Nothing & Nothing -> Head (Isolated)
            # 7. Middle -> Nothing & Nothing -> Middle (Isolated)
            # The 7 possible cases are divided into 4 categories:
            # 1. Perfect match
            # |  We use the offset coordinates of the head and middle keypoints as the final coordinates.
            # |  The confidence of the final coordinates is the average of the confidence of the head and middle keypoints.
            # 2. Overlap match (Case 2 as an example, vice versa for Case 3)
            # |  We use the offset coordinates of the head and the double-check middle coordinates
            # |  from the head (head_to_middle_doublecheck_coords) as the final coordinates.
            # |  The confidence of the final coordinates is the average of the confidence of the head and middle keypoints.
            # 3. Unilateral (Case 4 as an example, vice versa for Case 5)
            # |  There are two possible strategies to generate the final coordinates:
            # |  1. Use the offset coordinates of the head and the double-check middle coordinates
            # |     from the head (head_to_middle_doublecheck_coords).
            # |  2. Use the offset coordinates of the head and the offset coordinates of the middle.
            # |  We use the specified strategy as the final coordinates. The strategy is specified by the
            # |  argument 'unilateral_strategy'. Possible values are 'doublecheck', 'local_offset'.
            # |  The confidence is the confidence of the head keypoint for strategy 1 (doublecheck) and
            # |  the average of the confidence of the head and middle keypoints for strategy 2 (local_offset).
            # 4. Isolated (Case 6 as an example, vice versa for Case 7)
            # |  We use the offset coordinates of the head as the final coordinates.
            # |  The confidence of the final coordinates is the confidence of the head keypoint.

            perfect_match = []
            overlap_match = []
            unilateral = []
            isolated = []

            for head_index, middle_index in enumerate(map_head_to_middle):
                if middle_index == -1:
                    # Head -> Nothing
                    # The doublecheck at head coordinate perdicts a middle coordinate that is not in the middle coordinate list.
                    # We should decide if it is an isolated coordinate or a unilateral coordinate.
                    if head_index not in map_middle_to_head:
                        # Case 6
                        # Head -> Nothing & Nothing -> Head
                        isolated.append((head_cds[head_index],
                                         head_to_middle_cds[head_index],
                                         head_confs[head_index]))
                    else:
                        # Case 5
                        # Middle -> Head & Head -> Nothing
                        # Note that there may be multiple middle coordinates that are mapped to the same head coordinate.
                        middle_indexes = [i for i, x in enumerate(
                            map_middle_to_head) if x == head_index]
                        for middle_index in middle_indexes:
                            if unilateral_strategy == 'doublecheck':
                                unilateral.append((middle_to_head_cds[middle_index],
                                                   middle_cds[middle_index],
                                                   middle_confs[middle_index]))
                            elif unilateral_strategy == 'local_offset':
                                unilateral.append((head_cds[head_index],
                                                   middle_cds[middle_index],
                                                   (head_confs[head_index] + middle_confs[middle_index]) / 2))
                            else:
                                raise ValueError(
                                    f"Invalid Value for arg 'unilateral_strategy': '{unilateral_strategy}'\n"
                                    "Supported unilateral strategies: 'doublecheck', 'local_offset'")

                elif map_middle_to_head[middle_index] == head_index:
                    # Case 1
                    # Head -> Middle & Middle -> Head
                    # The doublecheck at head coordinate perdicts an existing middle coordinate and
                    # the corresponding doublecheck at the middle coordinate confirms the head coordinate.
                    perfect_match.append((head_cds[head_index],
                                          middle_cds[middle_index],
                                          (head_confs[head_index] + middle_confs[middle_index]) / 2))
                else:
                    # Head -> Middle
                    # The doublecheck at head coordinate perdicts an existing middle coordinate but
                    # the corresponding doublecheck at the middle coordinate does not confirm the head coordinate.
                    if map_middle_to_head[middle_index] != -1:
                        # Case 2
                        # Head -> Middle & Middle -> Another Head
                        # The doublecheck at middle coordinate perdicts another existing head coordinate
                        # This means that there may be an overlap at the middle coordinate, which means that
                        # there might be two head coordinates sharing the same middle coordinate.
                        if overlap_match_confidence_method == 'average':
                            overlap_match.append((head_cds[head_index],
                                                  head_to_middle_cds[head_index],
                                                  (head_confs[head_index] + middle_confs[middle_index]) / 2))
                        elif overlap_match_confidence_method == 'unilateral':
                            overlap_match.append((head_cds[head_index],
                                                  head_to_middle_cds[head_index],
                                                  head_confs[head_index]))
                        else:
                            raise ValueError(
                                f"Invalid Value for arg 'overlap_match_confidence_method': '{overlap_match_confidence_method}'\n"
                                "Supported overlap match confidence methods: 'average', 'unilateral'")
                    else:
                        # Case 4
                        # Head -> Middle & Middle -> Nothing
                        # The doublecheck at middle coordinate does not predict any head coordinate.
                        if unilateral_strategy == 'doublecheck':
                            unilateral.append((head_cds[head_index],
                                               head_to_middle_cds[head_index],
                                               head_confs[head_index]))
                        elif unilateral_strategy == 'local_offset':
                            unilateral.append((head_cds[head_index],
                                               middle_cds[middle_index],
                                               (head_confs[head_index] + middle_confs[middle_index]) / 2))
                        else:
                            raise ValueError(
                                f"Invalid Value for arg 'unilateral_strategy': '{unilateral_strategy}'\n"
                                "Supported unilateral strategies: 'doublecheck', 'local_offset'")
            for middle_index, head_index in enumerate(map_middle_to_head):
                if head_index == -1:
                    # Middle -> Nothing
                    if middle_index not in map_head_to_middle:
                        # Case 7
                        # Middle -> Nothing & Nothing -> Middle
                        isolated.append((middle_to_head_cds[middle_index],
                                         middle_cds[middle_index],
                                         middle_confs[middle_index]))
                elif map_head_to_middle[head_index] != middle_index and map_head_to_middle[head_index] != -1:
                    # Case 3
                    # Middle -> Head & Head -> Another Middle
                    if overlap_match_confidence_method == 'average':
                        overlap_match.append((middle_to_head_cds[middle_index],
                                              middle_cds[middle_index],
                                              (head_confs[head_index] + middle_confs[middle_index]) / 2))
                    elif overlap_match_confidence_method == 'unilateral':
                        overlap_match.append((middle_to_head_cds[middle_index],
                                              middle_cds[middle_index],
                                              middle_confs[middle_index]))
                    else:
                        raise ValueError(
                            f"Invalid Value for arg 'overlap_match_confidence_method': '{overlap_match_confidence_method}'\nSupported overlap match confidence methods: 'average', 'unilateral'")

            # Sort the results by confidence, in a descending order
            perfect_match.sort(key=lambda x: x[2], reverse=True)
            overlap_match.sort(key=lambda x: x[2], reverse=True)
            unilateral.sort(key=lambda x: x[2], reverse=True)
            isolated.sort(key=lambda x: x[2], reverse=True)

            all_result.append({'perfect_match': perfect_match,
                               'overlap_match': overlap_match,
                               'unilateral': unilateral,
                               'isolated': isolated})
        return all_result

    def select_matched_coords_no_strategy(self, matched_coords, process_log=False):

        selected_coords = []

        for matched_cds in matched_coords:
            selected_cds = []

            for key in ['perfect_match', 'overlap_match', 'unilateral', 'isolated']:
                for cd in matched_cds[key]:
                    if process_log:
                        selected_cds.append(cd+(key,))
                    else:
                        selected_cds.append(cd[:2])

            selected_coords.append(selected_cds)

        return selected_coords

    def get_dual_keypoints_offset_doublecheck_metrics(self,
                                                      head_offset_pred_coords: list,
                                                      middle_offset_pred_coords: list,
                                                      head_to_middle_doublecheck_coords: list,
                                                      middle_to_head_doublecheck_coords: list,
                                                      gt_coords: list,
                                                      names: list = [
                                                          'head', 'middle'],
                                                      detailed: bool = False):

        if isinstance(gt_coords[0][0], torch.Tensor):
            head_gt_coords = [i[0].cpu().numpy() for i in gt_coords]
            middle_gt_coords = [i[1].cpu().numpy() for i in gt_coords]
        else:
            head_gt_coords = [i[0] for i in gt_coords]
            middle_gt_coords = [i[1] for i in gt_coords]

        head_offset_metrics = self.add_prefix_to_dict(self.calculate_metrics(
            head_gt_coords, head_offset_pred_coords), names[0] + '_offset_')
        middle_offset_metrics = self.add_prefix_to_dict(self.calculate_metrics(
            middle_gt_coords, middle_offset_pred_coords), names[1] + '_offset_')

        head_doublecheck_metrics = self.add_prefix_to_dict(self.calculate_metrics(
            head_gt_coords, middle_to_head_doublecheck_coords), names[0] + '_doublecheck_')
        middle_doublecheck_metrics = self.add_prefix_to_dict(self.calculate_metrics(
            middle_gt_coords, head_to_middle_doublecheck_coords), names[1] + '_doublecheck_')

        result_dict = {**head_offset_metrics, **middle_offset_metrics,
                       **head_doublecheck_metrics, **middle_doublecheck_metrics}

        if detailed:
            return result_dict
        else:
            return {key: value
                    for key, value in result_dict.items()
                    if 'precision' in key or 'recall' in key or 'mae' in key}

    def select_matched_coords(self, matched_coords, process_log=False):
        return self.select_matched_coords_no_strategy(matched_coords, process_log=process_log)

    def evaluate(self, preds, gt_coords, apply_sigmoid=False, detailed=False, return_unmatched_metrics=False):

        dual_keypoints_offset_doublecheck_coords_and_confidences = self.get_pred_coords(
            preds, apply_sigmoid=apply_sigmoid, return_confidence=True)

        if return_unmatched_metrics:

            unmatched_metrics = self.get_dual_keypoints_offset_doublecheck_metrics(
                *dual_keypoints_offset_doublecheck_coords_and_confidences[:-2],
                gt_coords=gt_coords, detailed=detailed, names=['unmatched_head', 'unmatched_middle'])

        matched_coords = self.match_coords(
            *dual_keypoints_offset_doublecheck_coords_and_confidences)

        selected_coords = self.select_matched_coords(
            matched_coords, process_log=False)

        head_coords = [np.array([kp[0] for kp in cds])
                       for cds in selected_coords]
        middle_coords = [np.array([kp[1] for kp in cds])
                         for cds in selected_coords]

        if isinstance(gt_coords[0][0], torch.Tensor):
            head_gt_coords = [i[0].cpu().numpy() for i in gt_coords]
            middle_gt_coords = [i[1].cpu().numpy() for i in gt_coords]
        else:
            head_gt_coords = [i[0] for i in gt_coords]
            middle_gt_coords = [i[1] for i in gt_coords]

        return_false_positive = detailed

        head_metrics = self.add_prefix_to_dict(self.calculate_metrics(
            head_gt_coords, head_coords, return_false_positive), 'head_')
        middle_metrics = self.add_prefix_to_dict(self.calculate_metrics(
            middle_gt_coords, middle_coords, return_false_positive), 'middle_')

        if detailed == False:
            head_metrics = {key: value
                            for key, value in head_metrics.items()
                            if 'precision' in key or 'recall' in key or 'mae' in key}
            middle_metrics = {key: value
                              for key, value in middle_metrics.items()
                              if 'precision' in key or 'recall' in key or 'mae' in key}

        average_metrics = {'head_average': (head_metrics['head_precision'] + head_metrics['head_recall']) / 2,
                           'middle_average': (middle_metrics['middle_precision'] + middle_metrics['middle_recall']) / 2,
                           'total_average': (head_metrics['head_precision'] + head_metrics['head_recall'] + middle_metrics['middle_precision'] + middle_metrics['middle_recall']) / 4}

        if return_unmatched_metrics:
            return {**head_metrics, **middle_metrics, **unmatched_metrics, **average_metrics}
        else:
            return {**head_metrics, **middle_metrics, **average_metrics}
        
    def plot_tradeoff(self, predictions, gt_coords,
                      enumerate_peak_threshold_range=(5, 95)):
        """
        Plot the tradeoff between precision and recall for different peak and confidence thresholds.
        Args:
            predictions (torch.Tensor): Predictions of the model. Shape: (n, 10, height, width).
            gt_coords (list): List of ground truth coordinates. Each element is a list of numpy arrays of shape (n, 2).
            enumerate_peak_threshold_range (tuple): Tuple of two integers. The range of peak thresholds to be enumerated. Default: ``(5, 95)``.
                In actual evaluation, the peak threshold is set to ``peak_threshold / 100``.
        """

        previous_peak_threshold = self.peak_threshold
        metrics = {}

        for peak_threshold in tqdm(range(*enumerate_peak_threshold_range)):
            self.peak_threshold = peak_threshold / 100
            metrics[peak_threshold] = self.evaluate(predictions, gt_coords)

        head_recall = [metrics[i]['head_recall'] for i in metrics]
        head_precision = [metrics[i]['head_precision'] for i in metrics]
        middle_recall = [metrics[i]['middle_recall'] for i in metrics]
        middle_precision = [metrics[i]['middle_precision'] for i in metrics]
        total_average = [metrics[i]['total_average'] for i in metrics]

        plt.plot(list(metrics.keys()), head_recall, label="head_recall")
        plt.plot(list(metrics.keys()), head_precision, label="head_precision")
        plt.plot(list(metrics.keys()), middle_recall, label="middle_recall")
        plt.plot(list(metrics.keys()), middle_precision, label="middle_precision")
        plt.plot(list(metrics.keys()), total_average, label="total_average")

        max_index = total_average.index(max(total_average))
        max_position = list(metrics.keys())[max_index]
        max_peak_threshold = list(metrics.keys())[max_index] / 100

        plt.plot(max_position, total_average[max_index], label="max_average", marker='o')
        plt.annotate(f'({max_peak_threshold}, {format_percentage(total_average[max_index])})',
                        (max_position, total_average[max_index]))

        print(f'peak_threshold={max_peak_threshold}:')
        print(f"head_recall={format_percentage(head_recall[max_index])}, head_precision={format_percentage(head_precision[max_index])}")
        print(f"middle_recall={format_percentage(middle_recall[max_index])}, middle_precision={format_percentage(middle_precision[max_index])}")
        print(f"total_average={format_percentage(total_average[max_index])}")

        plt.legend()

        self.peak_threshold = previous_peak_threshold

        return plt


class CatagoryFirstDoublecheckKeypointEvaluator(DoublecheckKeypointEvaluator):
    """
    This evaluator is used to evaluate the performance of the model with
    offset regression and double-checking mechanism.
    The selection of the final coordinates is based on the catagory of the matching results.
    """

    def __init__(self,
                 max_detect: int = 100,
                 peak_threshold: float = 0.5,
                 peak_radius: int = 1,
                 local_distance_threshold: float = 1.5,
                 overlap_as_true_positive: bool = False,
                 average_2x2: bool = False,
                 doublecheck_normalization_factor: float = 8.0,
                 catagory_priority: float = ['perfect_match', 'overlap_match', 'unilateral', 'isolated'],
                 num_fish: int = 7,
                 doublecheck_threshold: float = 1.5,
                 head_mid_distance_range: tuple = (0.8, 1.2)):
        """
        Args:
            max_detect (int): The maximum number of peaks to detect.
            peak_threshold (float): The minimum intensity of peaks.
            peak_radius (int): The radius of peaks.
            local_distance_threshold (float): The maximum distance between a predicted coordinate and a ground truth coordinate to be considered as a true positive.
            overlap_as_true_positive (bool): Whether to allow two preds to match the same gt.
            average_2x2 (bool): Whether to average the 2x2 grid of offsets.
            doublecheck_normalization_factor (float): The normalization factor for the doublecheck coordinates.
            catagory_priority (list): List of catagories in a descending order. Default: ``['perfect_match', 'overlap_match', 'unilateral', 'isolated']``.
            num_fish (int): The number of fish to be detected.
            doublecheck_threshold (float): The threshold for the doublecheck coordinates.
        """
        super().__init__(max_detect, peak_threshold,
                         peak_radius, local_distance_threshold,
                         overlap_as_true_positive, average_2x2,
                         doublecheck_normalization_factor,
                         doublecheck_threshold,
                         head_mid_distance_range)

        self.num_fish = num_fish
        self.catagory_priority = catagory_priority

    def __repr__(self) -> str:
        return (f"NaiveDoublecheckKeypointEvaluator(max_detect={self.max_detect}, "
                f"peak_threshold={self.peak_threshold}, "
                f"peak_radius={self.peak_radius}, "
                f"local_distance_threshold={self.local_distance_threshold}, "
                f"overlap_as_true_positive={self.overlap_as_true_positive}, "
                f"average_2x2={self.average_2x2}, "
                f"doublecheck_normalization_factor={self.doublecheck_normalization_factor}, "
                f"catagory_priority={self.catagory_priority}, "
                f"num_fish={self.num_fish}, "
                f"doublecheck_threshold={self.doublecheck_threshold})")

    def select_matched_coords_catagory_first_strategy(self,
                                                      matched_coords: list,
                                                      num_fish: int,
                                                      catagory_priority: list,
                                                      process_log: bool = False):
        """
        Select the best coordinates from the matched coordinates. We use the following steps:

        1. Select the coordinates for catagory: perfect match.
        2. Select the coordinates for catagory: overlap match.
        3. Select the coordinates for catagory: unilateral.
        4. Select the coordinates for catagory: isolated.

        The priority of the catagories is specified by the argument 'catagory_priority' in a descending order.

        For each step, if the number of selected coordinates exceeds the number of fish, we stop.
        Since the coordinates are already sorted by confidence in a descending order,
        we can simply select the first 'num_fish' coordinates.

        Args:
            matched_coords (list): List of dictionaries. Each dictionary is expected to contain the following keys:
                'perfect_match': List of tuples. Tuples are in the format of (head_coord, middle_coord, confidence).
                'overlap_match': List of tuples. Tuples are in the same format as above.
                'unilateral': List of tuples. Tuples are in the same format as above.
                'isolated': List of tuples. Tuples are in the same format as above.
            num_fish (int): Number of fish to be detected.
            catagory_priority (list): List of catagories in a descending order. All keys in the dictionary should be in the list.
            process_log (bool): Whether to include the confidence value and match type in the result. Default: ``False``.
        Returns:
            list: List of coordinates. Each element is a list of tuples. Tuples are in the format of (head_coord, middle_coord).
                If 'process_log' is ``True``, the tuples are in the format of (head_coord, middle_coord, confidence, match_type).
                The coordinates are sorted by confidence in a descending order.
        """

        all_result = []

        for matched_cds in matched_coords:

            all_result.append([])

            remaining_fish = num_fish

            for catagory in catagory_priority:

                if process_log:
                    all_result[-1].extend(
                        [i+(catagory,) for i in matched_cds[catagory][:remaining_fish]])
                else:
                    # Get rid of the confidence value and make sure the lenth of the result is 'num_fish'
                    all_result[-1].extend(
                        [i[:2] for i in matched_cds[catagory][:remaining_fish]])

                num_matches = len(matched_cds[catagory])

                if num_matches >= remaining_fish:
                    break
                else:
                    remaining_fish -= num_matches

        return all_result

    def select_matched_coords(self, matched_coords, process_log=False):
        return self.select_matched_coords_catagory_first_strategy(matched_coords, num_fish=self.num_fish,
                                                                  catagory_priority=self.catagory_priority,
                                                                  process_log=process_log)
    
class ConfidenceFirstDoublecheckKeypointEvaluator(DoublecheckKeypointEvaluator):
    def __init__(self,
                 max_detect: int = 100,
                 peak_threshold: float = 0.5,
                 peak_radius: int = 1,
                 local_distance_threshold: float = 1.5,
                 overlap_as_true_positive: bool = False,
                 average_2x2: bool = False,
                 doublecheck_normalization_factor: float = 8.0,
                 num_fish: int = 7,
                 doublecheck_threshold: float = 1.5,
                 head_mid_distance_range: tuple = (0.8, 1.2)):
        """
        Args:
            max_detect (int): The maximum number of peaks to detect.
            peak_threshold (float): The minimum intensity of peaks.
            peak_radius (int): The radius of peaks.
            local_distance_threshold (float): The maximum distance between a predicted coordinate and a ground truth coordinate to be considered as a true positive.
            overlap_as_true_positive (bool): Whether to allow two preds to match the same gt.
            average_2x2 (bool): Whether to average the 2x2 grid of offsets.
            doublecheck_normalization_factor (float): The normalization factor for the doublecheck coordinates.
            num_fish (int): The number of fish to be detected.
            doublecheck_threshold (float): The threshold for the doublecheck coordinates.
        """
        super().__init__(max_detect, peak_threshold,
                         peak_radius, local_distance_threshold,
                         overlap_as_true_positive, average_2x2,
                         doublecheck_normalization_factor,
                         doublecheck_threshold,
                         head_mid_distance_range)
        self.num_fish = num_fish

    def select_matched_coords(self, matched_coords, process_log=False):
        return self.select_matched_coords_confidence_first_strategy(
            matched_coords, num_fish=self.num_fish, process_log=process_log)
    
    def select_matched_coords_confidence_first_strategy(self,
                                                      matched_coords: list,
                                                      num_fish: int,
                                                      process_log: bool = False):
        """
        Select the best coordinates from the matched coordinates. We select the coordinates
        only based on the confidence value. The coordinates are sorted by confidence in a descending order.
        Args:
            matched_coords (list): List of dictionaries. Each dictionary is expected to contain the following keys:
                'perfect_match': List of tuples. Tuples are in the format of (head_coord, middle_coord, confidence).
                'overlap_match': List of tuples. Tuples are in the same format as above.
                'unilateral': List of tuples. Tuples are in the same format as above.
                'isolated': List of tuples. Tuples are in the same format as above.
            num_fish (int): Number of fish to be detected.
            process_log (bool): Whether to include the confidence value and match type in the result. Default: ``False``.
        Returns:
            list: List of coordinates. Each element is a list of tuples.
                Tuples are in the format of (head_coord, middle_coord), each representing a predicted coordinate.
                If 'process_log' is ``True``, the tuples are in the format of (head_coord, middle_coord, confidence, match_type).
                The coordinates are sorted by confidence in a descending order.
        """

        all_result = []

        for matched_cds in matched_coords:

            all_result.append([])
            if process_log:
                matched_cds = [item+(key, ) for key, sublist in matched_cds.items()
                            for item in sublist]
            else:
                matched_cds = [item for sublist in matched_cds.values()
                            for item in sublist]

            sorted_matched_cds = sorted(matched_cds, key=lambda x: x[2], reverse=True)

            if process_log:
                all_result[-1].extend(sorted_matched_cds[:num_fish])
            else:
                all_result[-1].extend([i[:2] for i in sorted_matched_cds[:num_fish]])

        return all_result
    
class WeightedConfidenceDoublecheckKeypointEvaluator(DoublecheckKeypointEvaluator):
    def __init__(self,
                 max_detect: int = 100,
                 peak_threshold: float = 0.5,
                 peak_radius: int = 1,
                 local_distance_threshold: float = 1.5,
                 overlap_as_true_positive: bool = False,
                 average_2x2: bool = False,
                 doublecheck_normalization_factor: float = 8.0,
                 num_fish: int = 7,
                 confidence_weight: dict = {'perfect_match': 1.0, 'overlap_match': 1.0, 'unilateral': 1.0, 'isolated': 1.0},
                 doublecheck_threshold: float = 1.5,
                 head_mid_distance_range: tuple = (0.8, 1.2)):
        """
        Args:
            max_detect (int): The maximum number of peaks to detect.
            peak_threshold (float): The minimum intensity of peaks.
            peak_radius (int): The radius of peaks.
            local_distance_threshold (float): The maximum distance between a predicted coordinate and a ground truth coordinate to be considered as a true positive.
            overlap_as_true_positive (bool): Whether to allow two preds to match the same gt.
            average_2x2 (bool): Whether to average the 2x2 grid of offsets.
            doublecheck_normalization_factor (float): The normalization factor for the doublecheck coordinates.
            num_fish (int): The number of fish to be detected.
            confidence_weight (dict): The weight for each catagory. Default: ``{'perfect_match': 1.0, 'overlap_match': 1.0, 'unilateral': 1.0, 'isolated': 1.0}``.
            doublecheck_threshold (float): The threshold for the doublecheck coordinates.
        """
        super().__init__(max_detect, peak_threshold,
                         peak_radius, local_distance_threshold,
                         overlap_as_true_positive, average_2x2,
                         doublecheck_normalization_factor,
                         doublecheck_threshold,
                         head_mid_distance_range)
        self.num_fish = num_fish
        self.confidence_weight = confidence_weight

    def select_matched_coords(self, matched_coords, process_log=False):
        return self.select_matched_coords_weighted_confidence_strategy(
            matched_coords, num_fish=self.num_fish, 
            confidence_weight=self.confidence_weight, process_log=process_log)
    
    def select_matched_coords_weighted_confidence_strategy(self,
                                                      matched_coords: list,
                                                      num_fish: int,
                                                        confidence_weight: dict,
                                                        process_log: bool = False):
        """
        Select the best coordinates from the matched coordinates. We select the coordinates
        based on weighted confidence value. It is calculated by the confidence value multiplied by the weight.
        The weight for each catagory is specified by the argument 'confidence_weight'.
        The coordinates are sorted by weighted confidence in a descending order.
        Args:
            matched_coords (list): List of dictionaries. Each dictionary is expected to contain the following keys:
                'perfect_match': List of tuples. Tuples are in the format of (head_coord, middle_coord, confidence).
                'overlap_match': List of tuples. Tuples are in the same format as above.
                'unilateral': List of tuples. Tuples are in the same format as above.
                'isolated': List of tuples. Tuples are in the same format as above.
            num_fish (int): Number of fish to be detected.
            confidence_weight (dict): The weight for each catagory. All keys in the dictionary should be in the list.
            process_log (bool): Whether to include the confidence value and match type in the result. Default: ``False``.
        Returns:
            list: List of coordinates. Each element is a list of tuples.
                Tuples are in the format of (head_coord, middle_coord), each representing a predicted coordinate.
                If 'process_log' is ``True``, the tuples are in the format of (head_coord, middle_coord, confidence, match_type).
                The coordinates are sorted by weighted confidence in a descending order.
        """

        all_result = []

        for matched_cds in matched_coords:
                
            all_result.append([])
            matched_cds = [item+(key, ) for key, sublist in matched_cds.items()
                        for item in sublist]
            
            sorted_matched_cds = sorted(matched_cds, key=lambda x: x[2]*confidence_weight[x[3]], reverse=True)

            if process_log:
                all_result[-1].extend(sorted_matched_cds[:num_fish])
            else:
                all_result[-1].extend([i[:2] for i in sorted_matched_cds[:num_fish]])

        return all_result
    
class TripleKeypointsOffsetClassifyEvaluator(DoublecheckKeypointEvaluator):
    def __init__(self,
                 max_detect: int = 100,
                 peak_threshold: float = 0.5,
                 peak_radius: int = 1,
                 local_distance_threshold: float = 1.5,
                 overlap_as_true_positive: bool = False,
                 average_2x2: bool = False,
                 doublecheck_normalization_factor: dict = {('head', 'midsec'): 8.0,
                                                           ('head', 'tail'): 26.61078956931868,
                                                           ('midsec', 'tail'): 16.879570128434796},
                 num_fish: int = 7,
                 doublecheck_threshold: float = 1.5,
                 head_mid_distance_range: tuple = (0.8, 1.2)):
        """
        Args:
            max_detect (int): The maximum number of peaks to detect.
            peak_threshold (float): The minimum intensity of peaks.
            peak_radius (int): The radius of peaks.
            local_distance_threshold (float): The maximum distance between a predicted coordinate and a ground truth coordinate to be considered as a true positive.
            overlap_as_true_positive (bool): Whether to allow two preds to match the same gt.
            average_2x2 (bool): Whether to average the 2x2 grid of offsets.
            doublecheck_normalization_factor (float): The normalization factor for the doublecheck coordinates.
            num_fish (int): The number of fish to be detected.
            doublecheck_threshold (float): The threshold for the doublecheck coordinates.
        """
        super().__init__(max_detect, peak_threshold,
                         peak_radius, local_distance_threshold,
                         overlap_as_true_positive, average_2x2,
                         doublecheck_normalization_factor,
                         doublecheck_threshold,
                         head_mid_distance_range)
        self.num_fish = num_fish

    def get_pred_classifications(self, coords, classification_maps):
        pred_classifications = [[classification_map[:, cd[1], cd[0]] for cd in coord]
                                for coord, classification_map in zip(coords, classification_maps)]
        return pred_classifications

    def get_pred_coords(self, preds, apply_sigmoid=False, return_confidence=False):

        if apply_sigmoid:
            head_pred_heatmaps = torch.sigmoid(preds[:, 0])
            midsec_pred_heatmaps = torch.sigmoid(preds[:, 9])
            tail_pred_heatmaps = torch.sigmoid(preds[:, 18])
            head_pred_classifications = torch.sigmoid(preds[:, 7:9])
            midsec_pred_classifications = torch.sigmoid(preds[:, 16:18])
            tail_pred_classifications = torch.sigmoid(preds[:, 25:27])
        else:
            head_pred_heatmaps = preds[:, 0]
            midsec_pred_heatmaps = preds[:, 9]
            tail_pred_heatmaps = preds[:, 18]
            head_pred_classifications = preds[:, 7:9]
            midsec_pred_classifications = preds[:, 16:18]
            tail_pred_classifications = preds[:, 25:27]

        if return_confidence:
            head_peak_pred_coords, head_confidences = self.find_local_peaks(
                head_pred_heatmaps, return_confidence=True)
            midsec_peak_pred_coords, midsec_confidences = self.find_local_peaks(
                midsec_pred_heatmaps, return_confidence=True)
            tail_peak_pred_coords, tail_confidences = self.find_local_peaks(
                tail_pred_heatmaps, return_confidence=True)
        else:
            head_peak_pred_coords = self.find_local_peaks(head_pred_heatmaps)
            midsec_peak_pred_coords = self.find_local_peaks(midsec_pred_heatmaps)
            tail_peak_pred_coords = self.find_local_peaks(tail_pred_heatmaps)
        
        if return_confidence:
            head_peak_pred_coords, head_confidences = self.exclude_head_mid_distance_range(
                head_peak_pred_coords, preds[:, 3:5], head_confidences)
            midsec_peak_pred_coords, midsec_confidences = self.exclude_head_mid_distance_range(
                midsec_peak_pred_coords, preds[:, 12:14], midsec_confidences)
        else:
            head_peak_pred_coords = self.exclude_head_mid_distance_range(
                head_peak_pred_coords, preds[:, 3:5])
            midsec_peak_pred_coords = self.exclude_head_mid_distance_range(
                midsec_peak_pred_coords, preds[:, 12:14])
        
        head_pred_classifications = self.get_pred_classifications(
            head_peak_pred_coords, head_pred_classifications)
        midsec_pred_classifications = self.get_pred_classifications(
            midsec_peak_pred_coords, midsec_pred_classifications)
        tail_pred_classifications = self.get_pred_classifications(
            tail_peak_pred_coords, tail_pred_classifications)

        head_offset_pred_coords = self.apply_local_offsets(
            head_peak_pred_coords, preds[:, 1:3], average_2x2=self.average_2x2)
        head_doublechecked_from_midsec_pred_coords = self.apply_local_offsets(
            midsec_peak_pred_coords,
            preds[:, 12:14]*self.doublecheck_normalization_factor[('head', 'midsec')],
            average_2x2=self.average_2x2)
        head_doublechecked_from_tail_pred_coords = self.apply_local_offsets(
            tail_peak_pred_coords,
            preds[:, 21:23]*self.doublecheck_normalization_factor[('head', 'tail')],
            average_2x2=self.average_2x2)
        
        midsec_offset_pred_coords = self.apply_local_offsets(
            midsec_peak_pred_coords, preds[:, 10:12], average_2x2=self.average_2x2)
        midsec_doublechecked_from_head_pred_coords = self.apply_local_offsets(
            head_peak_pred_coords,
            preds[:, 3:5]*self.doublecheck_normalization_factor[('head', 'midsec')],
            average_2x2=self.average_2x2)
        midsec_doublechecked_from_tail_pred_coords = self.apply_local_offsets(
            tail_peak_pred_coords,
            preds[:, 23:25]*self.doublecheck_normalization_factor[('midsec', 'tail')],
            average_2x2=self.average_2x2)
        
        tail_offset_pred_coords = self.apply_local_offsets(
            tail_peak_pred_coords, preds[:, 19:21], average_2x2=self.average_2x2)
        tail_doublechecked_from_head_pred_coords = self.apply_local_offsets(
            head_peak_pred_coords,
            preds[:, 5:7]*self.doublecheck_normalization_factor[('head', 'tail')],
            average_2x2=self.average_2x2)
        tail_doublechecked_from_midsec_pred_coords = self.apply_local_offsets(
            midsec_peak_pred_coords,
            preds[:, 14:16]*self.doublecheck_normalization_factor[('midsec', 'tail')],
            average_2x2=self.average_2x2)
        
        if return_confidence:
            return ((head_offset_pred_coords, head_doublechecked_from_midsec_pred_coords,
                     head_doublechecked_from_tail_pred_coords, head_pred_classifications, head_confidences),
                    (midsec_offset_pred_coords, midsec_doublechecked_from_head_pred_coords,
                     midsec_doublechecked_from_tail_pred_coords, midsec_pred_classifications, midsec_confidences),
                    (tail_offset_pred_coords, tail_doublechecked_from_head_pred_coords,
                        tail_doublechecked_from_midsec_pred_coords, tail_pred_classifications, tail_confidences))
        else:
            return ((head_offset_pred_coords, head_doublechecked_from_midsec_pred_coords,
                     head_doublechecked_from_tail_pred_coords, head_pred_classifications),
                    (midsec_offset_pred_coords, midsec_doublechecked_from_head_pred_coords,
                     midsec_doublechecked_from_tail_pred_coords, midsec_pred_classifications),
                    (tail_offset_pred_coords, tail_doublechecked_from_head_pred_coords,
                        tail_doublechecked_from_midsec_pred_coords, tail_pred_classifications))
    
    def get_triple_keypoints_offset_doublecheck_metrics(self,
                                                        offset_pred_coords: list,
                                                        doublecheck_a_coords: list,
                                                        doublecheck_b_coords: list,
                                                        gt_coords: list,
                                                        pred_classifications: list,
                                                        gt_classifications: list,
                                                        names: list = ['head', 'midsec', 'tail'],
                                                        detailed: bool = False):
        if isinstance(gt_coords[0], torch.Tensor):
            gt_coords = [i.cpu().numpy() for i in gt_coords]
        
        offset_metrics = self.add_prefix_to_dict(self.calculate_metrics_with_classification(
            gt_coords, offset_pred_coords, gt_classifications, pred_classifications), names[0] + '_offset_')
        doublecheck_a_metrics = self.add_prefix_to_dict(self.calculate_metrics(
            gt_coords, doublecheck_a_coords), names[0] + '_from_' + names[1] + '_doublecheck_')
        doublecheck_b_coords = self.add_prefix_to_dict(self.calculate_metrics(
            gt_coords, doublecheck_b_coords), names[0] + '_from_' + names[2] + '_doublecheck_')
        
        result_dict = {**offset_metrics, **doublecheck_a_metrics, **doublecheck_b_coords}

        if detailed:
            return result_dict
        else:
            return {key: value
                    for key, value in result_dict.items()
                    if 'precision' in key or 'recall' in key or 'mae' in key or 'f1' in key}
        
    def calculate_classification_metrics(self, gts, preds):
        # Filter out -1 values in gts
        # At the same time, filter out the corresponding predictions
        gts = np.array(gts)
        preds = np.array(preds)
        preds = preds[gts != -1]
        gts = gts[gts != -1]

        # Calculate precision and recall for various thresholds
        precision, recall, thresholds = precision_recall_curve(gts, preds)
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall)
        
        # Replace any NaN F1 scores with 0
        f1_scores = np.nan_to_num(f1_scores)
        
        # Find the index of the maximum F1 score
        max_f1_index = np.argmax(f1_scores)
        best_precision = precision[max_f1_index]
        best_recall = recall[max_f1_index]
        
        # Return a dictionary with the precision and recall at the best F1 score
        performance_metrics = {
            'precision': best_precision,
            'recall': best_recall,
            'f1': f1_scores[max_f1_index]
        }
        
        return performance_metrics
        
    def calculate_metrics_with_classification(self,
                                              gt_coords: list,
                                              pred_coords: list,
                                              gt_classifications: list,
                                              pred_classifications: list,
                                              return_false_positive: bool = False):
        return self._calculate_metrics_with_classification(
            gt_coords, pred_coords,
            gt_classifications, pred_classifications,
            self.local_distance_threshold, return_false_positive)
        
    def _calculate_metrics_with_classification(self,
                                               gt_coords: list,
                                               pred_coords: list,
                                               gt_classifications: list,
                                               pred_classifications: list,
                                               local_distance_threshold: float,
                                               return_false_positive: bool):
        """
        Calculate the precision, recall, mean average error (MAE) and  of the predictions.
        Args:
            gt_coords (list): List of numpy arrays of shape (n, 2). Each array represents the ground truth coordinates.
            pred_coords (list): List of numpy arrays of shape (m, 2). Each array represents the predicted coordinates.
            gt_classifications (list): List of numpy arrays of shape (n, 2). Each array represents the ground truth classifications.
            pred_classifications (list): List of numpy arrays of shape (m, 2). Each array represents the predicted classifications.
        Returns:
            dict: Dictionary containing the precision, recall, and mean average error (MAE) of the predictions.
        """
        tp = 0
        fp = 0
        fn = 0
        mae = 0

        n_gts = sum(len(gt) for gt in gt_coords)
        n_preds = sum(len(pred) for pred in pred_coords)

        classification_gts = []
        classification_preds = []

        if return_false_positive:
            fp_list = []
        
        for gt_cds, pred_cds, gt_classes, pred_classes in zip(
            gt_coords, pred_coords, gt_classifications, pred_classifications):
            # Use Hungarian algorithm to find the optimal matching
            weights = []
            for gt in gt_cds:
                weight = []
                for pred in pred_cds:
                    distances = np.linalg.norm(pred - gt)
                    weight.append(distances)
                weights.append(weight)
            weights = np.array(weights)
            gt_indices, pred_indices = linear_sum_assignment(weights)
            if return_false_positive:
                fp_list.append([1] * len(pred_cds))
            for gt_i, pred_i in zip(gt_indices, pred_indices):
                if weights[gt_i, pred_i] < local_distance_threshold:
                    tp += 1
                    mae += weights[gt_i, pred_i]
                    classification_gts.append(gt_classes[gt_i])
                    classification_preds.append(pred_classes[pred_i])
                    if return_false_positive:
                        fp_list[-1][pred_i] = 0
                else:
                    fn += 1

        fp = n_preds - tp

        if n_preds == 0 or n_gts == 0:
            precision = False
            recall = False
            mae = False
        else:
            if tp == 0:
                precision = 0
                recall = 0
                mae = 0
            else:
                precision = tp / n_preds
                recall = tp / n_gts
                mae /= tp

        if len(classification_gts) > 0:
            if isinstance(classification_gts[0], torch.Tensor):
                classification_gts = [i.cpu().numpy() for i in classification_gts]
            if isinstance(classification_preds[0], torch.Tensor):
                classification_preds = [i.cpu().numpy() for i in classification_preds]

            covering_classification_metrics = self.add_prefix_to_dict(
                self.calculate_classification_metrics(
                    [i[0] for i in classification_gts],
                    [i[0] for i in classification_preds]),
                'covering_')
            covered_classification_metrics = self.add_prefix_to_dict(
                self.calculate_classification_metrics(
                    [i[1] for i in classification_gts],
                    [i[1] for i in classification_preds]),
                'covered_')
        else:
            covering_classification_metrics = {'covering_precision': 0, 'covering_recall': 0, 'covering_f1': 0}
            covered_classification_metrics = {'covered_precision': 0, 'covered_recall': 0, 'covered_f1': 0}

        result = {
            'precision': precision,
            'recall': recall,
            'mae': mae,
            'true_positive': tp,
            'false_positive': fp,
            'false_negative': fn,
            'n_predictions': n_preds,
            'n_ground_truths': n_gts,
            **covering_classification_metrics,
            **covered_classification_metrics
        }

        if return_false_positive:
            result['false_positive_list'] = fp_list
            
        return result
    
    def calculate_f1_score(self, precision, recall):
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    def evaluate(self, preds, gt_coords, apply_sigmoid=False, detailed=False):
        
        pred_coords = self.get_pred_coords(preds, apply_sigmoid=apply_sigmoid, return_confidence=True)

        if isinstance(gt_coords[0], torch.Tensor):
            gt_coords = [i.cpu().numpy() for i in gt_coords]
        
        head_metrics = self.get_triple_keypoints_offset_doublecheck_metrics(
            pred_coords[0][0], pred_coords[0][1], pred_coords[0][2],
            [gt_cds[:self.num_fish] for gt_cds in gt_coords],
            pred_coords[0][3], [gt_cds[-self.num_fish:] for gt_cds in gt_coords],
            names=['head', 'midsec', 'tail'], detailed=detailed)
        midsec_metrics = self.get_triple_keypoints_offset_doublecheck_metrics(
            pred_coords[1][0], pred_coords[1][1], pred_coords[1][2],
            [gt_cds[self.num_fish : self.num_fish * 2] for gt_cds in gt_coords],
            pred_coords[1][3], [gt_cds[-self.num_fish:] for gt_cds in gt_coords],
            names=['midsec', 'head', 'tail'], detailed=detailed)
        tail_metrics = self.get_triple_keypoints_offset_doublecheck_metrics(
            pred_coords[2][0], pred_coords[2][1], pred_coords[2][2],
            [gt_cds[self.num_fish * 2 : self.num_fish * 3] for gt_cds in gt_coords],
            pred_coords[2][3], [gt_cds[-self.num_fish:] for gt_cds in gt_coords],
            names=['tail', 'head', 'midsec'], detailed=detailed)
        
        f1_metrics = {
            'head_f1': self.calculate_f1_score(head_metrics['head_offset_precision'], head_metrics['head_offset_recall']),
            'midsec_f1': self.calculate_f1_score(midsec_metrics['midsec_offset_precision'], midsec_metrics['midsec_offset_recall']),
            'tail_f1': self.calculate_f1_score(tail_metrics['tail_offset_precision'], tail_metrics['tail_offset_recall']),
            'total_f1': (self.calculate_f1_score(head_metrics['head_offset_precision'], head_metrics['head_offset_recall']) +
                            self.calculate_f1_score(midsec_metrics['midsec_offset_precision'], midsec_metrics['midsec_offset_recall']) +
                            self.calculate_f1_score(tail_metrics['tail_offset_precision'], tail_metrics['tail_offset_recall'])) / 3
                            }
        
        classification_average = {'total_covering_f1': (head_metrics['head_offset_covering_f1'] +
                                                        midsec_metrics['midsec_offset_covering_f1'] +
                                                        tail_metrics['tail_offset_covering_f1']) / 3,
                                'total_covered_f1': (head_metrics['head_offset_covered_f1'] +
                                                        midsec_metrics['midsec_offset_covered_f1'] +
                                                        tail_metrics['tail_offset_covered_f1']) / 3}
        
        return {**head_metrics, **midsec_metrics, **tail_metrics, **f1_metrics, **classification_average}
    
class TripleKeypointsOffsetEvaluator(TripleKeypointsOffsetClassifyEvaluator):

    def __init__(self,
                 max_detect: int = 7,
                 peak_threshold: float = 0.25,
                 peak_radius: int = 1,
                 local_distance_threshold: float = 2,
                 overlap_as_true_positive: bool = False,
                 average_2x2: bool = False,
                 doublecheck_normalization_factor: dict = {
                    ('head', 'midsec'): 8,
                    ('head', 'tail'): 26.620351900862914,
                    ('midsec', 'tail'): 16.89075517685661}, # Dataset V1.0
                 num_fish: int = 7,
                 doublecheck_threshold: float = 4,
                 head_mid_distance_range: tuple = (0.5, 1.5),
                 tail_cluster: bool = True):
        super().__init__(max_detect, peak_threshold, peak_radius, local_distance_threshold, overlap_as_true_positive, average_2x2, doublecheck_normalization_factor, num_fish, doublecheck_threshold, head_mid_distance_range)
        self.tail_cluster = tail_cluster
        if tail_cluster and max_detect > num_fish:
            print('Warning: tail_cluster is set to True. We recommend setting max_detect equal to num_fish to avoid false positive detections.')

    def cluster_pred_coords(self, pred_coords, confidences=None):
        """
        ### IMPORTANT: This function assumes that the coordinates are sorted by confidence in a descending order. ###
        Cluster the predicted coordinates into max_detect groups.
        """
        if confidences is not None:
            clustered_coords = []
            clustered_confidences = []
            for coords, confs in zip(pred_coords, confidences):
                if len(coords) < self.max_detect:
                    clustered_coords.append(np.array(coords))
                    clustered_confidences.append(np.array(confs))
                    continue
                ac = AgglomerativeClustering(n_clusters=self.max_detect,
                                             metric='euclidean',
                                             linkage='ward')
                labels = ac.fit_predict(coords)
                seen_labels = []
                cds = []
                cfs = []
                for i, label in enumerate(labels):
                    if label not in seen_labels:
                        seen_labels.append(label)
                        cds.append(coords[i])
                        cfs.append(confs[i])
                clustered_coords.append(np.array(cds))
                clustered_confidences.append(np.array(cfs))
            return clustered_coords, clustered_confidences
        else:
            clustered_coords = []
            for coords in pred_coords:
                ac = AgglomerativeClustering(n_clusters=self.max_detect,
                                             affinity='euclidean',
                                             linkage='ward')
                labels = ac.fit_predict(coords)
                seen_labels = []
                cds = []
                for i, label in enumerate(labels):
                    if label not in seen_labels:
                        seen_labels.append(label)
                        cds.append(coords[i])
                clustered_coords.append(np.array(cds))
            return clustered_coords

    def get_pred_coords(self, preds, apply_sigmoid=False, return_confidence=False):

        if apply_sigmoid:
            head_pred_heatmaps = torch.sigmoid(preds[:, 0])
            midsec_pred_heatmaps = torch.sigmoid(preds[:, 7])
            tail_pred_heatmaps = torch.sigmoid(preds[:, 14])
        else:
            head_pred_heatmaps = preds[:, 0]
            midsec_pred_heatmaps = preds[:, 7]
            tail_pred_heatmaps = preds[:, 14]

        if return_confidence:
            head_peak_pred_coords, head_confidences = self.find_local_peaks(
                head_pred_heatmaps, return_confidence=True)
            midsec_peak_pred_coords, midsec_confidences = self.find_local_peaks(
                midsec_pred_heatmaps, return_confidence=True)
            if self.tail_cluster:
                tail_peak_pred_coords, tail_confidences = self._find_local_peaks(
                    tail_pred_heatmaps,
                    peak_threshold=self.peak_threshold, 
                    peak_radius=self.peak_radius,
                    max_detect=self.num_fish * 2,
                    return_confidence=True)
                tail_peak_pred_coords, tail_confidences = self.cluster_pred_coords(
                    tail_peak_pred_coords, tail_confidences)
            else:
                tail_peak_pred_coords, tail_confidences = self.find_local_peaks(
                    tail_pred_heatmaps, return_confidence=True)
        else:
            head_peak_pred_coords = self.find_local_peaks(head_pred_heatmaps)
            midsec_peak_pred_coords = self.find_local_peaks(midsec_pred_heatmaps)
            if self.tail_cluster:
                tail_peak_pred_coords = self._find_local_peaks(
                    tail_pred_heatmaps,
                    peak_threshold=self.peak_threshold, 
                    peak_radius=self.peak_radius,
                    max_detect=self.num_fish * 2,
                    return_confidence=False)
                tail_peak_pred_coords = self.cluster_pred_coords(tail_peak_pred_coords)
            else:
                tail_peak_pred_coords = self.find_local_peaks(tail_pred_heatmaps)

        if return_confidence:
            head_peak_pred_coords, head_confidences = self.exclude_head_mid_distance_range(
                head_peak_pred_coords, preds[:, 3:5], head_confidences)
            midsec_peak_pred_coords, midsec_confidences = self.exclude_head_mid_distance_range(
                midsec_peak_pred_coords, preds[:, 10:12], midsec_confidences)
        else:
            head_peak_pred_coords = self.exclude_head_mid_distance_range(
                head_peak_pred_coords, preds[:, 3:5])
            midsec_peak_pred_coords = self.exclude_head_mid_distance_range(
                midsec_peak_pred_coords, preds[:, 10:12])

        head_offset_pred_coords = self.apply_local_offsets(
            head_peak_pred_coords, preds[:, 1:3], average_2x2=self.average_2x2)
        head_doublechecked_from_midsec_pred_coords = self.apply_local_offsets(
            midsec_peak_pred_coords,
            preds[:, 10:12]*self.doublecheck_normalization_factor[('head', 'midsec')],
            average_2x2=self.average_2x2)
        head_doublechecked_from_tail_pred_coords = self.apply_local_offsets(
            tail_peak_pred_coords,
            preds[:, 17:19]*self.doublecheck_normalization_factor[('head', 'tail')],
            average_2x2=self.average_2x2)
        
        midsec_offset_pred_coords = self.apply_local_offsets(
            midsec_peak_pred_coords, preds[:, 8:10], average_2x2=self.average_2x2)
        midsec_doublechecked_from_head_pred_coords = self.apply_local_offsets(
            head_peak_pred_coords,
            preds[:, 3:5]*self.doublecheck_normalization_factor[('head', 'midsec')],
            average_2x2=self.average_2x2)
        midsec_doublechecked_from_tail_pred_coords = self.apply_local_offsets(
            tail_peak_pred_coords,
            preds[:, 19:21]*self.doublecheck_normalization_factor[('midsec', 'tail')],
            average_2x2=self.average_2x2)
        
        tail_offset_pred_coords = self.apply_local_offsets(
            tail_peak_pred_coords, preds[:, 15:17], average_2x2=self.average_2x2)
        tail_doublechecked_from_head_pred_coords = self.apply_local_offsets(
            head_peak_pred_coords,
            preds[:, 5:7]*self.doublecheck_normalization_factor[('head', 'tail')],
            average_2x2=self.average_2x2)
        tail_doublechecked_from_midsec_pred_coords = self.apply_local_offsets(
            midsec_peak_pred_coords,
            preds[:, 12:14]*self.doublecheck_normalization_factor[('midsec', 'tail')],
            average_2x2=self.average_2x2)
        
        if return_confidence:
            return ((head_offset_pred_coords, head_doublechecked_from_midsec_pred_coords,
                     head_doublechecked_from_tail_pred_coords, head_confidences),
                    (midsec_offset_pred_coords, midsec_doublechecked_from_head_pred_coords,
                     midsec_doublechecked_from_tail_pred_coords, midsec_confidences),
                    (tail_offset_pred_coords, tail_doublechecked_from_head_pred_coords,
                        tail_doublechecked_from_midsec_pred_coords, tail_confidences))
        else:
            return ((head_offset_pred_coords, head_doublechecked_from_midsec_pred_coords,
                     head_doublechecked_from_tail_pred_coords),
                    (midsec_offset_pred_coords, midsec_doublechecked_from_head_pred_coords,
                     midsec_doublechecked_from_tail_pred_coords),
                    (tail_offset_pred_coords, tail_doublechecked_from_head_pred_coords,
                        tail_doublechecked_from_midsec_pred_coords))
        
    def get_triple_keypoints_offset_doublecheck_metrics(self,
                                                        offset_pred_coords: list,
                                                        doublecheck_a_coords: list,
                                                        doublecheck_b_coords: list,
                                                        gt_coords: list,
                                                        names: list = ['head', 'midsec', 'tail'],
                                                        detailed: bool = False):
        if isinstance(gt_coords[0], torch.Tensor):
            gt_coords = [i.cpu().numpy() for i in gt_coords]
        
        offset_metrics = self.add_prefix_to_dict(self.calculate_metrics(
            gt_coords, offset_pred_coords), names[0] + '_offset_')
        doublecheck_a_metrics = self.add_prefix_to_dict(self.calculate_metrics(
            gt_coords, doublecheck_a_coords), names[0] + '_from_' + names[1] + '_doublecheck_')
        doublecheck_b_coords = self.add_prefix_to_dict(self.calculate_metrics(
            gt_coords, doublecheck_b_coords), names[0] + '_from_' + names[2] + '_doublecheck_')
        
        result_dict = {**offset_metrics, **doublecheck_a_metrics, **doublecheck_b_coords}

        if detailed:
            return result_dict
        else:
            return {key: value
                    for key, value in result_dict.items()
                    if 'mae' in key or 'f1' in key}
        
    def evaluate(self, preds, gt_coords, apply_sigmoid=False, detailed=False):
        
        pred_coords = self.get_pred_coords(preds, apply_sigmoid=apply_sigmoid, return_confidence=True)

        if isinstance(gt_coords[0], torch.Tensor):
            gt_coords = [i.cpu().numpy() for i in gt_coords]
        
        head_metrics = self.get_triple_keypoints_offset_doublecheck_metrics(
            pred_coords[0][0], pred_coords[0][1], pred_coords[0][2],
            [gt_cds[:self.num_fish] for gt_cds in gt_coords],
            names=['head', 'midsec', 'tail'], detailed=detailed)
        midsec_metrics = self.get_triple_keypoints_offset_doublecheck_metrics(
            pred_coords[1][0], pred_coords[1][1], pred_coords[1][2],
            [gt_cds[self.num_fish : self.num_fish * 2] for gt_cds in gt_coords],
            names=['midsec', 'head', 'tail'], detailed=detailed)
        tail_metrics = self.get_triple_keypoints_offset_doublecheck_metrics(
            pred_coords[2][0], pred_coords[2][1], pred_coords[2][2],
            [gt_cds[self.num_fish * 2 : self.num_fish * 3] for gt_cds in gt_coords],
            names=['tail', 'head', 'midsec'], detailed=detailed)
        
        return {**head_metrics, **midsec_metrics, **tail_metrics}
    
class TripleKeypointsOffsetDoublecheckEvaluator(TripleKeypointsOffsetEvaluator):

    def hungarian_matching(self, h_m, m, threshold):

        if len(h_m) == 0 or len(m) == 0:
            return {i: None
                for i in range(len(h_m))}

        mappings = {}

        weights = []
        for head_to_middle in h_m:
            weight = []
            for middle in m:
                weight.append(np.linalg.norm(head_to_middle - middle))
            weights.append(weight)
        row_ind, col_ind = linear_sum_assignment(weights)
        for r, c in zip(row_ind, col_ind):
            if weights[r][c] < threshold:
                mappings[r] = (c, weights[r][c])
            else:
                mappings[r] = None

        return mappings
    
    def select_fishes_from_graph_pool(self, graph_pool):
        sorted_pool = list(sorted(graph_pool,
                            key=lambda x: (x['active'], x['n_nodes'], x['n_edges'], x['original']),
                            reverse=True))[:self.num_fish]
        fishes = []
        types = []
        for i in sorted_pool:
            if i['active'] == False:
                continue
            nodes = list(i['graph'].nodes)
            if len(nodes) == 2:
                # We need to add additional information when we have 2 nodes
                # This is because when we have 2 nodes, we have two ways to determine the third node of the fish
                # However, when we have 1 or 3 nodes, there is no other options.
                if i['graph'].has_edge(nodes[0], nodes[1]) and i['graph'].has_edge(nodes[1], nodes[0]):
                    first_node = nodes[0] + (True, )
                    second_node = nodes[1] + (True, )
                    fishes.append([first_node, second_node])
                elif i['graph'].has_edge(nodes[0], nodes[1]):
                    first_node = nodes[0] + (True, )
                    second_node = nodes[1] + (False, )
                    fishes.append([first_node, second_node])
                elif i['graph'].has_edge(nodes[1], nodes[0]):
                    first_node = nodes[0] + (False, )
                    second_node = nodes[1] + (True, )
                    fishes.append([first_node, second_node])
                else:
                    raise ValueError('In the graph pool with 2 nodes, there should be at least one edge.')
            else:
                fishes.append(nodes)
            types.append(f"N{i['n_nodes']}_E{i['n_edges']}_O{int(i['original'])}")

        return fishes, types
    
    def reorder(self, current_order, *args):
        """
        Reorder the groups and the corresponding arguments.
        """
        target_order = ['head', 'midsec', 'tail']
        new_args = []
        for arg in args:
            new_arg = []
            for i in target_order:
                if i in current_order:
                    new_arg.append(arg[current_order.index(i)])
            new_args.append(new_arg)
        return new_args
    
    def get_mapping_from_pred_coords(self, pred_coords_with_confidence, index):
        h, h_m, h_t, h_c, m, m_h, m_t, m_c, t, t_h, t_m, t_c = (
            pred_coords_with_confidence[0][0][index], pred_coords_with_confidence[1][1][index], pred_coords_with_confidence[2][1][index], pred_coords_with_confidence[0][3][index],
            pred_coords_with_confidence[1][0][index], pred_coords_with_confidence[0][1][index], pred_coords_with_confidence[2][2][index], pred_coords_with_confidence[1][3][index],
            pred_coords_with_confidence[2][0][index], pred_coords_with_confidence[0][2][index], pred_coords_with_confidence[1][2][index], pred_coords_with_confidence[2][3][index])
        mapping = {}
        mapping['head'] = {'tail': self.hungarian_matching(h_t, t, self.doublecheck_threshold),
                        'midsec': self.hungarian_matching(h_m, m, self.doublecheck_threshold)}
        mapping['midsec'] = {'head': self.hungarian_matching(m_h, h, self.doublecheck_threshold),
                                'tail': self.hungarian_matching(m_t, t, self.doublecheck_threshold)}
        mapping['tail'] = {'head': self.hungarian_matching(t_h, h, self.doublecheck_threshold),
                                'midsec': self.hungarian_matching(t_m, m, self.doublecheck_threshold)}
        coords_dict = {'head': {'confs': h_c,
                                'coords': h,
                                'midsec': h_m,
                                'tail': h_t},
                        'midsec': {'confs': m_c,
                                    'coords': m,
                                    'head': m_h,
                                    'tail': m_t},
                        'tail': {'confs': t_c,
                                'coords': t,
                                'head': t_h,
                                'midsec': t_m}}
        return mapping, coords_dict
    
    def confirm_coords_from_nodes(self, fishes, types, coords_dict):
        matched_cds = []
        matched_cfs = []
        matched_tps = []
        for fish, tp in zip(fishes, types):
            if len(fish) == 3:
                current_order = [node[0] for node in fish]
                cds = [coords_dict[node[0]]['coords'][node[1]] for node in fish]
                cfs = [coords_dict[node[0]]['confs'][node[1]] for node in fish]
                cds, cfs = self.reorder(current_order, cds, cfs)
                matched_cds.append(np.array(cds))
                matched_cfs.append(np.array(cfs))
                matched_tps.append([tp] * 3)
            elif len(fish) == 2:

                # Find the group of the third node
                groups = set(('head', 'midsec', 'tail'))
                for node in fish:
                    groups.remove(node[0])
                third_group = groups.pop()

                if fish[0][2] == fish[1][2] == True:
                    # The nodes are pointing to each other
                    # To calculate the third node, we calculate the average of the two nodes
                    third_coord_from_node_0 = coords_dict[fish[0][0]][third_group][fish[0][1]]
                    conf_of_node_0 = coords_dict[fish[0][0]]['confs'][fish[0][1]]
                    third_coord_from_node_1 = coords_dict[fish[1][0]][third_group][fish[1][1]]
                    conf_of_node_1 = coords_dict[fish[1][0]]['confs'][fish[1][1]]
                    third_coord = (third_coord_from_node_0 * conf_of_node_0 +
                                    third_coord_from_node_1 * conf_of_node_1) / (conf_of_node_0 + conf_of_node_1)
                    third_conf = 0
                    tps = [tp, tp, tp + '_' + ''.join(sorted(fish[0][0][0].upper()+fish[1][0][0].upper())) + '>' + third_group[0].upper()] # Make sure that the characters are sorted
                elif fish[0][2] == True:
                    third_coord = coords_dict[fish[0][0]][third_group][fish[0][1]]
                    third_conf = 0
                    tps = [tp, tp, tp + '_' + fish[0][0][0].upper() + '>' + third_group[0].upper()]
                elif fish[1][2] == True:
                    third_coord = coords_dict[fish[1][0]][third_group][fish[1][1]]
                    third_conf = 0
                    tps = [tp, tp, tp + '_' + fish[1][0][0].upper() + '>' + third_group[0].upper()]
                else:
                    raise ValueError('The nodes should be pointing to each other or one of them should be pointing to the other.')
                current_order = [fish[0][0], fish[1][0], third_group]
                cds = [coords_dict[fish[0][0]]['coords'][fish[0][1]],
                    coords_dict[fish[1][0]]['coords'][fish[1][1]],
                    third_coord]
                cfs = [coords_dict[fish[0][0]]['confs'][fish[0][1]],
                    coords_dict[fish[1][0]]['confs'][fish[1][1]],
                    third_conf]
                cds, cfs, tps = self.reorder(current_order, cds, cfs, tps)
                matched_cds.append(np.array(cds))
                matched_cfs.append(np.array(cfs))
                matched_tps.append(tps)
            elif len(fish) == 1:
                remaining_groups = list(set(('head', 'midsec', 'tail')) - set([fish[0][0]]))
                remaining_cds = [coords_dict[fish[0][0]][group][fish[0][1]] for group in remaining_groups]
                current_order = [fish[0][0]] + remaining_groups
                cds = [coords_dict[fish[0][0]]['coords'][fish[0][1]]] + remaining_cds
                cfs = [coords_dict[fish[0][0]]['confs'][fish[0][1]]] * 3
                tps = [tp, tp+'_'+fish[0][0][0].upper(), tp+'_'+fish[0][0][0].upper()]
                cds, cfs, tps = self.reorder(current_order, cds, cfs, tps)
                matched_cds.append(np.array(cds))
                matched_cfs.append(np.array(cfs))
                matched_tps.append(tps)
            else:
                raise ValueError('The number of nodes should be 1, 2, or 3.')
            if len(matched_cds[-1]) != 3:
                ic([node for node in fish])
                ic(matched_cds[-1])
                raise ValueError('The number of nodes should be 3.')
        return np.array(matched_cds), np.array(matched_cfs), matched_tps

    def match_coords(self, pred_coords_with_confidence):
        matched_coords = []
        matched_confs = []
        matched_types = []
        for index in range(len(pred_coords_with_confidence[0][0])):
            mapping, coords_dict = self.get_mapping_from_pred_coords(pred_coords_with_confidence, index)
            full_graph = generate_full_graph(mapping)
            graph_pool = analyze_full_graph(full_graph)
            fishes, types = self.select_fishes_from_graph_pool(graph_pool)
            matched_cds, matched_cfs, matched_tps = self.confirm_coords_from_nodes(fishes, types, coords_dict)
            matched_coords.append(matched_cds)
            matched_confs.append(matched_cfs)
            matched_types.append(matched_tps)
        return matched_coords, matched_confs, matched_types
    
    def get_final_coords_from_pred_maps(self, preds, apply_sigmoid=False, return_conf_type=False):
        pred_coords = self.get_pred_coords(preds, apply_sigmoid=apply_sigmoid, return_confidence=True)
        matched_coords, matched_confs, matched_types = self.match_coords(pred_coords)
        if return_conf_type:
            return matched_coords, matched_confs, matched_types
        else:
            return matched_coords

    def evaluate(self, preds, gt_coords, apply_sigmoid=False, detailed=False):
        matched_coords = self.get_final_coords_from_pred_maps(preds, apply_sigmoid=apply_sigmoid)
        if isinstance(gt_coords[0], torch.Tensor):
            gt_coords = [i.cpu().numpy() for i in gt_coords]
        head_metrics = self.add_prefix_to_dict(self.calculate_metrics([gt_cds[:self.num_fish]
                                               for gt_cds in gt_coords],
                                              [[fish[0] for fish in coords] 
                                               for coords in matched_coords],
                                               return_false_positive=detailed),
                                               'head_')
        midsec_metrics = self.add_prefix_to_dict(self.calculate_metrics([gt_cds[self.num_fish:self.num_fish*2]
                                                 for gt_cds in gt_coords],
                                                [[fish[1] for fish in coords] 
                                                 for coords in matched_coords],
                                                 return_false_positive=detailed),
                                                'midsec_')
        tail_metrics = self.add_prefix_to_dict(self.calculate_metrics([gt_cds[self.num_fish*2:self.num_fish*3]
                                                for gt_cds in gt_coords],
                                                 [[fish[2] for fish in coords] 
                                                for coords in matched_coords],
                                                return_false_positive=detailed),
                                                    'tail_')
        if detailed:
            return {**head_metrics, **midsec_metrics, **tail_metrics}
        else:
            all_metrics = {**head_metrics, **midsec_metrics, **tail_metrics}
            return {key: value
                    for key, value in all_metrics.items()
                    if 'f1' in key or 'mae' in key}

class TripleKeypointsOffsetDoublecheckEvaluatorV2(TripleKeypointsOffsetDoublecheckEvaluator):

    def cluster_awaiting_graph_nodes(self, awaiting_graphs, num_missing, coords_dict):

        fishes = []
        corresponding_nodes = []
        corresponding_graphs = []

        for graph in awaiting_graphs:
            for node in graph.nodes:
                groups = set(('head', 'midsec', 'tail'))
                groups.remove(node[0]) # Nodes are represented as tuples (group, index)
                current_order = [node[0]] + list(groups)
                coords = [coords_dict[node[0]]['coords'][node[1]]] + [coords_dict[node[0]][group][node[1]] for group in groups]
                coords = self.reorder(current_order, coords)[0]
                fishes.append(np.concatenate(coords))
                corresponding_nodes.append(node)
                corresponding_graphs.append(graph)

        current_num_cluster = num_missing

        while True:

            ac = AgglomerativeClustering(n_clusters=current_num_cluster)
            labels = list(ac.fit_predict(fishes))

            # Count the number of fish for each label
            only_one = []
            more_than_three = []
            counter = Counter(labels)

            for label in range(current_num_cluster):
                if counter[label] == 1:
                    only_one.append(label)
                elif counter[label] > 3:
                    more_than_three.append(label)
            
            if len(only_one) and len(more_than_three):

                # If `more_than_three`` exists, it shows that the cluster can be further divided
                # The `only_one` is considered as noise
                fish_indices = [labels.index(label) for label in only_one]
                confs = [coords_dict[corresponding_nodes[i][0]]['confs'][corresponding_nodes[i][1]]
                         for i in fish_indices]
                min_conf_index = fish_indices[np.argmin(confs)]
                fishes.pop(min_conf_index)
                corresponding_nodes.pop(min_conf_index)
                corresponding_graphs.pop(min_conf_index)
            
            elif len(more_than_three):

                # The cluster is bad: the cluster can be further divided
                current_num_cluster += 1

            elif len(only_one):

                if current_num_cluster == num_missing:

                    # The cluster is good, but there may be repeated groups (e.g. two heads in a subgraph)
                    for label in range(num_missing):
                        if counter[label] > 1:
                            while True: # Add a loop to remove all repeated groups
                                seen_groups = []
                                repeat = False
                                for i, l in enumerate(labels):
                                    if l == label:
                                        group = corresponding_nodes[i][0]
                                        if group in seen_groups:
                                            repeat = group
                                            break
                                        else:
                                            seen_groups.append(group)
                                if repeat:
                                    # Find the node that is furthest from the average coord
                                    repeated_fish_indeces = [i
                                                    for i, l in enumerate(labels)
                                                    if l == label and corresponding_nodes[i][0] == repeat]
                                    repeated_coords = [fishes[i] for i in repeated_fish_indeces]
                                    to_be_removed = np.argmax(
                                        np.linalg.norm(
                                            repeated_coords - np.mean(repeated_coords, axis=0), axis=1))
                                    index_to_remove = repeated_fish_indeces[to_be_removed]
                                    labels[index_to_remove] = -1
                                else:
                                    break
                    # The repeated groups are removed, the cluster is good
                    break

                else:

                    # The cluster is bad: we need to remove the noise and reduce the number of clusters
                    fish_indices = [labels.index(label) for label in only_one]
                    confs = [coords_dict[corresponding_nodes[i][0]]['confs'][corresponding_nodes[i][1]]
                             for i in fish_indices]
                    min_conf_index = fish_indices[np.argmin(confs)]
                    fishes.pop(min_conf_index)
                    corresponding_nodes.pop(min_conf_index)
                    corresponding_graphs.pop(min_conf_index)
                    current_num_cluster -= 1

            else: # No `only_one` and no `more_than_three`

                if current_num_cluster == num_missing:

                    # The cluster is good, but there may be repeated groups (e.g. two heads in a subgraph)
                    for label in range(num_missing):
                        if counter[label] > 1:
                            while True: # Add a loop to remove all repeated groups
                                seen_groups = []
                                repeat = False
                                for i, l in enumerate(labels):
                                    if l == label:
                                        group = corresponding_nodes[i][0]
                                        if group in seen_groups:
                                            repeat = group
                                            break
                                        else:
                                            seen_groups.append(group)
                                if repeat:
                                    # Find the node that is furthest from the average coord
                                    repeated_fish_indeces = [i
                                                    for i, l in enumerate(labels)
                                                    if l == label and corresponding_nodes[i][0] == repeat]
                                    repeated_coords = [fishes[i] for i in repeated_fish_indeces]
                                    to_be_removed = np.argmax(
                                        np.linalg.norm(
                                            repeated_coords - np.mean(repeated_coords, axis=0), axis=1))
                                    index_to_remove = repeated_fish_indeces[to_be_removed]
                                    labels[index_to_remove] = -1
                                else:
                                    break
                    # The repeated groups are removed, the cluster is good
                    break

                else: # current_num_cluster > num_missing

                    # The cluster is bad: we remove the cluster with the smallest size and highest std
                    size_of_clusters = [counter[label] for label in range(current_num_cluster)]
                    min_size = min(size_of_clusters)
                    num_min_size = size_of_clusters.count(min_size)

                    # There is only one cluster with the smallest size
                    if num_min_size == 1:
                        label = size_of_clusters.index(min_size)
                        fish_indices = [i
                                        for i, l in enumerate(labels)
                                        if l == label]
                        fish_indices.sort(reverse=True) # Remove from the last index to avoid index error
                        for i in fish_indices:
                            fishes.pop(i)
                            corresponding_nodes.pop(i)
                            corresponding_graphs.pop(i)
                    # There are multiple clusters with the smallest size
                    # We remove the cluster with the highest std
                    else:
                        min_size_labels = [i
                                           for i, size in enumerate(size_of_clusters)
                                           if size == min_size]
                        label_stds = []
                        for label in min_size_labels:
                            fish_indices = [i for i, l in enumerate(labels) if l == label]
                            std = np.mean(np.std([fishes[i] for i in fish_indices], axis=0))
                            label_stds.append(std)
                        label_with_max_std = min_size_labels[np.argmax(label_stds)]
                        fish_indices = [i
                                        for i, l in enumerate(labels)
                                        if l == label_with_max_std]
                        fish_indices.sort(reverse=True) # Remove from the last index to avoid index error
                        for i in fish_indices:
                            fishes.pop(i)
                            corresponding_nodes.pop(i)
                            corresponding_graphs.pop(i)

                    # Reduce the number of clusters
                    current_num_cluster -= 1

        graphs = []
        for label in range(current_num_cluster):
            fish_indices = [i for i, l in enumerate(labels) if l == label]
            parent_graph = corresponding_graphs[fish_indices[0]]
            nodes = [corresponding_nodes[i] for i in fish_indices]
            graphs.append(parent_graph.subgraph(nodes))
        return graphs
    
    def select_fishes_from_graph_pool(self, graph_pools, coords_dict):
        fishes = []
        types = []
        
        for key, pool in graph_pools.items():
            for graph in pool:
                types.append(f"N{graph.number_of_nodes()}_E{graph.number_of_edges()}_{key[:3].upper()}")
                nodes = list(graph.nodes)
                if len(nodes) == 2:
                # We need to add additional information when we have 2 nodes
                # This is because when we have 2 nodes, we have two ways to determine the third node of the fish
                # However, when we have 1 or 3 nodes, there is no other options.
                    if graph.has_edge(nodes[0], nodes[1]) and graph.has_edge(nodes[1], nodes[0]):
                        first_node = nodes[0] + (True, )
                        second_node = nodes[1] + (True, )
                        fishes.append([first_node, second_node])
                    elif graph.has_edge(nodes[0], nodes[1]):
                        first_node = nodes[0] + (True, )
                        second_node = nodes[1] + (False, )
                        fishes.append([first_node, second_node])
                    elif graph.has_edge(nodes[1], nodes[0]):
                        first_node = nodes[0] + (False, )
                        second_node = nodes[1] + (True, )
                        fishes.append([first_node, second_node])
                    else:
                        first_node = nodes[0] + (True, )
                        second_node = nodes[1] + (True, )
                        fishes.append([first_node, second_node])
                else:
                    fishes.append(nodes)

        return fishes, types

    def select_fishes_from_graph_pool_v0(self, graph_pools, coords_dict):

        fishes = []
        types = []
        
        for key, pool in graph_pools.items():
            for graph in pool:
                types.append(f"N{graph.number_of_nodes()}_E{graph.number_of_edges()}_{key[:3].upper()}")
                nodes = list(graph.nodes)
                if len(nodes) == 2:
                    # We need to add additional information when we have 2 nodes
                    # This is because when we have 2 nodes, we have two ways to determine the third node of the fish
                    # However, when we have 1 or 3 nodes, there is no other options.
                    if graph.nodes[nodes[0]]['center'] == True:
                        first_node = nodes[0] + (False, ) # The center node is not reliable
                        second_node = nodes[1] + (True, )
                        fishes.append([first_node, second_node])
                    elif graph.nodes[nodes[1]]['center'] == True:
                        first_node = nodes[0] + (True, )
                        second_node = nodes[1] + (False, )
                        fishes.append([first_node, second_node])
                    else:
                        # No center node is found, we need to calculate the distances
                        #                                    1st group   2nd group    1st index
                        first_to_second_coord = coords_dict[nodes[0][0]][nodes[1][0]][nodes[0][1]]
                        #                           2nd group              2nd index
                        second_coord = coords_dict[nodes[1][0]]['coords'][nodes[1][1]]
                        first_to_second_dist = np.linalg.norm(first_to_second_coord - second_coord)

                        second_to_first_coord = coords_dict[nodes[1][0]][nodes[0][0]][nodes[1][1]]
                        first_coord = coords_dict[nodes[0][0]]['coords'][nodes[0][1]]
                        second_to_first_dist = np.linalg.norm(second_to_first_coord - first_coord)

                        if first_to_second_dist < second_to_first_dist:
                            first_node = nodes[0] + (True, ) # The first node can better determine the third node
                            second_node = nodes[1] + (False, )
                            fishes.append([first_node, second_node])
                        else:
                            first_node = nodes[0] + (False, )
                            second_node = nodes[1] + (True, )
                            fishes.append([first_node, second_node])
                else:
                    fishes.append(nodes)

        return fishes, types

    def match_coords(self, pred_coords_with_confidence):
        matched_coords = []
        matched_confs = []
        matched_types = []
        for index in range(len(pred_coords_with_confidence[0][0])):
            try:
                mapping, coords_dict = self.get_mapping_from_pred_coords(pred_coords_with_confidence, index)
                full_graph = generate_full_graph(mapping)
                active_graphs, awaiting_graphs, num_missing = analyze_full_graph_v2(full_graph, self.num_fish)
                if num_missing > 0:
                    awaiting_graphs = self.cluster_awaiting_graph_nodes(awaiting_graphs, num_missing, coords_dict)
                graph_pools = {'original': active_graphs,
                               'clustered': awaiting_graphs}
                fishes, types = self.select_fishes_from_graph_pool(graph_pools, coords_dict)
                matched_cds, matched_cfs, matched_tps = self.confirm_coords_from_nodes(fishes, types, coords_dict)
                matched_coords.append(matched_cds)
                matched_confs.append(matched_cfs)
                matched_types.append(matched_tps)
            except:
                ic(index)
                raise
        return matched_coords, matched_confs, matched_types
    
class TripleKeypointsOffsetDoublecheckEvaluatorV3(TripleKeypointsOffsetDoublecheckEvaluatorV2):

    def match_coords(self, pred_coords_with_confidence):
        matched_coords = []
        matched_confs = []
        matched_types = []
        for index in range(len(pred_coords_with_confidence[0][0])):
            try:
                mapping, coords_dict = self.get_mapping_from_pred_coords(pred_coords_with_confidence, index)
                full_graph = generate_full_graph(mapping)
                graph_pools, unprocessed_graphs = analyze_full_graph_v3(full_graph, self.num_fish)
                num_missing = self.num_fish - len(graph_pools['original']) - len(graph_pools['denoised']) - len(graph_pools['divided'])
                if num_missing > 0 and len(unprocessed_graphs) > 0:
                    graph_pools['clustered'] = self.cluster_awaiting_graph_nodes(unprocessed_graphs, num_missing, coords_dict)
                fishes, types = self.select_fishes_from_graph_pool(graph_pools, coords_dict)
                matched_cds, matched_cfs, matched_tps = self.confirm_coords_from_nodes(fishes, types, coords_dict)
                matched_coords.append(matched_cds)
                matched_confs.append(matched_cfs)
                matched_types.append(matched_tps)
            except:
                ic(index)
                raise
        return matched_coords, matched_confs, matched_types