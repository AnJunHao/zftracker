import numpy as np
from icecream import ic
from ..criterion.keypoint import CatagoryFirstDoublecheckKeypointEvaluator
from ..criterion.keypoint import TripleKeypointsOffsetDoublecheckEvaluator
from ..criterion.keypoint import TripleKeypointsOffsetDoublecheckEvaluatorV2
from ..criterion.keypoint import TripleKeypointsOffsetDoublecheckEvaluatorV3

class CategoryFirstDoublecheckKeypoint(object):
    """
    DoublecheckKeypoint class for inference.
    Provided a predicted heatmap (with doublecheck mechanism),
    this class can be used to generate keypoint predictions.
    The heatmap shape should be (N, 10, H, W) as the output
    of our model with doublecheck mechanism.
    """

    def __init__(self,
                 num_entities: int = 7,
                 max_detect: int = 20,
                 peak_threshold: float = 0.1,
                 peak_radius: int = 1,
                 average_2x2: bool = False,
                 doublecheck_normalization_factor: float = 8.0,
                 doublecheck_threshold: float = 1.5,
                 apply_sigmoid: bool = False,
                 category_priority: list = ['perfect_match', 'overlap_match', 'unilateral', 'isolated'],
                 head_mid_distance_range: tuple = (0.8, 1.2)):
        self.evaluator = CatagoryFirstDoublecheckKeypointEvaluator(
            num_fish=num_entities,
            max_detect=max_detect,
            peak_threshold=peak_threshold,
            peak_radius=peak_radius,
            # local_distance_threshold=local_distance_threshold,
            # overlap_as_true_positive=overlap_as_true_positive,
            average_2x2=average_2x2,
            doublecheck_normalization_factor=doublecheck_normalization_factor,
            doublecheck_threshold=doublecheck_threshold,
            catagory_priority=category_priority,
            head_mid_distance_range=head_mid_distance_range)
        self.apply_sigmoid = apply_sigmoid
    
    def __call__(self, preds, detailed=False):
        """
        Generate keypoint predictions from heatmap.
        Args:
            preds (torch.Tensor): Predicted heatmap with shape (N, 10, H, W).
        Returns:
            list: List of dictionaries. Each dictionary contains the following keys:
                'perfect_match': List of tuples. Tuples are in the format of (head_coord, middle_coord, confidence),
                    each representing a coordinate and its confidence, sorted by confidence in descending order.
                'overlap_match': List of tuples. Tuples are in the same format as above.
                'unilateral': List of tuples. Tuples are in the same format as above.
                'isolated': List of tuples. Tuples are in the same format as above.
        """
        dual_keypoints_offset_doublecheck_coords_and_confidences = self.evaluator.get_pred_coords(
            preds, apply_sigmoid=self.apply_sigmoid, return_confidence=True)

        matched_coords = self.evaluator.match_coords(
            *dual_keypoints_offset_doublecheck_coords_and_confidences)
        
        selected_coords = self.evaluator.select_matched_coords(matched_coords, process_log=True)

        coords = [[np.hstack(cd[:2]) for cd in cds]
                  for cds in selected_coords]
        conf_and_type = [[cd[2:] for cd in cds]
                         for cds in selected_coords]
        
        return coords, conf_and_type
    
class TripleKeypoint(object):

    def __init__(self,
                 num_entities: int = 7,
                 max_detect: int = 7,
                 peak_threshold: float = 0.25,
                 peak_radius: int = 1,
                 average_2x2: bool = False,
                 doublecheck_normalization_factor: dict = {
                    ('head', 'midsec'): 8,
                    ('head', 'tail'): 26.620351900862914,
                    ('midsec', 'tail'): 16.89075517685661}, # Dataset V1.0
                 doublecheck_threshold: float = 4,
                 head_mid_distance_range: tuple = (0.5, 1.5),
                 tail_cluster: bool = True,
                 doublecheck_version: int = 1):
        if doublecheck_version == 1:
            self.evaluator = TripleKeypointsOffsetDoublecheckEvaluator(
                max_detect=max_detect,
                peak_threshold=peak_threshold,
                peak_radius=peak_radius,
                average_2x2=average_2x2,
                doublecheck_normalization_factor=doublecheck_normalization_factor,
                num_fish=num_entities,
                doublecheck_threshold=doublecheck_threshold,
                head_mid_distance_range=head_mid_distance_range,
                tail_cluster=tail_cluster)
        elif doublecheck_version == 2:
            self.evaluator = TripleKeypointsOffsetDoublecheckEvaluatorV2(
                max_detect=max_detect,
                peak_threshold=peak_threshold,
                peak_radius=peak_radius,
                average_2x2=average_2x2,
                doublecheck_normalization_factor=doublecheck_normalization_factor,
                num_fish=num_entities,
                doublecheck_threshold=doublecheck_threshold,
                head_mid_distance_range=head_mid_distance_range,
                tail_cluster=tail_cluster)
        elif doublecheck_version == 3:
            self.evaluator = TripleKeypointsOffsetDoublecheckEvaluatorV3(
                max_detect=max_detect,
                peak_threshold=peak_threshold,
                peak_radius=peak_radius,
                average_2x2=average_2x2,
                doublecheck_normalization_factor=doublecheck_normalization_factor,
                num_fish=num_entities,
                doublecheck_threshold=doublecheck_threshold,
                head_mid_distance_range=head_mid_distance_range,
                tail_cluster=tail_cluster)
        else:
            raise ValueError('doublecheck_version should be 1 or 2 or 3')
    
    def __call__(self, preds, apply_sigmoid=False, return_raw_coords=False, raw_coords_only=False):

        pred_coords = self.evaluator.get_pred_coords(preds, apply_sigmoid=apply_sigmoid, return_confidence=True)
        if raw_coords_only:
            return pred_coords
        matched_coords, matched_confs, matched_types = self.evaluator.match_coords(pred_coords)
        
        if return_raw_coords:
            return (matched_coords, matched_confs, matched_types), pred_coords
        else:
            return matched_coords, matched_confs, matched_types