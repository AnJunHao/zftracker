from .util import format_scientific
from .util import TQDM as tqdm
import torch
import numpy as np
from skimage.feature import peak_local_max
from matplotlib import pyplot as plt



def train_model(
    model,
    dataloader,
    val_dataloader,
    device,
    num_epochs,
    loss_function,
    optimizer,
    lr_scheduler,
    randaug_scheduler,
    metrics,
    verbose=True,
):
    """
    Train a PyTorch model for a specified number of epochs, evaluating on a validation set and metrics.
    Args:
        model (torch.nn.Module): Model to be trained.
        dataloader (torch.utils.data.DataLoader): Training set dataloader.
        val_dataloader (torch.utils.data.DataLoader): Validation set dataloader.
        device (torch.device): Device to run the model on.
        num_epochs (int): Number of epochs to train for.
        loss_function (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        randaug_scheduler (WarmUpCosineRandaugmentScheduler): RandAugment scheduler.
        metrics (list): List of metrics to evaluate on.
        verbose (bool): Whether to print the current epoch's loss and learning rate.
    Returns:
        dict: Dictionary containing the training history.
    """

    model.to(device)

    # Initialize history dictionary
    history = {"loss": [], "lr": [], "randaug_n": [], "randaug_m": []}
    history.update({"val_" + metric.__name__: [] for metric in metrics})

    # Training loop
    for epoch in range(num_epochs):

        model.train()

        print("-" * 10)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        running_loss = 0.0

        for inputs, labels in tqdm(dataloader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            lr_scheduler.step()

        if epoch < num_epochs - 1:
            randaug_scheduler.step()

        epoch_loss = running_loss / len(dataloader.dataset)

        if verbose:
            print(
                f"Loss: {format_scientific(epoch_loss)}, Learning rate: {format_scientific(lr_scheduler.get_last_lr()[0])}"
            )

        model.eval()

        with torch.no_grad():

            all_outputs, all_labels, all_coords = [], [], []

            for val_inputs, val_labels, val_coords in tqdm(val_dataloader):

                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                outputs = model(val_inputs)

                all_outputs.append(outputs)
                all_labels.append(val_labels)
                all_coords.append(val_coords)

            all_outputs = torch.cat(all_outputs)
            all_labels = torch.cat(all_labels)
            all_coords = [
                coords for sublist in all_coords for coords in sublist]

            for metric in metrics:
                score = metric(all_outputs, all_labels, all_coords)
                if isinstance(score, dict):
                    for key in score:
                        name = "val_"+metric.__name__+"_"+key
                        if name not in history:
                            history[name] = []
                        history["val_"+metric.__name__ +
                                "_"+key].append(score[key])
                else:
                    if "val_"+metric.__name__ not in history:
                        history["val_"+metric.__name__] = []
                    history["val_" + metric.__name__].append(score)
                if verbose:
                    print(f"{'val_'+metric.__name__}: {score}")

        history["loss"].append(epoch_loss)
        history["lr"].append(lr_scheduler.get_last_lr()[0])
        history["randaug_n"].append(randaug_scheduler.latest_n)
        history["randaug_m"].append(randaug_scheduler.latest_m)

    return history

def find_local_peaks(data, peak_radius, peak_threshold, max_detect):
    # Data may be torch tensors or numpy arrays
    # Convert Data to numpy arrays before using peak_local_max
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    return [peak_local_max(image, min_distance=peak_radius, threshold_abs=peak_threshold, num_peaks=max_detect)[:, ::-1] for image in data]
    
def apply_local_offsets(pred_coords, preds_local_offsets):
    
    if isinstance(preds_local_offsets, torch.Tensor):
        preds_local_offsets = preds_local_offsets.detach().cpu().numpy()
    
    adjusted_coords = []

    for pred, offset in zip(pred_coords, preds_local_offsets):

        adjusted_coord = []
        
        for kp in pred:
            adjusted_coord.append(kp + np.array([offset[1, kp[1], kp[0]], offset[0, kp[1], kp[0]]]))
        adjusted_coords.append(np.array(adjusted_coord))
    
    return adjusted_coords

def calculate_metrics(gt_coords, pred_coords, diff_threshold):

    # gt_coords may be a list of torch tensors
    if isinstance(gt_coords[0], torch.Tensor):
        gt_coords = [gt.cpu().numpy() for gt in gt_coords]

    tp = 0
    fp = 0
    fn = 0
    true_overlap = 0
    mae = 0
    n_gts = sum(len(gt) for gt in gt_coords)
    n_preds = sum(len(pred) for pred in pred_coords)

    for gt, pred in zip(gt_coords, pred_coords):
        unpaired_preds = np.ones(len(pred), dtype=int)
        for kp in gt:
            distances = np.linalg.norm(pred - kp, axis=1)
            closest_candidate = np.argmin(distances)
            if distances[closest_candidate] < diff_threshold:
                tp += 1
                mae += distances[closest_candidate]
                if unpaired_preds[closest_candidate] == 0:
                    true_overlap += 1
                else:
                    unpaired_preds[closest_candidate] = 0
            else:
                fn += 1
        fp += np.sum(unpaired_preds)

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
            precision = (tp - true_overlap) / n_preds
            recall = tp / n_gts
            mae /= tp

    return {'precision': precision,
            'recall': recall,
            'mae': mae,
            'true_positive': tp,
            'false_positive': fp,
            'false_negative': fn,
            'n_predictions': n_preds,
            'n_ground_truths': n_gts,
            'true_overlap': true_overlap}

def evaluate_keypoints_with_local_offset(preds, gt_coords, max_detect=np.inf, peak_threshold=0.5, peak_radius=1, diff_threshold=3, detailed=False, apply_sigmoid=False, verbose=False):
    """
    Args:
        preds (torch.Tensor): Predictions from the model. The shape is (batch_size, 3, height, width)
            Note: The first channel is the heatmap, the second and third channels are the local offset.
        gts (numpy.ndarray): Ground truth coordinates.
        max_detect (int): Maximum number of keypoints to detect. Default: ``np.inf``.
        peak_threshold (float): Minimum value of a peak to be considered a keypoint. Default: ``0.5``.
        peak_radius (int): Minimum distance between two peaks. Default: ``1``.
        diff_threshold (int): Maximum distance between a prediction and a ground truth coordinate to be considered a true positive. Default: ``3``.
        detailed (bool): Whether to return detailed metrics. Default: ``False``.
        apply_sigmoid (bool): Whether to apply sigmoid on the input. Default: ``False``.
        verbose (bool): Whether to print progress. Default: ``False``.
    Returns:
        dict: The metrics.
    """
    # Get coordinates from heatmaps
    if apply_sigmoid:
        pred_heatmaps = torch.sigmoid(preds[: ,0])
    else:
        pred_heatmaps = preds[:, 0]
    pred_coords = find_local_peaks(pred_heatmaps, peak_radius, peak_threshold, max_detect)

    # Apply local offsets to coordinates
    pred_coords = apply_local_offsets(pred_coords, preds[:, 1:])
    
    # Calculate metrics
    metrics = calculate_metrics(gt_coords, pred_coords, diff_threshold)

    if detailed:
        return metrics
    else:
        return {'precision': metrics['precision'],
                'recall': metrics['recall'],
                'mae': metrics['mae']}

def evaluate_keypoints(preds: torch.Tensor,
                       gts: torch.Tensor,
                       max_detect=np.inf,
                       peak_threshold=0.5,
                       peak_radius=1,
                       diff_threshold=3,
                       detailed=False,
                       ground_truths_as_coords=True,
                       apply_sigmoid=False,
                       verbose=False):
    """
    Args:
        preds (torch.Tensor): Predictions from the model.
        gts (torch.Tensor): Ground truth coordinates.
        max_detect (int): Maximum number of keypoints to detect. Default: ``np.inf``.
        peak_threshold (float): Minimum value of a peak to be considered a keypoint. Default: ``0.5``.
        peak_radius (int): Minimum distance between two peaks. Default: ``1``.
        diff_threshold (int): Maximum distance between a prediction and a ground truth coordinate to be considered a true positive. Default: ``3``.
        detailed (bool): Whether to return detailed metrics. Default: ``False``.
        ground_truths_as_coords (bool): Whether the ground truths are coordinates or heatmaps. Default: ``True``.
        apply_sigmoid (bool): Whether to apply sigmoid on the input. Default: ``False``.
        verbose (bool): Whether to print progress. Default: ``False``.
    Returns:
        dict: The metrics.
    """
    # Convert tensors to numpy for skimage functions
    if apply_sigmoid:
        preds_np = torch.sigmoid(preds)
    pred_coords = find_local_peaks(preds_np, peak_radius, peak_threshold, max_detect)

    # Convert ground truth heatmaps into coordinates
    if not ground_truths_as_coords:
        gt_coords = find_local_peaks(gts, peak_radius, peak_threshold, max_detect)
    else:
        gt_coords = gts
    
    # Calculate metrics
    metrics = calculate_metrics(gt_coords, pred_coords, diff_threshold)

    if detailed:
        return metrics
    else:
        return {'precision': metrics['precision'],
                'recall': metrics['recall'],
                'mae': metrics['mae']}
    
def evaluate_and_plot(predictions,
                      test_coords,
                      eval_function=evaluate_keypoints_with_local_offset,
                      diff_threshold=3,
                      enumerate_range=(10, 91),
                      peak_radius=1,
                      apply_sigmoid=False):
    """
    Evaluate the predictions and plot the precision-recall curve.
    Args:
        predictions (torch.Tensor): Predictions from the model.
        test_coords (list): List of ground truth coordinates.
        diff_threshold (int): Maximum distance between a prediction and a ground truth coordinate to be considered a true positive. Default: ``3``.
        enumerate_range (tuple): Range of peak_thresholds to evaluate. Default: ``(10, 91)``.
        peak_radius (int): Minimum distance between two peaks. Default: ``1``.
        apply_sigmoid (bool): Whether to apply sigmoid on the input. Default: ``False``.
    Returns:
        plt: The plot.
    """
    x = []
    y = []
    for i in tqdm(range(*enumerate_range)):
        x.append(i / 100)
        y.append(
            eval_function(
                predictions,
                test_coords,
                apply_sigmoid=apply_sigmoid,
                peak_threshold=i / 100,
                diff_threshold=diff_threshold,
                peak_radius=peak_radius
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

    # Add maximum indicator for multiply and average
    max_multiply_index = y_multiply.index(max(y_multiply))
    max_average_index = y_average.index(max(y_average))

    # Annotate the points
    for idx, y_values, color, label in [(max_multiply_index, y_multiply, 'r', 'max multiply'),
                                        (max_average_index, y_average, 'b', 'max average')]:
        plt.plot(x[idx], y_values[idx], color+'o', label=label)
        plt.annotate(f'({x[idx]}, {y_values[idx]})', (x[idx], y_values[idx]))

        # Print precision and recall at maximum points
        print(f'{label} at peak_threshold={x[idx]}:')
        print(
            f'Precision / Recall: {100*y_precision[idx]:.2f}% & {100*y_recall[idx]:.2f}%')

        # Also print the MAE at the maximum
        print(f"MAE: {y[idx]['mae']:.4f}")

    plt.legend()

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
                 confidence_threshold: float = 0.5,
                 num_fish: int = 7,
                 doublecheck_threshold: float = 1.5):
        """
        Args:
            max_detect (int): The maximum number of peaks to detect.
            peak_threshold (float): The minimum intensity of peaks.
            peak_radius (int): The radius of peaks.
            local_distance_threshold (float): The maximum distance between a predicted coordinate and a ground truth coordinate to be considered as a true positive.
            overlap_as_true_positive (bool): Whether to allow two preds to match the same gt.
            average_2x2 (bool): Whether to average the 2x2 grid of offsets.
            doublecheck_normalization_factor (float): The normalization factor for the doublecheck coordinates.
            confidence_threshold (float): The confidence threshold for the final coordinates.
            num_fish (int): The number of fish to be detected.
            doublecheck_threshold (float): The threshold for the doublecheck coordinates.
        """
        super().__init__(max_detect, peak_threshold,
                         peak_radius, local_distance_threshold,
                         overlap_as_true_positive, average_2x2,
                         doublecheck_normalization_factor,
                         doublecheck_threshold)

        self.num_fish = num_fish

        # Convert the confidence threshold from a single value to a dictionary
        if isinstance(confidence_threshold, float):
            self.confidence_threshold = {'perfect_match': confidence_threshold,
                                         'overlap_match': confidence_threshold,
                                         'unilateral': confidence_threshold,
                                         'isolated': confidence_threshold}
        elif isinstance(confidence_threshold, dict):
            self.confidence_threshold = confidence_threshold
        else:
            raise ValueError(
                f"Invalid Value for arg 'confidence_threshold': '{confidence_threshold} \n Supported types: 'float', 'dict'")

    def __repr__(self) -> str:
        return (f"NaiveDoublecheckKeypointEvaluator(max_detect={self.max_detect}, "
                f"peak_threshold={self.peak_threshold}, "
                f"peak_radius={self.peak_radius}, "
                f"local_distance_threshold={self.local_distance_threshold}, "
                f"overlap_as_true_positive={self.overlap_as_true_positive}, "
                f"average_2x2={self.average_2x2}, "
                f"doublecheck_normalization_factor={self.doublecheck_normalization_factor}, "
                f"confidence_threshold={self.confidence_threshold}, "
                f"num_fish={self.num_fish}, "
                f"doublecheck_threshold={self.doublecheck_threshold})")

    def select_matched_coords_naive_strategy(self,
                                             matched_coords: list,
                                             num_fish: int,
                                             confidence_threshold: dict,
                                             process_log: bool = False):
        """
        Select the best coordinates from the matched coordinates. We use the following steps:

        1. Select the coordinates with confidence higher than the threshold for perfect match.
        2. Select the coordinates with confidence higher than the threshold for overlap match.
        3. Select the coordinates with confidence lower than the threshold for perfect match.
        4. Select the coordinates with confidence lower than the threshold for overlap match.
        5. Select the coordinates with confidence higher than the threshold for unilateral.
        6. Select the coordinates with confidence lower than the threshold for unilateral.
        7. Select the coordinates with confidence higher than the threshold for isolated.
        8. Select the coordinates with confidence lower than the threshold for isolated.

        For each step, if the number of selected coordinates exceeds the number of fish, we stop.

        When confidence_threshold == peak_threshold, the process is equivalent to the following steps:
        1. Select the coordinates with confidence higher than the threshold for perfect match.
        2. Select the coordinates with confidence higher than the threshold for overlap match.
        3. Select the coordinates with confidence higher than the threshold for unilateral.
        4. Select the coordinates with confidence higher than the threshold for isolated.

        Args:
            matched_coords (list): List of dictionaries. Each dictionary contains the following keys:
                'perfect_match': List of tuples. Tuples are in the format of (head_coord, middle_coord, confidence).
                'overlap_match': List of tuples. Tuples are in the same format as above.
                'unilateral': List of tuples. Tuples are in the same format as above.
                'isolated': List of tuples. Tuples are in the same format as above.
            num_fish (int): Number of fish to be detected.
            confidence_threshold (dict): Dictionary of confidence thresholds for each type of match.
            process_log (bool): Whether to include the confidence value and match type in the result. Default: ``False``.
        Returns:
            list: List of coordinates. Each element is a list of tuples. Tuples are in the format of (head_coord, middle_coord).
        """

        all_result = []

        for matched_cds in matched_coords:

            num_perfect_matches = sum(
                [1 for i in matched_cds['perfect_match'] if i[2] > confidence_threshold['perfect_match']])

            if num_perfect_matches >= num_fish:

                if process_log:
                    all_result.append(
                        [i+("perfect_match",) for i in matched_cds['perfect_match'][:num_fish]])
                else:
                    # Get rid of the confidence value and make sure the lenth of the result is 'num_fish'
                    all_result.append(
                        [i[:2] for i in matched_cds['perfect_match'][:num_fish]])
            else:
                perfect_match_gap = num_fish - num_perfect_matches
                num_overlap_matches = sum(
                    [1 for i in matched_cds['overlap_match'] if i[2] > confidence_threshold['overlap_match']])

                if num_overlap_matches >= perfect_match_gap:
                    if process_log:
                        all_result.append([i+("perfect_match",) for i in matched_cds['perfect_match']] +
                                          [i+("overlap_match",) for i in matched_cds['overlap_match'][:perfect_match_gap]])
                    else:
                        all_result.append([i[:2] for i in matched_cds['perfect_match'][:num_perfect_matches]] +
                                          [i[:2] for i in matched_cds['overlap_match'][:perfect_match_gap]])
                else:
                    overlap_match_gap = perfect_match_gap - num_overlap_matches
                    num_perfect_matches_lower = len(
                        matched_cds['perfect_match']) - num_perfect_matches

                    if num_perfect_matches_lower >= overlap_match_gap:
                        if process_log:
                            all_result.append([i+("perfect_match",) for i in matched_cds['perfect_match'][:num_perfect_matches+overlap_match_gap]] +
                                              [i+("overlap_match",) for i in matched_cds['overlap_match']])
                        else:
                            all_result.append([i[:2] for i in matched_cds['perfect_match'][:num_perfect_matches+overlap_match_gap]] +
                                              [i[:2] for i in matched_cds['overlap_match']])
                    else:
                        pm_lower_gap = overlap_match_gap - num_perfect_matches_lower
                        num_overlap_matches_lower = len(
                            matched_cds['overlap_match']) - num_overlap_matches

                        if num_overlap_matches_lower >= pm_lower_gap:
                            if process_log:
                                all_result.append([i+("perfect_match",) for i in matched_cds['perfect_match']] +
                                                  [i+("overlap_match",) for i in matched_cds['overlap_match'][:num_overlap_matches+pm_lower_gap]])
                            else:
                                all_result.append([i[:2] for i in matched_cds['perfect_match']] +
                                                  [i[:2] for i in matched_cds['overlap_match'][:num_overlap_matches+pm_lower_gap]])
                        else:
                            om_lower_gap = pm_lower_gap - num_overlap_matches_lower
                            num_unilateral = sum(
                                [1 for i in matched_cds['unilateral'] if i[2] > confidence_threshold['unilateral']])

                            if num_unilateral >= om_lower_gap:
                                if process_log:
                                    all_result.append([i+("perfect_match",) for i in matched_cds['perfect_match']] +
                                                      [i+("overlap_match",) for i in matched_cds['overlap_match']] +
                                                      [i+("unilateral",) for i in matched_cds['unilateral'][:om_lower_gap]])
                                else:
                                    all_result.append([i[:2] for i in matched_cds['perfect_match']] +
                                                      [i[:2] for i in matched_cds['overlap_match']] +
                                                      [i[:2] for i in matched_cds['unilateral'][:om_lower_gap]])
                            else:
                                unilateral_gap = om_lower_gap - num_unilateral
                                num_unilateral_lower = len(
                                    matched_cds['unilateral']) - num_unilateral

                                if num_unilateral_lower >= unilateral_gap:
                                    if process_log:
                                        all_result.append([i+("perfect_match",) for i in matched_cds['perfect_match']] +
                                                          [i+("overlap_match",) for i in matched_cds['overlap_match']] +
                                                          [i+("unilateral",) for i in matched_cds['unilateral'][:num_unilateral+unilateral_gap]])
                                    else:
                                        all_result.append([i[:2] for i in matched_cds['perfect_match']] +
                                                          [i[:2] for i in matched_cds['overlap_match']] +
                                                          [i[:2] for i in matched_cds['unilateral'][:num_unilateral+unilateral_gap]])
                                else:
                                    u_lower_gap = unilateral_gap - num_unilateral_lower
                                    num_isolated = sum(
                                        [1 for i in matched_cds['isolated'] if i[2] > confidence_threshold['isolated']])

                                    if num_isolated >= u_lower_gap:
                                        if process_log:
                                            all_result.append([i+("perfect_match",) for i in matched_cds['perfect_match']] +
                                                              [i+("overlap_match",) for i in matched_cds['overlap_match']] +
                                                              [i+("unilateral",) for i in matched_cds['unilateral']] +
                                                              [i+("isolated",) for i in matched_cds['isolated'][:u_lower_gap]])
                                        else:
                                            all_result.append([i[:2] for i in matched_cds['perfect_match']] +
                                                              [i[:2] for i in matched_cds['overlap_match']] +
                                                              [i[:2] for i in matched_cds['unilateral']] +
                                                              [i[:2] for i in matched_cds['isolated'][:u_lower_gap]])
                                    else:
                                        isolated_gap = u_lower_gap - num_isolated
                                        if process_log:
                                            all_result.append([i+("perfect_match",) for i in matched_cds['perfect_match']] +
                                                              [i+("overlap_match",) for i in matched_cds['overlap_match']] +
                                                              [i+("unilateral",) for i in matched_cds['unilateral']] +
                                                              [i+("isolated",) for i in matched_cds['isolated'][:num_isolated+isolated_gap]])
                                        else:
                                            all_result.append([i[:2] for i in matched_cds['perfect_match']] +
                                                              [i[:2] for i in matched_cds['overlap_match']] +
                                                              [i[:2] for i in matched_cds['unilateral']] +
                                                              [i[:2] for i in matched_cds['isolated'][:num_isolated+isolated_gap]])
        return all_result

    def _select_matched_coords_strategy(self, matched_coords, process_log=False):
        return self.select_matched_coords_naive_strategy(matched_coords, num_fish=self.num_fish,
                                                         confidence_threshold=self.confidence_threshold,
                                                         process_log=process_log)

    def plot_tradeoff(self, predictions, test_coords,
                      enumerate_peak_threshold_range=(1, 10),
                      enumerate_confidence_threshold_range=(1, 10),
                      division=10):
        """
        Plot the tradeoff between precision and recall for different peak and confidence thresholds.
        Args:
            predictions (torch.Tensor): Predictions of the model. Shape: (n, 10, height, width).
            test_coords (list): List of ground truth coordinates. Each element is a list of numpy arrays of shape (n, 2).
            enumerate_peak_threshold_range (tuple): Tuple of two integers. The range of peak thresholds to be enumerated.
            enumerate_confidence_threshold_range (tuple): Tuple of two integers. The range of confidence thresholds to be enumerated.
        """

        previous_peak_threshold = self.peak_threshold
        previous_confidence_threshold = self.confidence_threshold

        head_precision_matrix = np.zeros((enumerate_peak_threshold_range[1] - enumerate_peak_threshold_range[0],
                                          enumerate_confidence_threshold_range[1] - enumerate_confidence_threshold_range[0]))
        head_recall_matrix = np.zeros((enumerate_peak_threshold_range[1] - enumerate_peak_threshold_range[0],
                                       enumerate_confidence_threshold_range[1] - enumerate_confidence_threshold_range[0]))
        head_average_matrix = np.zeros((enumerate_peak_threshold_range[1] - enumerate_peak_threshold_range[0],
                                        enumerate_confidence_threshold_range[1] - enumerate_confidence_threshold_range[0]))
        middle_precision_matrix = np.zeros((enumerate_peak_threshold_range[1] - enumerate_peak_threshold_range[0],
                                            enumerate_confidence_threshold_range[1] - enumerate_confidence_threshold_range[0]))
        middle_recall_matrix = np.zeros((enumerate_peak_threshold_range[1] - enumerate_peak_threshold_range[0],
                                        enumerate_confidence_threshold_range[1] - enumerate_confidence_threshold_range[0]))
        middle_average_matrix = np.zeros((enumerate_peak_threshold_range[1] - enumerate_peak_threshold_range[0],
                                          enumerate_confidence_threshold_range[1] - enumerate_confidence_threshold_range[0]))
        total_average_matrix = np.zeros((enumerate_peak_threshold_range[1] - enumerate_peak_threshold_range[0],
                                         enumerate_confidence_threshold_range[1] - enumerate_confidence_threshold_range[0]))

        # Create an iterator for tqdm
        iterator = []

        for p_t in range(*enumerate_peak_threshold_range):
            for c_t in range(max(enumerate_confidence_threshold_range[0], p_t),
                             enumerate_confidence_threshold_range[1]):
                iterator.append((p_t, c_t))
        for p_t, c_t in tqdm(iterator):
            self.peak_threshold = p_t / division
            self.confidence_threshold = {'perfect_match': c_t / division,
                                         'overlap_match': c_t / division,
                                         'unilateral': c_t / division,
                                         'isolated': c_t / division}
            metrics = self.evaluate(
                predictions, test_coords, detailed=False)
            head_precision_matrix[p_t - enumerate_peak_threshold_range[0],
                                  c_t - enumerate_confidence_threshold_range[0]] = metrics['head_precision']
            head_recall_matrix[p_t - enumerate_peak_threshold_range[0],
                               c_t - enumerate_confidence_threshold_range[0]] = metrics['head_recall']
            head_average_matrix[p_t - enumerate_peak_threshold_range[0],
                                c_t - enumerate_confidence_threshold_range[0]] = (metrics['head_precision'] + metrics['head_recall']) / 2
            middle_precision_matrix[p_t - enumerate_peak_threshold_range[0],
                                    c_t - enumerate_confidence_threshold_range[0]] = metrics['middle_precision']
            middle_recall_matrix[p_t - enumerate_peak_threshold_range[0],
                                 c_t - enumerate_confidence_threshold_range[0]] = metrics['middle_recall']
            middle_average_matrix[p_t - enumerate_peak_threshold_range[0],
                                  c_t - enumerate_confidence_threshold_range[0]] = (metrics['middle_precision'] + metrics['middle_recall']) / 2
            total_average_matrix[p_t - enumerate_peak_threshold_range[0],
                                 c_t - enumerate_confidence_threshold_range[0]] = (metrics['head_precision'] + metrics['head_recall'] + metrics['middle_precision'] + metrics['middle_recall']) / 4

        self.peak_threshold = previous_peak_threshold
        self.confidence_threshold = previous_confidence_threshold

        # Set all elements in the matrixes to min(matrix) if they are 0
        head_precision_matrix[head_precision_matrix == 0] = np.min(
            head_precision_matrix[head_precision_matrix != 0])
        head_recall_matrix[head_recall_matrix == 0] = np.min(
            head_recall_matrix[head_recall_matrix != 0])
        head_average_matrix[head_average_matrix == 0] = np.min(
            head_average_matrix[head_average_matrix != 0])
        middle_precision_matrix[middle_precision_matrix == 0] = np.min(
            middle_precision_matrix[middle_precision_matrix != 0])
        middle_recall_matrix[middle_recall_matrix == 0] = np.min(
            middle_recall_matrix[middle_recall_matrix != 0])
        middle_average_matrix[middle_average_matrix == 0] = np.min(
            middle_average_matrix[middle_average_matrix != 0])
        total_average_matrix[total_average_matrix == 0] = np.min(
            total_average_matrix[total_average_matrix != 0])

        # Plot the 6 matrixes excluding the total average matrix
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Precision-Recall Tradeoff', fontsize=16)
        axs[0, 0].set_title('Head Precision')
        axs[0, 0].imshow(head_precision_matrix, cmap='hot',
                         interpolation='nearest')
        axs[0, 0].set_xlabel('Confidence Threshold')
        axs[0, 0].set_ylabel('Peak Threshold')
        axs[0, 1].set_title('Head Recall')
        axs[0, 1].imshow(head_recall_matrix, cmap='hot',
                         interpolation='nearest')
        axs[0, 1].set_xlabel('Confidence Threshold')
        axs[0, 1].set_ylabel('Peak Threshold')
        axs[0, 2].set_title('Head Average')
        axs[0, 2].imshow(head_average_matrix, cmap='hot',
                         interpolation='nearest')
        axs[0, 2].set_xlabel('Confidence Threshold')
        axs[0, 2].set_ylabel('Peak Threshold')
        axs[1, 0].set_title('Middle Precision')
        axs[1, 0].imshow(middle_precision_matrix,
                         cmap='hot', interpolation='nearest')
        axs[1, 0].set_xlabel('Confidence Threshold')
        axs[1, 0].set_ylabel('Peak Threshold')
        axs[1, 1].set_title('Middle Recall')
        axs[1, 1].imshow(middle_recall_matrix, cmap='hot',
                         interpolation='nearest')
        axs[1, 1].set_xlabel('Confidence Threshold')
        axs[1, 1].set_ylabel('Peak Threshold')
        axs[1, 2].set_title('Middle Average')
        axs[1, 2].imshow(middle_average_matrix, cmap='hot',
                         interpolation='nearest')
        axs[1, 2].set_xlabel('Confidence Threshold')
        axs[1, 2].set_ylabel('Peak Threshold')

        plt.show()

        # Print the highest average value of head, middle and total
        # Then print the corresponding peak and confidence thresholds
        print('Highest Head Average:', format_percentage(
            np.max(head_average_matrix)))
        print('Highest Middle Average:', format_percentage(
            np.max(middle_average_matrix)))
        print('Highest Total Average:', format_percentage(
            np.max(total_average_matrix)))
        print('Corresponding Head Precision & Recall: ',
              format_percentage(head_precision_matrix[np.unravel_index(
                  np.argmax(total_average_matrix), total_average_matrix.shape)]),
              format_percentage(head_recall_matrix[np.unravel_index(
                  np.argmax(total_average_matrix), total_average_matrix.shape)]))
        print('Corresponding Middle Precision & Recall: ',
              format_percentage(middle_precision_matrix[np.unravel_index(
                  np.argmax(total_average_matrix), total_average_matrix.shape)]),
              format_percentage(middle_recall_matrix[np.unravel_index(
                  np.argmax(total_average_matrix), total_average_matrix.shape)]))
        print('Corresponding Peak Threshold & Confidence Threshold: ',
              (np.unravel_index(np.argmax(total_average_matrix), total_average_matrix.shape)[
               0] + enumerate_peak_threshold_range[0]) / division,
              (np.unravel_index(np.argmax(total_average_matrix), total_average_matrix.shape)[1] + enumerate_confidence_threshold_range[0]) / division)

        return head_precision_matrix, head_recall_matrix, head_average_matrix, middle_precision_matrix, middle_recall_matrix, middle_average_matrix, total_average_matrix
    
from collections import Counter

class TrackerV0:

    def __init__(self,
                 num_keypoints=7,
                 distance_threshold_per_frame=1.5,
                 min_trajectory_length=2,
                 momentum_window=5):
        """
        Args:
            num_keypoints: The number of keypoints to be tracked.
            distance_threshold_per_frame: The maximum distance a keypoint can travel between frames.
        """
        self.num_keypoints = num_keypoints
        self.distance_threshold_per_frame = distance_threshold_per_frame
        self.min_trajectory_length = min_trajectory_length
        self.momentum_window = momentum_window
        self._build_initialized = False
        self._connect_initialized = False
    
    def initialize_build(self, initial_positions):
        """
        Args:
            initial_positions: The initial positions of the keypoints to be tracked.
            starting_frame: The frame number of the initial positions.
        """

        if self._build_initialized:
            raise ValueError("The tracker has already been initialized.")
        
        # Check if the shape of the initial_positions matches each other.
        if len(set([initial_position.shape for initial_position in initial_positions])) != 1:
            raise ValueError("The shape of the initial positions should match each other.")
        
        self.shape = initial_positions[0].shape
        self.trajectories = [
            {'trajectory': Trajectory(initial_position),
             'start': 0,
             'end': None,
             'merge': None,
             'connect_start': None,
             'connect_end': None}
            for initial_position in initial_positions
        ] 

        self.trajectory_timeline = [[id for id in range(len(initial_positions))]]

        self._build_initialized = True
    
    def step_build_trajectories(self, new_positions):
        """
        Args:
            new_positions: The new positions of the keypoints to be tracked.
        """
        if not self._build_initialized:
            raise ValueError("The tracker has not been initialized yet.")
        
        temp_timeline = []
        frame_number = len(self.trajectory_timeline)
        
        new_positions_status = [None for _ in range(len(new_positions))]
        
        for trajectory_id in self.trajectory_timeline[-1]:
            distances = self.trajectories[trajectory_id]['trajectory'].calculate_distances_momentum(new_positions,
                                                                                                    window=self.momentum_window)
            closest_id = np.argmin(distances)
            if distances[closest_id] < self.distance_threshold_per_frame:
                if new_positions_status[closest_id] is not None:
                    # Mark the trajectory as merged if the point is already connected to another trajectory.
                    self.trajectories[trajectory_id]['end'] = frame_number
                    self.trajectories[trajectory_id]['merge'] = new_positions_status[closest_id]
                else:
                    self.trajectories[trajectory_id]['trajectory'].append(new_positions[closest_id])
                    temp_timeline.append(trajectory_id)
                    new_positions_status[closest_id] = trajectory_id
            else:
                self.trajectories[trajectory_id]['end'] = frame_number
        
        for new_position_id, status in enumerate(new_positions_status):
            if status is None:
                # Check if the shape of the new_position matches the existing trajectories.
                if new_positions[new_position_id].shape != self.shape:
                    raise ValueError(
                        "The shape of the new position does not match the existing trajectories."
                    )
                self.trajectories.append({'trajectory': Trajectory(new_positions[new_position_id]),
                                            'start': frame_number,
                                            'end': None,
                                            'merge': None,
                                            'connect_start': None,
                                            'connect_end': None})
                temp_timeline.append(len(self.trajectories) - 1)
        
        self.trajectory_timeline.append(temp_timeline)
    
    def build_trajectories(self, coords):
        """
        Args:
            coords: List of numpy arrays with shape (num_positions, dimension).
        """
        self.initialize_build(coords[0])

        for frame in coords[1:]:
            self.step_build_trajectories(frame)
        
        for trajectory in self.trajectories:
            if trajectory['end'] == None:
                trajectory['end'] = len(self.trajectory_timeline)

    def initialize_connect(self, initial_frame):
        if self._connect_initialized:
            raise ValueError("The tracker has already been initialized.")
        
        self.reconstructed_trajectories = ReconstructedTrajectories(num_entities=self.num_keypoints)

        # RL = end_frame_index - current_frame_index: The remaining length of the trajectory.
        # We assume that trajectories with longer RL are more likely to be correct.
        sort_RL = list(sorted([(traj_id, self.trajectories[traj_id]['end'])
                        for traj_id in initial_frame], key=lambda x: x[1], reverse=True))
        for entity_id, (traj_id, _) in enumerate(sort_RL[:self.num_keypoints]):
            self.reconstructed_trajectories.append(entity_id, self.trajectories[traj_id]['trajectory'])
    
    def connect_trajectories(self):
        
        # Determine the first frame
        self.initialize_connect(self.trajectory_timeline[0])

        for frame_index, frame in enumerate(self.trajectory_timeline[1:]):
            self.step_connect_trajectories(frame_index + 1, frame)
        
    def track(self, matched_coords):
        """
        Args:
            matched_coords: List of dictionaries. Each dictionary contains the following keys:
                'perfect_match': List of tuples. Tuples are in the format of (head_coord, middle_coord, confidence).
                'overlap_match': List of tuples. Tuples are in the same format as above.
                'unilateral': List of tuples. Tuples are in the same format as above.
                'isolated': List of tuples. Tuples are in the same format as above.
                The matched coordinates is from zf.inference.keypoint.DoublecheckKeypoint
        """
        # Convert the matched_coords to a list of numpy arrays.
        matched_coords = [[np.hstack([coord[0], coord[1]])
                           for coords in list(matched_coord.values()) for coord in coords]
                           for matched_coord in matched_coords]
        
        self.build_trajectories(matched_coords)

        # self.connect_trajectories()
    
    def visualize_trajectories_length_distribution(self):
        trajectory_lengths = [trajectory['end'] - trajectory['start'] for trajectory in self.trajectories]
        
        counter = Counter(trajectory_lengths)
        sorted_items = sorted(counter.items())
        keys, values = zip(*sorted_items)
        keys = [str(key) for key in keys]
        plt.bar(keys, values)
        plt.title('Number Occurrences')
        plt.xlabel('Number')
        plt.ylabel('Occurrences')
        plt.show()

        return list(sorted_items)

    def visualize_timeline(self, min_trajectory_length=2):
        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 6))
        trajectory_ids = []

        # Add bars for the trajectory lifespans
        for (idx, start, duration) in zip(range(len(self.trajectories)),
                                          [trajectory['start']
                                           for trajectory in self.trajectories],
                                          [trajectory['end'] - trajectory['start']
                                           for trajectory in self.trajectories]):
            if duration >= min_trajectory_length:
                ax.broken_barh([(start, duration)],
                               (len(trajectory_ids) - 0.4, 0.8),
                               facecolors='tab:blue')
                trajectory_ids.append(idx)
            
        # Set the y-axis labels to the trajectory IDs
        ax.set_yticks(range(len(trajectory_ids)))
        ax.set_yticklabels(trajectory_ids)

        # Set the rest of the labels and title
        ax.set_xlabel('Frame')
        ax.set_ylabel('Trajectory ID')
        ax.set_title('Timelime Visualization')

        # Show grid and the plot
        ax.grid(True)
        plt.show()

    
class ConnectedTrajectories:
    def __init__(self, initial_position):
        """
        Args:
            initial_position: The starting position of the trajectories.
        """
        # Check if the initial_coord represents several points.
        if len(initial_position[0].shape) != 1:
            raise ValueError(
                f"The initial coordinates should represent several points, but got shape {initial_position.shape}."
            )
        self.trajectories = [initial_position]
        self.num_entities = len(initial_position)
        self.frame_state = [True for _ in range(len(initial_position))]
        self.shape = initial_position[0].shape

    def __len__(self):
        return len(self.trajectories)
    
    def frame(self, frame_index):
        """
        Args:
            frame_index: The index of the frame to be returned.
        Returns:
            The positions of all entities in the frame.
        """
        return self.trajectories[frame_index]
    
    def id(self, entity_index):
        """
        Args:
            entity_index: The index of the entity to be returned.
        Returns:
            The trajectory of the entity in all frames.
        """
        return [trajectory[entity_index] for trajectory in self.trajectories]

    def append(self, entity_id, new_position):
        """
        A consistency check is implemented to ensure that the new position is connected to the last frame.
        Args:
            entity_id: The index of the entity to be connected to the existing trajectory.
            new_position: A new point to be connected to the existing trajectory. This should be a 1D array or None.
        """
        # Check if the new_coords matches the dimension of the existing trajectory.
        if new_position is not None and new_position.shape != self.shape:
            raise ValueError(
                f"The new coordinates with shape {new_position.shape}"
                "does not match the dimension of the existing trajectory with shape {self.shape}."
            )
        
        # Check if the last frame is complete and append a new frame if it is.
        if all(self.frame_state):
            self.trajectories.append([None for _ in range(self.num_entities)])
            self.frame_state = [False for _ in range(self.num_entities)]
        
        # Check if the entity has already been appended to the last frame.
        if self.frame_state[entity_id] == False:
            self.trajectories[-1][entity_id] = new_position
        else:
            raise ValueError(
                f"The entity with id {entity_id} has already been appended to the last frame."
            )
    
    def append_frame(self, new_frame):
        """
        Args:
            new_frame: A new frame to be connected to the existing trajectories.
        """
        # Check if the new_frame matches the dimension of the existing trajectories.
        if not all([position.shape == self.shape for position in new_frame]):
            raise ValueError(
                f"The new frame with shape {new_frame.shape}"
                "does not match the dimension of the existing trajectories with shape {self.shape}."
            )
        elif len(new_frame) != self.num_entities:
            raise ValueError(
                f"The number of entities in the new frame {len(new_frame)} does not match the number of entities in the existing trajectories {self.num_entities}."
            )
        self.trajectories.append(new_frame)
        self.frame_state = [True for _ in range(self.num_entities)]

    def calculate_distances(self, entity_id, new_positions):
        """
        Consider both the head position and the midsection position.
        """
        return np.linalg.norm(self.trajectories[-1][entity_id] - new_positions, axis=1)
    
    def calculate_distances_momentum(self, entity_id, new_positions, window=5):
        """
        Consider the head position, the midsection position, 
        and the momentum of the trajectory based on a moving average of the last `window` positions.

        Args:
            entity_id: The index of the entity to be connected to the existing trajectory.
            new_positions: A new point to be compared to the predicted next position.
            window: The number of positions to consider for the moving average.

        Returns:
            The distance between the predicted next position and the new position.
        """
        trajectory_length = len(self.trajectories)
        
        # Fall back to the regular distance calculation if the trajectory is too short.
        if trajectory_length <= 1 or window == 0:
            return self.calculate_distances(entity_id, new_positions)
        
        window = min(window, trajectory_length - 1)
        velocities = [self.trajectories[-1-i][entity_id] - self.trajectories[-2-i][entity_id] for i in range(0, window)]
        avg_velocity = np.sum(velocities, axis=0) / window
        momentum_pred = self.trajectories[-1][entity_id] + avg_velocity

        return np.linalg.norm(momentum_pred - new_positions, axis=1)
    
"""
def initialize_connect(self, frame):
        if not self._build_initialized:
            raise ValueError("The tracker has not been initialized yet.")
        elif self._connect_initialized:
            raise ValueError("The tracker has already been initialized for connecting trajectories.")
        
        trajectory_duration = [(id, self.trajectories[id]['end'] - self.trajectories[id]['start'])
                               for id in frame]
        trajectory_duration.sort(key=lambda x: x[1], reverse=True) # Sort by duration in descending order.

        if len(trajectory_duration) >= self.num_keypoints:

            self.connect_timeline = [[]]
            
            initial_positions = []
            for key in range(self.num_keypoints):
                # We assume that longer trajectories are more likely to be correct.
                initial_positions.append(self.trajectories[trajectory_duration[key][0]]['trajectory'][0])
                self.connect_timeline[-1].append(trajectory_duration[key][0])
                self.trajectories[trajectory_duration[key][0]]['connect_start'] = 0
            
            self.connect_trajectories = ConnectedTrajectories(initial_positions)

        else:
            raise ValueError("The number of trajectories is less than the number of keypoints during initialization.")
        
        self._connect_initialized = True
    
    def step_connect_trajectories(self, frame):

        if not self._connect_initialized:
            raise ValueError("The tracker has not been initialized for connecting trajectories yet.")
        
        # During the trajectory build, there are 3 scenarios:
        # Notations:
        #  | N_Enti: The number of entities to be tracked
        #  | N_Pred: The number of predicted trajectories in the current frame.
        #  | N_CP: The number of predicted trajectories in the current frame that is connected
        #  |       to an entity in the previous frame, according to momentum nearest neighbor.
        #  | N_CE: The number of entities in the previous frame that is connected to a trajectory
        #  |       in the current frame, according to momentum nearest neighbor.
        #  | -- Note that N_CP and N_CE are not the same, since two trajectory can be connected to
        #  |    one entities.
        #  | RL = end_frame_index - current_frame_index: The remaining length of the trajectory.
        #  | -- We assume that trajectories with longer RL are more likely to be correct.
        #  |
        # 1. True positive: Given N entities, N trajectories are predicted in the current frame,
        #  | and each trajectory is correctly connected to an entity in the previous frame,
        #  | according to momentum nearest neighbor.
        # 2. False positive: Given N entities, M trajectories are predicted in the current frame, where M > N.
        #  | In this case, there are M - N extra trajectories that are not connected to any entity in the previous frame.
        #  | We assume that longer trajectories (starting from the current frame) are more likely to be correct.
        #  | Thus, we evaluate whether the 
        # The trajectory connection is done with 3 steps:
        # 1. Locate the longest trajectories that is active in the current frame.
        #  | Check if these trajectories are active in the previous frame.
        #  | If they are, connect them to the existing  corresponding entity trajectory.
        #  | If they are not, append them to the disconnected trajectories for later connection.
        # 2. 
        
        
        temp_timeline = [None for _ in range(self.num_keypoints)]
        frame_number = len(self.connect_timeline)

        trajectory_duration = [(id, self.trajectories[id]['end'] - self.trajectories[id]['start'])
                                 for id in frame]
        trajectory_duration.sort(key=lambda x: x[1], reverse=True)

        disconnected_traj_ids = []

        for trajectory_id, _ in trajectory_duration[:self.num_keypoints]:
            if trajectory_id in self.connect_timeline[-1]:
                entity_id = self.connect_timeline[-1].index(trajectory_id)
                temp_timeline[entity_id] = trajectory_id
                trajectory_index = frame_number - self.trajectories[trajectory_id]['start']
                position = self.trajectories[trajectory_id]['trajectory'][trajectory_index]
                self.connect_trajectories.append(entity_id, position)
            else:
                disconnected_traj_ids.append(trajectory_id)

"""

# Test code for the ReconstructedTrajectory class

"""
import numpy as np

# Mocking a simple Trajectory class since it's referenced in the append method
class Trajectory:
    def __init__(self, trajectory):
        self.trajectory = trajectory

    def __getitem__(self, index):
        return self.trajectory[index]

# Create an instance of ReconstructedTrajectories
reconstructed_trajectories = ReconstructedTrajectories(num_entities=3)

# Adding mock trajectories to entity_id 1
trajectories_to_add = [
    Trajectory([None, None]),
    Trajectory([None, None]),
    Trajectory([np.array([5, 5]), None])
]

# Append these trajectories to our reconstructed_trajectories instance
for traj in trajectories_to_add:
    reconstructed_trajectories.append(entity_id=1, trajectory=traj)

# Define new mock trajectories to compare against the existing
# Let's assume we are checking against two new trajectory starting points
new_trajectories = [Trajectory([
    np.array([7, 7]),  # A logical continuation of the trajectory
    np.array([10, 10]) # A point further away
])]

# Test the calculate_distances_momentum function for entity_id 1
distances = reconstructed_trajectories.calculate_distances_momentum(
    entity_id=1,
    new_trajectories=new_trajectories,
    window=5
)

print(distances)  # Expected output: array with distances from momentum_pred to each new_trajectory
"""