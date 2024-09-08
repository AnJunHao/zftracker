from .trajectory import TrajectoryPoolV2, JoinedTrajectoryV2, distance_to_ray
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..criterion.graph import get_subgraphs_from_mapping, analyze_full_graph
from ..preprocess.evaluate_annotation_v2 import calculate_absolute_radian, find_arc_center, is_clockwise
import networkx as nx
from collections import Counter
from icecream import ic
from ..util.tqdm import TQDM as tqdm
from time import time
from scipy.optimize import root_scalar

def arc_angle_equation(theta, arc_length, ht_distance):
    k = (ht_distance ** 2) / (2 * arc_length ** 2)
    return theta**2 * k - 1 + np.cos(theta)

def solve_arc_angle(arc_length, mid_tail_distance):
    if arc_length >= mid_tail_distance * np.pi / 2:
        bracket = [np.pi, 2 * np.pi]
    else:
        bracket = [0, np.pi]
    sol = root_scalar(arc_angle_equation, args=(arc_length, mid_tail_distance), bracket=bracket)
    return sol.root

def get_pred_hm_absolute_angles(gt_hm_absolute_angles, arc_angles, exponent_base, rescale, moving_window, return_stabalizer, gap=1):
    """
    This function is intended to test the performance by calling on a complete trajectory.
    This should not be used in the actual tracking process.
    In our preliminary test, we recommend the following parameters on 120fps data:
        exponent_base=0.26
        rescale=0.27
        moving_window=10
        return_stabalizer=0.17
    Args:
        gt_hm_absolute_angles: The ground truth head-mid absolute angles.
        arc_angles: The arc angles of the fish.
        exponent_base: The base of the exponent.
        rescale: The rescale factor. The moving sum is multiplied by this factor.
        moving_window: The moving window.
        return_stabalizer: The return stabalizer, the value is between 0 and 1.
        gap: The gap between the ground truth and the predicted angles.
    Returns:
        The predicted head-mid absolute angles.
    """

    predicted_hm_absolute_angles = [[] for _ in range(gap)]

    arc_angle_diff = angle_diffs(arc_angles)

    for i in range(gap):
        for index in range(len(gt_hm_absolute_angles)):
            moving_sum = 0
            for moving_index in range(index, index-moving_window, -1):
                if moving_index < 0:
                    break
                moving_sum += arc_angle_diff[moving_index] * exponent_base ** (index-moving_index)
            if index == 0:
                previous_angle = 0
                stabalizer = 1
            else:
                if i == 0:
                    previous_angle = gt_hm_absolute_angles[index-1]
                else:
                    previous_angle = predicted_hm_absolute_angles[i-1][index-1]
                current_arc_angle = arc_angles[index]
                current_arc_angle_diff = arc_angle_diff[index]
                if current_arc_angle * current_arc_angle_diff >=0 :
                    stabalizer = 1
                else:
                    stabalizer = return_stabalizer
            predicted_hm_absolute_angles[i].append(previous_angle + moving_sum * rescale * stabalizer)
    
    return np.array(predicted_hm_absolute_angles[-1])
    
def predict_head_midsec_absolute_angle(trajectory: np.ndarray, # Also supports TrajectoryV3 object
                                        last_arc_angle: float,
                                        exponent_base: float,
                                        rescale: float,
                                        moving_window: int,
                                        return_stabalizer: float):
    """
    This function is used to predict the head-mid absolute angles of a fish given the trajectory in the actual tracking process.
    Args:
        trajectory (np.ndarray or TrajectoryV3): The trajectory of the fish.
        exponent_base (float): The base of the exponent.
        rescale (float): The rescale factor. The moving sum is multiplied by this factor.
        moving_window (int): The moving window.
        return_stabalizer (float): The return stabalizer, the value is between 0 and 1.
    Returns:
        The predicted head-mid absolute angle in the next frame.
    """

    previous_positions = trajectory[-moving_window:] # May get less than moving_window positions
    if len(previous_positions) == 0:
        raise ValueError('No available previous positions.')
    arc_angles = stable_get_arc_angles(previous_positions) # Length is moving_window (or len(previous_positions))

    """
    # We don't need this part, because we are using stable_get_arc_angles
    if last_arc_angle - arc_angles[-1] > np.pi: # last_arc_angle is too large
        # diff = last_arc_angle - arc_angles[-1] > np.pi
        # np.pi < diff < 3 * np.pi ----> last_arc_angle - 2 * np.pi
        # 3 * np.pi < diff < 5 * np.pi ----> last_arc_angle - 4 * np.pi
        last_arc_angle -= ((last_arc_angle - arc_angles[-1] + np.pi) // (2 * np.pi)) * 2 * np.pi
    elif arc_angles[-1] - last_arc_angle > np.pi: # last_arc_angle is too small
        # diff = arc_angles[-1] - last_arc_angle > np.pi
        # np.pi < diff < 3 * np.pi ----> last_arc_angle + 2 * np.pi
        # 3 * np.pi < diff < 5 * np.pi ----> last_arc_angle + 4 * np.pi
        last_arc_angle += ((arc_angles[-1] - last_arc_angle + np.pi) // (2 * np.pi)) * 2 * np.pi

    if abs(last_arc_angle - arc_angles[-1]) > np.pi:
        raise ValueError(f'last_arc_angle: {last_arc_angle}, arc_angles[-1]: {arc_angles[-1]}')
    """
    
    arc_angles.append(last_arc_angle) # Length is moving_window + 1 (or len(previous_positions) + 1)
    arc_angles_diff = angle_diffs(arc_angles, keep_length=False) # Length is moving_window (or len(previous_positions))

    moving_sum = 0
    for moving_index in range(len(arc_angles_diff)):
        moving_sum += arc_angles_diff[-1 - moving_index] * exponent_base ** moving_index
    
    previous_hm_angle = get_head_mid_angles([previous_positions[-1]])[0]
    current_arc_angle = arc_angles[-1]
    current_arc_angle_diff = arc_angles_diff[-1]
    if current_arc_angle * current_arc_angle_diff >= 0:
        # The tail angle is high and keeps going up (or low and keeps going down)
        # This means that the is a dramatic change in head-mid angle and
        # the tail angle is following the change
        stabalizer = 1
    else:
        # The arc angle is still high but is going down (or low but going back to 0)
        # We need to lower the moving_sum value, because the tail angle
        # slowly goes back to 0 after a dramatic change in head-mid angle.
        # This going-back-to-0 process does not mean a change in head-mid angle.
        stabalizer = return_stabalizer # The value is between 0 and 1, usually below 0.5
    
    # print(f'previous_hm_angle: {previous_hm_angle}, arc_angles_diff:{arc_angles_diff[-3:]}, moving_sum: {moving_sum}, rescale: {rescale}, stabalizer: {stabalizer}')

    return previous_hm_angle + moving_sum * rescale * stabalizer

def get_pred_arc_angles(head_midsec_angles, gt_arc_angles, exponent_base, rescale, moving_window, gap=1, return_factor=1):
    """
    This function is intended to test the performance by calling on a complete trajectory.
    This should not be used in the actual tracking process.
    In our preliminary test, we recommend the following parameters on 120fps data:
        exponent_base=0.93;
        rescale=2.51;
        moving_window=60; # Higher, better
        return_factor=1.18.
    Args:
        head_midsec_angles: The head-midsec angles.
        gt_arc_angles: The ground truth arc angles.
        exponent_base: The base of the exponent.
        rescale: The rescale factor. The moving sum is multiplied by this factor.
        moving_window: The moving window.
        gap: The gap between the ground truth and the predicted angles.
        return_factor: The return factor, the value should be larger than 1.
    Returns:
        The predicted arc angles.
    """

    predicted_arc_angles = [[] for _ in range(gap)]

    head_midsec_accels = angle_accelerations(head_midsec_angles)

    for i in range(gap):
        for index in range(len(head_midsec_accels)):
            moving_sum = 0
            for moving_index in range(index, index-moving_window, -1):
                if moving_index < 0:
                    break
                moving_sum += head_midsec_accels[moving_index] * exponent_base**(index-moving_index)
            if index == 0:
                previous_arc_angle = 0
            else:
                if i == 0:
                    previous_arc_angle = gt_arc_angles[index-1]
                else:
                    previous_arc_angle = predicted_arc_angles[i-1][index-1]
            if previous_arc_angle * moving_sum >= 0:
                predicted_arc_angles[i].append(previous_arc_angle + moving_sum * rescale)
            else:
                # The previous arc angle and the moving sum are in different directions
                # We will accelerate the move
                predicted_arc_angles[i].append(previous_arc_angle + moving_sum * rescale * return_factor)

    return np.array(predicted_arc_angles[-1])

def predict_arc_angle(trajectory: np.ndarray, # Also supports TrajectoryV3 object
                        last_hm_angle: float,
                        exponent_base: float,
                        rescale: float,
                        moving_window: int,
                        return_factor: float):
    """
    This function is used to predict the arc angles of a fish given the trajectory in the actual tracking process.
    Args:
        trajectory (np.ndarray or TrajectoryV3): The trajectory of the fish.
        exponent_base (float): The base of the exponent.
        rescale (float): The rescale factor. The moving sum is multiplied by this factor.
        moving_window (int): The moving window.
        return_factor (float): The return factor, the value should be larger than 1.
    Returns:
        The predicted arc angle in the next frame.
    """
    
    previous_positions = trajectory[-moving_window-1:] # Get one more for calculating acceleration
    if len(previous_positions) == 0:
        raise ValueError('No available previous positions.')
    
    head_midsec_angles = get_head_mid_angles(previous_positions)
    head_midsec_angles.append(last_hm_angle) # Length is moving_window + 1 (or len(previous_positions) + 1)
    head_midsec_accels = angle_accelerations(head_midsec_angles) # Length is moving_window (or len(previous_positions))

    moving_sum = 0
    for moving_index in range(len(head_midsec_accels) - 1, len(head_midsec_accels) - 1 - moving_window, -1):
        if moving_index < 0:
            break
        moving_sum += head_midsec_accels[moving_index] * exponent_base ** (len(head_midsec_accels) - 1 - moving_index)
    
    # TODO Check this part: We suspect that even if we use multiple arc angles, the result will not be more accurate
    # previous_arc_angle = previous_arc_angle = stable_get_arc_angles(trajectory[-moving_window:])[-1] # We get multiple arc angles, to get more accurate return_factor
    previous_arc_angle = stable_get_arc_angles([trajectory[-1]])[0]
    if previous_arc_angle * moving_sum >= 0:
        return previous_arc_angle + moving_sum * rescale
    else:
        return previous_arc_angle + moving_sum * rescale * return_factor

def get_head_mid_angles(trajectory):

    hm_angles = [calculate_absolute_radian(
        trajectory[0][0:2], trajectory[0][2:4])]
    for pos in trajectory[1:]:
        radian = calculate_absolute_radian(pos[0:2], pos[2:4])
        if radian - hm_angles[-1] > np.pi:
            radian -= 2*np.pi * ((radian - hm_angles[-1] - np.pi) // (2*np.pi) + 1)
        elif hm_angles[-1] - radian > np.pi:
            radian += 2*np.pi * ((hm_angles[-1] - radian - np.pi) // (2*np.pi) + 1)
        hm_angles.append(radian)

    return hm_angles

def get_mid_tail_angles(trajectory):

    mid_tail_angles = [calculate_absolute_radian(
        trajectory[0][2:4], trajectory[0][4:6])]
    for pos in trajectory[1:]:
        radian = calculate_absolute_radian(pos[2:4], pos[4:6])
        if radian - mid_tail_angles[-1] > np.pi:
            radian -= 2*np.pi * ((radian - mid_tail_angles[-1] - np.pi) // (2 * np.pi) + 1)
        elif mid_tail_angles[-1] - radian > np.pi:
            radian += 2*np.pi * ((mid_tail_angles[-1] - radian - np.pi) // (2 * np.pi) + 1)
        mid_tail_angles.append(radian)

    return mid_tail_angles

def angle_diffs(angles, keep_length=True):
    raw_diffs = np.diff(angles)
    if keep_length:
        diffs = [0.]
    else:
        diffs = []
    for diff in raw_diffs:
        diff = diff % (2*np.pi) # Will always result in value between 0 and 2pi
        if diff > np.pi:
            diff -= 2*np.pi
        diffs.append(diff) # Will always result in value between -pi and pi
    return np.array(diffs)

def angle_error(angles1, angles2):

    raw_diffs = np.array(angles1) - np.array(angles2)
    diffs = []
    for diff in raw_diffs:
        diff = diff % (2*np.pi)
        if diff > np.pi:
            diff -= 2*np.pi # Or we can use diff = 2 * np.pi - diff, which will result in negative values, but is still reasonable
        diffs.append(diff)
    return np.array(diffs)

def angle_accelerations(angles):
    raw_diffs = np.diff(angles) # length is len(angles) - 1
    diffs = [0.] # This is to keep the length of the list the same as the length of angles
    for diff in raw_diffs:
        diff = diff % (2*np.pi)
        if diff > np.pi:
            diff -= 2*np.pi
        diffs.append(diff)
    raw_accels = np.diff(diffs) # length is len(angles) - 1
    accels = [0.]
    for accel in raw_accels:
        accel = accel % (2*np.pi)
        if accel > np.pi:
            accel -= 2*np.pi
        accels.append(accel)
    return np.array(accels)

def get_arc_angles(trajectory):
    
    arc_angles = []
    for pos in trajectory:
        center = find_arc_center(pos[0:2], pos[2:4], pos[4:6])
        arc_angle = calculate_angle(pos[4:6], center, pos[2:4])
        if len(arc_angles) == 0:
            pass
        elif arc_angle - arc_angles[-1] > np.pi:
            arc_angle -= 2*np.pi
        elif arc_angles[-1] - arc_angle > np.pi:
            arc_angle += 2*np.pi
        arc_angles.append(arc_angle)
    return arc_angles

def stable_get_arc_angles(trajectory):
    arc_angles = []
    for pos in trajectory:
        center = find_arc_center(pos[0:2], pos[2:4], pos[4:6])
        arc_angle = calculate_angle(pos[4:6], center, pos[2:4]) # The value is between -pi and pi
        hmt_clockwise = is_clockwise(pos[0:2], pos[2:4], pos[4:6])
        mtc_clockwise = is_clockwise(pos[2:4], pos[4:6], center)
        if hmt_clockwise != mtc_clockwise:
            if arc_angle > 0:
                arc_angle -= 2*np.pi
            else:
                arc_angle += 2*np.pi
        arc_angles.append(arc_angle)

    return arc_angles

def get_mean_arc_lengths(trajectory):

    arc_lengths = []
    for pos in trajectory:
        center, radius = find_arc_center(pos[0:2], pos[2:4], pos[4:6], return_radius=True)
        arc_angle = calculate_angle(pos[4:6], center, pos[2:4])
        hmt_clockwise = is_clockwise(pos[0:2], pos[2:4], pos[4:6])
        mtc_clockwise = is_clockwise(pos[2:4], pos[4:6], center)
        if hmt_clockwise != mtc_clockwise:
            if arc_angle > 0:
                arc_angle -= 2*np.pi
            else:
                arc_angle += 2*np.pi
        arc_lengths.append(abs(arc_angle) * radius)

    return np.mean(arc_lengths)

def translate_two_points(hmt, target, target_node):
    # Find the translated triangle DEF given the triangle ABC and the point D
    h, m, t = hmt[0:2], hmt[2:4], hmt[4:6]
    if target_node == 0: # target is head
        h_to_target =target- h
        # Return translated m and t
        return m + h_to_target, t + h_to_target
    elif target_node == 1: # target is midsec
        m_to_target =target- m
        # Return translated h and t
        return h + m_to_target, t + m_to_target
    elif target_node == 2: # target is tail
        t_to_target =target- t
        # Return translated h and m
        return h + t_to_target, m + t_to_target
    else:
        raise ValueError('The node index should be 0, 1, or 2.')

def calculate_angle(A, B, C):
    """
    The output is a radian value between -pi and pi, representing
    the angle: A - B - C, where B is the pivot point. We turn AB
    counter-clockwise (right-hand) to reach BC.
    """

    # Convert points to numpy arrays if they aren't already
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    
    # Vector from B to A and B to C
    BA = A - B
    BC = C - B
    
    # Dot and cross products
    dot_product = np.dot(BA, BC)
    cross_product = np.cross(BA, BC)
    
    # Calculate the angle using atan2 (this gives angle in radians)
    angle = np.arctan2(cross_product, dot_product)
    
    return angle # The value is between -pi and pi

def mean_angle(angles, threshold=np.pi):
    """
    Calculate the mean of a list of angles.
    The purpose of this function is to handle edge cases when 
    the radian values change from somewhere close to 0
    to somewhere close to 2pi.
    """
    angle_sum = 0
    previous_angle = angles[0]
    for angle in angles:

        # Ajdust the angle if it has a large difference with the previous angle
        if angle - previous_angle > threshold: # current angle has a large value
            angle -= 2 * np.pi
        elif previous_angle - angle > threshold: # current angle has a small value
            angle += 2 * np.pi
        
        angle_sum += angle
        previous_angle = angle

    return angle_sum / len(angles)

def sas_find_point(A, B, angle_ABC, length_BC):
    """
    Calculate the coordinates of point C given point A, point B, angle ABC, and the length of BC.
    
    Args:
    A (np.array): The coordinates of point A.
    B (np.array): The coordinates of point B.
    angle_ABC (float): The angle in radians between AB and BC.
    length_BC (float): The length of the segment BC.
    
    Returns:
    np.array: The coordinates of point C.
    """
    # Calculate the angle of vector BA with respect to the x-axis
    BA = A - B
    angle_BA = np.arctan2(BA[1], BA[0])
    
    # Calculate the angle of BC with respect to the x-axis
    angle_BC = angle_BA + angle_ABC
    
    # Calculate the coordinates of C
    C_x = B[0] + length_BC * np.cos(angle_BC)
    C_y = B[1] + length_BC * np.sin(angle_BC)
    
    return np.array([C_x, C_y])

def translate_abc(abc):
    """
    Given the coordinates of the head, midsec, and tail, translate them
    so that the head is at the origin.
    """
    a = abc[0:2]
    b = abc[2:4] - a
    c = abc[4:6] - a
    return np.array([0, 0, *b, *c])

class JoinedTrajectoryV3(JoinedTrajectoryV2):

    def __init__(self, *trajectories, break_type=None):
        super().__init__(*trajectories, break_type=break_type)
        self.geometric_prediction_weight = trajectories[0].geometric_prediction_weight
        self.arc_pred_params = trajectories[0].arc_pred_params
        self.hm_pred_params = trajectories[0].hm_pred_params
    
    def calculate_distance_momentum_weighted(self, new_position, nodes):
        is_initial, velocity = self.get_velocity(-1-self.gap)
        predicted_position = velocity * (self.gap + 1) + self.trajectory[-1-self.gap]
        differences = new_position - predicted_position
        normed_differences = [np.linalg.norm(differences[i*2 : i*2+2])
                              if nodes[i] else 0 # nodes is a list of 3 integers (0 or 1)
                              for i in range(3)]
        normed_differences[2] *= self.tail_weight
        normed_differences = [nd
                              for nd in normed_differences
                              if nd != 0]
        return is_initial, np.mean(normed_differences)

    def backward_calculate_distance_momentum_weighted(self, new_position, gap=0):
        is_initial, velocity = self.get_backward_velocity(0)
        predicted_position = velocity * (gap + 1) + self.trajectory[0]
        differences = new_position - predicted_position
        normed_differences = [np.linalg.norm(differences[i*2 : i*2+2])
                              for i in range(3)]
        normed_differences[2] *= self.tail_weight
        normed_differences = [nd for nd in normed_differences]
        return is_initial, np.mean(normed_differences)
    
    def get_average_fish(self, index=-1):
        if index < 0:
            index = len(self.trajectory) + index
            if index < 0:
                raise ValueError('The index is too small.')
        translated_fishes = [translate_abc(self.trajectory[i])
                             for i in range(index,
                                            max(0, index - self.momentum_window),
                                            -1)
                             if self.trajectory[i] is not None]
        if len(translated_fishes) == 0:
            return False, None
        return True, np.mean(translated_fishes, axis=0) # The result is a 6-element array
    
    def append_predict(self, position, confidence, nodes, match_type):
        """
        ### Important: We don't support gaps in the trajectory. ###
        Args:
            new_coords: A new point to be connected to the existing trajectory.
            confidence: The confidence of the new point.
            match_type: The type of the match for the new point.
            velocity: The velocity of the new point.
        """
        # Check if the new_coords matches the dimension of the existing trajectory.
        if position.shape != self.shape:
            raise ValueError(
                f"The new coordinates with shape {position.shape}"
                f"does not match the dimension of the existing trajectory with shape {self.shape}."
            )
        if sum(nodes) == 2:
            
            # Two nodes are present, we may predict the third node using:
            # 1. predicting midsec:
            #    | similar triangle method: Side(t-h) Angle(t-h-m) Side(h-m)
            # 2. predicting head:
            #    | line arc method: Arc points(t, m), Arc angle(t, m), Line(m, h)
            # 3. predicting tail:
            #    | line arc method: Line(h, m), Arc angle(m, t), Arc length(m, t)
            # In the two existing nodes, we will prefer the head and midsec nodes

            # The head is missing, we will use the line arc method to predict the head
            # In the prediction, we will provide:
            # 1. Previous trajectory
            # 2. Current frame arc angle (presented with tail and midsec)
            # To get the current frame arc angle, we will use the mean length of the previous arcs.
            if nodes[0] == 0:
                
                if len(self.trajectory) == 0:
                    # No previous fish to calculate the average SAS
                    predicted_position = position
                    match_type += '_net'
                else:

                    # Get the average arc length
                    average_arc_length = get_mean_arc_lengths(self.trajectory[-self.momentum_window:])

                    # Get the arc angle of the current frame.
                    # Note that the sign of the value is to be determined.
                    current_arc_angle_raw = solve_arc_angle(average_arc_length,
                                                            np.linalg.norm(position[2:4] - position[4:6]))
                    
                    # Use previous angles to determine the sign of the current arc angle
                    last_arc_angles = stable_get_arc_angles(self.trajectory[-self.momentum_window-1:])
                    if len(last_arc_angles) >= 2:
                        angle_velocity = np.mean(angle_diffs(last_arc_angles))
                        velocity_pred = angle_velocity + last_arc_angles[-1]
                    elif len(last_arc_angles) == 1:
                        velocity_pred = last_arc_angles[0]
                    else:
                        raise ValueError('No previous arc angles to determine the sign of the current arc angle.')
                    if velocity_pred >= 0:
                        current_arc_angle = current_arc_angle_raw
                    else:
                        current_arc_angle = -current_arc_angle_raw

                    # Now we can predict the head-midsec angle
                    predicted_head_midsec_angle = predict_head_midsec_absolute_angle(
                        self.trajectory, current_arc_angle, **self.hm_pred_params)
                    
                    # Then, we should calculate the average head-midsec length in the window
                    head_mid_sides = [np.linalg.norm(cd[2:4] - cd[0:2])
                                      for cd in self.trajectory[-self.momentum_window:]]
                    average_head_mid_side = np.mean(head_mid_sides)

                    # With the angle and the side, we can predict the head position
                    head_x = position[2] - average_head_mid_side * np.cos(predicted_head_midsec_angle)
                    head_y = position[3] - average_head_mid_side * np.sin(predicted_head_midsec_angle)
                    predicted_head = np.array([head_x, head_y])

                    # Finally, we will provide the weighted prediction
                    weighted_predicted_head = predicted_head * self.geometric_prediction_weight + position[0:2] * (1 - self.geometric_prediction_weight)
                    position[0:2] = weighted_predicted_head
                    predicted_position = position
                    match_type += '_geo'


            
            # The midsec is missing, we will predict the midsec with Side (t-h) Angle (t-h-m) Side (h-m)
            # In the prediction, we will provide side TH, angle THM, and average HM
            elif nodes[1] == 0:

                # Get the average SAS
                head_mid_sides = [np.linalg.norm(cd[2:4] - cd[0:2])
                                    for cd in self.trajectory[-self.momentum_window:]
                                    if cd is not None]
                if len(head_mid_sides) == 0:
                    # No previous fish to calculate the average SAS
                    predicted_position = position
                    match_type += '_net'
                else:
                    average_head_mid_side = np.mean(head_mid_sides)
                    t_h_m_angles = [calculate_angle(cd[4:6], cd[0:2], cd[2:4])
                                    for cd in self.trajectory[-self.momentum_window:]
                                    if cd is not None]
                    average_t_h_m_angle = mean_angle(t_h_m_angles) # We use mean_angle function to handle edge cases
                    predicted_midsec = sas_find_point(position[4:6], position[0:2], average_t_h_m_angle, average_head_mid_side)
                    weighted_predicted_midsec = predicted_midsec * self.geometric_prediction_weight + position[2:4] * (1 - self.geometric_prediction_weight)
                    position[2:4] = weighted_predicted_midsec
                    predicted_position = position
                    match_type += '_geo'

            # The tail is missing, we will predict the tail with Side (h-m) Angle (h-m-t) Side (m-t)
            # In the prediction, we will provide side HM, angle HMT, and average MT
            elif nodes[2] == 0:

                # Get the average SAS
                mid_tail_sides = [np.linalg.norm(cd[2:4] - cd[4:6])
                                    for cd in self.trajectory[-self.momentum_window:]
                                    if cd is not None]
                if len(mid_tail_sides) == 0:
                    # No previous fish to calculate the average SAS
                    predicted_position = position
                    match_type += '_net'
                else:
                    average_mid_tail_side = np.mean(mid_tail_sides)
                    h_m_t_angles = [calculate_angle(cd[0:2], cd[2:4], cd[4:6])
                                    for cd in self.trajectory[-self.momentum_window:]
                                    if cd is not None]
                    average_h_m_t_angle = mean_angle(h_m_t_angles)
                    predicted_tail = sas_find_point(position[0:2], position[2:4], average_h_m_t_angle, average_mid_tail_side)
                    weighted_predicted_tail = predicted_tail * self.geometric_prediction_weight + position[4:6] * (1 - self.geometric_prediction_weight)
                    position[4:6] = weighted_predicted_tail
                    predicted_position = position
                    match_type += '_geo'

            else:
                raise ValueError("Cannot find the missing node.")

        elif sum(nodes) == 1:

            # Only one node is present, we may predict by translating previous fish
            average_fish_exists, average_fish = self.get_average_fish()

            if average_fish_exists:
                # There are two unseen nodes
                unseen_nodes = [0, 1, 2]
                unseen_nodes.remove(nodes[0])

                # Get the geometric prediction
                pred_a, pred_b = translate_two_points(average_fish,
                                                 position[nodes[0]*2:nodes[0]*2+2],
                                                 nodes[0])
                predicted_position = np.zeros(6, dtype=float)
                predicted_position[nodes[0]*2:nodes[0]*2+2] = position[nodes[0]*2:nodes[0]*2+2]
                predicted_position[unseen_nodes[0]*2:unseen_nodes[0]*2+2] = pred_a
                predicted_position[unseen_nodes[1]*2:unseen_nodes[1]*2+2] = pred_b

                # Get the final position
                # We don't care about whether we are dealing with [head midsec] or tail here.
                predicted_position = predicted_position * self.geometric_prediction_weight + position * (1 - self.geometric_prediction_weight)
                match_type += '_geo'
            else:
                predicted_position = position
                match_type += '_net'

        elif sum(nodes) == 0:
            raise ValueError("The new position is missing all 3 coordinates.")
        
        else:
            predicted_position = position
        
        self.append(predicted_position, confidence, match_type)

    def prepend_predict(self, position, confidence, nodes, match_type):
        pass

    def ray_distance(self, new_position):
        ray_distances = [distance_to_ray(self.trajectory[-1-self.gap][:2],
                                            self.trajectory[-1-self.gap][2:4],
                                            new_position[2*i:2*i+2])
                            for i in range(2)] # Only calculate the distance for head and midsec
        return np.mean(ray_distances)

class TrajectoryV3(JoinedTrajectoryV3):
    def __init__(self,
                 initial_position: np.ndarray,
                 initial_confidence: float,
                 initial_match_type: str,
                 momentum_window: int,
                 tail_weight: float,
                 geometric_prediction_weight: float,
                 arc_pred_params: dict,
                 hm_pred_params: dict):
        self.trajectory = [initial_position]
        self.confidences = [initial_confidence]
        self.match_types = [initial_match_type]
        self.sources = ['append']
        self.velocities = [None]
        self.shape = initial_position.shape
        self.breakpoints = dict()
        
        if momentum_window <= 0:
            raise ValueError("The momentum window should be a positive integer.")
        self.momentum_window = momentum_window
        self.tail_weight = tail_weight
        self.geometric_prediction_weight = geometric_prediction_weight
        self.arc_pred_params = arc_pred_params
        self.hm_pred_params = hm_pred_params
        self.gap = 0
    
class TrajectoryPoolV3(TrajectoryPoolV2):

    def __init__(self,
                 initial_positions,
                 initial_confidences,
                 initial_match_types,
                 momentum_window,
                 tail_weight,   # Note that in this version, the tail_weight should be better
                                # considered as a factor to normalize the mae (so that the mae
                                # of the tail is comparable to the mae of the head and midsec)
                 max_gap,
                 geometric_prediction_weight,
                 arc_pred_params,
                 hm_pred_params):
        # Check if the shape of the initial_coords is all the same.
        if len(set([initial_position.shape for initial_position in initial_positions])) != 1:
            raise ValueError(
                "The initial coordinates should all have the same shape, but got different shapes."
            )
        self.all_trajectories = [{'trajectory': TrajectoryV3(initial_position,
                                                           initial_confidence,
                                                           initial_match_type,
                                                           momentum_window=momentum_window,
                                                           tail_weight=tail_weight,
                                                           geometric_prediction_weight=geometric_prediction_weight,
                                                           arc_pred_params=arc_pred_params,
                                                           hm_pred_params=hm_pred_params),
                                  'start': 0,
                                  'end': None,
                                  'discard': False,
                                  'prepend': 0,
                                  'fragment': False,
                                  'child_of': None}
                                 for initial_position, initial_confidence, initial_match_type in zip(
                                 initial_positions, initial_confidences, initial_match_types)]

        self.shape = initial_positions[0].shape
        self.momentum_window = momentum_window
        self.tail_weight = tail_weight
        self.geometric_prediction_weight = geometric_prediction_weight
        self.arc_pred_params = arc_pred_params
        self.hm_pred_params = hm_pred_params
        self.max_gap = max_gap
        self.end_value_assigned = False
        self.prepend_paired = False
        self.coarse_paired = False
        self.fine_paired = False

        # In Tracker object:
        # 1. Initialize the TrajectoryPool
        # 2. Enter Loop, in each loop:
        #   a. Call add_trajectory or update_trajectory
        #   b. Call end_time_step
        #      i. Check if there is any trajectory terminated in the current frame
        #      ii. Update the active trajs in the current frame to the timeline
        #      iii. Current frame + 1 (New frame)
        self.active_traj_id = [i for i in range(len(self.all_trajectories))]
        self.timeline = [tuple(self.active_traj_id)]
        self.current_frame = 1 # The 0 th frame is the initial frame and is completed

        # For debugging
        self.debug_log = []

    def add_trajectory(self, start_position, start_confidence, start_match_type):

        if start_position.shape != self.shape:
            raise ValueError(
                f"The new coordinates with shape {start_position.shape}"
                "does not match the dimension of the existing trajectories with shape {self.shape}."
            )

        self.active_traj_id.append(len(self.all_trajectories))
        self.all_trajectories.append({'trajectory': TrajectoryV3(start_position,
                                                               start_confidence,
                                                               start_match_type,
                                                               momentum_window=self.momentum_window,
                                                               tail_weight=self.tail_weight,
                                                               geometric_prediction_weight=self.geometric_prediction_weight,
                                                               arc_pred_params=self.arc_pred_params,
                                                               hm_pred_params=self.hm_pred_params),
                                      'start': self.current_frame,
                                      'end': None,
                                      'discard': False,
                                      'prepend': 0,
                                      'fragment': False,
                                      'child_of': None})

    def update_predict_trajectory(self, traj_id, new_position, new_confidence, new_nodes, new_match_type, prepend=None):
        if prepend is None:
            # This is for building the traj_pool, executed by Tracker
            if traj_id not in self.active_traj_id:
                raise ValueError(
                    f"Trajectory with ID {traj_id} is not active.")
            self.all_trajectories[traj_id]['trajectory'].append_predict(new_position,
                                                                new_confidence,
                                                                new_nodes,
                                                                new_match_type)
        else:
            # This is executed by self.prepend_pair()
            for i in range(prepend):
                self.all_trajectories[traj_id]['trajectory'].prepend(None, 0, 'None')
                self.all_trajectories[traj_id]['start'] -= 1
                self.all_trajectories[traj_id]['prepend'] += 1
                self.timeline[self.all_trajectories[traj_id]
                          ['start']].append(traj_id)
            self.all_trajectories[traj_id]['trajectory'].prepend_predict(new_position,
                                                                 new_confidence,
                                                                 new_nodes,
                                                                 new_match_type)
            self.all_trajectories[traj_id]['start'] -= 1
            self.all_trajectories[traj_id]['prepend'] += 1
            self.timeline[self.all_trajectories[traj_id]
                          ['start']].append(traj_id)

class TrackerV3:

    def __init__(self,
                 num_keypoints=7,
                 distance_threshold_per_frame=4,
                 traj_length_belief=4,
                 full_state_length_belief=4,
                 momentum_window=3,
                 overlap_length_threshold=2,
                 max_gap=0,
                 tail_weight=0.5,   # Note that in this version, the tail_weight should be better
                                    # considered as a factor to normalize the mae (so that the mae
                                    # of the tail is comparable to the mae of the head and midsec)
                 geometric_prediction_weight=0.5,
                 doublecheck_threshold=2,
                 arc_pred_params={'exponent_base': 0.93,
                                    'rescale': 2.51,
                                    'moving_window': 60,
                                    'return_factor': 1.18},
                 hm_pred_params={'exponent_base': 0.26,
                                    'rescale': 0.27,
                                    'moving_window': 10,
                                    'return_stabalizer': 0.17}):
        self.num_keypoints = num_keypoints
        self.distance_threshold_per_frame = distance_threshold_per_frame
        self.traj_length_belief = traj_length_belief
        self.full_state_length_belief = full_state_length_belief
        self.momentum_window = momentum_window
        self.overlap_length_threshold = overlap_length_threshold
        self.max_gap = max_gap
        self.tail_weight = tail_weight
        self.geometric_prediction_weight = geometric_prediction_weight
        self.doublecheck_threshold = doublecheck_threshold
        self.arc_pred_params = arc_pred_params
        self.hm_pred_params = hm_pred_params

    def initialize_build(self, all_raw_coords):
        self.length = len(all_raw_coords[0][0])
        self.all_coords_dicts = self.get_all_coords_dicts_from_all_raw_coords(all_raw_coords)
        initial_subgraphs = self.get_subgraphs_from_coords_dict(self.all_coords_dicts[0])
        good_subgraphs, bad_subgraphs = self.analyze_subgraphs(initial_subgraphs)
        if len(good_subgraphs) != self.num_keypoints:
            print(f'Warning: the number of initial good subgraphs {len(good_subgraphs)} is not equal to the number of keypoints.')
        initial_coords, initial_confs, initial_nodes, initial_types = self.get_coords_from_good_graphs(good_subgraphs, self.all_coords_dicts[0])
        self.traj_pool = TrajectoryPoolV3(initial_coords, initial_confs, initial_types,
                                        self.momentum_window, self.tail_weight, self.max_gap, self.geometric_prediction_weight,
                                        self.arc_pred_params, self.hm_pred_params)
        
    def step_build_traj_pool(self, coords_dict):

        num_current_active_traj = len(self.traj_pool.active_traj_id)

        subgraphs = self.get_subgraphs_from_coords_dict(coords_dict)
        good_subgraphs, bad_graph = self.analyze_subgraphs(subgraphs)
        good_coords, good_confs, good_nodes, good_types = self.get_coords_from_good_graphs(good_subgraphs, coords_dict)

        # Use the Hungarian algorithm to connect the new positions to the existing trajectories.
        initial_trajectory_ids = set()
        weights = []
        for trajectory_id in self.traj_pool.active_traj_id:
            weights.append([])
            for cds, nodes in zip(good_coords, good_nodes):
                is_initial, distance = self.traj_pool[trajectory_id].calculate_distance_momentum_weighted(cds, nodes)
                weights[-1].append(distance)
                if is_initial:
                    initial_trajectory_ids.add(trajectory_id)

        # Each row represents a previous trajectory, each column represents a new position.
        weights = np.array(weights)
        row_ind, col_ind = linear_sum_assignment(weights)

        not_assigned_traj_ids = set(self.traj_pool.active_traj_id)
        not_assigned_positions = set(range(len(good_coords)))

        for previous_index, new_position_index in zip(row_ind, col_ind):
            if weights[previous_index, new_position_index] < self.distance_threshold_per_frame:
                # If the distance is within the threshold, connect the new position to the existing trajectory.
                trajectory_id = self.traj_pool.active_traj_id[previous_index]
                self.traj_pool.update_predict_trajectory(trajectory_id,
                                                 good_coords[new_position_index],
                                                 good_confs[new_position_index],
                                                 good_nodes[new_position_index],
                                                 good_types[new_position_index])
                not_assigned_traj_ids.remove(trajectory_id)
                not_assigned_positions.remove(new_position_index)
        
        # If we have remaining not-assigned and 'initial' trajs, we need to pair them using ray distance.
        if len(initial_trajectory_ids.intersection(not_assigned_traj_ids)) > 0:
            weights = []
            awaiting_traj_ids = list(initial_trajectory_ids.intersection(not_assigned_traj_ids))
            awaiting_position_indices = list(not_assigned_positions)
            for traj_id in awaiting_traj_ids:
                weights.append([])
                for pos_index in awaiting_position_indices:
                    distance = self.traj_pool[traj_id].ray_distance(good_coords[pos_index])
                    weights[-1].append(distance)
            weights = np.array(weights)
            row_ind, col_ind = linear_sum_assignment(weights)
            for i, j in zip(row_ind, col_ind):
                if weights[i, j] < self.distance_threshold_per_frame:
                    self.traj_pool.update_predict_trajectory(awaiting_traj_ids[i],
                                                     good_coords[awaiting_position_indices[j]],
                                                     good_confs[awaiting_position_indices[j]],
                                                     good_nodes[awaiting_position_indices[j]],
                                                     good_types[awaiting_position_indices[j]]+'_ray')
                    not_assigned_traj_ids.remove(awaiting_traj_ids[i])
                    not_assigned_positions.remove(awaiting_position_indices[j])

        # This logic is discarded, now we consider not-assigned positions as bad positions
        # If we still have not-assigned good positions, we need to add them as new trajectories.
        # for position_index in not_assigned_positions:
        #    self.traj_pool.add_trajectory(good_coords[position_index],
        #                                   good_confs[position_index],
        #                                   good_types[position_index])

        # If we have remaining not-assigned bad positions, add them into the bad graph.
        for position_index in not_assigned_positions:
            subgraph = good_subgraphs[position_index]
            bad_graph.add_nodes_from(subgraph.nodes)
            bad_graph.add_edges_from(subgraph.edges)

        # Then, we should check the num_fish (num_keypoints) and number of active trajectrories
        # Case 1: num_active_traj == num_fish: no need to do anything
        # Case 2: num_active_traj < num_fish:
        #       a. If there is remaining not-assigned trajs, we use bad coords to fill them
        #       b. If there is no remaining not assigned trajs, we add the bad coords as new trajs
        # Case 3: num_active_traj > num_fish: should NEVER happen

        num_active_traj = num_current_active_traj - len(not_assigned_traj_ids) # Updated active trajs
                          # + len(not_assigned_positions)) # Newly added active trajs
                          # We discarded the feature of directly adding not-assigned positions as new trajs
        
        if num_active_traj < self.num_keypoints:

            # Case 2: num_active_traj < num_fish:
            
            bad_dict = self.get_coords_from_bad_graph(bad_graph, coords_dict)
            remaining_gap = self.num_keypoints - num_active_traj # Prevent updating too many old trajs with bad coords

            if len(not_assigned_traj_ids) > 0:

                # Case 2: a. If there is remaining not-assigned trajs, we use bad coords to fill them

                awaiting_traj_ids = list(not_assigned_traj_ids)
                
                corresponding_assigned_nodes = [[None, None, None, 0, None, None, None] # head, midsec, tail, conf, node_head, node_midsec, node_tail
                                                for _ in range(len(awaiting_traj_ids))]
                corresponding_scores = [0] * len(awaiting_traj_ids) # The score is the number of assigned nodes for each traj
                
                for node, node_dict in bad_dict.items():

                    if len(node_dict['coords']) == 0:
                        continue

                    weights = []
                    for traj_id in awaiting_traj_ids:
                        weights.append([])
                        for pos in node_dict['coords']:
                            is_initial, distance = self.traj_pool[traj_id].calculate_distance_momentum_weighted(pos, node)
                            weights[-1].append(distance)

                    weights = np.array(weights)
                    row_ind, col_ind = linear_sum_assignment(weights)
                    pos_indices_to_remove = []

                    for i, j in zip(row_ind, col_ind):
                        if weights[i, j] < self.distance_threshold_per_frame:
                            node_hmt_index = node.index(1) # 0: head, 1: midsec, 2: tail
                            if corresponding_assigned_nodes[i][node_hmt_index] is None:
                                corresponding_assigned_nodes[i][node_hmt_index] = node_dict['coords'][j][node_hmt_index*2:node_hmt_index*2+2]
                                corresponding_assigned_nodes[i][3] += node_dict['confs'][j]
                                corresponding_assigned_nodes[i][node_hmt_index + 4] = node_dict['node_name'][j]
                                corresponding_scores[i] += 100
                                if node_hmt_index == 2: # Tail
                                    corresponding_scores[i] -= 1 # We prioritize head and midsec
                                pos_indices_to_remove.append(j)
                            else:
                                raise ValueError('The node is already assigned.')

                    for index in sorted(pos_indices_to_remove, reverse=True): # Remove from the last index
                        bad_dict[node]['coords'].pop(index)
                        bad_dict[node]['confs'].pop(index)
                        bad_dict[node]['node_name'].pop(index)

                index_with_high_scores = np.argsort(corresponding_scores)[::-1][:remaining_gap] # With more nodes (prioritize head and midsec)

                for i in index_with_high_scores:
                    # assigned_nodes is a list of 7 elements:
                    # 0: head_coord, 1: midsec_coord, 2: tail_coord, 3: total_conf, 4: node_head, 5: node_midsec, 6: node_tail
                    assigned_nodes = corresponding_assigned_nodes[i]

                    num_assigned_nodes = sum([1 for node in assigned_nodes[:3] if node is not None])

                    if num_assigned_nodes == 3:

                        not_assigned_traj_ids.remove(awaiting_traj_ids[i])

                        # All nodes are assigned
                        subgraph = bad_graph.subgraph([assigned_nodes[4], assigned_nodes[5], assigned_nodes[6]])
                        bad_graph.remove_nodes_from([assigned_nodes[4], assigned_nodes[5], assigned_nodes[6]])
                        try:
                            self.traj_pool.update_predict_trajectory(awaiting_traj_ids[i],
                                                            np.ravel(assigned_nodes[:3]),
                                                            assigned_nodes[3], # No need to divide by 3, because it is already divided in `get_coords_from_bad_graphs`
                                                            [1, 1, 1],
                                                            f'N3_E{len(subgraph.edges)}_bad')
                        except:
                            ic(assigned_nodes[:3], assigned_nodes[3:])
                            raise
                    
                    elif num_assigned_nodes == 2:

                        not_assigned_traj_ids.remove(awaiting_traj_ids[i])

                        # One node is missing, we use the other two nodes to predict the missing node

                        if assigned_nodes[0] is None:
                            # Head is missing
                            pred_head_from_midsec = coords_dict['midsec']['head'][assigned_nodes[5][1]]
                            pred_head_from_tail = coords_dict['tail']['head'][assigned_nodes[6][1]]
                            weighted_pred_head = (pred_head_from_midsec + pred_head_from_tail * self.tail_weight) / (1 + self.tail_weight)
                            subgraph = bad_graph.subgraph([assigned_nodes[5], assigned_nodes[6]])
                            bad_graph.remove_nodes_from([assigned_nodes[5], assigned_nodes[6]])
                            self.traj_pool.update_predict_trajectory(awaiting_traj_ids[i],
                                                            np.ravel([weighted_pred_head, assigned_nodes[1], assigned_nodes[2]]),
                                                            assigned_nodes[3], # Confidence
                                                            [0, 1, 1],
                                                            f'N2_E{len(subgraph.edges)}_mt_bad')

                        elif assigned_nodes[1] is None:
                            # Midsec is missing
                            pred_midsec_from_head = coords_dict['head']['midsec'][assigned_nodes[4][1]]
                            pred_midsec_from_tail = coords_dict['tail']['midsec'][assigned_nodes[6][1]]
                            weighted_pred_midsec = (pred_midsec_from_head + pred_midsec_from_tail * self.tail_weight) / (1 + self.tail_weight)
                            subgraph = bad_graph.subgraph([assigned_nodes[4], assigned_nodes[6]])
                            bad_graph.remove_nodes_from([assigned_nodes[4], assigned_nodes[6]])
                            self.traj_pool.update_predict_trajectory(awaiting_traj_ids[i],
                                                            np.ravel([assigned_nodes[0], weighted_pred_midsec, assigned_nodes[2]]),
                                                            assigned_nodes[3], # Confidence
                                                            [1, 0, 1],
                                                            f'N2_E{len(subgraph.edges)}_ht_bad')
                        
                        elif assigned_nodes[2] is None:
                            # Tail is missing
                            pred_tail_from_head = coords_dict['head']['tail'][assigned_nodes[4][1]]
                            pred_tail_from_midsec = coords_dict['midsec']['tail'][assigned_nodes[5][1]]
                            weighted_pred_tail = (pred_tail_from_head + pred_tail_from_midsec) / 2
                            subgraph = bad_graph.subgraph([assigned_nodes[4], assigned_nodes[5]])
                            bad_graph.remove_nodes_from([assigned_nodes[4], assigned_nodes[5]])
                            self.traj_pool.update_predict_trajectory(awaiting_traj_ids[i],
                                                            np.ravel([assigned_nodes[0], assigned_nodes[1], weighted_pred_tail]),
                                                            assigned_nodes[3], # Confidence
                                                            [1, 1, 0],
                                                            f'N2_E{len(subgraph.edges)}_hm_bad')

                        else:
                            raise ValueError('There are only 2 assigned node, but we cannot find the missing node')

                    elif num_assigned_nodes == 1:

                        not_assigned_traj_ids.remove(awaiting_traj_ids[i])

                        # One node is assigned, we will predict the other two nodes using the assigned node
                        assigned_node_name = [node for node in assigned_nodes[4:7] if node is not None][0] # assigned_node_name may look like ('head', 2)
                        assigned_node_hmt_index = assigned_nodes[4:7].index(assigned_node_name) # 0: head, 1: midsec, 2: tail
                        cds = [None, None, None]
                        cds[assigned_node_hmt_index] = assigned_nodes[assigned_node_hmt_index]
                        unseen_groups = set(['head', 'midsec', 'tail']) - set([assigned_node_name[0]])
                        for unseen_group in unseen_groups:
                            pred_from_seen = coords_dict[assigned_node_name[0]][unseen_group][assigned_node_name[1]]
                            unseen_index = ['head', 'midsec', 'tail'].index(unseen_group)
                            cds[unseen_index] = pred_from_seen
                        nodes_state_list = [0, 0, 0]
                        nodes_state_list[assigned_node_hmt_index] = 1
                        bad_graph.remove_node(assigned_node_name)
                        self.traj_pool.update_predict_trajectory(awaiting_traj_ids[i],
                                                         np.ravel(cds),
                                                         assigned_nodes[3], # Confidence
                                                         nodes_state_list,
                                                         f'N1_E0_{assigned_node_name[0][0]}_bad')
                        
                    else:
                        # No node is assigned to this trajectory
                        # This trajectory will be gapped (and then terminated)
                        pass
            
            # Case 2: b. If there is no remaining not assigned trajs, we add the bad coords as new trajs
            # Now, we have attempted to assign all not-assigned trajs using bad coords
            # We should again check the num_active_traj, because the length of not_assigned_traj_ids has changed
            # If the num_active_traj is still less than num_fish, we need to add new trajs from bad coords
            num_active_traj = num_current_active_traj - len(not_assigned_traj_ids) # Updated active trajs
                                #  + len(not_assigned_positions)) # Newly added *good* active trajs (This feature is discarded)
            
            if num_active_traj < self.num_keypoints:
                
                graph_pool = analyze_full_graph(bad_graph)
                sorted_pool = list(sorted(graph_pool,
                            key=lambda i: (i['active'], i['n_nodes'], i['n_edges'], i['original']),
                            reverse=True))[:self.num_keypoints - num_active_traj]
                
                for i in sorted_pool:

                    if i['active'] == False:
                        break

                    cds = np.zeros(6, dtype=float)
                    conf = 0
                    seen_nodes = [0, 0, 0]
                    for node in i['graph'].nodes:
                        node_hmt_index = ['head', 'midsec', 'tail'].index(node[0])
                        cds[node_hmt_index*2:node_hmt_index*2+2] = coords_dict[node[0]]['coords'][node[1]]
                        conf += coords_dict[node[0]]['confs'][node[1]]
                        seen_nodes[node_hmt_index] = 1

                    if sum(seen_nodes) == 3:
                        self.traj_pool.add_trajectory(cds, conf, f'N3_E{i["n_edges"]}_bad')

                    elif sum(seen_nodes) == 2:
                        # One node is missing, we use the other two nodes to predict the missing node

                        if seen_nodes[0] == 0: # Head is missing
                            for node in i['graph'].nodes:
                                if node[0] == 'midsec':
                                    head_from_midsec = coords_dict['midsec']['head'][node[1]]
                                elif node[0] == 'tail':
                                    head_from_tail = coords_dict['tail']['head'][node[1]]
                                else:
                                    raise ValueError(f'Head is missing, but the node type is not midsec or tail.')
                            weighted_pred_head = (head_from_midsec + head_from_tail * self.tail_weight) / (1 + self.tail_weight)
                            cds[0:2] = weighted_pred_head
                            self.traj_pool.add_trajectory(cds,
                                                          conf / 2,
                                                          f'N2_E{i["n_edges"]}_mt_bad')
                        
                        elif seen_nodes[1] == 0: # Midsec is missing
                            for node in i['graph'].nodes:
                                if node[0] == 'head':
                                    midsec_from_head = coords_dict['head']['midsec'][node[1]]
                                elif node[0] == 'tail':
                                    midsec_from_tail = coords_dict['tail']['midsec'][node[1]]
                                else:
                                    raise ValueError(f'Midsec is missing, but the node type is not head or tail.')
                            weighted_pred_midsec = (midsec_from_head + midsec_from_tail * self.tail_weight) / (1 + self.tail_weight)
                            cds[2:4] = weighted_pred_midsec
                            self.traj_pool.add_trajectory(cds,
                                                          conf / 2,
                                                          f'N2_E{i["n_edges"]}_ht_bad')

                        elif seen_nodes[2] == 0: # Tail is missing
                            for node in i['graph'].nodes:
                                if node[0] == 'head':
                                    tail_from_head = coords_dict['head']['tail'][node[1]]
                                elif node[0] == 'midsec':
                                    tail_from_midsec = coords_dict['midsec']['tail'][node[1]]
                                else:
                                    raise ValueError(f'Tail is missing, but the node type is not head or midsec.')
                            weighted_pred_tail = (tail_from_head + tail_from_midsec) / 2
                            cds[4:6] = weighted_pred_tail
                            self.traj_pool.add_trajectory(cds,
                                                          conf / 2,
                                                          f'N2_E{i["n_edges"]}_hm_bad')

                        else:
                            raise ValueError('There are only 2 seen nodes, but we cannot find the missing node.')

                    elif sum(seen_nodes) == 1:
                        # One node is assigned, we will predict the other two nodes using the assigned node
                        the_node = list(i['graph'].nodes)[0] # the_node may look like ('head', 2)
                        assigned_node_hmt_index = seen_nodes.index(1)
                        unseen_groups = set(['head', 'midsec', 'tail']) - set([the_node[0]])
                        for unseen_group in unseen_groups:
                            pred_from_seen = coords_dict[the_node[0]][unseen_group][node[1]]
                            unseen_index = ['head', 'midsec', 'tail'].index(unseen_group)
                            cds[unseen_index*2:unseen_index*2+2] = pred_from_seen
                        self.traj_pool.add_trajectory(cds,
                                                      conf / 3,
                                                      f'N1_E0_{the_node[0][0]}_bad')
                    
                    else:
                        raise ValueError('Empty graph is not expected.')
                
            elif num_active_traj > self.num_keypoints:
                ic(num_current_active_traj, not_assigned_traj_ids, not_assigned_positions)
                raise ValueError('The number of active trajectories is greater than the number of keypoints.')
            else:
                # num_active_traj == num_fish, no need to do anything
                pass

        elif num_active_traj > self.num_keypoints:
            ic(num_active_traj, self.num_keypoints, self.traj_pool.active_traj_id, not_assigned_traj_ids, not_assigned_positions)
            raise ValueError('The number of active trajectories is greater than the number of keypoints.')
        else:
            # num_active_traj == num_fish, no need to do anything
            pass

        # We need to gap the not-assigned trajectories
        for traj_id in not_assigned_traj_ids:
            self.traj_pool.gap_trajectory(traj_id)

        try:
            self.traj_pool.end_time_step()
        except:
            ic(not_assigned_traj_ids, not_assigned_positions)
            raise
    
    def build_traj_pool(self, all_raw_coords, verbose):
        self.initialize_build(all_raw_coords)
        for coords_dict in tqdm(self.all_coords_dicts[1:], disable=not verbose):
            self.step_build_traj_pool(coords_dict)

    def get_coords_from_bad_graph(self, bad_graph, coords_dict):
        
        output = {(1, 0, 0): {'coords': [],
                                'confs': [],
                                'node_name': []},
                    (0, 1, 0): {'coords': [],
                                'confs': [],
                                'node_name': []},
                    (0, 0, 1): {'coords': [],
                                'confs': [],
                                'node_name': []}}
        
        for node in bad_graph.nodes:
            if node[0] == 'head':
                cds = [*coords_dict['head']['coords'][node[1]],
                        *coords_dict['head']['midsec'][node[1]],
                        *coords_dict['head']['tail'][node[1]]]
                output[(1, 0, 0)]['coords'].append(np.array(cds))
                output[(1, 0, 0)]['confs'].append(coords_dict['head']['confs'][node[1]] / 3)
                output[(1, 0, 0)]['node_name'].append(node)
            elif node[0] == 'midsec':
                cds = [*coords_dict['midsec']['head'][node[1]],
                        *coords_dict['midsec']['coords'][node[1]],
                        *coords_dict['midsec']['tail'][node[1]]]
                output[(0, 1, 0)]['coords'].append(np.array(cds))
                output[(0, 1, 0)]['confs'].append(coords_dict['midsec']['confs'][node[1]] / 3)
                output[(0, 1, 0)]['node_name'].append(node)
            elif node[0] == 'tail':
                cds = [*coords_dict['tail']['head'][node[1]],
                        *coords_dict['tail']['midsec'][node[1]],
                        *coords_dict['tail']['coords'][node[1]]]
                output[(0, 0, 1)]['coords'].append(np.array(cds))
                output[(0, 0, 1)]['confs'].append(coords_dict['tail']['confs'][node[1]] / 3)
                output[(0, 0, 1)]['node_name'].append(node)
            else:
                raise ValueError(f'Unknown node type {node[0]}')
        
        return output
    
    def get_coords_from_good_graphs(self, good_subgraphs, coords_dict):

        fish_coords = []
        fish_confs = []
        fish_nodes = []
        fish_string_types = []

        for subgraph in good_subgraphs:

            cds = np.zeros(6, dtype=float)
            conf = 0
            seen_nodes = np.zeros(3, dtype=int)
            string_seen_nodes = []

            for node in subgraph.nodes:
                if node[0] == 'head':
                    cds[0:2] = coords_dict['head']['coords'][node[1]]
                    seen_nodes[0] = 1
                    conf += coords_dict['head']['confs'][node[1]]
                    string_seen_nodes.append('head')
                elif node[0] == 'midsec':
                    cds[2:4] = coords_dict['midsec']['coords'][node[1]]
                    seen_nodes[1] = 1
                    conf += coords_dict['midsec']['confs'][node[1]]
                    string_seen_nodes.append('midsec')
                elif node[0] == 'tail':
                    cds[4:6] = coords_dict['tail']['coords'][node[1]]
                    seen_nodes[2] = 1
                    conf += coords_dict['tail']['confs'][node[1]]
                    string_seen_nodes.append('tail')
                else:
                    raise ValueError(f'Unknown node type.')
                
            if sum(seen_nodes) == 3:
                pass # All coords are filled

            elif sum(seen_nodes) == 2:
                # One coord is missing, we use the other two coords to predict the missing coord

                if seen_nodes[0] == 0: # head is missing
                    for node in subgraph.nodes:
                        if node[0] == 'midsec':
                            head_from_midsec = coords_dict['midsec']['head'][node[1]]
                        elif node[0] == 'tail':
                            head_from_tail = coords_dict['tail']['head'][node[1]]
                        else:
                            raise ValueError(f'Head is missing, but the node type is not midsec or tail.')
                    weighted_pred_head = (head_from_midsec + head_from_tail * self.tail_weight) / (1 + self.tail_weight)
                    cds[0:2] = weighted_pred_head

                elif seen_nodes[1] == 0: # midsec is missing
                    for node in subgraph.nodes:
                        if node[0] == 'head':
                            midsec_from_head = coords_dict['head']['midsec'][node[1]]
                        elif node[0] == 'tail':
                            midsec_from_tail = coords_dict['tail']['midsec'][node[1]]
                        else:
                            raise ValueError(f'Midsec is missing, but the node type is not head or tail.')
                    weighted_pred_midsec = (midsec_from_head + midsec_from_tail * self.tail_weight) / (1 + self.tail_weight)
                    cds[2:4] = weighted_pred_midsec
                
                elif seen_nodes[2] == 0: # tail is missing
                    for node in subgraph.nodes:
                        if node[0] == 'head':
                            tail_from_head = coords_dict['head']['tail'][node[1]]
                        elif node[0] == 'midsec':
                            tail_from_midsec = coords_dict['midsec']['tail'][node[1]]
                        else:
                            raise ValueError(f'Tail is missing, but the node type is not head or midsec.')
                    pred_tail = (tail_from_head + tail_from_midsec) / 2
                    cds[4:6] = pred_tail
                
                else:
                    raise ValueError(f'Unknown missing coord.')

            elif sum(seen_nodes) == 1:
                # Two coords are missing, we use the remaining coord to predict the missing coords
                unseen_groups = set(['head', 'midsec', 'tail']) - set(string_seen_nodes)
                node = list(subgraph.nodes)[0]
                for unseen_group in unseen_groups:
                    pred_from_seen = coords_dict[node[0]][unseen_group][node[1]]
                    if unseen_group == 'head':
                        cds[0:2] = pred_from_seen
                    elif unseen_group == 'midsec':
                        cds[2:4] = pred_from_seen
                    elif unseen_group == 'tail':
                        cds[4:6] = pred_from_seen
            
            else:
                raise ValueError(f'Empty subgraph.')
            
            fish_coords.append(cds)
            fish_confs.append(conf / 3)
            fish_nodes.append(seen_nodes)
            if len(subgraph.nodes) == 3:
                fish_string_types.append(f'N3_E{len(subgraph.edges)}')
            else:
                fish_string_types.append(f"N{len(subgraph.nodes)}_E{len(subgraph.edges)}_{''.join(sorted([n[0] for n in string_seen_nodes]))}")

        return fish_coords, fish_confs, fish_nodes, fish_string_types
    
    def analyze_subgraphs(self, subgraphs):

        num_nodes = [len(subgraph.nodes) for subgraph in subgraphs]
        counter = Counter(num_nodes)
        remaining_graphs = self.num_keypoints - counter[3]

        if remaining_graphs == 0:
            # All subgraphs are 3-node good subgraphs
            return self.get_good_bad_subgraphs(subgraphs, [3])
        elif remaining_graphs < 0:
            raise ValueError('Too many 3-node subgraphs, the max_detect is too high.')
        
        num_nodes_more_than_three = [num for num in num_nodes if num > 3]
        if len(num_nodes_more_than_three) == 0:

            # No subgraphs have more than 3 nodes, all remaining subgraphs are 1-node or 2-node subgraphs
            remaining_graphs -= counter[2]
            if remaining_graphs == 0:
                # 2-nodes and 3-nodes subgraphs are enough
                return self.get_good_bad_subgraphs(subgraphs, [2, 3])
            elif remaining_graphs < 0:
                # there may be errors in the 2-node subgraphs, because there are too many 2-node subgraphs
                return self.get_good_bad_subgraphs(subgraphs, [3])
            
            # there are not enough 2-node subgraphs, we need to use 1-node subgraphs
            remaining_graphs -= counter[1]
            if remaining_graphs >= 0:
                # inluding 1-node subgraphs will not exceed the maximum number of detections
                return self.get_good_bad_subgraphs(subgraphs, [1, 2, 3])
            else:
                # there may be errors in the 1-node subgraphs, because there are too many 1-node subgraphs
                return self.get_good_bad_subgraphs(subgraphs, [2, 3])
            
        # There are subgraphs with more than 3 nodes
        remaining_graphs -= counter[2]
        for num in num_nodes_more_than_three:
            remaining_graphs -= num // 2 # Assume that 4 or 5 nodes represent 2 fishes; 6 or 7 nodes represent 3 fishes; and so on
        if remaining_graphs >= 0:
            # including 2-node graphs will not exceed the maximum number of detections
            return self.get_good_bad_subgraphs(subgraphs, [2, 3])
        else:
            return self.get_good_bad_subgraphs(subgraphs, [3])
    
    def get_good_bad_subgraphs(self, subgraphs, good_nodes):
        good_subgraphs = []
        bad_subgraphs = []
        for subgraph in subgraphs:
            if len(subgraph.nodes) in good_nodes:
                good_subgraphs.append(subgraph)
            else:
                bad_subgraphs.append(subgraph)

        # Create an empty directed graph
        bad_full_graph = nx.DiGraph()

        # Loop through each subgraph in the list
        for subgraph in bad_subgraphs:
            # Add nodes and edges from the subgraph to the full graph
            bad_full_graph.add_nodes_from(subgraph.nodes(data=True))
            bad_full_graph.add_edges_from(subgraph.edges(data=True))

        return good_subgraphs, bad_full_graph
        
    def hungarian_matching(self, h_m, m, threshold):

        if len(h_m) == 0 or len(m) == 0:
            return {i: None
                for i in range(len(h_m))}

        mappings = {}

        weights = []
        for head_to_middle in h_m:
            weights.append([])
            for middle in m:
                weights[-1].append(np.linalg.norm(head_to_middle - middle))
        row_ind, col_ind = linear_sum_assignment(weights)
        for r, c in zip(row_ind, col_ind):
            if weights[r][c] < threshold:
                mappings[r] = (c, weights[r][c])
            else:
                mappings[r] = None

        return mappings
    
    def get_all_coords_dicts_from_all_raw_coords(self, all_raw_coords):
        all_coords_dicts = []
        for index in range(len(all_raw_coords[0][0])):
            h, h_m, h_t, h_c, m, m_h, m_t, m_c, t, t_h, t_m, t_c = (
                all_raw_coords[0][0][index], all_raw_coords[1][1][index], all_raw_coords[2][1][index], all_raw_coords[0][3][index],
                all_raw_coords[1][0][index], all_raw_coords[0][1][index], all_raw_coords[2][2][index], all_raw_coords[1][3][index],
                all_raw_coords[2][0][index], all_raw_coords[0][2][index], all_raw_coords[1][2][index], all_raw_coords[2][3][index])
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
            all_coords_dicts.append(coords_dict)
        return all_coords_dicts
    
    def get_subgraphs_from_coords_dict(self, coords_dict):
        mapping = {}
        mapping['head'] = {'tail': self.hungarian_matching(coords_dict['head']['tail'], coords_dict['tail']['coords'], self.doublecheck_threshold / self.tail_weight),
                        'midsec': self.hungarian_matching(coords_dict['head']['midsec'], coords_dict['midsec']['coords'], self.doublecheck_threshold)}
        mapping['midsec'] = {'head': self.hungarian_matching(coords_dict['midsec']['head'], coords_dict['head']['coords'], self.doublecheck_threshold),
                            'tail': self.hungarian_matching(coords_dict['midsec']['tail'], coords_dict['tail']['coords'], self.doublecheck_threshold / self.tail_weight)}
        mapping['tail'] = {'head': self.hungarian_matching(coords_dict['tail']['head'], coords_dict['head']['coords'], self.doublecheck_threshold / self.tail_weight),
                            'midsec': self.hungarian_matching(coords_dict['tail']['midsec'], coords_dict['midsec']['coords'], self.doublecheck_threshold / self.tail_weight)}
        return get_subgraphs_from_mapping(mapping)

    def track(self, raw_coords, verbose=False):
        """
        Args:
            matched_coords: List of dictionaries. Each dictionary contains the following keys:
                'perfect_match': List of tuples. Tuples are in the format of (head_coord, middle_coord, confidence).
                'overlap_match': List of tuples. Tuples are in the same format as above.
                'unilateral': List of tuples. Tuples are in the same format as above.
                'isolated': List of tuples. Tuples are in the same format as above.
                The matched coordinates is from zf.inference.keypoint.DoublecheckKeypoint
            confidences: List of lists of floats. Each float represents the confidence of the corresponding matched_coords.
            match_types: List of lists of strings. Each string represents the match type of the corresponding matched_coords.
            verbose: If True, print the tracking process.
        """
        # The tracking is performed in three steps:
        # 1. Build the trajectory pool: A set of trajectories built from the matched_coords.
        # 2. Pair trajectories within the pool, to form one consistent trajectory for each entity.
        # 3. Refine the trajectories to remove noise and fill in the gaps.

        # Step 1: Build the trajectory pool
        # We firstly use forward-backward matching to build the trajectory pool.
        # Then, we attempt connect unpaired positions (called fragments).
        if verbose:
            print("Step 1: Building the trajectory pool.")
            current_time = time()
        self.build_traj_pool(raw_coords, verbose=verbose)
        if verbose:
            forward_time = time() - current_time
        self.traj_pool.assign_end_value()
        self.traj_pool.prepend_pair(self.distance_threshold_per_frame,
                                    self.traj_length_belief,
                                    verbose=False) # The time cost for prepend_pair is negligible.
        if verbose:
            backward_time = time() - current_time - forward_time
        if verbose:
            print(f"Forward: {forward_time:.2f}s, Backward: {backward_time:.2f}s, "
                  f"Total: {time() - current_time:.2f}s")

        # Step 2: Pair trajectories within the pool
        # Coarse Pair and Fine Pair are based on the belief that trajectories with longer length
        # are more likely to be correct. In this code, we assume that trajectories with length >=
        # length_belief_threshold are ground truth and are used for pairing.
        if verbose:
            print("Step 2: Pairing the trajectories.")
            current_time = time()
        self.traj_pool.coarse_pair(traj_length_belief=self.traj_length_belief,
                                   full_state_length_belief=self.full_state_length_belief,
                                   num_entities=self.num_keypoints,
                                   verbose=verbose,
                                   detailed_verbose=False)
        if verbose:
            coarse_pair_time = time() - current_time
        self.traj_pool.fine_pair(full_state_length_belief=self.full_state_length_belief,
                                 overlap_length_threshold=self.overlap_length_threshold,
                                 num_entities=self.num_keypoints,
                                 verbose=verbose,
                                 detailed_verbose=False)
        if verbose:
            fine_pair_time = time() - current_time - coarse_pair_time
            print(f"Coarse Pair: {coarse_pair_time:.2f}s, Fine Pair: {fine_pair_time:.2f}s, "
                  f"Total: {time() - current_time:.2f}s")

        # Step 3: Refine the trajectories: Fill the gaps with fragment positions.
        if verbose:
            print("Step 3: Refining the trajectories.")
            current_time = time()

        self.traj_pool.refine_trajectories(verbose=verbose, detailed_verbose=False)

        if verbose:
            refine_time = time() - current_time
        for traj in self.traj_pool.fine_paired_trajectories:
            traj['trajectory'].interpolate_gaps()
        if verbose:
            interpolate_time = time() - current_time - refine_time
            print(f"Refine: {refine_time:.2f}s, Interpolate: {interpolate_time:.2f}s, "
                  f"Total: {time() - current_time:.2f}s")
            
        for traj in self.traj_pool.fine_paired_trajectories:
            traj['trajectory'].get_all_break_distances()

        return self.traj_pool