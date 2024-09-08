from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment
from icecream import ic
from mpl_toolkits.mplot3d import Axes3D
from time import time
from ..util.tqdm import TQDM as tqdm
from ..preprocess.evaluate_annotation_v2 import create_fish

INFINITY = 100000

def distance_to_ray(A, B, C):
    """
    Calculate the distance from point C to the ray extending from A through B.
    
    Parameters:
    A (tuple): Coordinates of point A (xa, ya).
    B (tuple): Coordinates of point B (xb, yb).
    C (tuple): Coordinates of point C (xc, yc).
    
    Returns:
    float: The shortest distance from point C to the ray from A through B.
    """
    # Convert points to numpy arrays
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    
    # Vectors AB and AC
    AB = B - A
    AC = C - A
    
    # Project vector AC onto AB using dot product
    projection_length = np.dot(AC, AB) / np.dot(AB, AB)
    projection_vector = projection_length * AB
    
    # Calculate the vector from A to the projection point on AB
    projection_point = A + projection_vector

    # Check if the projection falls behind point A on the ray
    if projection_length < 0:
        # If the projection is behind A, use the distance from C to A
        distance = np.linalg.norm(AC)
    else:
        # Otherwise, calculate the distance from C to the projection point
        distance = np.linalg.norm(C - projection_point)

    return distance

def calculate_distance_between_trajectories(prev_traj, later_traj, gap, window=3, per_frame=False):
    """
    Calculate the distance between two trajectories by predicting positions at the middle timestep.

    Args:
        prev_traj: The previous trajectory (JoinedTrajectory object).
        later_traj: The later trajectory (JoinedTrajectory object).
        gap: The gap between the two trajectories (int).
        window: The number of momentums to consider for the moving average.
        per_frame: Whether to calculate the distance per frame or not.

    Returns:
        The distance between the predicted positions at the middle timestep.
    """

    if gap < 0:
        # Directly calculate the distance between the overlapping positions
        # The length of the overlapping positions is the absolute value of the gap
        overlap_length = abs(gap)

        prev_traj_overlap = prev_traj[-overlap_length:]
        later_traj_overlap = later_traj[:overlap_length]

        distances = [np.linalg.norm(prev_pos - later_pos)
                     for prev_pos, later_pos in zip(prev_traj_overlap, later_traj_overlap)]

        distance = np.mean(distances)

        return distance

    if len(prev_traj) == 1 and len(later_traj) == 1:
        distance = calc_dist_between_short_trajs(prev_traj, later_traj)
    elif len(prev_traj) == 1 and len(later_traj) > 1:
        distance = calc_dist_between_short_traj_and_normal_traj(prev_traj, later_traj, gap=gap, window=window)
    elif len(later_traj) == 1 and len(prev_traj) > 1:
        distance = calc_dist_between_normal_traj_and_short_traj(prev_traj, later_traj, gap=gap, window=window)
    else:
        half_time_between_traj = (gap + 1) / 2

        prev_traj_end_pos = prev_traj[-1]
        later_traj_start_pos = later_traj[0]

        prev_velocities = [prev_traj[-i - 1] - prev_traj[-i - 2]
                        for i in range(min(window, len(prev_traj) - 1))
                        if prev_traj[-i - 1] is not None and prev_traj[-i - 2] is not None]
        later_velocities = [later_traj[i + 1] - later_traj[i]
                            for i in range(min(window, len(later_traj) - 1))
                            if later_traj[i + 1] is not None and later_traj[i] is not None]
        
        if len(prev_velocities) == 0 and len(later_velocities) == 0:
            distance = calc_dist_between_short_trajs(prev_traj[-1:], later_traj[:1])
        elif len(prev_velocities) == 0:
            distance = calc_dist_between_short_traj_and_normal_traj(
                prev_traj[-1:], later_traj, gap=gap, window=window)
        elif len(later_velocities) == 0:
            distance = calc_dist_between_normal_traj_and_short_traj(
                prev_traj, later_traj[:1], gap=gap, window=window)
        else:

            prev_avg_velocity = np.mean(prev_velocities, axis=0)
            later_avg_velocity = np.mean(later_velocities, axis=0)

            pred_from_prev = prev_traj_end_pos + half_time_between_traj * prev_avg_velocity
            pred_from_later = later_traj_start_pos - \
                half_time_between_traj * later_avg_velocity

            distance = np.linalg.norm(pred_from_prev - pred_from_later)

    if per_frame:
        distance /= gap + 1

    return distance


def calc_dist_between_short_traj_and_normal_traj(short_traj, normal_traj, gap=0, window=3):
    """
    Calculate the distance between a short trajectory and a normal trajectory.

    Args:
        short_traj: The short trajectory (JoinedTrajectory object), with length == 1.
        normal_traj: The normal trajectory (JoinedTrajectory object).
        gap: The gap between the two trajectories (int).
        window: The number of momentums to consider for the moving average.

    Returns:
        The distance between the short trajectory and the normal trajectory.
    """

    if len(short_traj) != 1:
        raise ValueError("The short trajectory should have length == 1.")

    prev_traj_end_pos = short_traj[-1]
    later_traj_start_pos = normal_traj[0]

    later_velocities = [normal_traj[i + 1] - normal_traj[i]
                        for i in range(min(window, len(normal_traj) - 1))
                        if normal_traj[i + 1] is not None and normal_traj[i] is not None]
    later_avg_velocity = np.mean(later_velocities, axis=0)

    pred_from_later = later_traj_start_pos - later_avg_velocity * (gap + 1)

    distance = np.linalg.norm(pred_from_later - prev_traj_end_pos)

    return distance


def calc_dist_between_normal_traj_and_short_traj(normal_traj, short_traj, gap=0, window=3):
    """
    Calculate the distance between a normal trajectory and a short trajectory.

    Args:
        normal_traj: The normal trajectory (JoinedTrajectory object).
        short_traj: The short trajectory (JoinedTrajectory object), with length == 1.
        gap: The gap between the two trajectories (int).
        window: The number of momentums to consider for the moving average.

    Returns:
        The distance between the normal trajectory and the short trajectory.
    """

    if len(short_traj) != 1:
        raise ValueError("The short trajectory should have length == 1.")

    prev_traj_end_pos = normal_traj[-1]
    later_traj_start_pos = short_traj[0]

    prev_velocities = [normal_traj[-i - 1] - normal_traj[-i - 2]
                       for i in range(min(window, len(normal_traj) - 1))
                          if normal_traj[-i - 1] is not None and normal_traj[-i - 2] is not None]
    prev_avg_velocity = np.mean(prev_velocities, axis=0)

    pred_from_prev = prev_traj_end_pos + prev_avg_velocity * (gap + 1)

    distance = np.linalg.norm(pred_from_prev - later_traj_start_pos)

    return distance


def calc_dist_between_short_trajs(short_traj1, short_traj2):
    """
    Calculate the distance between two short trajectories.

    Args:
        short_traj1: The first short trajectory (JoinedTrajectory object), with length == 1.
        short_traj2: The second short trajectory (JoinedTrajectory object), with length == 1.

    Returns:
        The distance between the two short trajectories.
    """

    if len(short_traj1) != 1 or len(short_traj2) != 1:
        raise ValueError("The short trajectories should have length == 1.")

    distance = np.linalg.norm(short_traj1[-1] - short_traj2[0])

    return distance


class JoinedTrajectory:

    def __init__(self, *trajectories, break_type=None):
        """
        Args:
            trajectories: A list of trajectories to be joined together.
        """

        self.trajectory = []
        self.confidences = []
        self.match_types = []
        self.sources = []
        self.shape = None

        self.breakpoints = {}
        accumulated_length = 0
        merge = False

        for traj in trajectories:
            if isinstance(traj, int):
                if traj >= 0:
                    self.trajectory.extend((None, ) * traj)
                    self.confidences.extend((0, ) * traj)
                    self.match_types.extend(('None', ) * traj)
                    self.sources.extend(('None', ) * traj)
                else:
                    merge = abs(traj) # > 0
                    self.trajectory[-merge:] = [cd / 2 for cd in self.trajectory[-merge:]]
                    self.confidences[-merge:] = [cf / 2 for cf in self.confidences[-merge:]]
                    self.sources[-merge:] = ['merge'] * merge
                self.breakpoints[accumulated_length] = {'length': traj,
                                                        'distance_per_frame': None,
                                                        'type': break_type,
                                                        'refined': False}
                accumulated_length += traj
            else:
                if self.shape is None:
                    self.shape = traj.shape
                elif self.shape != traj.shape:
                    raise ValueError(
                        "All trajectories should have the same shape.")
                if merge:
                    distance_per_frame = np.mean([np.linalg.norm(cd_a * 2 - cd_b) for cd_a, cd_b in zip(self.trajectory[-merge:], traj.trajectory)])
                    self.breakpoints[accumulated_length+merge]['distance_per_frame'] = distance_per_frame
                    self.trajectory[-merge:] = [cd_a + cd_b / 2
                                                for cd_a, cd_b in zip(self.trajectory[-merge:], traj.trajectory)]
                    self.confidences[-merge:] = [cf_a + cf_b / 2
                                                for cf_a, cf_b in zip(self.confidences[-merge:], traj.confidences)]
                self.trajectory.extend(traj.trajectory[merge:])
                self.confidences.extend(traj.confidences[merge:])
                self.match_types.extend(traj.match_types[merge:])
                self.sources.extend(traj.sources[merge:])
                merge = False
                for k, v in traj.breakpoints.items():
                    self.breakpoints[k + accumulated_length] = v
                accumulated_length += len(traj)

        if self.shape is None:
            raise ValueError(
                "At least one non-empty trajectory should be provided.")

        self.break_distances = {k: None
                                 for k in self.breakpoints.keys()}

    def get_break_distance(self, index, window=3):
        if self.breakpoints[index]['type'] == 'padding':
            return None
        if self.breakpoints[index]['distance_per_frame'] is None:
            if (self.trajectory[index+self.breakpoints[index]['length']+1] is None
                and self.trajectory[index-2] is None):
                raise ValueError(
                    f"Index {index} is not a valid breakpoint. It's surrounded by lenght=1 trajs.")
            elif self.trajectory[index-2] is None:
                self.breakpoints[index]['distance_per_frame'] = calculate_distance_between_trajectories(
                    self.trajectory[index-1:index],
                    self.trajectory[index + self.breakpoints[index]['length']:
                                    index + self.breakpoints[index]['length'] + window + 1],
                    gap=self.breakpoints[index]['length'],
                    window=window,
                    per_frame=True)
            elif self.trajectory[index+self.breakpoints[index]['length']+1] is None:
                self.breakpoints[index]['distance_per_frame'] = calculate_distance_between_trajectories(
                    self.trajectory[max(index-window-1, 0):index],
                    self.trajectory[index + self.breakpoints[index]['length']:
                                    index + self.breakpoints[index]['length']+1],
                    gap=self.breakpoints[index]['length'],
                    window=window,
                    per_frame=True)
            elif self.trajectory[index-2] is not None and self.trajectory[index-1] is not None:
                self.breakpoints[index]['distance_per_frame'] = calculate_distance_between_trajectories(
                    self.trajectory[max(index-window-1, 0):index],
                    self.trajectory[index + self.breakpoints[index]['length']:
                                    index + self.breakpoints[index]['length']+window+1],
                    gap=self.breakpoints[index]['length'],
                    window=window,
                    per_frame=True)
            else:
                raise ValueError(
                    f"Index {index} is not a valid breakpoint. Its previous position is None.")
        return self.breakpoints[index]['distance_per_frame']

    def get_all_break_distances(self):
        for index in self.breakpoints:
            self.get_break_distance(index)

    def evaluate_insertion(self, index, new_traj, window=3):
        """
        Evaluate an insertion of a new trajectory at a specific index.
        The index and length of the insertion must be within the range of a breakpoint.

        Args:
            index: The index of the insertion.
            new_traj: The new trajectory to be inserted.
            window: The number of momentums to consider for the moving average.
        
        Returns:
            The change in the breakpoint distance if the insertion is accepted.
            If the value is negative, it means that the insertion successfully refines the trajectory.
            The more negative the value, the better the refinement.
        """

        # Find the breakpoint that the index belongs to
        break_index = None
        for breakpoint, info in self.breakpoints.items():
            if breakpoint <= index < breakpoint + info['length']:
                if index + len(new_traj) > breakpoint + info['length']:
                    return INFINITY
                break_index = breakpoint
                break
        if (break_index is None or
            break_index == 0 or
            break_index + self.breakpoints[break_index]['length'] == len(self.trajectory)):
            return INFINITY

        if (len(new_traj) == 1 and # It's a length=1 traj
            (index != break_index and # It's a disconnected traj
             index != break_index + self.breakpoints[break_index]['length'] - 1) and
            (self.trajectory[break_index-2] is None or # Adjacent traj is length=1
             self.trajectory[break_index + self.breakpoints[break_index]['length'] + 1] is None)):
            # Does not allow insertion of disconnected length=1 traj next to
            # another disconnected length=1 traj
            return INFINITY

        # Calculate the breakpoint distance if the insertion was accepted
        if (len(new_traj) == 1 and
            index == break_index and
            self.trajectory[break_index-2] is None):
            prev_break_index = None
            for breakpoint, info in self.breakpoints.items():
                if breakpoint < break_index and breakpoint + info['length'] == break_index - 1:
                    prev_break_index = breakpoint
                    break
            if prev_break_index is None:
                raise ValueError(
                    f"Index {break_index} does not have a valid previous breakpoint.")
            prev_part = calculate_distance_between_trajectories(
                self.trajectory[
                    max(prev_break_index - window - 1, 0) : prev_break_index],
                [self.trajectory[break_index-1], new_traj[0]],
                gap=self.breakpoints[prev_break_index]['length'],
                window=window,
                per_frame=True)
            later_part = calculate_distance_between_trajectories(
                [self.trajectory[break_index-1], new_traj[0]],
                self.trajectory[
                    break_index + self.breakpoints[break_index]['length'] :
                    break_index + self.breakpoints[break_index]['length'] + window + 1],
                gap=self.breakpoints[break_index]['length']-1,
                window=window,
                per_frame=True)
            final_dist = (prev_part + later_part) / 2
            old_dist = (self.get_break_distance(prev_break_index) +
                        self.get_break_distance(break_index)) / 2
            return final_dist - old_dist
        elif (len(new_traj) == 1 and
              index == break_index + self.breakpoints[break_index]['length'] - 1 and
              self.trajectory[break_index-2] is None):
            new_dist = calculate_distance_between_trajectories(
                [self.trajectory[break_index-1]],
                [new_traj[0]] + self.trajectory[
                    index + 1 : index + window + 2], 
                gap=self.breakpoints[break_index]['length']-1,
                window=window,
                per_frame=True)
            return new_dist - self.get_break_distance(break_index)
        elif (len(new_traj) == 1 and
                index == break_index and
                self.trajectory[break_index + self.breakpoints[break_index]['length'] + 1] is None):
            new_dist = calculate_distance_between_trajectories(
                self.trajectory[max(0, index-window-1):index] + [new_traj[0]],
                [self.trajectory[break_index + self.breakpoints[break_index]['length']]],
                gap=self.breakpoints[break_index]['length'] - 1,
                per_frame=True,
                window=window)
            return new_dist - self.get_break_distance(break_index)
        elif (len(new_traj) == 1 and
                index == break_index + self.breakpoints[break_index]['length'] - 1 and
                self.trajectory[index + 2] is None):
            prev_part = calculate_distance_between_trajectories(
                self.trajectory[
                    max(0, break_index - window - 1) : break_index],
                [new_traj[0], self.trajectory[index + 1]],
                gap=self.breakpoints[break_index]['length'] - 1,
                window=window,
                per_frame=True)
            next_break_index = index + 2
            later_part = calculate_distance_between_trajectories(
                [new_traj[0], self.trajectory[index + 1]],
                self.trajectory[
                    next_break_index + self.breakpoints[next_break_index]['length']:
                    next_break_index + self.breakpoints[next_break_index]['length'] + window + 1],
                gap=self.breakpoints[break_index]['length'] - 1,
                window=window,
                per_frame=True)
            new_dist = (prev_part + later_part) / 2
            old_dist = (self.get_break_distance(break_index) +
                        self.get_break_distance(next_break_index)) / 2
            return new_dist - old_dist
        else:
            prev_part = calculate_distance_between_trajectories(
                self.trajectory[
                    max(0, break_index - window - 1) : break_index],
                new_traj,
                gap=index-break_index,
                window=window,
                per_frame=True)
            later_part = calculate_distance_between_trajectories(
                new_traj,
                self.trajectory[break_index + self.breakpoints[break_index]['length']:
                                break_index + self.breakpoints[break_index]['length'] + window + 1],
                gap=self.breakpoints[break_index]['length']+break_index-index-len(new_traj),
                window=window,
                per_frame=True)
            final_dist = (prev_part + later_part) / 2

            return final_dist - self.get_break_distance(break_index)

    def insert(self, index, new_traj):
        """
        Insert a new trajectory at a specific index.
        This is only used for refinement.

        Args:
            index: The index of the insertion.
            new_traj: The new trajectory to be inserted.
        """

        # Find the breakpoint that the index belongs to
        break_index = None
        for breakpoint, info in self.breakpoints.items():
            if breakpoint <= index < breakpoint + info['length']:
                if index + len(new_traj) > breakpoint + info['length']:
                    raise ValueError(
                        "The length of the new trajectory exceeds the length of the breakpoint.")
                break_index = breakpoint
                break
        if break_index is None:
            raise ValueError(
                "The index does not belong to any breakpoint.")
        
        if (len(new_traj) == 1 and # It's a length=1 traj
            (index != break_index and # It's a disconnected traj
             index != break_index + self.breakpoints[break_index]['length'] - 1) and
            (self.trajectory[break_index-2] is None or # Adjacent traj is length=1
             self.trajectory[break_index + self.breakpoints[break_index]['length'] + 1] is None)):
            raise ValueError(
                "The insertion of a disconnected trajectory next to another disconnected trajectory is not allowed.")

        # Insert the new trajectory
        self.trajectory[index:index + len(new_traj)] = new_traj.trajectory
        self.confidences[index:index + len(new_traj)] = new_traj.confidences
        self.match_types[index:index + len(new_traj)] = new_traj.match_types
        self.sources[index:index + len(new_traj)] = ['refine'] * len(new_traj)
        
        self.breakpoints[index+len(new_traj)] = {
            'length': break_index + self.breakpoints[break_index]['length'] - index - len(new_traj),
            'distance_per_frame': None,
            'type': self.breakpoints[break_index]['type'],
            'refined': True}
        
        self.breakpoints[break_index]['length'] = index - break_index
        self.breakpoints[break_index]['distance_per_frame'] = None
        self.breakpoints[break_index]['refined'] = True
    
    def get_pairing_timeline(self):
        
        pairing_methods_timeline = []
        pairing_method = 'initial'
        current_index = 0
        for key in sorted(self.breakpoints.keys()):
            pairing_methods_timeline.extend(
                [pairing_method] * (key - current_index))
            pairing_method = self.breakpoints[key]['type']
            current_index = key
        pairing_methods_timeline.extend(
            [pairing_method] * (len(self.trajectory) - current_index))

        return pairing_methods_timeline

    def __len__(self):
        return len(self.trajectory)

    def __getitem__(self, index):
        return self.trajectory[index]

    def average_confidence(self):
        return np.mean(self.confidences)

    def append(self, new_position, confidence, match_type):
        """
        Args:
            new_coords: A new point to be connected to the existing trajectory.
            confidence: The confidence of the new point.
            match_type: The type of the match for the new point.
        """
        # Check if the new_coords matches the dimension of the existing trajectory.
        if new_position.shape != self.shape:
            raise ValueError(
                f"The new coordinates with shape {new_position.shape}"
                "does not match the dimension of the existing trajectory with shape {self.shape}."
            )
        self.trajectory.append(new_position)
        self.confidences.append(confidence)
        self.match_types.append(match_type)
        self.sources.append('append')

    def prepend(self, new_position, confidence, match_type):
        """
        Args:
            new_coords: A new point to be connected to the existing trajectory.
            confidence: The confidence of the new point.
            match_type: The type of the match for the new point.
        """
        # Check if the new_coords matches the dimension of the existing trajectory.
        if new_position.shape != self.shape:
            raise ValueError(
                f"The new coordinates with shape {new_position.shape}"
                "does not match the dimension of the existing trajectory with shape {self.shape}."
            )
        self.trajectory.insert(0, new_position)
        self.confidences.insert(0, confidence)
        self.match_types.insert(0, match_type)
        self.sources.insert(0, 'prepend')

    def calculate_distances_head_only(self, new_positions):
        """
        Only consider the head position.
        This is deprecated and should not be used.
        """
        return np.linalg.norm(self.trajectory[-1][:2] - np.array(new_positions)[:, :2], axis=1)

    def calculate_distances(self, new_positions):
        """
        Consider both the head position and the midsection position.
        """
        return np.linalg.norm(self.trajectory[-1] - new_positions, axis=1)

    def calculate_distances_momentum(self, new_position, window=3):
        """
        Consider the head position, the midsection position, 
        and the momentum of the trajectory based on a moving average of the last `window` positions.

        Args:
            new_position: A new point to be compared to the predicted next position.
            window: The number of positions to consider for the moving average.

        Returns:
            The distance between the predicted next position and the new position.
        """
        trajectory_length = len(self.trajectory)

        # Fall back to the regular distance calculation if the trajectory is too short.
        if trajectory_length <= 1 or window == 0:
            return self.calculate_distances(new_position)

        window = min(window, trajectory_length - 1)
        velocities = [self.trajectory[-1-i] - self.trajectory[-2-i]
                      for i in range(0, window)]
        avg_velocity = np.sum(velocities, axis=0) / window
        momentum_pred = self.trajectory[-1] + avg_velocity

        return np.linalg.norm(momentum_pred - new_position, axis=1)

    
    def visualize(self, size=5, aspect_ratio=(1, 1, 0.5)):
        # Filter out None values
        trajectory = [x for x in self.trajectory if x is not None]

        # Determine the number of points per position
        points_per_position = self.shape[0] // 2

        # Create a 3D plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(points_per_position):
            # Extract x and y coordinates for the current point
            x = [pos[i * 2] for pos in trajectory]
            y = [pos[i * 2 + 1] for pos in trajectory]

            # Create time values for each point
            time = list(range(len(x)))

            # Set the transparency (alpha) for each point
            alpha = 0.75 if i == 0 else 0.25

            # Plot the trajectory in 3D
            ax.scatter(x, y, time, c='blue', alpha=alpha, s=size, label=f"Point {i + 1}")

            # Plot breakpoints with 'X' marker
            for breakpoint in self.breakpoints:
                if breakpoint < len(self.trajectory):
                    breakpoint_pos_start = self.trajectory[breakpoint-1]
                    ax.scatter(
                        breakpoint_pos_start[i * 2], breakpoint_pos_start[i * 2 + 1], breakpoint, 
                        marker='X', s=size*5, c='red'
                    )
                    breakpoint_pos_end = self.trajectory[breakpoint + self.breakpoints[breakpoint]['length']]
                    ax.scatter(
                        breakpoint_pos_end[i * 2], breakpoint_pos_end[i * 2 + 1],
                        breakpoint + self.breakpoints[breakpoint]['length'],
                        marker='D', s=size*5, c='black'
                    )

        # Set the title and labels for the plot
        ax.set_box_aspect(aspect_ratio)  # Adjust the ratio as needed
        ax.set_title("Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Time")

        # Add a legend
        ax.legend()

        # Display the plot
        plt.show()

    def mean_distance_per_frame(self):
        """
        Returns:
            The mean distance between consecutive frames in the trajectory.
        """
        output = []
        for i in range(self.shape[0] // 2):
            output.append(np.mean(np.linalg.norm(
                np.diff(np.array(self.trajectory)[:, i*2: i*2+2], axis=0), axis=1)))
        return output

    def join(self, trajectory, gap=0, break_type=None):
        """
        Args:
            trajectory: Another trajectory to be connected to the existing trajectory.
        """
        if trajectory.shape != self.shape:
            raise ValueError(
                "The new coordinates with shape {trajectory.shape}"
                "does not match the dimension of the existing trajectory with shape {self.shape}."
            )
        return JoinedTrajectory(self, gap, trajectory, break_type=break_type)

    def interpolate_gaps(self, window=3):
        """
        Fill in the None values at the breakpoints by predicting the positions
        according to the previous and future positions with momentum.
        """
        for start, info in self.breakpoints.items():

            length = info['length']
            if length == 0:
                continue
            elif start == 0:
                next_pos = self.trajectory[start + length]
                next_velocities = [self.trajectory[start + length + i + 1] - self.trajectory[start + length + i]
                                    for i in range(min(window, len(self.trajectory) - start - length - 1))
                                    if self.trajectory[start + length + i + 1] is not None and self.trajectory[start + length + i] is not None]
                if len(next_velocities) >= 1:
                    next_avg_velocity = np.mean(next_velocities, axis=0)
                    for i in range(length):
                        interpolated_pos = next_pos - next_avg_velocity * (length - i)
                        self.trajectory[start + i] = interpolated_pos
                        if self.sources[start + i] == 'none':
                            self.sources[start + i] = 'interpolate'
                elif len(next_velocities) == 0 and self.trajectory[start + length] is not None:
                    for i in range(length):
                        self.trajectory[start + i] = self.trajectory[start + length]
                        if self.sources[start + i] == 'none':
                            self.sources[start + i] = 'interpolate'
                else:
                    raise ValueError(
                        f"Index {start} does not have valid future positions.")
                continue
            elif start + length == len(self.trajectory):
                prev_pos = self.trajectory[start - 1]
                prev_velocities = [self.trajectory[start - i - 1] - self.trajectory[start - i - 2]
                                    for i in range(min(window, start))
                                    if self.trajectory[start - i - 1] is not None and self.trajectory[start - i - 2] is not None]
                if len(prev_velocities) >= 1:
                    prev_avg_velocity = np.mean(prev_velocities, axis=0)
                    for i in range(length):
                        interpolated_pos = prev_pos + prev_avg_velocity * (i + 1)
                        self.trajectory[start + i] = interpolated_pos
                        if self.sources[start + i] == 'none':
                            self.sources[start + i] = 'interpolate'
                elif len(prev_velocities) == 0 and self.trajectory[start - 1] is not None:
                    for i in range(length):
                        self.trajectory[start + i] = self.trajectory[start - 1]
                        if self.sources[start + i] == 'none':
                            self.sources[start + i] = 'interpolate'
                else:
                    raise ValueError(
                        f"Index {start} does not have valid previous positions.")
                continue

            prev_pos = self.trajectory[start - 1]
            next_pos = self.trajectory[start + length]

            # Calculate the velocities based on the previous and future positions
            prev_velocities = [self.trajectory[start - i - 1] - self.trajectory[start - i - 2]
                                for i in range(min(window, start))
                                if self.trajectory[start - i - 1] is not None and self.trajectory[start - i - 2] is not None]
            next_velocities = [self.trajectory[start + length + i + 1] - self.trajectory[start + length + i]
                                for i in range(min(window, len(self.trajectory) - start - length - 1))
                                if self.trajectory[start + length + i + 1] is not None and self.trajectory[start + length + i] is not None]

            if len(prev_velocities) == 0 and len(next_velocities) == 0:
                raise ValueError(
                    f"Index {start} does not have valid previous and future positions.")
            elif len(prev_velocities) == 0:
                # Assume that the previous velocity is the same as the next velocity
                prev_avg_velocity = np.mean(next_velocities, axis=0)
                next_avg_velocity = prev_avg_velocity
            elif len(next_velocities) == 0:
                # Assume that the next velocity is the same as the previous velocity
                next_avg_velocity = np.mean(prev_velocities, axis=0)
                prev_avg_velocity = next_avg_velocity
            else:
                prev_avg_velocity = np.mean(prev_velocities, axis=0)
                next_avg_velocity = np.mean(next_velocities, axis=0)

            # Interpolate the positions within the gap

            for i in range(length):
                # Predict position based on previous positions and velocities
                pred_from_past = prev_pos + prev_avg_velocity * (i + 1)

                # Predict position based on future positions and velocities (in a backward manner)
                pred_from_future = next_pos - next_avg_velocity * (length - i)

                # Calculate the weights for averaging the predictions
                past_weight = (length - i) / (length + 1)
                future_weight = (i + 1) / (length + 1)

                # Average the predictions with the calculated weights
                interpolated_pos = past_weight * pred_from_past + future_weight * pred_from_future
                self.trajectory[start + i] = interpolated_pos
                self.sources[start + i] = 'interpolate'


class DirectTrajectory(JoinedTrajectory):
    def __init__(self,
                 trajectory: list,
                 confidences: list,
                 match_types: list,
                 sources: list = None):
        self.trajectory = trajectory
        self.confidences = confidences
        self.match_types = match_types
        if sources is None:
            self.sources = ['direct'] * len(trajectory)
        else:
            self.sources = sources
        if not len(trajectory) == len(confidences) == len(self.sources) == len(match_types):
            raise ValueError(
                "The length of trajectory, confidences, match_types and sources should be the same.")
        self.shape = trajectory[0].shape
        self.breakpoints = dict()

class Trajectory(JoinedTrajectory):
    def __init__(self, initial_position, initial_confidence, initial_match_type):
        """
        Args:
            initial_position: The starting position of the trajectory.
            initial_confidence: The confidence of the starting position.
            initial_match_type: The type of the match for the starting position.
        """
        # Check if the initial_coord represents a single point.
        if len(initial_position.shape) != 1:
            raise ValueError(
                f"The initial coordinates should represent a single point, but got shape {initial_position.shape}."
            )
        self.trajectory = [initial_position]
        self.confidences = [initial_confidence]
        self.match_types = [initial_match_type]
        self.sources = ['append']
        self.shape = initial_position.shape
        self.breakpoints = dict()


class TrajectoryPool:

    def __init__(self, initial_positions, initial_confidences, initial_match_types):
        # Check if the shape of the initial_coords is all the same.
        if len(set([initial_position.shape for initial_position in initial_positions])) != 1:
            raise ValueError(
                "The initial coordinates should all have the same shape, but got different shapes."
            )
        self.all_trajectories = [{'trajectory': Trajectory(initial_position,
                                                           initial_confidence,
                                                           initial_match_type),
                                  'start': 0,
                                  'end': None,
                                  'discard': False,
                                  'prepend': 0,
                                  'fragment': False,
                                  'child_of': None}
                                 for initial_position, initial_confidence, initial_match_type in zip(
                                 initial_positions, initial_confidences, initial_match_types)]

        self.shape = initial_positions[0].shape
        self.end_value_assigned = False
        self.prepend_paired = False
        self.fragment_paired = False
        self.coarse_paired = False
        self.fine_paired = False

        # In Tracker object:
        # 1. Initialize the TrajectoryPool
        # 2. Enter Loop, in each loop:
        #   a. Call new_time_step
        #      i. Check if there is any trajectory terminated in the current frame
        #      ii. Update the active trajs in the current frame to the timeline
        #      iii. Current frame + 1 (New frame)
        #   b. Call add_trajectory or update_trajectory
        self.current_frame = 0
        self.timeline = []
        self.updated_traj_id = [i for i in range(len(initial_positions))]

    def new_time_step(self):

        if len(self.timeline) > 0:
            for traj_id in self.timeline[-1]:
                if traj_id not in self.updated_traj_id:
                    self.all_trajectories[traj_id]['end'] = self.current_frame

        self.timeline.append(self.updated_traj_id)
        self.current_frame += 1
        self.updated_traj_id = []

    def get_previous_traj_ids(self):
        return self.timeline[-1]

    def add_trajectory(self, start_position, start_confidence, start_match_type):

        if start_position.shape != self.shape:
            raise ValueError(
                f"The new coordinates with shape {start_position.shape}"
                "does not match the dimension of the existing trajectories with shape {self.shape}."
            )

        self.updated_traj_id.append(len(self.all_trajectories))
        self.all_trajectories.append({'trajectory': Trajectory(start_position,
                                                               start_confidence,
                                                               start_match_type),
                                      'start': self.current_frame,
                                      'end': None,
                                      'discard': False,
                                      'prepend': 0,
                                      'fragment': False,
                                      'child_of': None})

    def add_full_trajectory(self, positions, confidences, match_types, start_frame):
        '''
        Add a full trajectory to the pool. This is used during fragment pairing.

        Args:
            positions: A list of positions of the trajectory.
            confidences: A list of confidences of the trajectory.
            match_types: A list of match types of the trajectory.
            start_frame: The frame number of the starting frame of the trajectory.
        '''
        if not len(positions) == len(confidences) == len(match_types):
            raise ValueError(
                "The length of positions, confidences and match_types should be the same.")
        if positions[0].shape != self.shape:
            raise ValueError(
                f"The new coordinates with shape {positions[0].shape}"
                "does not match the dimension of the existing trajectories with shape {self.shape}."
            )
        self.all_trajectories.append({
            'trajectory': DirectTrajectory(positions,
                                           confidences,
                                            match_types,
                                           ['fragment'] * len(positions)),
            'start': start_frame,
            'end': start_frame + len(positions),
            'discard': False,
            'prepend': 0,
            'fragment': True,
            'child_of': None})
        for i in range(start_frame, start_frame + len(positions)):
            self.timeline[i].append(len(self.all_trajectories) - 1)

    def update_trajectory(self, traj_id, new_position, new_confidence, new_match_type, prepend=False):

        if not prepend:
            # This is for building the traj_pool, executed by Tracker
            if traj_id not in self.timeline[-1]:
                raise ValueError(
                    f"Trajectory with ID {traj_id} is not active.")

            if traj_id in self.updated_traj_id:
                raise ValueError(
                    f"Trajectory with ID {traj_id} is already updated.")

            self.updated_traj_id.append(traj_id)
            self.all_trajectories[traj_id]['trajectory'].append(new_position,
                                                                new_confidence,
                                                                new_match_type)
        else:
            # This is executed by short_pair()
            self.all_trajectories[traj_id]['trajectory'].prepend(new_position,
                                                                 new_confidence,
                                                                 new_match_type)
            self.all_trajectories[traj_id]['start'] -= 1
            self.all_trajectories[traj_id]['prepend'] += 1
            self.timeline[self.all_trajectories[traj_id]
                          ['start']].append(traj_id)

    def assign_end_value(self):
        for traj in self.all_trajectories:
            if traj['end'] is None:
                traj['end'] = len(self.timeline)
        self.end_value_assigned = True

    def join_trajectories(self, chain, from_pool, to_pool, join_type=None):
        for traj_id in chain:
            if from_pool[traj_id]['child_of'] is None:
                from_pool[traj_id]['child_of'] = (join_type, len(to_pool))
            else:
                ic(chain, traj_id, from_pool[traj_id])
                raise ValueError(
                    "The trajectory is already a child of another trajectory.")
        if len(chain) > 1:
            trajs_to_be_joined = [from_pool[chain[0]]['trajectory']]
            end_frame = from_pool[chain[0]]['end']
            for traj_id in chain[1:]:
                start_frame = from_pool[traj_id]['start']
                trajs_to_be_joined.append(start_frame - end_frame)
                trajs_to_be_joined.append(from_pool[traj_id]['trajectory'])
                end_frame = from_pool[traj_id]['end']
            to_pool.append({'trajectory': JoinedTrajectory(*trajs_to_be_joined, break_type=join_type),
                            'start': from_pool[chain[0]]['start'],
                            'end': from_pool[chain[-1]]['end'],
                            'chain': chain,
                            'child_of': None})
        else:
            to_pool.append({'trajectory': from_pool[chain[0]]['trajectory'],
                            'start': from_pool[chain[0]]['start'],
                            'end': from_pool[chain[0]]['end'],
                            'chain': chain,
                            'child_of': None})

    def rearrange_fragments(self, fragments, verbose=False):
        """
        Rearrange the fragments to form a complete trajectory.
        Args:
            fragments: A list of fragments to be rearranged.
            verbose: Whether to print out the rearrangement process.
        """

        smaller_fragments = []

        for i, frag in tqdm(list(enumerate(fragments)), desc='Rearranging Fragments', disable=not verbose):
            breakpoints = []
            for j, frame in enumerate(list(zip(frag['positions'],
                                               frag['confidences'],
                                               frag['match_types']))):
                if len(frame[0]) == 0:
                    breakpoints.append(j)
            # Break fragment into smaller fragments
            if len(breakpoints) == 0:
                smaller_fragments.append(frag)
            else:
                previous_index = 0
                for k, breakpoint in enumerate(breakpoints):
                    if breakpoint > previous_index:
                        smaller_fragments.append({'positions': frag['positions'][previous_index:breakpoint],
                                                  'confidences': frag['confidences'][previous_index:breakpoint],
                                                  'match_types': frag['match_types'][previous_index:breakpoint],
                                                  'start_frame': frag['start_frame'] + previous_index,
                                                  'end_frame': frag['start_frame'] + breakpoint})
                    if k == len(breakpoints) - 1:
                        if breakpoint < len(frag['positions']) - 1:
                            smaller_fragments.append({'positions': frag['positions'][breakpoint+1:],
                                                      'confidences': frag['confidences'][breakpoint+1:],
                                                        'match_types': frag['match_types'][breakpoint+1:],
                                                      'start_frame': frag['start_frame'] + breakpoint + 1,
                                                      'end_frame': frag['end_frame']})
                    previous_index = breakpoint + 1

        return smaller_fragments

    def create_fragment_timeline(self, verbose=False):
        
        self.fragment_timeline = [[] for _ in range(len(self.timeline))]

        for i, frag in tqdm(list(enumerate(self.fragments)),
                            desc='Fragment Timeline',
                            disable=not verbose):
            for j in range(frag['start_frame'], frag['end_frame']):
                self.fragment_timeline[j].append(i)

    def prepend_pair(self, distance_threshold_per_frame, max_prepend_length=2, verbose=False):
        """
        Pair fragments that fail to form a complete trajectory with existing long trajectories.
        The fragments consist of short trajectories that are no more than 2 frames long.
        Args:
            distance_threshold_per_frame: The distance threshold for pairing.
            max_prepend_length: The maximum length of the short trajectory to be prepended.
                                Available options are 0, 1 and 2.
            verbose: Whether to print out the pairing process.
        """
        if max_prepend_length not in (0, 1, 2):
            raise ValueError(
                "The max_prepend_length should be 1 or 2.")

        # 1. Extract all short trajectories and form fragments
        # 2. Remove the short trajectories from the timeline

        fragments = []
        current_fragment = None

        for i, frame in tqdm(list(enumerate(self.timeline)),
                            desc='Fragment Extraction',
                            disable=not verbose):

            updated = False
            traj_ids_to_remove = []

            for traj_id in frame:

                if (len(self.all_trajectories[traj_id]['trajectory']) <= 2
                        and self.all_trajectories[traj_id]['start'] == i):

                    if current_fragment is None:
                        current_fragment = {'positions': [[self.all_trajectories[traj_id]['trajectory'][0]]],
                                            'confidences': [[self.all_trajectories[traj_id]['trajectory'].confidences[0]]],
                                            'match_types': [[self.all_trajectories[traj_id]['trajectory'].match_types[0]]],
                                            'start_frame': i,
                                            'end_frame': None}
                    else:
                        current_fragment['positions'][-1].append(
                            self.all_trajectories[traj_id]['trajectory'][0])
                        current_fragment['confidences'][-1].append(
                            self.all_trajectories[traj_id]['trajectory'].confidences[0])
                        current_fragment['match_types'][-1].append(
                            self.all_trajectories[traj_id]['trajectory'].match_types[0])

                    updated = True
                    traj_ids_to_remove.append(traj_id)

                elif (len(self.all_trajectories[traj_id]['trajectory']) == 2
                      and self.all_trajectories[traj_id]['end'] == i + 1):
                    current_fragment['positions'][-1].append(
                        self.all_trajectories[traj_id]['trajectory'][1])
                    current_fragment['confidences'][-1].append(
                        self.all_trajectories[traj_id]['trajectory'].confidences[1])
                    current_fragment['match_types'][-1].append(
                        self.all_trajectories[traj_id]['trajectory'].match_types[1])

                    updated = True
                    traj_ids_to_remove.append(traj_id)

            if updated:
                current_fragment['positions'].append([])
                current_fragment['confidences'].append([])
                current_fragment['match_types'].append([])
            else:
                if current_fragment is not None:
                    current_fragment['end_frame'] = i
                    current_fragment['positions'] = current_fragment['positions'][:-1]
                    current_fragment['confidences'] = current_fragment['confidences'][:-1]
                    current_fragment['match_types'] = current_fragment['match_types'][:-1]
                    fragments.append(current_fragment)
                    current_fragment = None

            for traj_id in traj_ids_to_remove:
                frame.remove(traj_id)
                self.all_trajectories[traj_id]['discard'] = True

        # 3. Return if the max_prepend_length is 0

        if max_prepend_length == 0:
            self.fragments = fragments
            self.prepend_paired = True
            return

        # 4. Try to connect length=1 trajs with existing long trajs

        for frag in tqdm(fragments, disable=not verbose, desc='Length=1 Prepend'):
            for i, frame in enumerate(reversed(list(zip(frag['positions'],
                                                        frag['confidences'],
                                                        frag['match_types'])))):

                pos_and_conf_to_be_removed = []

                frame_index = frag['end_frame'] - i - 1
                candidate_long_traj_ids = [traj_id for traj_id in self.timeline[frame_index+1]
                                           if (self.all_trajectories[traj_id]['start'] == frame_index+1)]
                if len(candidate_long_traj_ids) > 0:
                    weights = []
                    for pos in frame[0]:
                        temp_traj = np.array([pos])
                        weight = []
                        for traj_id in candidate_long_traj_ids:
                            weight.append(calc_dist_between_short_traj_and_normal_traj(
                                temp_traj,
                                self.all_trajectories[traj_id]['trajectory']))
                        weights.append(weight)
                    weights = np.array(weights)
                    row_ind, col_ind = linear_sum_assignment(weights)
                    for i, j in zip(row_ind, col_ind):
                        if weights[i, j] < distance_threshold_per_frame:
                            self.update_trajectory(candidate_long_traj_ids[j],
                                                   frame[0][i],
                                                   frame[1][i],
                                                    frame[2][i],
                                                   prepend=True)
                            pos_and_conf_to_be_removed.append(i)
                for index in sorted(pos_and_conf_to_be_removed, reverse=True):
                    frame[0].pop(index)
                    frame[1].pop(index)

        # 4. Rearrange the fragments
        fragments = self.rearrange_fragments(fragments)

        # 5. Return if the max_prepend_length is 1

        if max_prepend_length == 1:
            self.fragments = fragments
            self.prepend_paired = True
            return

        # 6. Try to connect length=2 trajs with existing long trajs

        for frag in tqdm(fragments, desc='Length=2 Prepend', disable=not verbose):

            if frag['end_frame'] - frag['start_frame'] == 1:
                continue

            awaiting_state = [[True] * len(frame)
                              for frame in frag['positions']]
            current_index = -1

            while -current_index <= len(frag['positions']) - 1:

                current_last_frame = frag['end_frame'] + current_index
                candidate_short_trajs = []
                for i in range(len(frag['positions'][current_index-1])):
                    for j in range(len(frag['positions'][current_index])):
                        if awaiting_state[current_index-1][i] and awaiting_state[current_index][j]:
                            candidate_short_trajs.append(
                                (current_index-1, i, current_index, j))

                if len(candidate_short_trajs) > 0:
                    candidate_long_traj_ids = [traj_id for traj_id in self.timeline[current_last_frame+1]
                                               if (self.all_trajectories[traj_id]['start'] == current_last_frame+1)]

                    if len(candidate_long_traj_ids) > 0:
                        weights = []
                        for short_traj_index in candidate_short_trajs:
                            weight = []
                            a, b, c, d = short_traj_index
                            short_traj = np.array([frag['positions'][a][b],
                                                   frag['positions'][c][d]])
                            for traj_id in candidate_long_traj_ids:
                                weight.append(calculate_distance_between_trajectories(
                                    short_traj,
                                    self.all_trajectories[traj_id]['trajectory'],
                                    current_index))
                            weights.append(weight)
                        weights = np.array(weights)
                        row_ind, col_ind = linear_sum_assignment(weights)
                        for i, j in zip(row_ind, col_ind):
                            if weights[i, j] < distance_threshold_per_frame:
                                a, b, c, d = candidate_short_trajs[i]
                                self.update_trajectory(candidate_long_traj_ids[j],
                                                       frag['positions'][c][d],
                                                       frag['confidences'][c][d],
                                                       frag['match_types'][c][d],
                                                       prepend=True)
                                self.update_trajectory(candidate_long_traj_ids[j],
                                                       frag['positions'][a][b],
                                                       frag['confidences'][a][b],
                                                       frag['match_types'][a][b],
                                                       prepend=True)
                                awaiting_state[a][b] = False
                                awaiting_state[c][d] = False
                pos_and_conf_to_be_removed = [x for x in range(len(awaiting_state[current_index]))
                                              if not awaiting_state[current_index][x]]
                for index in sorted(pos_and_conf_to_be_removed, reverse=True):
                    frag['positions'][current_index].pop(index)
                    frag['confidences'][current_index].pop(index)
                    frag['match_types'][current_index].pop(index)
                current_index -= 1
            pos_and_conf_to_be_removed = [x for x in range(len(awaiting_state[0]))
                                          if not awaiting_state[0][x]]
            for index in sorted(pos_and_conf_to_be_removed, reverse=True):
                frag['positions'][0].pop(index)
                frag['confidences'][0].pop(index)
                frag['match_types'][0].pop(index)

        # 7. Rearrange the fragments
        self.fragments = self.rearrange_fragments(fragments)
        self.prepend_paired = True

        return

    def fragment_pair(self, distance_threshold_per_frame, verbose=False):

        if not self.prepend_paired:
            raise ValueError(
                "The trajectory pool must be prepended before performing a fragment merge.")

        for frag in tqdm(self.fragments, desc='Fragment Pairing', disable=not verbose):

            # In each fragment, explore all possible pairs of short trajs
            candidate_prev_trajs = []
            pairs = []

            for frame_index in range(len(frag['positions'])-2):

                candidate_prev_trajs.extend([[(frame_index, a), (frame_index+1, b)]
                                             for a in range(len(frag['positions'][frame_index]))
                                             for b in range(len(frag['positions'][frame_index+1]))])
                candidate_later_trajs = [(frame_index+2, a)
                                         for a in range(len(frag['positions'][frame_index+2]))]
                if len(candidate_prev_trajs) > 0 and len(candidate_later_trajs) > 0:
                    weights = []
                    for prev_traj_index in candidate_prev_trajs:
                        weight = []
                        prev_traj = np.array([frag['positions'][i[0]][i[1]]
                                              for i in prev_traj_index])
                        for traj_index in candidate_later_trajs:
                            a, b = traj_index
                            later_traj = np.array([frag['positions'][a][b]])
                            weight.append(calc_dist_between_normal_traj_and_short_traj(
                                prev_traj,
                                later_traj,
                                window=1))  # Set window = 1 because we expect accelerations to be sharp in fragments.
                        weights.append(weight)
                    weights = np.array(weights)
                    row_ind, col_ind = linear_sum_assignment(weights)
                    temp_candidate_prev_trajs = []
                    for i, j in zip(row_ind, col_ind):
                        if weights[i, j] < distance_threshold_per_frame:
                            candidate_prev_trajs[i].append(
                                candidate_later_trajs[j])
                            temp_candidate_prev_trajs.append(
                                candidate_prev_trajs[i])
                    candidate_prev_trajs = temp_candidate_prev_trajs
                    pairs.extend(temp_candidate_prev_trajs)

            # Exlucde shorter duplicates from pairs
            simplified_pairs = []
            while len(pairs) > 0:
                longest = pairs[np.argmax([len(i) for i in pairs])]
                simplified_pairs.append(longest)
                pairs = [pair
                         for pair in pairs
                         if not pair == longest[:len(pair)]]

            # Exclude pairs that share the same position, and keep the logest and most confident one
            unique_pairs = []
            while len(simplified_pairs) > 0:
                longest = simplified_pairs.pop(
                    np.argmax([len(i) for i in simplified_pairs]))
                simplified_pairs = [pair
                                    for pair in simplified_pairs
                                    if (len(pair) < len(longest)
                                        and all([pos not in longest for pos in pair]))]
                # Check if equal length pairs
                long_pairs = [(i, pair)
                              for i, pair in enumerate(simplified_pairs)
                              if (len(pair) == len(longest)
                                  and not all([pos not in longest for pos in pair]))]
                if len(long_pairs) > 0:
                    long_pairs_average_conf = [(i, np.mean([frag['confidences'][i[0]][i[1]]
                                                            for i in pair]))
                                               for i, pair in long_pairs]
                    longest_average_conf = np.mean([frag['confidences'][i[0]][i[1]]
                                                    for i in longest])
                    if max([i[1] for i in long_pairs_average_conf]) > longest_average_conf:
                        unique_pairs.append(
                            long_pairs[np.argmax([i[1] for i in long_pairs_average_conf])][1])
                    else:
                        unique_pairs.append(longest)
                    # Remove the long pairs
                    for i, pair in reversed(long_pairs):
                        simplified_pairs.pop(i)
                else:
                    unique_pairs.append(longest)

            for pair in unique_pairs:
                self.add_full_trajectory([frag['positions'][i[0]][i[1]] for i in pair],
                                         [frag['confidences'][i[0]][i[1]] for i in pair],
                                         [frag['match_types'][i[0]][i[1]] for i in pair],
                                         frag['start_frame'] + pair[0][0])
            pos_and_conf_to_be_removed = [
                i for pair in unique_pairs for i in pair]
            for index in sorted(pos_and_conf_to_be_removed, reverse=True):
                frag['positions'][index[0]].pop(index[1])
                frag['confidences'][index[0]].pop(index[1])

        self.fragment_paired = True
        self.fragments = self.rearrange_fragments(self.fragments)

    def coarse_pair(self, traj_length_belief, full_state_length_belief, num_entities=7, verbose=False, detailed_verbose=False):

        if full_state_length_belief > traj_length_belief:
            raise ValueError(
                "The full state length should be smaller than the resolution.")

        traj_ids_to_be_paired = []
        entry_timeline = [[] for _ in range(len(self.timeline))]
        exit_timeline = [[] for _ in range(len(self.timeline) + 1)]
        for traj_id, traj in tqdm(list(enumerate(self.all_trajectories)),
                                  desc='Initializing Coarse Timeline',
                                disable=not detailed_verbose):
            if (traj['end'] - traj['start'] >= traj_length_belief
                    and not traj['discard']):
                entry_timeline[traj['start']].append(traj_id)
                exit_timeline[traj['end']].append(traj_id)
                traj_ids_to_be_paired.append(traj_id)
        traj_ids_to_be_paired.sort(
            key=lambda x: self.all_trajectories[x]['start'])

        active_traj_ids = []
        pairs = []
        pairing_frames = []  # the pairing happens in which frame

        # state = 0: After two or more trajectories are terminated from a full state,
        #            we can not determine which trajectory is going to be paired with an incoming trajectory.
        #            Thus, before the next full state, we can not determine the pairing.
        # state = 1: One and only one trajectory is terminated from a full state,
        #            this trajectory is going to be paired with an incoming trajectory.
        # state = 2: The number of active trajectories matches the number of entities. It's a full state.

        state = 0
        awaiting_traj_id = None

        for entry_ids, exit_ids in tqdm(list(zip(entry_timeline, exit_timeline)),
                                        desc='Coarse Pairing',
                                       disable=not verbose):

            active_traj_ids.extend(entry_ids)
            for traj_id in exit_ids:
                active_traj_ids.remove(traj_id)

            # 1 -> 2: Entering Full State, Pairing
            if (len(active_traj_ids) == num_entities and
                len(exit_ids) == 0 and
                len(entry_ids) == 1 and
                    state == 1):

                state = 2
                pairs.append((awaiting_traj_id, entry_ids[0]))
                pairing_frames.append((self.all_trajectories[awaiting_traj_id]['end'],
                                       self.all_trajectories[entry_ids[0]]['start']))
                awaiting_traj_id = None

            # 1 -> 2: Entering Full State, NO Pairing
            elif (len(active_traj_ids) == num_entities and
                  len(exit_ids) >= 1 and
                  state == 1):

                state = 2
                awaiting_traj_id = None

            # 2 -> 1: Entering Awaiting State
            elif (len(active_traj_ids) == num_entities - 1 and
                  len(exit_ids) == 1 and
                  len(entry_ids) == 0 and
                  state == 2):

                state = 1
                awaiting_traj_id = exit_ids[0]

            # 2 -> 0: Entering Ambiguous State
            elif (len(active_traj_ids) < num_entities and
                  len(exit_ids) >= 2 and
                  state == 2):

                state = 0

            # 1 -> 0: Entering Ambiguous State
            elif (len(active_traj_ids) < num_entities and
                  len(exit_ids) >= 1 and
                  state == 1):

                state = 0
                awaiting_traj_id = None

            # 0 -> 2: Entering Full State
            elif (len(active_traj_ids) == num_entities and
                  state == 0):

                state = 2

            # 0 -> 0: Remaining Ambiguous State
            elif (len(active_traj_ids) < num_entities and
                    state == 0):

                state = 0

            # 2 -> 2: Remaining Full State, Pairing
            elif (len(active_traj_ids) == num_entities and
                    len(exit_ids) == 1 and
                    len(entry_ids) == 1 and
                    state == 2):

                state = 2
                pairs.append((exit_ids[0], entry_ids[0]))
                pairing_frames.append((self.all_trajectories[exit_ids[0]]['end'],
                                       self.all_trajectories[entry_ids[0]]['start']))

            # 2 -> 2: Remaining Full State
            elif (len(active_traj_ids) == num_entities and
                    state == 2):

                state = 2

            # 1 -> 1: Remaining Awaiting State
            elif (len(active_traj_ids) == num_entities - 1 and
                  len(exit_ids) == 0 and
                    len(entry_ids) == 0 and
                    state == 1):

                state = 1

            elif len(active_traj_ids) > num_entities:
                raise ValueError(
                    "The number of active trajectories exceeds the number of entities.")

            else:
                raise ValueError("The trajectory pool is in an invalid state.")

        # This part is to prevent that two pairing events are too close to each other,
        # which may cause identity confusion when two occlusions happen (almost) at the same time.
        # The threshold that determines whether two pairing events are too close is the full_state_length_belief.
        if detailed_verbose:
            current_time = time()
        pairs_to_be_removed = set()
        pairs_gap = [pairing_frames[i+1][0] - pairing_frames[i][1]
                     for i in range(len(pairing_frames) - 1)]
        for i, gap in enumerate(pairs_gap):
            if gap < full_state_length_belief:
                pairs_to_be_removed.add(i)
                pairs_to_be_removed.add(i+1)
        pairs = {pair[0]: pair[1]
                 for i, pair in enumerate(pairs)
                 if i not in pairs_to_be_removed}
        if detailed_verbose:
            print(f"Time used for removing close pairing events: {time() - current_time:.2f}s")

        # This part is to join the paired trajectories.
        # Firstly, we build chains of paired trajectories.
        if detailed_verbose:
            current_time = time()
        seen_ids = []
        paired_traj_ids = []
        for id in traj_ids_to_be_paired:
            if id not in seen_ids:
                seen_ids.append(id)
                if id in pairs:
                    chain = [id]
                    next_id = pairs[id]
                    while next_id in pairs:
                        chain.append(next_id)
                        seen_ids.append(next_id)
                        next_id = pairs[next_id]
                    chain.append(next_id)
                    seen_ids.append(next_id)
                    paired_traj_ids.append(chain)
                else:
                    paired_traj_ids.append([id])
        if detailed_verbose:
            print(f"Time used for building chains of paired trajectories: {time() - current_time:.2f}s")
        # Then, we join the trajectories in each chain.
        self.coarse_paired_trajectories = []
        for pair in tqdm(paired_traj_ids, desc='Joining Trajectories', disable=not detailed_verbose):
            self.join_trajectories(
                pair, self.all_trajectories, self.coarse_paired_trajectories, join_type='coarse')

        self.coarse_paired_trajectories = tuple(self.coarse_paired_trajectories)
        self.coarse_paired = True

        return pairs

    def fine_pair(self, overlap_length_threshold, full_state_length_belief, num_entities=7, verbose=False, detailed_verbose=False):

        if not self.coarse_paired:
            raise ValueError(
                "The trajectory pool must be coarse paired before performing a fine pair.")

        entry_timeline = [[] for _ in range(len(self.timeline))]
        exit_timeline = [[] for _ in range(len(self.timeline) + 1)]
        for traj_id, traj in tqdm(list(enumerate(self.coarse_paired_trajectories)),
                                  desc='Initializing Fine Timeline',
                                disable=not detailed_verbose):
            entry_timeline[traj['start']].append(traj_id)
            exit_timeline[traj['end']].append(traj_id)

        # We pair every imcoming new trajectories with existing unpaired terminated trajectories, which terminates
        # 1. no more than overlap_threshold frames after the incoming new trajectory.
        # 2. after the last full state.
        active_traj_ids = []
        awaiting_exit_ids = []
        awaiting_entry_ids = []
        full_state_length = 0
        pairs = []

        for entry_ids, exit_ids in tqdm(list(zip(entry_timeline, exit_timeline)),
                                        desc='Analyzing Fine Timeline',
                                       disable=not detailed_verbose):

            active_traj_ids.extend(entry_ids)
            for traj_id in exit_ids:
                active_traj_ids.remove(traj_id)

            # Full State -> Full State, Reset full state length
            if (len(active_traj_ids) == num_entities and
                len(exit_ids) > 0 and
                len(entry_ids) > 0 and
                    full_state_length > 0):

                full_state_length = 1
                awaiting_exit_ids.extend(exit_ids)
                awaiting_entry_ids.extend(entry_ids)

            # Full State -> Full State, Increase full state length
            elif (len(active_traj_ids) == num_entities and
                  len(exit_ids) == 0 and
                  len(entry_ids) == 0 and
                  full_state_length > 0):

                full_state_length += 1

            # Full State -> Ambiguous State
            elif (len(active_traj_ids) < num_entities and
                  full_state_length > 0):

                full_state_length = 0
                awaiting_exit_ids.extend(exit_ids)
                awaiting_entry_ids.extend(entry_ids)

            # Ambiguous State -> Full State
            elif (len(active_traj_ids) == num_entities and
                  full_state_length == 0):

                full_state_length = 1
                awaiting_exit_ids.extend(exit_ids)
                awaiting_entry_ids.extend(entry_ids)

            # Ambiguous State -> Ambiguous State
            elif (len(active_traj_ids) < num_entities and
                  full_state_length == 0):

                awaiting_exit_ids.extend(exit_ids)
                awaiting_entry_ids.extend(entry_ids)

            # Error: active_traj_ids exceeds num_entities
            elif len(active_traj_ids) > num_entities:
                ic(active_traj_ids, full_state_length, entry_ids, exit_ids)
                raise ValueError(
                    "The number of active trajectories exceeds the number of entities.")

            # Error: The trajectory pool is in an invalid state.
            else:
                ic(active_traj_ids, full_state_length, entry_ids, exit_ids)
                raise ValueError("The trajectory pool is in an invalid state.")

            if full_state_length == full_state_length_belief + 1:
                pairs.append((tuple(awaiting_exit_ids),
                              tuple(awaiting_entry_ids)))
                awaiting_exit_ids = []
                awaiting_entry_ids = []

        id_mapping = {}

        for exit_ids, entry_ids in tqdm(pairs, desc='Fine Pairing', disable=not verbose):

            if len(exit_ids) == 0:
                continue

            # Use Hungarian algorithm to pair the trajectories
            weights = []
            for exit_id in exit_ids:
                current_weights = []
                for entry_id in entry_ids:
                    if (self.coarse_paired_trajectories[exit_id]['end']
                            - self.coarse_paired_trajectories[entry_id]['start']) > overlap_length_threshold:
                        current_weights.append(np.inf)
                    else:
                        current_weights.append(calculate_distance_between_trajectories(
                            self.coarse_paired_trajectories[exit_id]['trajectory'],
                            self.coarse_paired_trajectories[entry_id]['trajectory'],
                            self.coarse_paired_trajectories[entry_id]['start']
                            - self.coarse_paired_trajectories[exit_id]['end']))
                weights.append(current_weights)
            weights = np.array(weights)
            row_ind, col_ind = linear_sum_assignment(weights)

            # Update id_mapping
            for i, j in zip(row_ind, col_ind):
                if weights[i, j] != np.inf:
                    id_mapping[exit_ids[i]] = entry_ids[j]
                else:
                    raise ValueError(
                        "The trajectory pairing yeilds an infinite distance.")

        # Join the paired trajectories.
        # Firstly, we build chains of paired trajectories.
        if detailed_verbose:
            current_time = time()
        seen_ids = []
        paired_traj_ids = []  # The chains of paired trajectories
        traj_ids = [i for i in range(len(self.coarse_paired_trajectories))]
        for id in traj_ids:
            if id not in seen_ids:
                seen_ids.append(id)
                if id in id_mapping:
                    chain = [id]
                    next_id = id_mapping[id]
                    while next_id in id_mapping:
                        chain.append(next_id)
                        seen_ids.append(next_id)
                        next_id = id_mapping[next_id]
                    chain.append(next_id)
                    seen_ids.append(next_id)
                    paired_traj_ids.append(chain)
                else:
                    paired_traj_ids.append([id])
        if detailed_verbose:
            print(f"Time used for building chains of paired trajectories: {time() - current_time:.2f}s")
        # Then, we join the trajectories in each chain.
        self.fine_paired_trajectories = []
        for pair in tqdm(paired_traj_ids, desc='Joining Trajectories', disable=not detailed_verbose):
            self.join_trajectories(
                pair, self.coarse_paired_trajectories, self.fine_paired_trajectories, join_type='fine')

        # Padding the trajectories to make sure that they cover the whole timeline
        for traj in self.fine_paired_trajectories:

            trajs_to_be_joined = []
            if traj['start'] > 0:
                trajs_to_be_joined.append(traj['start'])
            trajs_to_be_joined.append(traj['trajectory'])
            if traj['end'] < len(self.timeline):
                trajs_to_be_joined.append(len(self.timeline) - traj['end'])

            if len(trajs_to_be_joined) > 1:
                traj['trajectory'] = JoinedTrajectory(*trajs_to_be_joined, break_type='padding')
                traj['start'] = 0
                traj['end'] = len(self.timeline)

        self.fine_paired_trajectories = tuple(self.fine_paired_trajectories)
        self.fine_paired = True

    def refine_trajectories(self, verbose=False, detailed_verbose=False):
        break_entry_timeline = [[] for _ in range(len(self.timeline))]
        break_exit_timeline = [[] for _ in range(len(self.timeline) + 1)]
        for i, traj in tqdm(list(enumerate(self.fine_paired_trajectories)),
                            desc='Initializing Refine Timeline',
                           disable=not detailed_verbose):
            for k, v in traj['trajectory'].breakpoints.items():
                if v['length'] > 0:
                    break_entry_timeline[k].append(i)
                    break_exit_timeline[k + v['length']].append(i)

        self.create_fragment_timeline()
        
        active_breaks = []
        full_state = True
        awaiting_traj_id = []
        break_start_index = None

        for index, (entry_ids, exit_ids) in tqdm(list(enumerate(list(zip(break_entry_timeline, break_exit_timeline)))),
                                                 desc='Refining Trajectories',
                                                 disable=not verbose):

            active_breaks.extend(entry_ids)
            for traj_id in exit_ids:
                active_breaks.remove(traj_id)

            # Full State -> Full State
            if len(active_breaks) == 0 and full_state:

                pass

            # Full State -> Ambiguous State
            elif len(active_breaks) > 0 and full_state:

                full_state = False
                break_start_index = index
                awaiting_traj_id.extend([traj_id for traj_id in entry_ids])
                
            # Ambiguous State -> Ambiguous State
            elif len(active_breaks) > 0 and not full_state:

                awaiting_traj_id.extend([traj_id for traj_id in entry_ids])
                
            # Ambiguous State -> Full State, Refine
            elif ((len(active_breaks) == 0 and not full_state) or
                  index == len(self.timeline)):

                full_state = True

                updated = True
                loop_counter = 0
                awaiting_traj_id = list(set(awaiting_traj_id))
                fragments_to_be_removed = []
                
                while updated:
                    updated = False
                    loop_counter += 1
                    # Firstly, try to refine with the unpaired trajectories
                    candidate_traj_ids = set()
                    for frame in self.timeline[break_start_index:index]:
                        for traj_id in frame:
                            if self.all_trajectories[traj_id]['child_of'] == None:
                                candidate_traj_ids.add(traj_id)
                    candidate_traj_ids = list(candidate_traj_ids)

                    if len(candidate_traj_ids) > 0:
                        weights = []
                        for traj_id in awaiting_traj_id:
                            weight = []
                            for cand_traj_id in candidate_traj_ids:
                                distance = self.fine_paired_trajectories[traj_id]['trajectory'].evaluate_insertion(
                                    self.all_trajectories[cand_traj_id]['start'],
                                    self.all_trajectories[cand_traj_id]['trajectory'])
                                weight.append(distance)
                            weights.append(weight)

                        weights = np.array(weights)
                        row_ind, col_ind = linear_sum_assignment(weights)
                        for i, j in zip(row_ind, col_ind):
                            if weights[i, j] < 0:
                                self.fine_paired_trajectories[awaiting_traj_id[i]]['trajectory'].insert(
                                    self.all_trajectories[candidate_traj_ids[j]]['start'],
                                    self.all_trajectories[candidate_traj_ids[j]]['trajectory'])
                                self.all_trajectories[
                                    candidate_traj_ids[j]]['child_of'] = ('refine', awaiting_traj_id[i])
                                # ic(loop_counter, awaiting_traj_id[i], candidate_traj_ids[j])
                                updated = True

                    # Secondly, try to refine with fragments (length=1)
                    candidate_frag_ids = set()
                    for frame in self.fragment_timeline[break_start_index:index]:
                        for frag_id in frame:
                            candidate_frag_ids.add(frag_id)
                    candidate_frag_ids = list(candidate_frag_ids)
                    candidate_positions = [(i, j, k) for i in candidate_frag_ids
                                        for j in range(len(self.fragments[i]['positions']))
                                        for k in range(len(self.fragments[i]['positions'][j]))
                                        if (i, j, k) not in fragments_to_be_removed]

                    if len(candidate_positions) >= 0:
                        weights = []
                        for traj_id in awaiting_traj_id:
                            weight = []
                            for pos_index in candidate_positions:
                                distance = self.fine_paired_trajectories[traj_id]['trajectory'].evaluate_insertion(
                                    self.fragments[pos_index[0]]['start_frame'] + pos_index[1],
                                    [self.fragments[pos_index[0]]['positions'][pos_index[1]][pos_index[2]]])
                                weight.append(distance)
                            weights.append(weight)
                        
                        weights = np.array(weights)
                        row_ind, col_ind = linear_sum_assignment(weights)
                        for i, j in zip(row_ind, col_ind):
                            if weights[i, j] < 0:
                                a, b, c = candidate_positions[j]
                                self.fine_paired_trajectories[
                                    awaiting_traj_id[i]]['trajectory'].insert(
                                        self.fragments[a]['start_frame'] + b,
                                        DirectTrajectory([self.fragments[a]['positions'][b][c]],
                                                        [self.fragments[a]['confidences'][b][c]],
                                                        [self.fragments[a]['match_types'][b][c]]))
                                updated = True
                                fragments_to_be_removed.append((a, b, c))

                        
                        # Thirdly, try to refine with fragments (length=2)
                        candidate_positions = [(i, j, k, l)
                                               for i in candidate_frag_ids
                                               for j in range(len(self.fragments[i]['positions']) - 1)
                                               for k in range(len(self.fragments[i]['positions'][j]))
                                               for l in range(len(self.fragments[i]['positions'][j+1]))
                                               if ((i, j, k) not in fragments_to_be_removed and
                                                   (i, j+1, l) not in fragments_to_be_removed)]
                        if len(candidate_positions) >= 0:
                            weights = []
                            for traj_id in awaiting_traj_id:
                                weight = []
                                for pos_index in candidate_positions:
                                    distance = self.fine_paired_trajectories[traj_id]['trajectory'].evaluate_insertion(
                                        self.fragments[pos_index[0]]['start_frame'] + pos_index[1],
                                        [self.fragments[pos_index[0]]['positions'][pos_index[1]][pos_index[2]],
                                         self.fragments[pos_index[0]]['positions'][pos_index[1]+1][pos_index[3]]])
                                    weight.append(distance)
                                weights.append(weight)
                            weights = np.array(weights)
                            row_ind, col_ind = linear_sum_assignment(weights)
                            for i, j in zip(row_ind, col_ind):
                                if weights[i, j] < 0:
                                    a, b, c, d = candidate_positions[j]
                                    if ((a, b, c) not in fragments_to_be_removed and
                                        (a, b+1, d) not in fragments_to_be_removed):
                                        self.fine_paired_trajectories[
                                            awaiting_traj_id[i]]['trajectory'].insert(
                                                self.fragments[a]['start_frame'] + b,
                                                DirectTrajectory([self.fragments[a]['positions'][b][c],
                                                                self.fragments[a]['positions'][b+1][d]],
                                                                [self.fragments[a]['confidences'][b][c],
                                                                self.fragments[a]['confidences'][b+1][d]],
                                                                [self.fragments[a]['match_types'][b][c],
                                                                self.fragments[a]['match_types'][b+1][d]]))
                                        updated = True
                                        fragments_to_be_removed.append((a, b, c))
                                        fragments_to_be_removed.append((a, b+1, d))
                        

                awaiting_traj_id = []
                break_start_index = None

                for i, j, k in sorted(fragments_to_be_removed, reverse=True):
                    self.fragments[i]['positions'][j].pop(k)
                    self.fragments[i]['confidences'][j].pop(k)
                    self.fragments[i]['match_types'][j].pop(k)

            else:
                raise ValueError("The trajectory pool is in an invalid state.")

        self.fragments = self.rearrange_fragments(self.fragments)

    def __getitem__(self, traj_id):
        if self.fine_paired:
            return self.fine_paired_trajectories[traj_id]['trajectory']
        elif self.coarse_paired:
            return self.coarse_paired_trajectories[traj_id]['trajectory']
        else:
            return self.all_trajectories[traj_id]['trajectory']

    def visualize_raw_timeline(self, min_traj_length, start_from=0, end_at=np.inf):

        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 6))
        trajectory_ids = []

        # Add bars for the trajectory lifespans
        for (idx, start, duration) in zip(range(len(self.all_trajectories)),
                                          [trajectory['start']
                                           for trajectory in self.all_trajectories],
                                          [trajectory['end'] - trajectory['start']
                                           for trajectory in self.all_trajectories]):
            end = start + duration
            if duration >= min_traj_length and not self.all_trajectories[idx]['discard'] and end > start_from and start <= end_at:
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

        ax.grid(True)
        plt.show()

    def __len__(self):
        if self.fine_paired:
            return len(self.fine_paired_trajectories)
        elif self.coarse_paired:
            return len(self.coarse_paired_trajectories)
        else:
            return len(self.all_trajectories)

    def visualize_paired_timeline(self, pair_type='refine', linewidth=4):

        if pair_type == 'fine' or pair_type == 'refine':
            paired_trajectories = self.fine_paired_trajectories
        elif pair_type == 'coarse':
            paired_trajectories = self.coarse_paired_trajectories
        else:
            raise ValueError(
                "The pair type should be 'refine', 'fine' or 'coarse'.")

        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 6))
        trajectory_ids = []

        coarse_events = []
        fine_events = []
        refine_events = []

        # Add horizontal lines for the trajectory lifespans
        for idx, trajectory in enumerate(paired_trajectories):
            start = trajectory['start']
            end = trajectory['end']
            ax.hlines(y=idx, xmin=start, xmax=end,
                      color='tab:blue', linewidth=linewidth)
            trajectory_ids.append(idx)
            if trajectory['trajectory'].breakpoints is not None:
                for k, v in trajectory['trajectory'].breakpoints.items():
                    if v['type'] == 'coarse':
                        coarse_events.append((idx, k + trajectory['start']))
                    elif v['type'] == 'fine':
                        fine_events.append((idx, k + trajectory['start']))
                    if v['refined'] == True:
                        refine_events.append((idx, k + trajectory['start']))
                    
        # Add vertical lines for key events
        for event in coarse_events:
            traj_id, frame_index = event
            # Calculate the ymin and ymax for the red lines in data coordinates
            ymin = traj_id - 0.4
            ymax = traj_id + 0.4
            ax.vlines(x=frame_index, ymin=ymin,
                      ymax=ymax, color='green', linewidth=1)
        if pair_type == 'fine' or pair_type == 'refine':
            for event in fine_events:
                traj_id, frame_index = event
                # Calculate the ymin and ymax for the red lines in data coordinates
                ymin = traj_id - 0.4
                ymax = traj_id + 0.4
                ax.vlines(x=frame_index, ymin=ymin,
                          ymax=ymax, color='orange', linewidth=1)
        # Add vertical lines for key events
        if pair_type == 'refine':
            for event in refine_events:
                traj_id, frame_index = event
                # Calculate the ymin and ymax for the red lines in data coordinates
                ymin = traj_id - 0.4
                ymax = traj_id + 0.4
                ax.vlines(x=frame_index, ymin=ymin,
                          ymax=ymax, color='red', linewidth=1)

        # Set the y-axis labels to the trajectory IDs
        ax.set_yticks(range(len(trajectory_ids)))
        ax.set_yticklabels(trajectory_ids)

        # Set the rest of the labels and title
        ax.set_xlabel('Frame')
        ax.set_ylabel('Trajectory ID')
        ax.set_title('Paired Timeline Visualization')

        ax.grid(True)
        plt.show()

    def visualize_trajectories_length_cdf(self, markersize=1):

        trajectory_lengths = [trajectory['end'] - trajectory['start']
                              if trajectory['end'] is not None
                              else len(self.timeline) - trajectory['start']
                              for trajectory in self.all_trajectories
                              if not trajectory['discard']]
        if self.prepend_paired:
            num_fragments = sum([len(frag['positions'])
                                for frag in self.fragments])
            trajectory_lengths.extend([1.5 for i in range(num_fragments)])
            fragment_percentage = num_fragments / len(trajectory_lengths) * 100
            annotation_text = f"Fragments: {num_fragments} ({fragment_percentage:.2f}%)"
            plt.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                         bbox=dict(boxstyle="round", fc="white", ec="gray", pad=0.2))
        trajectory_lengths = np.array(trajectory_lengths)
        sorted_lengths = np.sort(trajectory_lengths)
        yvals = np.arange(1, len(sorted_lengths)+1) / len(sorted_lengths)

        plt.plot(sorted_lengths, yvals, marker='.',
                 linestyle='none', markersize=markersize)
        plt.title('CDF of Trajectory Lengths')
        plt.xlabel('Length')
        plt.xscale('log')
        plt.ylabel('CDF')
        plt.grid(True)
        plt.show()

    def lower_fps(self, divisor):
            
        if divisor < 1:
            raise ValueError("The divisor should be a positive integer.")

        if not self.fine_paired:
            raise ValueError(
                "The trajectory pool must be fine paired before lowering the fps.")
        
        for traj in self.fine_paired_trajectories:
            traj['trajectory'].interpolate_gaps()

        output = []

        for traj in self.fine_paired_trajectories:
            positions = [traj['trajectory'][i] for i in range(0, len(traj['trajectory']), divisor)]
            confidences = [traj['confidences'][i] for i in range(0, len(traj['trajectory']), divisor)]
            match_types = [traj['match_types'][i] for i in range(0, len(traj['trajectory']), divisor)]
            sources = [traj['sources'][i] for i in range(0, len(traj['trajectory']), divisor)]
            output.append(DirectTrajectory(positions, confidences, match_types, sources))

        return DirectTrajectoryPool(output, state='fine_paired')
    
    def stats(self):
    
        false_negative = 0
        refine = 0
        merge = 0
        fragment = 0
        prepend = 0
        append = 0
        coarse_pair = 0
        fine_pair = 0
        padding = 0

        for traj in self:
            for source in traj.sources:
                if source == 'none' or source == 'interpolate':
                    false_negative += 1
                elif source == 'refine':
                    refine += 1
                elif source == 'fragment':
                    fragment += 1
                elif source == 'prepend':
                    prepend += 1
                elif source == 'append':
                    append += 1
                elif source == 'merge':
                    merge += 1
                else:
                    raise ValueError(f"Unknown source: {source}")
            
            for breakpoint, info in traj.breakpoints.items():
                if info['type'] == 'coarse':
                    coarse_pair += 1
                elif info['type'] == 'fine':
                    fine_pair += 1
                elif info['type'] == 'padding':
                    padding += 1
                else:
                    raise ValueError(f"Unknown breakpoint type: {info['type']}")
                
        true_positive = refine + fragment + prepend + append + merge
        false_positive = sum([len(frame)
                            for frag in self.fragments
                            for frame in frag['positions']]) + merge
        assert true_positive + false_negative == len(self) * len(self[0])
                
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        
        f1 = 2 * (precision * recall) / (precision + recall)
                
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'append_contribution': append / true_positive,
            'prepend_contribution': prepend / true_positive,
            'refine_contribution': refine / true_positive,
            'fragment_contribution': fragment / true_positive,
            'merge_contribution': merge / true_positive,
            'coarse_pair_contribution': coarse_pair / (coarse_pair + fine_pair + padding),
            'fine_pair_contribution': fine_pair / (coarse_pair + fine_pair + padding),
            'padding_contribution': padding / (coarse_pair + fine_pair + padding),
        }

class DirectTrajectoryPool(TrajectoryPool):

    def __init__(self, trajectories, state='fine_paired'):
        if state == 'fine_paired':
            self.fine_paired_trajectories = trajectories
            self.coarse_paired_trajectories = None
            self.all_trajectories = None
            self.fine_paired = True
            self.coarse_paired = True
            self.prepend_paired = True
            self.fragment_paired = True
            self.end_value_assigned = True
        elif state == 'coarse_paired':
            self.coarse_paired_trajectories = trajectories
            self.all_trajectories = None
            self.fine_paired = False
            self.coarse_paired = True
            self.prepend_paired = True
            self.fragment_paired = True
            self.end_value_assigned = True
        elif state == 'initial':
            self.all_trajectories = trajectories
            self.fine_paired = False
            self.coarse_paired = False
            self.prepend_paired = True
            self.fragment_paired = True
            self.end_value_assigned = True
        else:
            raise ValueError("The state should be 'fine_paired', 'coarse_paired' or 'initial'.")


class Tracker:

    def __init__(self,
                 num_keypoints=7,
                 distance_threshold_per_frame=1.5,
                 traj_length_belief=3,
                 full_state_length_belief=3,
                 max_prepend_length=2,
                 fragment_pair=True,
                 momentum_window=3,
                 overlap_length_threshold=3):
        """
        Args:
            num_keypoints: The number of keypoints to be tracked.
            distance_threshold_per_frame: The maximum distance a keypoint can travel between frames.
        """
        self.num_keypoints = num_keypoints
        self.distance_threshold_per_frame = distance_threshold_per_frame
        self.momentum_window = momentum_window
        self._build_initialized = False
        self.traj_length_belief = traj_length_belief
        self.full_state_length_belief = full_state_length_belief
        self.overlap_length_threshold = overlap_length_threshold
        self.max_prepend_length = max_prepend_length
        self.fragment_pair = fragment_pair

    def initialize_build(self, initial_positions, initial_confidences, initial_match_types):
        """
        Args:
            initial_positions: The initial positions of the keypoints to be tracked.
            initial_confidences: The initial confidences of the keypoints to be tracked.
            initial_match_types: The initial match types of the keypoints to be tracked.
        """

        if self._build_initialized:
            raise ValueError("The tracker has already been initialized.")

        # Check if the shape of the initial_positions matches each other.
        if len(set([initial_position.shape for initial_position in initial_positions])) != 1:
            raise ValueError(
                "The shape of the initial positions should match each other.")

        self.traj_pool = TrajectoryPool(initial_positions, initial_confidences, initial_match_types)

        self._build_initialized = True

    def step_build_traj_pool(self, new_positions, new_confidences, new_match_types):
        """
        Args:
            new_positions: The new positions of the keypoints to be tracked.
        """
        if not self._build_initialized:
            raise ValueError("The tracker has not been initialized yet.")

        new_positions_status = [None] * len(new_positions)
        self.traj_pool.new_time_step()

        # Use the Hungarian algorithm to connect the new positions to the existing trajectories.
        weights = []
        previous_timestep_traj_ids = self.traj_pool.get_previous_traj_ids()
        for trajectory_id in previous_timestep_traj_ids:
            distances = self.traj_pool[trajectory_id].calculate_distances_momentum(new_positions,
                                                                                   window=self.momentum_window)
            weights.append(distances)
        # Each row represents a previous trajectory, each column represents a new position.
        weights = np.array(weights)
        row_ind, col_ind = linear_sum_assignment(weights)

        for previous_index, new_position_index in zip(row_ind, col_ind):
            if weights[previous_index, new_position_index] < self.distance_threshold_per_frame:
                # If the distance is within the threshold, connect the new position to the existing trajectory.
                trajectory_id = previous_timestep_traj_ids[previous_index]
                self.traj_pool.update_trajectory(trajectory_id,
                                                 new_positions[new_position_index],
                                                 new_confidences[new_position_index],
                                                 new_match_types[new_position_index])
                new_positions_status[new_position_index] = trajectory_id

        for new_position_index, status in enumerate(new_positions_status):
            if status is None:
                self.traj_pool.add_trajectory(new_positions[new_position_index],
                                              new_confidences[new_position_index],
                                              new_match_types[new_position_index])

    def build_traj_pool(self, coords, all_confidences, all_match_types, verbose=False):
        """
        Args:
            coords: List of numpy arrays with shape (num_positions, dimension).
        """
        self.initialize_build(coords[0], all_confidences[0], all_match_types[0])

        if verbose:
            iterator = tqdm(list(zip(coords[1:], all_confidences[1:], all_match_types[1:])),
                            desc="Forward Build")
        else:
            iterator = zip(coords[1:], all_confidences[1:], all_match_types[1:])

        for positions, confidences, match_types in iterator:
            self.step_build_traj_pool(positions, confidences, match_types)
        self.traj_pool.new_time_step()

    def track(self, coords, confidences, match_types, verbose=False):
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

        if type(coords) == dict:
            # Convert the matched_coords to a list of numpy arrays.
            coords = [[np.hstack([coord[0], coord[1]])
                       for coords in list(matched_coord.values()) for coord in coords]
                      for matched_coord in coords]

        # Step 1: Build the trajectory pool
        # We firstly use forward-backward matching to build the trajectory pool.
        # Then, we attempt connect unpaired positions (called fragments).
        if verbose:
            print("Step 1: Building the trajectory pool.")
            current_time = time()
        self.build_traj_pool(coords, confidences, match_types, verbose=verbose)
        if verbose:
            forward_time = time() - current_time
        self.traj_pool.prepend_pair(self.distance_threshold_per_frame,
                                    self.max_prepend_length,
                                    verbose=False) # The time cost for prepend_pair is negligible.
        if verbose:
            backward_time = time() - current_time - forward_time
        if self.fragment_pair:
            self.traj_pool.fragment_pair(self.distance_threshold_per_frame,
                                         verbose=False) # The time cost for fragment_pair is negligible.
        if verbose:
            fragment_time = time() - current_time - forward_time - backward_time
            print(f"Forward: {forward_time:.2f}s, Backward: {backward_time:.2f}s, "
                  f"Fragment: {fragment_time:.2f}s, Total: {time() - current_time:.2f}s")
        self.traj_pool.assign_end_value()

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
    
class JoinedTrajectoryV2(JoinedTrajectory):

    def __init__(self, *trajectories, break_type=None):
        """
        Args:
            trajectories: A list of trajectories to be joined together.
        """

        self.trajectory = []
        self.confidences = []
        self.match_types = []
        self.sources = []
        self.velocities = []
        self.shape = None
        self.momentum_window = None
        self.tail_weight = None
        self.breakpoints = {}
        accumulated_length = 0
        merge = False

        if isinstance(trajectories[0].confidences[0], list):
            confidence_type = 'list'
        elif isinstance(trajectories[0].confidences[0], (int, float)):
            confidence_type = 'float'
        else:
            raise ValueError("The confidence type should be either list or float.")

        self.gap = 0

        for traj in trajectories:
            if isinstance(traj, int):
                if traj >= 0:
                    self.trajectory.extend((None, ) * traj)

                    if confidence_type == 'float':
                        self.confidences.extend((0., ) * traj)
                    else:
                        self.confidences.extend(([0., 0., 0.], ) * traj)

                    self.match_types.extend(('None', ) * traj)
                    self.sources.extend(('None', ) * traj)
                    self.velocities.extend([None] * traj)
                else:
                    merge = abs(traj)
                    self.trajectory[-merge:] = [cd / 2 for cd in self.trajectory[-merge:]]
                    if confidence_type == 'float':
                        self.confidences[-merge:] = [cf / 2 for cf in self.confidences[-merge:]]
                    else:
                        self.confidences[-merge:] = [np.array(cf) / 2 for cf in self.confidences[-merge:]]
                    self.sources[-merge:] = ['merge'] * merge
                    self.velocities[-merge:] = [None] * merge
                self.breakpoints[accumulated_length] = {'length': traj,
                                                        'distance_per_frame': None,
                                                        'type': break_type,
                                                        'refined': False}
                accumulated_length += traj
            else:

                if self.shape is None:
                    self.shape = traj.shape
                elif self.shape != traj.shape:
                    raise ValueError(
                        "All trajectories should have the same shape.")
                if self.momentum_window is None:
                    self.momentum_window = traj.momentum_window
                elif self.momentum_window != traj.momentum_window:
                    raise ValueError(
                        "All trajectories should have the same momentum window.")
                if self.tail_weight is None:
                    self.tail_weight = traj.tail_weight
                elif self.tail_weight != traj.tail_weight:
                    raise ValueError(
                        "All trajectories should have the same tail weight.")
                if traj.gap != 0:
                    raise ValueError(
                        "All trajectories should have gap=0.")
                
                if merge:
                    distance_per_frame = np.mean([np.linalg.norm(cd_a * 2 - cd_b) for cd_a, cd_b in zip(self.trajectory[-merge:], traj.trajectory)])
                    self.breakpoints[accumulated_length+merge]['distance_per_frame'] = distance_per_frame
                    self.trajectory[-merge:] = [cd_a + cd_b / 2
                                                for cd_a, cd_b in zip(self.trajectory[-merge:], traj.trajectory)]
                    if confidence_type == 'float':
                        self.confidences[-merge:] = [cf_a + cf_b / 2
                                                    for cf_a, cf_b in zip(self.confidences[-merge:], traj.confidences)]
                    else:
                        self.confidences[-merge:] = [cf_a + np.array(cf_b) / 2
                                                    for cf_a, cf_b in zip(self.confidences[-merge:], traj.confidences)]
                self.trajectory.extend(traj.trajectory[merge:])
                self.confidences.extend(traj.confidences[merge:])
                self.match_types.extend(traj.match_types[merge:])
                self.sources.extend(traj.sources[merge:])
                self.velocities.extend(traj.velocities[merge:])
                merge = False
                for k, v in traj.breakpoints.items():
                    self.breakpoints[k + accumulated_length] = v
                accumulated_length += len(traj)

        if self.shape is None:
            raise ValueError(
                "At least one non-empty trajectory should be provided.")

        self.break_distances = {k: None
                                 for k in self.breakpoints.keys()}

    def append(self, new_position, confidence, match_type):
        """
        Args:
            new_coords: A new point to be connected to the existing trajectory.
            confidence: The confidence of the new point.
            match_type: The type of the match for the new point.
            velocity: The velocity of the new point.
        """
        # Check if the new_coords matches the dimension of the existing trajectory.
        if new_position.shape != self.shape:
            raise ValueError(
                f"The new coordinates with shape {new_position.shape}"
                "does not match the dimension of the existing trajectory with shape {self.shape}."
            )
        self.trajectory.append(new_position)
        self.confidences.append(confidence)
        self.match_types.append(match_type)
        self.sources.append('append')
        self.velocities.append(None)
        self.gap = 0

    def append_none(self):
        self.trajectory.append(None)
        self.confidences.append(0)
        self.match_types.append('None')
        self.sources.append('gap')
        self.velocities.append(None)
        self.gap += 1

    def prepend(self, new_position, confidence, match_type):
        """
        Args:
            new_coords: A new point to be connected to the existing trajectory.
            confidence: The confidence of the new point.
            match_type: The type of the match for the new point.
            velocity: The velocity of the new point.
        """
        # Check if the new_coords matches the dimension of the existing trajectory.
        if new_position.shape != self.shape:
            raise ValueError(
                f"The new coordinates with shape {new_position.shape}"
                "does not match the dimension of the existing trajectory with shape {self.shape}."
            )
        self.trajectory.insert(0, new_position)
        self.confidences.insert(0, confidence)
        self.match_types.insert(0, match_type)
        self.sources.insert(0, 'prepend')
        self.velocities.insert(0, None)

    def calculate_distances_head_only(self):
        raise NotImplementedError
    
    def calculate_distances(self):
        raise NotImplementedError

    def calculate_distances_momentum(self):
        raise NotImplementedError
    
    def calculate_distance_momentum_weighted(self, new_position):
        is_initial, velocity = self.get_velocity(-1-self.gap)
        predicted_position = velocity * (self.gap + 1) + self.trajectory[-1-self.gap]
        difference = new_position - predicted_position
        difference[-2:] = difference[-2:] * self.tail_weight
        return is_initial, np.linalg.norm(difference)

    def backward_calculate_distance_momentum_weighted(self, new_position, gap=0):
        is_initial, velocity = self.get_backward_velocity(0)
        predicted_position = velocity * (gap + 1) + self.trajectory[0]
        difference = new_position - predicted_position
        difference[-2:] = difference[-2:] * self.tail_weight
        return is_initial, np.linalg.norm(difference)

    def get_backward_velocity(self, index=0):
        if index < 0:
            index = len(self.trajectory) + index
            if index < 0:
                raise ValueError(
                    f"The index {index} is out of range for the trajectory with length {len(self.trajectory)}.")
        # The backward velocity is not recorded in self.velocities
        velocities = [self.trajectory[i] - self.trajectory[i+1]
                        for i in range(index,
                                        min(index + self.momentum_window, len(self.trajectory) - 1))
                        if self.trajectory[i] is not None and self.trajectory[i+1] is not None]
        if len(velocities) == 0:
            return True, 0
        else:
            avg_velocity = np.mean(velocities, axis=0)
            return False, avg_velocity
    
    def ray_distance(self, new_position):
        ray_distances = [distance_to_ray(self.trajectory[-1-self.gap][:2],
                                            self.trajectory[-1-self.gap][2:4],
                                            new_position[2*i:2*i+2])
                            for i in range(len(new_position) // 2)]
        return np.mean(ray_distances)

    def get_velocity(self, index=-1):
        if index < 0:
            index = len(self.trajectory) + index
            if index < 0:
                raise ValueError(
                    f"The index {index} is out of range for the trajectory with length {len(self.trajectory)}.")
        if self.velocities[index] is not None:
            return False, self.velocities[index]
        elif index == 0:
            return True, 0
        else:
            velocities = [self.trajectory[i] - self.trajectory[i-1]
                          for i in range(index,
                                         max(index - self.momentum_window, 0), # The range function is exclusive on the second bound
                                         -1)
                          if self.trajectory[i] is not None and self.trajectory[i-1] is not None]
            if len(velocities) == 0:
                return True, 0
            else:
                avg_velocity = np.mean(velocities, axis=0)
                self.velocities[index] = avg_velocity
                return False, avg_velocity
            
    def end(self):
        self.trajectory = self.trajectory[:len(self.trajectory) - self.gap]
        self.confidences = self.confidences[:len(self.confidences) - self.gap]
        self.match_types = self.match_types[:len(self.match_types) - self.gap]
        self.sources = self.sources[:len(self.sources) - self.gap]
        self.velocities = self.velocities[:len(self.velocities) - self.gap]
        output = self.gap
        self.gap = 0
        return output

class TrajectoryV2(JoinedTrajectoryV2):
    def __init__(self,
                 initial_position,
                 initial_confidence,
                 initial_match_type,
                 momentum_window,
                 tail_weight):
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
        self.gap = 0

class DirectTrajectoryV2(JoinedTrajectoryV2):
    def __init__(self,
                trajectory: list,
                confidences: list,
                match_types: list,
                momentum_window: int,
                tail_weight: float,
                sources: list = None,
                velocities: list = None):
        self.trajectory = trajectory
        self.confidences = confidences
        self.match_types = match_types
        if sources is None:
            self.sources = ['direct'] * len(trajectory)
        else:
            self.sources = sources
        if velocities is None:
            self.velocities = [None] * len(trajectory)
        else:
            self.velocities = velocities
        if not len(trajectory) == len(confidences) == len(self.sources) == len(match_types) == len(self.velocities):
            raise ValueError(
                "The length of trajectory, confidences, match_types, sources and velocities should be the same.")
        self.shape = trajectory[0].shape
        self.breakpoints = dict()
        self.momentum_window = momentum_window
        self.tail_weight = tail_weight

class TrajectoryPoolV2:

    def __init__(self,
                 initial_positions,
                 initial_confidences,
                 initial_match_types,
                 momentum_window,
                 tail_weight,
                 max_gap):
        # Check if the shape of the initial_coords is all the same.
        if len(set([initial_position.shape for initial_position in initial_positions])) != 1:
            raise ValueError(
                "The initial coordinates should all have the same shape, but got different shapes."
            )
        self.all_trajectories = [{'trajectory': TrajectoryV2(initial_position,
                                                           initial_confidence,
                                                           initial_match_type,
                                                           momentum_window=momentum_window,
                                                           tail_weight=tail_weight),
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

    def end_time_step(self):

        if len(self.timeline) > 0:
            ids_to_remove = []
            for traj_id in self.active_traj_id:
                if self.all_trajectories[traj_id]['trajectory'].gap > self.max_gap:
                    gap = self.all_trajectories[traj_id]['trajectory'].end() # Gap must be >= 1
                    self.all_trajectories[traj_id]['end'] = self.current_frame - gap + 1
                    ids_to_remove.append(traj_id)
                    for i in range(gap-1):
                        self.timeline[-1-i].remove(traj_id)
            for traj_id in ids_to_remove:
                self.active_traj_id.remove(traj_id)

        self.timeline.append(list(tuple(self.active_traj_id)))
        self.current_frame += 1

    def add_trajectory(self, start_position, start_confidence, start_match_type):

        if start_position.shape != self.shape:
            raise ValueError(
                f"The new coordinates with shape {start_position.shape}"
                "does not match the dimension of the existing trajectories with shape {self.shape}."
            )

        self.active_traj_id.append(len(self.all_trajectories))
        self.all_trajectories.append({'trajectory': TrajectoryV2(start_position,
                                                               start_confidence,
                                                               start_match_type,
                                                               momentum_window=self.momentum_window,
                                                               tail_weight=self.tail_weight),
                                      'start': self.current_frame,
                                      'end': None,
                                      'discard': False,
                                      'prepend': 0,
                                      'fragment': False,
                                      'child_of': None})

    def update_trajectory(self, traj_id, new_position, new_confidence, new_match_type, prepend=None):

        if prepend is None:
            # This is for building the traj_pool, executed by Tracker
            if traj_id not in self.active_traj_id:
                raise ValueError(
                    f"Trajectory with ID {traj_id} is not active.")
            self.all_trajectories[traj_id]['trajectory'].append(new_position,
                                                                new_confidence,
                                                                new_match_type)
        else:
            # This is executed by self.prepend_pair()
            for i in range(prepend):
                self.all_trajectories[traj_id]['trajectory'].prepend(None, 0, 'None')
                self.all_trajectories[traj_id]['start'] -= 1
                self.all_trajectories[traj_id]['prepend'] += 1
                self.timeline[self.all_trajectories[traj_id]
                          ['start']].append(traj_id)
            self.all_trajectories[traj_id]['trajectory'].prepend(new_position,
                                                                 new_confidence,
                                                                 new_match_type)
            self.all_trajectories[traj_id]['start'] -= 1
            self.all_trajectories[traj_id]['prepend'] += 1
            self.timeline[self.all_trajectories[traj_id]
                          ['start']].append(traj_id)

    def gap_trajectory(self, traj_id):
            
        if traj_id not in self.active_traj_id:
            raise ValueError(
                f"Trajectory with ID {traj_id} is not active.")
        self.all_trajectories[traj_id]['trajectory'].append_none()
            
    def assign_end_value(self):
        for traj in self.active_traj_id:
            gap = self.all_trajectories[traj]['trajectory'].end()
            self.all_trajectories[traj]['end'] = self.current_frame - gap
            for i in range(gap):
                self.timeline[-1-i].remove(traj)
        self.end_value_assigned = True

    def join_trajectories(self, chain, from_pool, to_pool, join_type=None):
        for traj_id in chain:
            if from_pool[traj_id]['child_of'] is None:
                from_pool[traj_id]['child_of'] = (join_type, len(to_pool))
            else:
                ic(chain, traj_id, from_pool[traj_id])
                raise ValueError(
                    "The trajectory is already a child of another trajectory.")
        if len(chain) > 1:
            trajs_to_be_joined = [from_pool[chain[0]]['trajectory']]
            end_frame = from_pool[chain[0]]['end']
            for traj_id in chain[1:]:
                start_frame = from_pool[traj_id]['start']
                trajs_to_be_joined.append(start_frame - end_frame)
                trajs_to_be_joined.append(from_pool[traj_id]['trajectory'])
                end_frame = from_pool[traj_id]['end']
            to_pool.append({'trajectory': JoinedTrajectoryV2(*trajs_to_be_joined, break_type=join_type),
                            'start': from_pool[chain[0]]['start'],
                            'end': from_pool[chain[-1]]['end'],
                            'chain': chain,
                            'child_of': None})
        else:
            to_pool.append({'trajectory': from_pool[chain[0]]['trajectory'],
                            'start': from_pool[chain[0]]['start'],
                            'end': from_pool[chain[0]]['end'],
                            'chain': chain,
                            'child_of': None})

    def prepend_pair(self, distance_threshold_per_frame, traj_length_belief, verbose=False):
        """
        Pair fragments that fail to form a complete trajectory with existing long trajectories.
        The fragments consist of short trajectories that are no more than traj_length_belief frames long.
        Args:
            distance_threshold_per_frame: The distance threshold for pairing.
            traj_length_belief: The belief of the length of a trajectory.
            verbose: Whether to print out the pairing process.
        """

        fragment_timeline = [[] for _ in range(len(self.timeline))]
        for i, traj in enumerate(self.all_trajectories):
            if traj['discard'] == False and traj['end'] - traj['start'] < traj_length_belief:
                start_frame = traj['start']
                end_frame = traj['end']
                for i in range(start_frame, end_frame):
                    if traj['trajectory'].trajectory[i-start_frame] is not None:
                        fragment_timeline[i].append(
                            (traj['trajectory'].trajectory[i-start_frame],
                                traj['trajectory'].confidences[i-start_frame],
                                traj['trajectory'].match_types[i-start_frame]))
                traj['discard'] = True
        
        active_traj_ids = set()
        for frame_index in range(len(self.timeline) - 1, -1, -1):

            current_traj_ids = self.timeline[frame_index]
            for traj_id in current_traj_ids:
                if self.all_trajectories[traj_id]['discard'] == False:
                    active_traj_ids.add(traj_id)

            traj_ids_to_be_paired = []
            traj_ids_to_be_removed = []
            for traj_id in active_traj_ids:
                if frame_index - self.all_trajectories[traj_id]['start'] >= 1:
                    # This means that can be prepended
                    if frame_index - self.all_trajectories[traj_id]['start'] - 1 > self.max_gap:
                        traj_ids_to_be_removed.append(traj_id)
                    else:
                        traj_ids_to_be_paired.append((traj_id,
                                                      frame_index - self.all_trajectories[traj_id]['start'] - 1))
            
            for traj_id in traj_ids_to_be_removed:
                active_traj_ids.remove(traj_id)

            if len(fragment_timeline[frame_index]) == 0 or len(traj_ids_to_be_paired) == 0:
                continue

            weights = []
            for traj_id, gap in traj_ids_to_be_paired:
                weights.append([])
                for cds, _1, _2 in fragment_timeline[frame_index]:
                    is_initial, distance = self.all_trajectories[traj_id]['trajectory'].backward_calculate_distance_momentum_weighted(cds, gap=gap)
                    weights[-1].append(distance)
            
            weights = np.array(weights)
            row_ind, col_ind = linear_sum_assignment(weights)
            connected_frag_ids = []
            for i, j in zip(row_ind, col_ind):
                if weights[i, j] < distance_threshold_per_frame:
                    self.update_trajectory(traj_ids_to_be_paired[i][0],
                                           fragment_timeline[frame_index][j][0],
                                             fragment_timeline[frame_index][j][1],
                                                fragment_timeline[frame_index][j][2],
                                                prepend=traj_ids_to_be_paired[i][0])
                    connected_frag_ids.append(j)
            
            connected_frag_ids.sort(reverse=True) # Remove from the last index
            for frag_id in connected_frag_ids:
                fragment_timeline[frame_index].pop(frag_id)

        self.fragment_timeline = fragment_timeline # A list containing the fragments in each frame. The fragments in each frame is a list of tuples (position, confidence, match_type).

    def coarse_pair(self, traj_length_belief, full_state_length_belief, num_entities=7, verbose=False, detailed_verbose=False):

        if full_state_length_belief > traj_length_belief:
            raise ValueError(
                "The full state length should be smaller than the resolution.")

        traj_ids_to_be_paired = []
        entry_timeline = [[] for _ in range(len(self.timeline))]
        exit_timeline = [[] for _ in range(len(self.timeline) + 1)]
        for traj_id, traj in tqdm(list(enumerate(self.all_trajectories)),
                                  desc='Initializing Coarse Timeline',
                                disable=not detailed_verbose):
            if (traj['end'] - traj['start'] >= traj_length_belief
                    and not traj['discard']):
                entry_timeline[traj['start']].append(traj_id)
                exit_timeline[traj['end']].append(traj_id)
                traj_ids_to_be_paired.append(traj_id)
        traj_ids_to_be_paired.sort(
            key=lambda x: self.all_trajectories[x]['start'])

        active_traj_ids = []
        pairs = []
        pairing_frames = []  # the pairing happens in which frame

        # state = 0: After two or more trajectories are terminated from a full state,
        #            we can not determine which trajectory is going to be paired with an incoming trajectory.
        #            Thus, before the next full state, we can not determine the pairing.
        # state = 1: One and only one trajectory is terminated from a full state,
        #            this trajectory is going to be paired with an incoming trajectory.
        # state = 2: The number of active trajectories matches the number of entities. It's a full state.

        state = 0
        awaiting_traj_id = None

        for frame_id, (entry_ids, exit_ids) in enumerate(tqdm(list(zip(entry_timeline, exit_timeline)),
                                        desc='Coarse Pairing',
                                       disable=not verbose)):

            active_traj_ids.extend(entry_ids)
            for traj_id in exit_ids:
                active_traj_ids.remove(traj_id)

            # 1 -> 2: Entering Full State, Pairing
            if (len(active_traj_ids) == num_entities and
                len(exit_ids) == 0 and
                len(entry_ids) == 1 and
                    state == 1):

                state = 2
                pairs.append((awaiting_traj_id, entry_ids[0]))
                pairing_frames.append((self.all_trajectories[awaiting_traj_id]['end'],
                                       self.all_trajectories[entry_ids[0]]['start']))
                awaiting_traj_id = None

            # 1 -> 2: Entering Full State, NO Pairing
            elif (len(active_traj_ids) == num_entities and
                  len(exit_ids) >= 1 and
                  state == 1):

                state = 2
                awaiting_traj_id = None

            # 2 -> 1: Entering Awaiting State
            elif (len(active_traj_ids) == num_entities - 1 and
                  len(exit_ids) == 1 and
                  len(entry_ids) == 0 and
                  state == 2):

                state = 1
                awaiting_traj_id = exit_ids[0]

            # 2 -> 0: Entering Ambiguous State
            elif (len(active_traj_ids) < num_entities and
                  len(exit_ids) >= 2 and
                  state == 2):

                state = 0

            # 1 -> 0: Entering Ambiguous State
            elif (len(active_traj_ids) < num_entities and
                  len(exit_ids) >= 1 and
                  state == 1):

                state = 0
                awaiting_traj_id = None

            # 0 -> 2: Entering Full State
            elif (len(active_traj_ids) == num_entities and
                  state == 0):

                state = 2

            # 0 -> 0: Remaining Ambiguous State
            elif (len(active_traj_ids) < num_entities and
                    state == 0):

                state = 0

            # 2 -> 2: Remaining Full State, Pairing
            elif (len(active_traj_ids) == num_entities and
                    len(exit_ids) == 1 and
                    len(entry_ids) == 1 and
                    state == 2):

                state = 2
                pairs.append((exit_ids[0], entry_ids[0]))
                pairing_frames.append((self.all_trajectories[exit_ids[0]]['end'],
                                       self.all_trajectories[entry_ids[0]]['start']))

            # 2 -> 2: Remaining Full State
            elif (len(active_traj_ids) == num_entities and
                    state == 2):

                state = 2

            # 1 -> 1: Remaining Awaiting State
            elif (len(active_traj_ids) == num_entities - 1 and
                  len(exit_ids) == 0 and
                    len(entry_ids) == 0 and
                    state == 1):

                state = 1

            elif len(active_traj_ids) > num_entities:
                raise ValueError(
                    f"The number of active trajectories exceeds the number of entities at frame {frame_id}.")

            else:
                raise ValueError("The trajectory pool is in an invalid state.")

        # This part is to prevent that two pairing events are too close to each other,
        # which may cause identity confusion when two occlusions happen (almost) at the same time.
        # The threshold that determines whether two pairing events are too close is the full_state_length_belief.
        if detailed_verbose:
            current_time = time()
        pairs_to_be_removed = set()
        pairs_gap = [pairing_frames[i+1][0] - pairing_frames[i][1]
                     for i in range(len(pairing_frames) - 1)]
        for i, gap in enumerate(pairs_gap):
            if gap < full_state_length_belief:
                pairs_to_be_removed.add(i)
                pairs_to_be_removed.add(i+1)
        pairs = {pair[0]: pair[1]
                 for i, pair in enumerate(pairs)
                 if i not in pairs_to_be_removed}
        if detailed_verbose:
            print(f"Time used for removing close pairing events: {time() - current_time:.2f}s")

        # This part is to join the paired trajectories.
        # Firstly, we build chains of paired trajectories.
        if detailed_verbose:
            current_time = time()
        seen_ids = []
        paired_traj_ids = []
        for id in traj_ids_to_be_paired:
            if id not in seen_ids:
                seen_ids.append(id)
                if id in pairs:
                    chain = [id]
                    next_id = pairs[id]
                    while next_id in pairs:
                        chain.append(next_id)
                        seen_ids.append(next_id)
                        next_id = pairs[next_id]
                    chain.append(next_id)
                    seen_ids.append(next_id)
                    paired_traj_ids.append(chain)
                else:
                    paired_traj_ids.append([id])
        if detailed_verbose:
            print(f"Time used for building chains of paired trajectories: {time() - current_time:.2f}s")
        # Then, we join the trajectories in each chain.
        self.coarse_paired_trajectories = []
        for pair in tqdm(paired_traj_ids, desc='Joining Trajectories', disable=not detailed_verbose):
            self.join_trajectories(
                pair, self.all_trajectories, self.coarse_paired_trajectories, join_type='coarse')

        self.coarse_paired_trajectories = tuple(self.coarse_paired_trajectories)
        self.coarse_paired = True

        return pairs

    def fine_pair(self, overlap_length_threshold, full_state_length_belief, num_entities=7, verbose=False, detailed_verbose=False):

        if not self.coarse_paired:
            raise ValueError(
                "The trajectory pool must be coarse paired before performing a fine pair.")

        entry_timeline = [[] for _ in range(len(self.timeline))]
        exit_timeline = [[] for _ in range(len(self.timeline) + 1)]
        for traj_id, traj in tqdm(list(enumerate(self.coarse_paired_trajectories)),
                                  desc='Initializing Fine Timeline',
                                disable=not detailed_verbose):
            entry_timeline[traj['start']].append(traj_id)
            exit_timeline[traj['end']].append(traj_id)

        # We pair every imcoming new trajectories with existing unpaired terminated trajectories, which terminates
        # 1. no more than overlap_threshold frames after the incoming new trajectory.
        # 2. after the last full state.
        active_traj_ids = []
        awaiting_exit_ids = []
        awaiting_entry_ids = []
        full_state_length = 0
        pairs = []

        for entry_ids, exit_ids in tqdm(list(zip(entry_timeline, exit_timeline)),
                                        desc='Analyzing Fine Timeline',
                                       disable=not detailed_verbose):

            active_traj_ids.extend(entry_ids)
            for traj_id in exit_ids:
                active_traj_ids.remove(traj_id)

            # Full State -> Full State, Reset full state length
            if (len(active_traj_ids) == num_entities and
                len(exit_ids) > 0 and
                len(entry_ids) > 0 and
                    full_state_length > 0):

                full_state_length = 1
                awaiting_exit_ids.extend(exit_ids)
                awaiting_entry_ids.extend(entry_ids)

            # Full State -> Full State, Increase full state length
            elif (len(active_traj_ids) == num_entities and
                  len(exit_ids) == 0 and
                  len(entry_ids) == 0 and
                  full_state_length > 0):

                full_state_length += 1

            # Full State -> Ambiguous State
            elif (len(active_traj_ids) < num_entities and
                  full_state_length > 0):

                full_state_length = 0
                awaiting_exit_ids.extend(exit_ids)
                awaiting_entry_ids.extend(entry_ids)

            # Ambiguous State -> Full State
            elif (len(active_traj_ids) == num_entities and
                  full_state_length == 0):

                full_state_length = 1
                awaiting_exit_ids.extend(exit_ids)
                awaiting_entry_ids.extend(entry_ids)

            # Ambiguous State -> Ambiguous State
            elif (len(active_traj_ids) < num_entities and
                  full_state_length == 0):

                awaiting_exit_ids.extend(exit_ids)
                awaiting_entry_ids.extend(entry_ids)

            # Error: active_traj_ids exceeds num_entities
            elif len(active_traj_ids) > num_entities:
                ic(active_traj_ids, full_state_length, entry_ids, exit_ids)
                raise ValueError(
                    "The number of active trajectories exceeds the number of entities.")

            # Error: The trajectory pool is in an invalid state.
            else:
                ic(active_traj_ids, full_state_length, entry_ids, exit_ids)
                raise ValueError("The trajectory pool is in an invalid state.")

            if full_state_length == full_state_length_belief + 1:
                pairs.append((tuple(awaiting_exit_ids),
                              tuple(awaiting_entry_ids)))
                awaiting_exit_ids = []
                awaiting_entry_ids = []

        id_mapping = {}

        for exit_ids, entry_ids in tqdm(pairs, desc='Fine Pairing', disable=not verbose):

            if len(exit_ids) == 0:
                continue

            # Use Hungarian algorithm to pair the trajectories
            weights = []
            for exit_id in exit_ids:
                current_weights = []
                for entry_id in entry_ids:
                    if (self.coarse_paired_trajectories[exit_id]['end']
                            - self.coarse_paired_trajectories[entry_id]['start']) > overlap_length_threshold:
                        current_weights.append(np.inf)
                    else:
                        current_weights.append(calculate_distance_between_trajectories(
                            self.coarse_paired_trajectories[exit_id]['trajectory'],
                            self.coarse_paired_trajectories[entry_id]['trajectory'],
                            self.coarse_paired_trajectories[entry_id]['start']
                            - self.coarse_paired_trajectories[exit_id]['end']))
                weights.append(current_weights)
            weights = np.array(weights)
            row_ind, col_ind = linear_sum_assignment(weights)

            # Update id_mapping
            for i, j in zip(row_ind, col_ind):
                if weights[i, j] != np.inf:
                    id_mapping[exit_ids[i]] = entry_ids[j]
                else:
                    raise ValueError(
                        "The trajectory pairing yeilds an infinite distance.")

        # Join the paired trajectories.
        # Firstly, we build chains of paired trajectories.
        if detailed_verbose:
            current_time = time()
        seen_ids = []
        paired_traj_ids = []  # The chains of paired trajectories
        traj_ids = [i for i in range(len(self.coarse_paired_trajectories))]
        for id in tqdm(traj_ids, desc='Building Chains', disable=not verbose):
            if id not in seen_ids:
                seen_ids.append(id)
                if id in id_mapping:
                    chain = [id]
                    next_id = id_mapping[id]
                    while next_id in id_mapping:
                        chain.append(next_id)
                        seen_ids.append(next_id)
                        next_id = id_mapping[next_id]
                    chain.append(next_id)
                    seen_ids.append(next_id)
                    paired_traj_ids.append(chain)
                else:
                    paired_traj_ids.append([id])
        if detailed_verbose:
            print(f"Time used for building chains of paired trajectories: {time() - current_time:.2f}s")
        # Then, we join the trajectories in each chain.
        self.fine_paired_trajectories = []
        for pair in tqdm(paired_traj_ids, desc='Joining Trajectories', disable=not detailed_verbose):
            self.join_trajectories(
                pair, self.coarse_paired_trajectories, self.fine_paired_trajectories, join_type='fine')

        # Padding the trajectories to make sure that they cover the whole timeline
        for traj in self.fine_paired_trajectories:

            trajs_to_be_joined = []
            if traj['start'] > 0:
                trajs_to_be_joined.append(traj['start'])
            trajs_to_be_joined.append(traj['trajectory'])
            if traj['end'] < len(self.timeline):
                trajs_to_be_joined.append(len(self.timeline) - traj['end'])

            if len(trajs_to_be_joined) > 1:
                traj['trajectory'] = JoinedTrajectory(*trajs_to_be_joined, break_type='padding')
                traj['start'] = 0
                traj['end'] = len(self.timeline)

        self.fine_paired_trajectories = tuple(self.fine_paired_trajectories)
        self.fine_paired = True

    def refine_trajectories(self, verbose=False, detailed_verbose=False):
        break_entry_timeline = [[] for _ in range(len(self.timeline))]
        break_exit_timeline = [[] for _ in range(len(self.timeline) + 1)]
        for i, traj in tqdm(list(enumerate(self.fine_paired_trajectories)),
                            desc='Initializing Refine Timeline',
                           disable=not detailed_verbose):
            for k, v in traj['trajectory'].breakpoints.items():
                if v['length'] > 0:
                    break_entry_timeline[k].append(i)
                    break_exit_timeline[k + v['length']].append(i)
        
        active_breaks = []
        full_state = True
        awaiting_traj_id = []
        break_start_index = None

        for index, (entry_ids, exit_ids) in tqdm(list(enumerate(list(zip(break_entry_timeline, break_exit_timeline)))),
                                                 desc='Refining Trajectories',
                                                 disable=not verbose):

            active_breaks.extend(entry_ids)
            for traj_id in exit_ids:
                active_breaks.remove(traj_id)

            # Full State -> Full State
            if len(active_breaks) == 0 and full_state:

                pass

            # Full State -> Ambiguous State
            elif len(active_breaks) > 0 and full_state:

                full_state = False
                break_start_index = index
                awaiting_traj_id.extend([traj_id for traj_id in entry_ids])
                
            # Ambiguous State -> Ambiguous State
            elif len(active_breaks) > 0 and not full_state:

                awaiting_traj_id.extend([traj_id for traj_id in entry_ids])
                
            # Ambiguous State -> Full State, Refine
            elif ((len(active_breaks) == 0 and not full_state) or
                  index == len(self.timeline)):

                full_state = True

                updated = True
                loop_counter = 0
                awaiting_traj_id = list(set(awaiting_traj_id))
                fragments_to_be_removed = []
                
                while updated:
                    updated = False
                    loop_counter += 1

                    # Try to refine with fragments (length=1)
                    candidate_frag_ids = set()
                    for i, frame in enumerate(self.fragment_timeline[break_start_index:index]):
                        for j in range(len(frame)):
                            candidate_frag_ids.add((break_start_index + i, j))
                    candidate_frag_ids = list(candidate_frag_ids)
                    candidate_positions = [frag_id # (frame_index, position_index)
                                           for frag_id in candidate_frag_ids
                                           if frag_id not in fragments_to_be_removed]

                    if len(candidate_positions) >= 0:
                        weights = []
                        for traj_id in awaiting_traj_id:
                            weight = []
                            for pos_index in candidate_positions:
                                distance = self.fine_paired_trajectories[traj_id]['trajectory'].evaluate_insertion(
                                    pos_index[0],
                                    [self.fragment_timeline[pos_index[0]][pos_index[1]][0]])
                                weight.append(distance)
                            weights.append(weight)
                        
                        weights = np.array(weights)
                        row_ind, col_ind = linear_sum_assignment(weights)
                        for i, j in zip(row_ind, col_ind):
                            if weights[i, j] < 0:
                                frame_index, index = candidate_positions[j]
                                self.fine_paired_trajectories[
                                    awaiting_traj_id[i]]['trajectory'].insert(
                                        frame_index,
                                        DirectTrajectory([self.fragment_timeline[frame_index][index][0]],
                                                         [self.fragment_timeline[frame_index][index][1]],
                                                         [self.fragment_timeline[frame_index][index][2]]))
                                updated = True
                                fragments_to_be_removed.append((frame_index, index))

                awaiting_traj_id = []
                break_start_index = None

                for frame_index, index in sorted(fragments_to_be_removed, reverse=True):
                    self.fragment_timeline[frame_index].pop(index)

            else:
                raise ValueError("The trajectory pool is in an invalid state.")

    def __getitem__(self, traj_id):
        if self.fine_paired:
            return self.fine_paired_trajectories[traj_id]['trajectory']
        elif self.coarse_paired:
            return self.coarse_paired_trajectories[traj_id]['trajectory']
        else:
            return self.all_trajectories[traj_id]['trajectory']

    def visualize_raw_timeline(self, min_traj_length, start_from=0, end_at=np.inf):

        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 6))
        trajectory_ids = []

        # Add bars for the trajectory lifespans
        for (idx, start, duration) in zip(range(len(self.all_trajectories)),
                                          [trajectory['start']
                                           for trajectory in self.all_trajectories],
                                          [trajectory['end'] - trajectory['start']
                                           for trajectory in self.all_trajectories]):
            end = start + duration
            if duration >= min_traj_length and not self.all_trajectories[idx]['discard'] and end > start_from and start <= end_at:
                plot_start = max(start, start_from)
                plot_duration = min(end-plot_start, end_at-plot_start)
                ax.broken_barh([(plot_start, plot_duration)],
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

        ax.grid(True)
        plt.show()

    def __len__(self):
        if self.fine_paired:
            return len(self.fine_paired_trajectories)
        elif self.coarse_paired:
            return len(self.coarse_paired_trajectories)
        else:
            return len(self.all_trajectories)

    def visualize_paired_timeline(self, pair_type='refine', linewidth=4):

        if pair_type == 'fine' or pair_type == 'refine':
            paired_trajectories = self.fine_paired_trajectories
        elif pair_type == 'coarse':
            paired_trajectories = self.coarse_paired_trajectories
        else:
            raise ValueError(
                "The pair type should be 'refine', 'fine' or 'coarse'.")

        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 6))
        trajectory_ids = []

        coarse_events = []
        fine_events = []
        refine_events = []

        # Add horizontal lines for the trajectory lifespans
        for idx, trajectory in enumerate(paired_trajectories):
            start = trajectory['start']
            end = trajectory['end']
            ax.hlines(y=idx, xmin=start, xmax=end,
                      color='tab:blue', linewidth=linewidth)
            trajectory_ids.append(idx)
            if trajectory['trajectory'].breakpoints is not None:
                for k, v in trajectory['trajectory'].breakpoints.items():
                    if v['type'] == 'coarse':
                        coarse_events.append((idx, k + trajectory['start']))
                    elif v['type'] == 'fine':
                        fine_events.append((idx, k + trajectory['start']))
                    if v['refined'] == True:
                        refine_events.append((idx, k + trajectory['start']))
                    
        # Add vertical lines for key events
        for event in coarse_events:
            traj_id, frame_index = event
            # Calculate the ymin and ymax for the red lines in data coordinates
            ymin = traj_id - 0.4
            ymax = traj_id + 0.4
            ax.vlines(x=frame_index, ymin=ymin,
                      ymax=ymax, color='green', linewidth=1)
        if pair_type == 'fine' or pair_type == 'refine':
            for event in fine_events:
                traj_id, frame_index = event
                # Calculate the ymin and ymax for the red lines in data coordinates
                ymin = traj_id - 0.4
                ymax = traj_id + 0.4
                ax.vlines(x=frame_index, ymin=ymin,
                          ymax=ymax, color='orange', linewidth=1)
        # Add vertical lines for key events
        if pair_type == 'refine':
            for event in refine_events:
                traj_id, frame_index = event
                # Calculate the ymin and ymax for the red lines in data coordinates
                ymin = traj_id - 0.4
                ymax = traj_id + 0.4
                ax.vlines(x=frame_index, ymin=ymin,
                          ymax=ymax, color='red', linewidth=1)

        # Set the y-axis labels to the trajectory IDs
        ax.set_yticks(range(len(trajectory_ids)))
        ax.set_yticklabels(trajectory_ids)

        # Set the rest of the labels and title
        ax.set_xlabel('Frame')
        ax.set_ylabel('Trajectory ID')
        ax.set_title('Paired Timeline Visualization')

        ax.grid(True)
        plt.show()

    def visualize_trajectories_length_cdf(self, markersize=1):

        trajectory_lengths = [trajectory['end'] - trajectory['start']
                              if trajectory['end'] is not None
                              else len(self.timeline) - trajectory['start']
                              for trajectory in self.all_trajectories
                              if not trajectory['discard']]
        if self.prepend_paired:
            num_fragments = sum([len(frag['positions'])
                                for frag in self.fragments])
            trajectory_lengths.extend([1.5 for i in range(num_fragments)])
            fragment_percentage = num_fragments / len(trajectory_lengths) * 100
            annotation_text = f"Fragments: {num_fragments} ({fragment_percentage:.2f}%)"
            plt.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                         bbox=dict(boxstyle="round", fc="white", ec="gray", pad=0.2))
        trajectory_lengths = np.array(trajectory_lengths)
        sorted_lengths = np.sort(trajectory_lengths)
        yvals = np.arange(1, len(sorted_lengths)+1) / len(sorted_lengths)

        plt.plot(sorted_lengths, yvals, marker='.',
                 linestyle='none', markersize=markersize)
        plt.title('CDF of Trajectory Lengths')
        plt.xlabel('Length')
        plt.xscale('log')
        plt.ylabel('CDF')
        plt.grid(True)
        plt.show()
    
    def stats(self):
    
        false_negative = 0
        refine = 0
        merge = 0
        fragment = 0
        prepend = 0
        append = 0
        coarse_pair = 0
        fine_pair = 0
        padding = 0

        for traj in self:
            for source in traj.sources:
                if source == 'none' or source == 'interpolate':
                    false_negative += 1
                elif source == 'refine':
                    refine += 1
                elif source == 'fragment':
                    fragment += 1
                elif source == 'prepend':
                    prepend += 1
                elif source == 'append':
                    append += 1
                elif source == 'merge':
                    merge += 1
                else:
                    raise ValueError(f"Unknown source: {source}")
            
            for breakpoint, info in traj.breakpoints.items():
                if info['type'] == 'coarse':
                    coarse_pair += 1
                elif info['type'] == 'fine':
                    fine_pair += 1
                elif info['type'] == 'padding':
                    padding += 1
                else:
                    raise ValueError(f"Unknown breakpoint type: {info['type']}")
                
        true_positive = refine + fragment + prepend + append + merge
        false_positive = sum([len(frame)
                            for frag in self.fragments
                            for frame in frag['positions']]) + merge
        assert true_positive + false_negative == len(self) * len(self[0])
                
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        
        f1 = 2 * (precision * recall) / (precision + recall)
                
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'append_contribution': append / true_positive,
            'prepend_contribution': prepend / true_positive,
            'refine_contribution': refine / true_positive,
            'fragment_contribution': fragment / true_positive,
            'merge_contribution': merge / true_positive,
            'coarse_pair_contribution': coarse_pair / (coarse_pair + fine_pair + padding),
            'fine_pair_contribution': fine_pair / (coarse_pair + fine_pair + padding),
            'padding_contribution': padding / (coarse_pair + fine_pair + padding),
        }

class TrackerV2:

    def __init__(self,
                 num_keypoints=7, # Number of fish being tracked in each frame
                 distance_threshold_per_frame=1.5, # The maximum distance of a fish can travel between frames
                 traj_length_belief=3, # The minimum frames to make a trajectory valid ground truth
                 full_state_length_belief=3, # The minimum frames to consider a timeline as full state
                 momentum_window=3, # The window size for calculating the momentum (velocity)
                 overlap_length_threshold=3, # The maximum overlappping frames to fine pair two trajectories
                 max_gap=1,  # The maximum gap inside a trajectory
                 tail_weight=0.5):
        """
        Args:
            num_keypoints: The number of keypoints to be tracked.
            distance_threshold_per_frame: The maximum distance a keypoint can travel between frames.
        """
        if traj_length_belief <= overlap_length_threshold:
            raise ValueError(
                "The overlap length threshold should be strictly smaller than the trajectory length belief.")
        self.num_keypoints = num_keypoints
        self.distance_threshold_per_frame = distance_threshold_per_frame
        self.traj_length_belief = traj_length_belief
        self.full_state_length_belief = full_state_length_belief
        self.momentum_window = momentum_window
        self.overlap_length_threshold = overlap_length_threshold
        self.max_gap = max_gap
        self.tail_weight = tail_weight
        self._build_initialized = False

    def initialize_build(self, initial_positions, initial_confidences, initial_match_types):
        """
        Args:
            initial_positions: The initial positions of the keypoints to be tracked.
            initial_confidences: The initial confidences of the keypoints to be tracked.
            initial_match_types: The initial match types of the keypoints to be tracked.
        """

        if self._build_initialized:
            raise ValueError("The tracker has already been initialized.")

        # Check if the shape of the initial_positions matches each other.
        if len(set([initial_position.shape for initial_position in initial_positions])) != 1:
            raise ValueError(
                "The shape of the initial positions should match each other.")

        self.traj_pool = TrajectoryPoolV2(
            initial_positions, initial_confidences, initial_match_types,
            momentum_window=self.momentum_window, tail_weight=self.tail_weight, max_gap=self.max_gap)

        self._build_initialized = True

    def step_build_traj_pool(self, new_positions, new_confidences, new_match_types):
        """
        Args:
            new_positions: The new positions of the keypoints to be tracked.
        """
        if not self._build_initialized:
            raise ValueError("The tracker has not been initialized yet.")

        # Use the Hungarian algorithm to connect the new positions to the existing trajectories.
        initial_trajectory_ids = set()
        weights = []
        for trajectory_id in self.traj_pool.active_traj_id:
            weights.append([])
            for pos in new_positions:
                is_initial, distance = self.traj_pool[trajectory_id].calculate_distance_momentum_weighted(pos)
                weights[-1].append(distance)
                if is_initial:
                    initial_trajectory_ids.add(trajectory_id)

        # Each row represents a previous trajectory, each column represents a new position.
        weights = np.array(weights)
        row_ind, col_ind = linear_sum_assignment(weights)

        not_assigned_traj_ids = set(self.traj_pool.active_traj_id)
        not_assigned_positions = set(range(len(new_positions)))

        for previous_index, new_position_index in zip(row_ind, col_ind):
            if weights[previous_index, new_position_index] < self.distance_threshold_per_frame:
                # If the distance is within the threshold, connect the new position to the existing trajectory.
                trajectory_id = self.traj_pool.active_traj_id[previous_index]
                self.traj_pool.update_trajectory(trajectory_id,
                                                 new_positions[new_position_index],
                                                 new_confidences[new_position_index],
                                                 new_match_types[new_position_index])
                not_assigned_traj_ids.remove(trajectory_id)
                not_assigned_positions.remove(new_position_index)

        if len(initial_trajectory_ids.intersection(not_assigned_traj_ids)) > 0:
            weights = []
            awaiting_traj_ids = list(initial_trajectory_ids.intersection(not_assigned_traj_ids))
            awaiting_position_indices = list(not_assigned_positions)
            for traj_id in awaiting_traj_ids:
                weights.append([])
                for pos_index in awaiting_position_indices:
                    distance = self.traj_pool[traj_id].ray_distance(new_positions[pos_index])
                    weights[-1].append(distance)
            weights = np.array(weights)
            row_ind, col_ind = linear_sum_assignment(weights)
            for i, j in zip(row_ind, col_ind):
                if weights[i, j] < self.distance_threshold_per_frame:
                    self.traj_pool.update_trajectory(awaiting_traj_ids[i],
                                                     new_positions[awaiting_position_indices[j]],
                                                     new_confidences[awaiting_position_indices[j]],
                                                     new_match_types[awaiting_position_indices[j]])
                    not_assigned_traj_ids.remove(awaiting_traj_ids[i])
                    not_assigned_positions.remove(awaiting_position_indices[j])

        for trajectory_id in not_assigned_traj_ids:
            # If the trajectory is not connected to any new position, give it a gap
            self.traj_pool.gap_trajectory(trajectory_id)
        for position_index in not_assigned_positions:
            # If the new position is not connected to any existing trajectory, create a new trajectory
            self.traj_pool.add_trajectory(new_positions[position_index],
                                           new_confidences[position_index],
                                           new_match_types[position_index])

        self.traj_pool.end_time_step()

    def build_traj_pool(self, coords, all_confidences, all_match_types, verbose=False):
        """
        Args:
            coords: List of numpy arrays with shape (num_positions, dimension).
        """
        self.initialize_build(coords[0], all_confidences[0], all_match_types[0])

        if verbose:
            iterator = tqdm(list(zip(coords[1:], all_confidences[1:], all_match_types[1:])),
                            desc="Forward Build")
        else:
            iterator = zip(coords[1:], all_confidences[1:], all_match_types[1:])

        for positions, confidences, match_types in iterator:
            self.step_build_traj_pool(positions, confidences, match_types)

        self.traj_pool.timeline = [list(ids) for ids in self.traj_pool.timeline]

    def track(self, coords, confidences, match_types, verbose=False):
        """
        Args:
            matched_coords: List of coordinates of the keypoints to be tracked. Each coordinate is a numpy array.
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
        if verbose:
            print("Step 1: Building the trajectory pool.")
            current_time = time()
        self.build_traj_pool(coords, confidences, match_types, verbose=verbose)
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