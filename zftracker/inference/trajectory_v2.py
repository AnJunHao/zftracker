from .trajectory import TrajectoryPoolV2, JoinedTrajectoryV2, distance_to_ray
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..criterion.graph import get_subgraphs_from_mapping, analyze_full_graph
import networkx as nx
from collections import Counter
from icecream import ic
from ..util.tqdm import TQDM as tqdm
from time import time
from scipy.optimize import root_scalar
import torch

class JoinedTrajectoryV3(JoinedTrajectoryV2):

    def __init__(self, *trajectories, break_type=None):
        super().__init__(*trajectories, break_type=break_type)
        self.geometric_prediction_param = trajectories[0].geometric_prediction_param
    
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

                if len(self) < self.geometric_prediction_param['length']:
                    predicted_position = position
                    match_type += '_net'
                else:
                    predicted_head = self.geometric_prediction_param['predictor'](
                        self.trajectory[-self.geometric_prediction_param['length']:],
                        {'h': position[2:6]}
                    )
                    weighted_predicted_head = (predicted_head * self.geometric_prediction_param['weight']['head'] +
                                                position[0:2] * (1 - self.geometric_prediction_param['weight']['head']))
                    predicted_position = np.array([*weighted_predicted_head, *position[2:6]])
                    match_type += '_geo'
            
            # The midsec is missing, we will predict the midsec with Side (t-h) Angle (t-h-m) Side (h-m)
            # In the prediction, we will provide side TH, angle THM, and average HM
            elif nodes[1] == 0:

                if len(self) < self.geometric_prediction_param['length']:
                    predicted_position = position
                    match_type += '_net'
                else:
                    predicted_midsec = self.geometric_prediction_param['predictor'](
                        self.trajectory[-self.geometric_prediction_param['length']:],
                        {'m': position[[0, 1, 4, 5]]}
                    )
                    weighted_predicted_midsec = (predicted_midsec * self.geometric_prediction_param['weight']['midsec'] +
                                                position[2:4] * (1 - self.geometric_prediction_param['weight']['midsec']))
                    predicted_position = np.array([*position[0:2], *weighted_predicted_midsec, *position[4:6]])
                    match_type += '_geo'

            # The tail is missing, we will predict the tail with Side (h-m) Angle (h-m-t) Side (m-t)
            # In the prediction, we will provide side HM, angle HMT, and average MT
            elif nodes[2] == 0:
                
                if len(self) < self.geometric_prediction_param['length']:
                    predicted_position = position
                    match_type += '_net'
                else:
                    predicted_tail = self.geometric_prediction_param['predictor'](
                        self.trajectory[-self.geometric_prediction_param['length']:],
                        {'t': position[0:4]}
                    )
                    weighted_predicted_tail = (predicted_tail * self.geometric_prediction_param['weight']['tail'] +
                                                position[4:6] * (1 - self.geometric_prediction_param['weight']['tail']))
                    predicted_position = np.array([*position[0:4], *weighted_predicted_tail])
                    match_type += '_geo'

            else:
                raise ValueError("Cannot find the missing node.")

        elif sum(nodes) == 1:

            # The head is present. We will predict the midsec and tail using the head.
            if nodes[0] == 1:

                if len(self) < self.geometric_prediction_param['length']:
                    predicted_position = position
                    match_type += '_net'
                else:
                    predicted_midsec_tail = self.geometric_prediction_param['predictor'](
                        self.trajectory[-self.geometric_prediction_param['length']:],
                        {'mt': position[0:2]}
                    )
                    weighted_predicted_midsec = (predicted_midsec_tail[0:2] * self.geometric_prediction_param['weight']['mt']['midsec'] +
                                                position[2:4] * (1 - self.geometric_prediction_param['weight']['mt']['midsec']))
                    weighted_predicted_tail = (predicted_midsec_tail[2:4] * self.geometric_prediction_param['weight']['mt']['tail'] +
                                                position[4:6] * (1 - self.geometric_prediction_param['weight']['mt']['tail']))
                    predicted_position = np.array([*position[0:2], *weighted_predicted_midsec, *weighted_predicted_tail])
                    match_type += '_geo'

            # The midsec is present. We will predict the head and tail using the midsec.
            elif nodes[1] == 1:
                    
                if len(self) < self.geometric_prediction_param['length']:
                    predicted_position = position
                    match_type += '_net'
                else:
                    predicted_head_tail = self.geometric_prediction_param['predictor'](
                        self.trajectory[-self.geometric_prediction_param['length']:],
                        {'ht': position[2:4]}
                    )
                    weighted_predicted_head = (predicted_head_tail[0:2] * self.geometric_prediction_param['weight']['ht']['head'] +
                                                position[0:2] * (1 - self.geometric_prediction_param['weight']['ht']['head']))
                    weighted_predicted_tail = (predicted_head_tail[2:4] * self.geometric_prediction_param['weight']['ht']['tail'] +
                                                position[4:6] * (1 - self.geometric_prediction_param['weight']['ht']['tail']))
                    predicted_position = np.array([*weighted_predicted_head, *position[2:4], *weighted_predicted_tail])
                    match_type += '_geo'
            
            # The tail is present. We will predict the head and midsec using the tail.
            elif nodes[2] == 1:

                if len(self) < self.geometric_prediction_param['length']:
                    predicted_position = position
                    match_type += '_net'
                else:
                    predicted_head_midsec = self.geometric_prediction_param['predictor'](
                        self.trajectory[-self.geometric_prediction_param['length']:],
                        {'hm': position[4:6]}
                    )
                    weighted_predicted_head = (predicted_head_midsec[0:2] * self.geometric_prediction_param['weight']['hm']['head'] +
                                                position[0:2] * (1 - self.geometric_prediction_param['weight']['hm']['head']))
                    weighted_predicted_midsec = (predicted_head_midsec[2:4] * self.geometric_prediction_param['weight']['hm']['midsec'] +
                                                position[2:4] * (1 - self.geometric_prediction_param['weight']['hm']['midsec']))
                    predicted_position = np.array([*weighted_predicted_head, *weighted_predicted_midsec, *position[4:6]])
                    match_type += '_geo'

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
    
    def meet_conf_belief(self, conf_belief):
        accumulated_conf = 0
        for conf in self.confidences:
            accumulated_conf += sum(conf) / 3
            if accumulated_conf >= conf_belief:
                return True
        return False

class TrajectoryV3(JoinedTrajectoryV3):
    def __init__(self,
                 initial_position: np.ndarray,
                 initial_confidence: np.ndarray,
                 initial_match_type: str,
                 momentum_window: int,
                 tail_weight: float,
                 geometric_prediction_param: float):
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
        self.geometric_prediction_param = geometric_prediction_param
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
                 geometric_prediction_param):
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
                                                           geometric_prediction_param=geometric_prediction_param),
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
        self.geometric_prediction_param = geometric_prediction_param
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

    def coords_numpy(self):
        return np.array(self).transpose(1, 0, 2).reshape(-1, 6 * len(self))
    
    def save_coords_numpy(self, path):
        np.save(path, self.coords_numpy())
    
    def confs_numpy(self):
        return np.array([traj.confidences for traj in self]).transpose(1, 0, 2).reshape(-1, 3 * len(self))
    
    def save_confs_numpy(self, path):
        np.save(path, self.confs_numpy())

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
                                                               geometric_prediction_param=self.geometric_prediction_param),
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
    
    def build_fragment_timeline(self, traj_conf_belief):
        fragment_timeline = [[] for _ in range(len(self.timeline))]
        for i, traj in enumerate(self.all_trajectories):
            if traj['discard'] == False and not traj['trajectory'].meet_conf_belief(traj_conf_belief):
                start_frame = traj['start']
                end_frame = traj['end']
                for i in range(start_frame, end_frame):
                    if traj['trajectory'].trajectory[i-start_frame] is not None:
                        fragment_timeline[i].append(
                            (traj['trajectory'].trajectory[i-start_frame],
                                traj['trajectory'].confidences[i-start_frame],
                                traj['trajectory'].match_types[i-start_frame]))
                traj['discard'] = True
        self.fragment_timeline = fragment_timeline # A list containing the fragments in each frame. The fragments in each frame is a list of tuples (position, confidence, match_type).
    
    def prepend_pair(self, distance_threshold_per_frame, traj_conf_belief, verbose=False):
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
            if traj['discard'] == False and not traj['trajectory'].meet_conf_belief(traj_conf_belief):
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

    def coarse_pair(self, traj_conf_belief, full_state_length_belief, num_entities=7, verbose=False, detailed_verbose=False):

        traj_ids_to_be_paired = []
        entry_timeline = [[] for _ in range(len(self.timeline))]
        exit_timeline = [[] for _ in range(len(self.timeline) + 1)]
        for traj_id, traj in tqdm(list(enumerate(self.all_trajectories)),
                                  desc='Initializing Coarse Timeline',
                                disable=not detailed_verbose):
            if (traj['trajectory'].meet_conf_belief(traj_conf_belief)
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

class TrackerV3:

    def __init__(self,
                 num_keypoints=7,
                 distance_threshold_per_frame=3,
                 traj_conf_belief=3,
                 full_state_length_belief=3,
                 momentum_window=3,
                 overlap_length_threshold=1,
                 tail_weight=0.5,   # Note that in this version, the tail_weight should be better
                                    # considered as a factor to normalize the mae (so that the mae
                                    # of the tail is comparable to the mae of the head and midsec)
                 doublecheck_threshold=2,
                 geometric_prediction_param={},
                 good_graph_priority=True):
        self.num_keypoints = num_keypoints
        self.distance_threshold_per_frame = distance_threshold_per_frame
        self.traj_conf_belief = traj_conf_belief
        self.full_state_length_belief = full_state_length_belief
        self.momentum_window = momentum_window
        self.overlap_length_threshold = overlap_length_threshold
        self.max_gap = 0
        self.tail_weight = tail_weight
        self.geometric_prediction_param = geometric_prediction_param
        self.doublecheck_threshold = doublecheck_threshold
        self.good_graph_priority = good_graph_priority

        if int(traj_conf_belief) + 1 <= overlap_length_threshold:
            raise ValueError(
                "The trajectory confidence belief should be larger than the overlap length threshold.")

    def initialize_build(self, all_raw_coords):
        self.length = len(all_raw_coords[0][0])
        self.all_coords_dicts = self.get_all_coords_dicts_from_all_raw_coords(all_raw_coords)
        initial_subgraphs = self.get_subgraphs_from_coords_dict(self.all_coords_dicts[0])
        good_subgraphs, bad_subgraphs = self.analyze_subgraphs(initial_subgraphs)
        if len(good_subgraphs) != self.num_keypoints:
            print(f'Warning: the number of initial good subgraphs {len(good_subgraphs)} is not equal to the number of keypoints.')
        initial_coords, initial_confs, initial_nodes, initial_types = self.get_coords_from_good_graphs(good_subgraphs, self.all_coords_dicts[0])
        self.traj_pool = TrajectoryPoolV3(initial_coords, initial_confs, initial_types,
                                        self.momentum_window, self.tail_weight, self.max_gap, self.geometric_prediction_param)
        
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

        if self.good_graph_priority:
            # If we still have not-assigned good positions, we need to add them as new trajectories.
            for position_index in not_assigned_positions:
                self.traj_pool.add_trajectory(good_coords[position_index],
                                            good_confs[position_index],
                                            good_types[position_index])
        else:
            # If we have remaining not-assigned bad positions, add them into the bad graph.
            for position_index in not_assigned_positions:
                subgraph = good_subgraphs[position_index]
                bad_graph.add_nodes_from(subgraph.nodes(data=True))
                bad_graph.add_edges_from(subgraph.edges(data=True))

        # Then, we should check the num_fish (num_keypoints) and number of active trajectrories
        # Case 1: num_active_traj == num_fish: no need to do anything
        # Case 2: num_active_traj < num_fish:
        #       a. If there is remaining not-assigned trajs, we use bad coords to fill them
        #       b. If there is no remaining not assigned trajs, we add the bad coords as new trajs
        # Case 3: num_active_traj > num_fish: should NEVER happen

        if self.good_graph_priority:
            num_active_traj = (num_current_active_traj - len(not_assigned_traj_ids)  # Updated active trajs
                               + len(not_assigned_positions)) # Newly added active trajs
        else: 
            num_active_traj = num_current_active_traj - len(not_assigned_traj_ids) # Updated active trajs
        
        if num_active_traj < self.num_keypoints:

            # Case 2: num_active_traj < num_fish:
            
            bad_dict = self.get_coords_from_bad_graph(bad_graph, coords_dict)
            remaining_gap = self.num_keypoints - num_active_traj # Prevent updating too many old trajs with bad coords

            if len(not_assigned_traj_ids) > 0:

                # Case 2: a. If there is remaining not-assigned trajs, we use bad coords to fill them

                awaiting_traj_ids = list(not_assigned_traj_ids)
                
                corresponding_assigned_nodes = [[None, None, None, [0, 0, 0], None, None, None] # head, midsec, tail, conf, node_head, node_midsec, node_tail
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
                                corresponding_assigned_nodes[i][3][node_hmt_index] = node_dict['confs'][j]
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
            if self.good_graph_priority:
                num_active_traj = (num_current_active_traj - len(not_assigned_traj_ids) # Updated active trajs
                                   + len(not_assigned_positions)) # Newly added *good* active trajs

            else:
                num_active_traj = num_current_active_traj - len(not_assigned_traj_ids) # Updated active trajs
            
            if num_active_traj < self.num_keypoints:
                
                graph_pool = analyze_full_graph(bad_graph)
                sorted_pool = list(sorted(graph_pool,
                            key=lambda i: (i['active'], i['n_nodes'], i['n_edges'], i['original']),
                            reverse=True))[:self.num_keypoints - num_active_traj]
                
                for i in sorted_pool:

                    if i['active'] == False:
                        break

                    cds = np.zeros(6, dtype=float)
                    conf = [0, 0, 0]
                    seen_nodes = [0, 0, 0]
                    for node in i['graph'].nodes:
                        node_hmt_index = ['head', 'midsec', 'tail'].index(node[0])
                        cds[node_hmt_index*2:node_hmt_index*2+2] = coords_dict[node[0]]['coords'][node[1]]
                        conf[node_hmt_index] = coords_dict[node[0]]['confs'][node[1]]
                        seen_nodes[node_hmt_index] = 1

                    if sum(seen_nodes) == 3:
                        self.traj_pool.add_trajectory(cds, conf, f'N3_E{i["n_edges"]}_ini')

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
                                                          conf,
                                                          f'N2_E{i["n_edges"]}_mt_ini')
                        
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
                                                          conf,
                                                          f'N2_E{i["n_edges"]}_ht_ini')

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
                                                          conf,
                                                          f'N2_E{i["n_edges"]}_hm_ini')

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
                                                      conf,
                                                      f'N1_E0_{the_node[0][0]}_ini')
                    
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
    
    def build_traj_pool(self, all_raw_coords, verbose, stop_at=None):
        self.initialize_build(all_raw_coords)
        if stop_at is None:
            for coords_dict in tqdm(self.all_coords_dicts[1:], disable=not verbose):
                self.step_build_traj_pool(coords_dict)
        else:
            for coords_dict in tqdm(self.all_coords_dicts[1:stop_at], disable=not verbose):
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
                output[(1, 0, 0)]['confs'].append(coords_dict['head']['confs'][node[1]])
                output[(1, 0, 0)]['node_name'].append(node)
            elif node[0] == 'midsec':
                cds = [*coords_dict['midsec']['head'][node[1]],
                        *coords_dict['midsec']['coords'][node[1]],
                        *coords_dict['midsec']['tail'][node[1]]]
                output[(0, 1, 0)]['coords'].append(np.array(cds))
                output[(0, 1, 0)]['confs'].append(coords_dict['midsec']['confs'][node[1]])
                output[(0, 1, 0)]['node_name'].append(node)
            elif node[0] == 'tail':
                cds = [*coords_dict['tail']['head'][node[1]],
                        *coords_dict['tail']['midsec'][node[1]],
                        *coords_dict['tail']['coords'][node[1]]]
                output[(0, 0, 1)]['coords'].append(np.array(cds))
                output[(0, 0, 1)]['confs'].append(coords_dict['tail']['confs'][node[1]])
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
            conf = [0, 0, 0]
            seen_nodes = np.zeros(3, dtype=int)
            string_seen_nodes = []

            for node in subgraph.nodes:
                if node[0] == 'head':
                    cds[0:2] = coords_dict['head']['coords'][node[1]]
                    seen_nodes[0] = 1
                    conf[0] = coords_dict['head']['confs'][node[1]]
                    string_seen_nodes.append('head')
                elif node[0] == 'midsec':
                    cds[2:4] = coords_dict['midsec']['coords'][node[1]]
                    seen_nodes[1] = 1
                    conf[1] = coords_dict['midsec']['confs'][node[1]]
                    string_seen_nodes.append('midsec')
                elif node[0] == 'tail':
                    cds[4:6] = coords_dict['tail']['coords'][node[1]]
                    seen_nodes[2] = 1
                    conf[2] = coords_dict['tail']['confs'][node[1]]
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
            fish_confs.append(conf)
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

    def track(self, raw_coords, verbose=False, stop_at=None):
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
        self.build_traj_pool(raw_coords, verbose=verbose, stop_at=stop_at)
        if verbose:
            forward_time = time() - current_time
        self.traj_pool.assign_end_value()
        self.traj_pool.build_fragment_timeline(traj_conf_belief=self.traj_conf_belief)
        """
        self.traj_pool.prepend_pair(self.distance_threshold_per_frame,
                                    self.traj_conf_belief,
                                    verbose=False) # The time cost for prepend_pair is negligible.
        if verbose:
            backward_time = time() - current_time - forward_time
        if verbose:
            print(f"Forward: {forward_time:.2f}s, Backward: {backward_time:.2f}s, "
        """
        if verbose:
            print(f"Total: {time() - current_time:.2f}s")

        # Step 2: Pair trajectories within the pool
        # Coarse Pair and Fine Pair are based on the belief that trajectories with longer length
        # are more likely to be correct. In this code, we assume that trajectories with length >=
        # length_belief_threshold are ground truth and are used for pairing.
        if verbose:
            print("Step 2: Pairing the trajectories.")
            current_time = time()
        self.traj_pool.coarse_pair(traj_conf_belief=self.traj_conf_belief,
                                   full_state_length_belief=self.full_state_length_belief,
                                   num_entities=self.num_keypoints,
                                   verbose=False, # The time cost for coarse_pair is negligible
                                   detailed_verbose=False)
        if verbose:
            coarse_pair_time = time() - current_time
        self.traj_pool.fine_pair(full_state_length_belief=self.full_state_length_belief,
                                 overlap_length_threshold=self.overlap_length_threshold,
                                 num_entities=self.num_keypoints,
                                 verbose=False, # The time cost for fine_pair is negligible
                                 detailed_verbose=False)
        if verbose:
            fine_pair_time = time() - current_time - coarse_pair_time
            print(f"Coarse Pair: {coarse_pair_time:.2f}s, Fine Pair: {fine_pair_time:.2f}s, "
                  f"Total: {time() - current_time:.2f}s")

        # Step 3: Refine the trajectories: Fill the gaps with fragment positions.
        if verbose:
            print("Step 3: Refining the trajectories.")
            current_time = time()

        self.traj_pool.refine_trajectories(verbose=False, detailed_verbose=False)

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
    
class GeometricPredictor:

    def __init__(self,
                 model,
                 device,
                 window_length):
        
        self.model = model
        self.device = device
        self.window_length = window_length

        self.model.eval()
        self.model.to(self.device)

    def __call__(self, trajectory, coords_dict):
        if len(trajectory) != self.window_length:
            raise ValueError('Trajectory length must be equal to window length + 1')
        normed_trajectory = torch.tensor(trajectory[1:], dtype=torch.float32) - torch.tensor(trajectory[0], dtype=torch.float32)
        key = list(coords_dict.keys())[0]
        if key == 'h':
            # To predict head, we need midsec & tail coords
            normed_coords = torch.tensor(coords_dict[key], dtype=torch.float32) - torch.tensor(trajectory[0][2:6], dtype=torch.float32)
            reference_coord = trajectory[-1][0:2]
        elif key == 'm':
            # To predict midsec, we need head & tail coords
            normed_coords = torch.tensor(coords_dict[key], dtype=torch.float32) - torch.tensor(trajectory[0][[0, 1, 4, 5]], dtype=torch.float32)
            reference_coord = trajectory[-1][2:4]
        elif key == 't':
            # To predict tail, we need head & midsec coords
            normed_coords = torch.tensor(coords_dict[key], dtype=torch.float32) - torch.tensor(trajectory[0][0:4], dtype=torch.float32)
            reference_coord = trajectory[-1][4:6]
        elif key == 'hm':
            # To predict head & midsec, we need tail coords
            normed_coords = torch.tensor(coords_dict[key], dtype=torch.float32) - torch.tensor(trajectory[0][4:6], dtype=torch.float32)
            reference_coord = trajectory[-1][0:4]
        elif key == 'ht':
            # To predict head & tail, we need midsec coords
            normed_coords = torch.tensor(coords_dict[key], dtype=torch.float32) - torch.tensor(trajectory[0][2:4], dtype=torch.float32)
            reference_coord = trajectory[-1][[0, 1, 4, 5]]
        elif key == 'mt':
            # To predict midsec & tail, we need head coords
            normed_coords = torch.tensor(coords_dict[key], dtype=torch.float32) - torch.tensor(trajectory[0][0:2], dtype=torch.float32)
            reference_coord = trajectory[-1][2:6]
        elif key == 'hmt':
            # To predict head, midsec & tail, we need no additional coords
            normed_coords = torch.tensor([], dtype=torch.float32)
            reference_coord = trajectory[-1]
        else:
            raise ValueError('Invalid key')
        input_dict = {'seq': torch.flatten(normed_trajectory).unsqueeze(0).to(self.device),
                      key: normed_coords.unsqueeze(0).to(self.device)}
        with torch.no_grad():
            output = self.model(input_dict)
        return output[key].cpu().numpy()[0] + reference_coord