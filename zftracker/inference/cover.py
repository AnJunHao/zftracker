import numpy as np
import shapely
from math import pi
from ..util.tqdm import TQDM as tqdm
from ..preprocess.evaluate_annotation_v2 import create_fish

def get_overlaps_from_traj_pool(traj_pool, max_distance=0, verbose=True):
    frames = [[[traj.trajectory[i][:2],
            traj.trajectory[i][2:4],
            traj.trajectory[i][4:6]]
           for traj in traj_pool]
           for i in range(len(traj_pool[0]))]
    overlaps = []

    for index, frame in enumerate(tqdm(frames, disable=not verbose)):
        for i in range(len(frame)):
            fish = create_fish(*frame[i])
            for j in range(i + 1, len(frame)):
                fish2 = create_fish(*frame[j])
                distance = fish.distance(fish2)
                if distance <= max_distance:
                    overlaps.append((index, i, j))

    return overlaps

def get_crop_center(keypoints):
    fish = create_fish(keypoints[:2], keypoints[2:4], keypoints[4:])
    coords = np.array([coord
                       for linestring in fish.geoms
                       for coord in linestring.coords])
    x, y = np.mean([np.max(coords, axis=0), np.min(coords, axis=0)], axis=0)
    return int(x), int(y)

def get_crop_dict(overlaps, traj_pool, frame_index_minus=0, frame_index_multiply=2, rescale_factor=4, frame_index_upper_bound=np.inf):
    crop_dict = {frame_index * frame_index_multiply - frame_index_minus: []
                    for frame_index, _, _ in overlaps}
    for frame_index, traj_id_a, traj_id_b in overlaps:
        if frame_index >= frame_index_upper_bound:
            break
        for traj_id in [traj_id_a, traj_id_b]:
            keypoints = traj_pool[traj_id][frame_index] * rescale_factor
            center = get_crop_center(keypoints)
            crop_dict[frame_index * frame_index_multiply - frame_index_minus].append(
                {'center': center, 'keypoints': keypoints})
    return crop_dict

def score_intersection(fish_a_coords, fish_b_coords, head_mid_distance=32, ambiguous_range=4, tail_range=8):
    print('Warning: This function is deprecated. There is no appropriate use case for this function.')
    angle_a, fish_a = create_fish(
        *[fish_a_coords[i*2:i*2+2] for i in range(3)], return_angle=True)
    angle_b, fish_b = create_fish(
        *[fish_b_coords[i*2:i*2+2] for i in range(3)], return_angle=True)
    intersection_points = fish_a.intersection(fish_b)
    score = 0
    if isinstance(intersection_points, shapely.geometry.Point):
        iterator = [intersection_points]
    elif isinstance(intersection_points, shapely.geometry.MultiPoint):
        iterator = intersection_points.geoms
    else:
        return 0
    for point in iterator:
        this_point_score = 1
        for angle, fish in ((angle_a, fish_a), (angle_b, fish_b)):
            location  = shapely.line_locate_point(fish, point)
            if location <= head_mid_distance:
                if location >= ambiguous_range:
                    this_point_score = min(this_point_score, 1)
                else:
                    this_point_score = min(this_point_score,
                                           location / ambiguous_range)
            else:
                mid_tail_length = fish.length - head_mid_distance
                relative_location = min(location - head_mid_distance,
                                        fish.length - location)
                angle_factor = 1 - abs(angle) / pi / 2 # The factor is lower when the angle is larger (0 at 360 degrees)
                angle_factor_weight = relative_location / (mid_tail_length / 2) # The factor is lower when the location is closer to the peak of angle
                angle_score = 1 * (1 - angle_factor_weight) + angle_factor_weight * angle_factor
                if fish.length - location < tail_range:
                    this_point_score = min(this_point_score,
                                           angle_score * (fish.length - location) / tail_range)
                else:
                    this_point_score = min(this_point_score, angle_score)
        score += this_point_score
    return score