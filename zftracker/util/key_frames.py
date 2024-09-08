from ..preprocess.video import get_single_frame_from_clip, video_total_frames
import os
from PIL import Image
from .tqdm import TQDM as tqdm

def get_key_frames(traj_pool, threshold=0.5, max_gap=2, min_duration=6, skip=480, mode='low'):

    # find all frames with low / high confidence
    all_frames = []
    for i in range(len(traj_pool.timeline)):
        confs = [traj.confidences[i] for traj in traj_pool]
        if mode == 'low':
            if any([conf <= threshold for conf in confs]):
                all_frames.append(True)
            else:
                all_frames.append(False)
        elif mode == 'high':
            if all([conf >= threshold for conf in confs]):
                all_frames.append(True)
            else:
                all_frames.append(False)
    
    current_group = []
    gap_count = 0
    key_frames = []
    i = 0

    while i < len(all_frames):

        frame = all_frames[i]

        if frame:
            current_group.append(i)
            gap_count = 0
        else:
            gap_count += 1
        
        if gap_count > max_gap and len(current_group) > 0:
            if len(current_group) > min_duration:
                key_frames.append(current_group[len(current_group)//2])
                i += skip
            current_group = []
            gap_count = 0
            
        else:
            i += 1
    
    if len(current_group) > min_duration:
        key_frames.append(current_group[len(current_group)//2])
    
    return key_frames

def get_low_confidence_key_frames(traj_pool, threshold=0.75, window=240, min_duration=5):

    print('Deprecated function. Use get_key_frames instead.')

    # find all frames with low confidence
    all_frames = []
    for i in range(len(traj_pool.timeline)):
        for traj in traj_pool:
            if traj.confidences[i] < threshold:
                all_frames.append(i)
                break
    
    index = 0
    min_frame_number = 0
    key_frames = []
    while index < len(all_frames) - 1:
        if all_frames[index] > min_frame_number:
            # find nearby frames
            current_frame_number = all_frames[index]
            nearby_frames = [i for i in all_frames if 0 <= (i - current_frame_number) < window]
            # iterate to find more nearby frames
            if len(nearby_frames) >= 1:
                last_nearby_index = all_frames.index(nearby_frames[-1])
                while last_nearby_index < len(all_frames) - 2:
                    last_nearby_index += 1
                    if all_frames[last_nearby_index] == nearby_frames[-1] + 1:
                        nearby_frames.append(all_frames[last_nearby_index])
                    else:
                        break
                # make sure the duration is long enough
                if len(nearby_frames) >= min_duration:
                    # find the median frame
                    key_frames.append(nearby_frames[len(nearby_frames) // 2])
                    min_frame_number = all_frames[last_nearby_index] + window * 2
                else:
                    index = last_nearby_index + 1
            else:
                index += 1
        else:
            index += 1
    
    return key_frames

def save_key_frames_from_video_list(file_path_list, key_frames, output_path, rotate=False, horizontal_cut=(0, -1), verbose=True):
    total_frames = [video_total_frames(file_path) for file_path in file_path_list]
    
    for frame in tqdm(key_frames, desc="Saving key frames", disable=not verbose):
        cumulative_frames = 0
        for i, total_frame in enumerate(total_frames):
            cumulative_frames += total_frame
            if frame < cumulative_frames:
                frame_in_video = frame - (cumulative_frames - total_frame)
                image_array = get_single_frame_from_clip(
                    file_path_list[i], frame_in_video, rotate=rotate)[:, horizontal_cut[0]:horizontal_cut[1], :]
                video_name = os.path.splitext(os.path.basename(file_path_list[i]))[0]
                image_filename = f"{video_name}_{str(frame_in_video).zfill(4)}.png"
                image_path = os.path.join(output_path, image_filename)
                image = Image.fromarray(image_array)
                image.save(image_path)
                break
            else:
                continue

def save_frame_from_video(video_path, frame_index, output_path, rotate=False, horizontal_cut=(0, -1)):
    image_array = get_single_frame_from_clip(video_path, frame_index, rotate=rotate)[:, horizontal_cut[0]:horizontal_cut[1], :]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    image_filename = f"{video_name}_{str(frame_index).zfill(4)}.png"
    image_path = os.path.join(output_path, image_filename)
    image = Image.fromarray(image_array)
    image.save(image_path)