from PIL import Image
import numpy as np
import os
import subprocess
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from math import log10, ceil
import re
import sys
import shlex
import albumentations as A

from ..util.tqdm import TQDM as tqdm
    
def resize_frames(frames, target_size):
    """
    Resizes a batch of frames to a target size.

    Args:
        frames (numpy.ndarray): Array of frames to resize.
        target_size (tuple): Target size for the frames, in the format (height, width).

    Returns:
        numpy.ndarray: Array of the resized frames.
    """
    transforms = A.Compose([A.Resize(*target_size, interpolation=cv2.INTER_AREA)])
    return np.array([transforms(image=f)['image'] for f in frames])

def save_clips_from_video(file, duration, destination):
    """
    DEPRECATED: Use save_clips_from_video_ffmpeg instead.
    Splits a video file into clips of the specified duration and saves them to a folder.
    
    Args:
        file (str): Path to the video file.
        duration (int): Duration of each clip, in seconds.
        destination (str): Path to the folder to save the clips.
    """

    print('This function is deprecated. Use save_clips_from_video_ffmpeg instead.')

    # Open the video file
    video = cv2.VideoCapture(file)

    # Get the video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    num_clips = int(total_duration / duration) + 1
    print('Duration of video file is %.1f seconds.' % total_duration)
    print('Total number of clips is %d.' % num_clips)

    # Calculate the number of frames per clip
    frames_per_clip = int(fps * duration)

    # Split the video into clips
    zfill_arg = ceil(log10(num_clips))
    for i in tqdm(range(num_clips)):
        # Create a VideoWriter object for the current clip
        clip_path = os.path.join(destination, file.split('.')[0]+'_clip_'+str(i).zfill(zfill_arg)+'.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        clip_writer = cv2.VideoWriter(clip_path, fourcc, fps, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        # Write frames to the clip
        for _ in range(frames_per_clip):
            ret, frame = video.read()
            if not ret:
                break
            clip_writer.write(frame)

        # Release the VideoWriter object
        clip_writer.release()

    # Release the video file
    video.release()

def save_clips_from_video_ffmpeg(file, duration, destination, path_to_ffmpeg='ffmpeg'):
    if not os.path.exists(destination):
        os.makedirs(destination)

    base_name = os.path.splitext(os.path.basename(file))[0]
    output_template = os.path.join(destination, base_name + '_%03d.mp4').replace("\\", "/")

    clip_duration = str(duration)

    # Initialize total_seconds to None
    total_seconds = None

    # Get the total duration of the video
    cmd = f"{path_to_ffmpeg} -i {shlex.quote(file)} 2>&1"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = result.stdout.decode()
    match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})", output)
    if match is not None:
        hours, minutes, seconds = match.groups()
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    if total_seconds is None:
        raise ValueError("Could not parse total video duration from FFmpeg output.")

    # Call FFmpeg and capture the stderr output
    process = subprocess.Popen([path_to_ffmpeg,
                                '-i', file,
                                '-c', 'copy',
                                '-map', '0',
                                '-segment_time', clip_duration,
                                '-f', 'segment',
                                '-reset_timestamps', '1',
                                output_template],
                               stderr=subprocess.PIPE, universal_newlines=True)

    # Parse FFmpeg output in real-time to show progress
    while True:
        line = process.stderr.readline()
        if not line:
            break

        # Extract the time from the current line
        time_match = re.search(r"time=(\d{2}):(\d{2}):(\d{2}\.\d{2})", line)
        if time_match is not None and total_seconds is not None:
            hours, minutes, seconds = time_match.groups()
            current_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            progress = current_seconds / total_seconds * 100
            print(f"Progress: {progress:.2f}%", end='\r', flush=True)

    # Wait for FFmpeg to finish
    process.wait()

    # Check if process was successful
    if process.returncode != 0:
        print("FFmpeg encountered an error")
    else:
        print("Video splitting completed successfully.")

def rotate_video(file, destination, path_to_ffmpeg='ffmpeg', rotate_option='1'):
    print('Deprecated: Rotating the video will cause dramatic loss in quality.')

    if not os.path.exists(destination):
        os.makedirs(destination)

    # Construct the output file path
    base_name = os.path.splitext(os.path.basename(file))[0]
    output_file = os.path.join(destination, base_name + '_rotated.mp4')

    # Call FFmpeg to rotate the video
    process = subprocess.Popen([path_to_ffmpeg, '-i', file, '-vf', f"transpose={rotate_option}",
                                '-c:a', 'copy', output_file], 
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    # Regex pattern to match the time
    time_pattern = re.compile(r"time=\d+:\d+:\d+.\d+")

    # Read the output line by line and search for the time pattern
    while True:
        line = process.stdout.readline()
        if not line:
            break

        # Search for the time pattern and print it
        match = time_pattern.search(line)
        if match:
            sys.stdout.write(f"\r{match.group()}")
            sys.stdout.flush()

    # Wait for the FFmpeg process to finish
    process.wait()
    
    # Check if process was successful
    if process.returncode != 0:
        print("FFmpeg encountered an error")
    else:
        print("\nVideo rotation completed successfully.")

def save_frames_from_clips(from_folder, to_folder, verbose=True):
    """
    Extracts frames from a folder of video clips.
    
    Args:
        from_folder (str): Path to the folder containing the video clips.
        to_folder (str): Path to the folder to save the extracted frames.
        verbose (bool, optional): If True, prints a message confirming the save operation. Default is True.
    """

    clip_files = [os.path.join(from_folder, name)
                  for name in os.listdir(from_folder)]  # Video files

    if not os.path.exists(to_folder):
        os.makedirs(to_folder)

    if verbose:
        clip_files = tqdm(clip_files)

    for clip_file in clip_files:
        if clip_file[-4:] not in ('.mp4', ):
            continue
        clip = VideoFileClip(clip_file)
        duration = clip.duration
        zfill_arg = ceil(log10(duration))
        for time in range(1, int(duration)):
            frame = clip.get_frame(time)
            save_name = os.path.join(to_folder,
                                     os.path.splitext(
                                         os.path.basename(clip_file))[0]
                                     + '_' + str(time).zfill(zfill_arg) + '.jpg')  # 文件名为帧数
            img = Image.fromarray(frame.astype(np.uint8))
            img.save(save_name)
        clip.close()

def get_frames_from_clip(file, fps_divisor=1, rotate=False, verbose=True, to_rgb=True, horizontal_cut=False):
    """
    Extracts all frames from a video clip and converts them from BGR to RGB.
    Optionally rotates the frames to convert a vertical video to horizontal.

    Args:
        file (str): Path to the video clip.
        fps_divisor (int, optional): Divisor for the frame rate. Defaults to 1.
        rotate (bool, optional): Whether to rotate frames to make the video horizontal. Defaults to False.
        verbose (bool, optional): If True, uses tqdm to display a progress bar. Defaults to True.
        to_rgb (bool, optional): Whether to convert the frames from BGR to RGB. Defaults to True.

    Returns:
        numpy.ndarray: Array of the frames in RGB format.
    """
    frames = []
    video = cv2.VideoCapture(file)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps_divisor < 1 or not isinstance(fps_divisor, int):
        raise ValueError("fps_divisor must be a positive integer")

    for i in tqdm(range(total_frames), disable=not verbose):
        ret, frame = video.read()
        if not ret:
            break
        if i % fps_divisor != 0:
            continue
        # Convert BGR to RGB
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if rotate:
            # Rotate the frame to convert vertical video to horizontal
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if horizontal_cut:
            frame = frame[:, horizontal_cut[0]:horizontal_cut[1]]
        frames.append(frame)

    video.release()

    return np.array(frames)

def get_frames_from_clip_generator(file, fps_divisor=1, rotate=False, verbose=True, to_rgb=True, horizontal_cut=None):
    """
    Yields frames from a video clip one at a time. Converts frames from BGR to RGB, rotates if specified,
    and applies a horizontal cut if specified.

    Args:
        file (str): Path to the video clip.
        fps_divisor (int, optional): Divisor for the frame rate. Defaults to 1.
        rotate (bool, optional): Whether to rotate frames to make the video horizontal. Defaults to False.
        verbose (bool, optional): If True, uses tqdm to display a progress bar. Defaults to True.
        to_rgb (bool, optional): Whether to convert the frames from BGR to RGB. Defaults to True.
        horizontal_cut (tuple, optional): A tuple (start, end) to slice the frame horizontally.

    Yields:
        numpy.ndarray: A single frame in RGB format.
    """
    video = cv2.VideoCapture(file)
    if not video.isOpened():
        raise IOError("Error opening video file")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps_divisor < 1 or not isinstance(fps_divisor, int):
        raise ValueError("fps_divisor must be a positive integer")

    try:
        frame_count = 0
        for i in tqdm(range(total_frames), disable=not verbose):
            ret, frame = video.read()
            if not ret:
                break

            if i % fps_divisor == 0:
                if rotate:
                    # Rotate the frame to convert vertical video to horizontal
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                if to_rgb:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if horizontal_cut:
                    # Apply horizontal cut if specified
                    frame = frame[:, horizontal_cut[0]:horizontal_cut[1]]

                yield frame
                frame_count += 1

    finally:
        video.release()

def get_specific_frames_from_clip(file, frame_indices, rotate=False, to_rgb=True, horizontal_cut=False, verbose=True):
    """
    Extracts specific frames from a video clip.

    Args:
        file (str): Path to the video clip.
        frame_indices (list): List of indices of the frames to extract.
        rotate (bool, optional): Whether to rotate the frames to make the video horizontal. Defaults to False.
        to_rgb (bool, optional): Whether to convert the frames from BGR to RGB. Defaults to True.
        horizontal_cut (bool, optional): Whether to cut the frames horizontally. Defaults to False.

    Returns:
        numpy.ndarray: Array of the extracted frames.
    """
    frames = []
    video = cv2.VideoCapture(file)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(total_frames), disable=not verbose):
        ret, frame = video.read()
        if not ret:
            break
        if i in frame_indices:
            if to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            if horizontal_cut:
                frame = frame[:, horizontal_cut[0]:horizontal_cut[1]]
            frames.append(frame)

    video.release()

    return np.array(frames)

def get_cropped_frames_and_keypoints_from_clip(file, crop_dict, crop_range=80, rotate=False, to_rgb=True, horizontal_cut=False, verbose=True):
    """
    Extracts specific frames from a video clip,
    then crop the frames based on the crop_dict.

    Args:
        file (str): Path to the video clip.
        crop_dict (dict): Dictionary of the crop parameters. Keys are the frame indices, and values are the crop parameters.
        rotate (bool, optional): Whether to rotate the frames to make the video horizontal. Defaults to False.
        to_rgb (bool, optional): Whether to convert the frames from BGR to RGB. Defaults to True.
        horizontal_cut (bool, optional): Whether to cut the frames horizontally. Defaults to False.

    Returns:
        numpy.ndarray: Array of the extracted frames.
    """
    all_images = []
    all_keypoints = []

    video = cv2.VideoCapture(file)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(total_frames), disable=not verbose):
        ret, frame = video.read()
        if not ret:
            break
        if i in crop_dict:
            if to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            if horizontal_cut:
                frame = frame[:, horizontal_cut[0]:horizontal_cut[1]]

            for config in crop_dict[i]:
                crop_x, crop_y = config['center']
                frame_shape = frame.shape

                # Calculate the slicing indices
                top = max(0, crop_y - crop_range)
                bottom = min(frame_shape[0], crop_y + crop_range)
                left = max(0, crop_x - crop_range)
                right = min(frame_shape[1], crop_x + crop_range)

                # Crop the image
                cropped_frame = frame[top:bottom, left:right]

                # Calculate padding if necessary
                pad_top = -min(0, crop_y - crop_range)
                pad_bottom = max(0, (crop_y + crop_range) - frame_shape[0])
                pad_left = -min(0, crop_x - crop_range)
                pad_right = max(0, (crop_x + crop_range) - frame_shape[1])

                # Create padding tuple for np.pad
                padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

                # Pad the image
                cropped_padded_frame = np.pad(cropped_frame, padding, 'constant', constant_values=0)

                all_images.append(cropped_padded_frame)

                keypoints = config['keypoints']
                all_keypoints.append(keypoints-np.array([left-pad_left, top-pad_top]*3))

    video.release()

    return all_images, all_keypoints

def get_frames_from_clip_moviepy(file):
    """
    DEPRECATED: There is an issue with VideoFileClip.
    Out of unknown reasons, it returns 2408 frames from a 2400-frame video.
    The extra 8 frames come from nowhere.

    Extracts all frames from a video clip.

    Args:
        file (str): Path to the video clip.

    Returns:
        numpy.ndarray: Array of the frames.
    """
    print('This function is deprecated. Use get_frames_from_clip instead.')

    # Load the video file
    video = VideoFileClip(file)

    # Extract the frames
    frames = []
    for frame in video.iter_frames():
        frames.append(frame)

    return np.array(frames)

def get_single_frame_from_clip_stable(file, frame_index, rotate=False, to_rgb=True):
    """
    Extracts a single frame from a video clip.
    This function works slower than get_single_frame_from_clip, but it aligns with the total number of frames in the video.

    Args:
        file (str): Path to the video clip.
        frame_index (int): Index of the frame to extract.
        rotate (bool, optional): Whether to rotate the frame to make the video horizontal. Defaults to False.
        to_rgb (bool, optional): Whether to convert the frame from BGR to RGB. Defaults to True.

    Returns:
        numpy.ndarray: Array of the extracted frame.
    """
    # Open the video file
    cap = cv2.VideoCapture(file)

    # Check if video opened successfully
    if not cap.isOpened():
        raise Exception(f"Error: Could not open video file {file}")

    for i in range(frame_index+1):
        ret, frame = cap.read()
        if not ret:
            raise Exception(
                f"Error: While attempting to read frame {frame_index} in a consecutive manner from video file {file}"
                f", the frame index {i} is out of range.")

    if ret:
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    else:
        raise Exception(
            f"Error: Could not read frame {frame_index} from video file {file}")

    # When everything done, release the video capture object
    cap.release()

    return frame

def get_single_frame_from_clip(file, frame_index, rotate=False, to_rgb=True, verbose=True):
    """
    Extracts a single frame from a video clip.

    Args:
        file (str): Path to the video clip.
        frame_index (int): Index of the frame to extract.
        rotate (bool, optional): Whether to rotate the frame to make the video horizontal. Defaults to False.
        to_rgb (bool, optional): Whether to convert the frame from BGR to RGB. Defaults to True.

    Returns:
        numpy.ndarray: Array of the extracted frame.
    """
    # Open the video file
    cap = cv2.VideoCapture(file)

    # Check if video opened successfully
    if not cap.isOpened():
        raise Exception(f"Error: Could not open video file {file}")
    
    ret = False
    warned = False
    
    while not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read the frame
        ret, frame = cap.read()

        # Check if we got the frame successfully
        if ret:
            if to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        else:
            if not warned and verbose:
                print("This function (get_single_frame_from_clip) is faster than get_single_frame_from_clip_stable, but it may cause the last few frames to disappear due to unknown issue.")
                print("It does not align with zf.preprocess.video_total_frames() and zf.preprocess.get_frames_from_clip(), especially for later frames. ")
                print("If you need consistency, use get_single_frame_from_clip_stable instead.")
                print(f"Could not read frame {frame_index} from video file {file}, attempting to read previous frames.")
            warned = True
            frame_index -= 1

    if warned and verbose:
        print(f"Successfully read frame {frame_index} from video file {file}")

    # When everything done, release the video capture object
    cap.release()

    return frame

def video_total_frames(file):
    """
    Get the total number of frames in a video file.

    Args:
        file (str): Path to the video file.

    Returns:
        int: Total number of frames in the video file.
    """

    cap = cv2.VideoCapture(file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return total_frames