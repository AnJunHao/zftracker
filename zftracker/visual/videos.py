from PIL import Image
import io
import numpy as np
import ipywidgets as widgets
from ipywidgets import interact
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from icecream import ic

from ..preprocess.video import video_total_frames, get_frames_from_clip_generator
from ..util.tqdm import TQDM as tqdm

def draw_trajectory_on_video(file_path_list,
                             traj_pool,
                             output_file,
                             thickness=2,
                             multiply_height=4,
                             multiply_width=4,
                             add=145,
                             h264=False,
                             fps_divisor=1,
                             interpolate_gaps=False,
                             rotate=False,
                             release_memory=True,
                             confidence_type='list'): # 'list' or 'single'
    """
    Draws trajectories on a video and saves the result as a new video file.

    Args:
        file_path_list (list): List of file paths for the input videos.
        traj_pool (list): List of trajectories to be drawn on the video.
        output_file (str): Path to save the output video file.
        thickness (int): Thickness of the trajectory line (default: 2).
        multiply_height (int): Multiplication factor for trajectory coordinates (default: 4).
        multiply_width (int): Multiplication factor for trajectory coordinates (default: 4).
        add (int): Addition factor for trajectory coordinates (default: 145).
        h264 (bool): Whether to use H.264 codec for the output video (default: False).
        fps_divisor (int): Divisor for the frame rate of the output video (default: 1).
        interpolate_gaps (bool): Whether to interpolate gaps in the trajectories (default: False).
        rotate (bool): Whether to rotate the video 90 degrees clockwise (default: False).
        release_memory (bool): Whether to release memory after each video is processed (default: True).
        confidence_type (str): Type of confidence to display on the video ('list' or 'single') (default: 'list').
    """
    try:
        cap = cv2.VideoCapture(file_path_list[0])
        fps = round(cap.get(cv2.CAP_PROP_FPS) / fps_divisor)
        if rotate:
            width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    except:
        cap.release()
        raise ValueError("The input video file is not accessible.")

    if h264:
        fourcc = cv2.VideoWriter_fourcc(*'h264')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    try:
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        num_frames = [round(video_total_frames(file_path) / fps_divisor)
                        for file_path in file_path_list]

        total_frames = sum(num_frames)
        frame_index = 0
        video_index = 0

        # Generate color mapping for each trajectory
        num_trajectories = len(traj_pool)
        hue_values = np.linspace(0, 179, num_trajectories + 1, dtype=np.uint8)
        saturation = 255
        value = 255
        color_mapping = [cv2.cvtColor(np.array([[[h, saturation, value]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0].tolist()
                        for h in hue_values[:-1]] # This is in BGR format
        
        # Interpolate the trajectory
        if interpolate_gaps:
            for traj in traj_pool:
                traj.interpolate_gaps()

        pairing_timelines = []
        for traj in traj_pool:
            pairing_timelines.append(traj.get_pairing_timeline())

        loaded_frames = None
        points_per_position = len(traj_pool[0][0]) // 2

        for i in tqdm(range(total_frames)):
            if frame_index >= num_frames[video_index]:
                frame_index = 0
                video_index += 1
                if release_memory:
                    del loaded_frames
                loaded_frames = None

            if loaded_frames is None:
                loaded_frames = get_frames_from_clip_generator(file_path_list[video_index],
                                                     fps_divisor=fps_divisor,
                                                     rotate=rotate,
                                                     to_rgb=False,
                                                     verbose=False)

            frame = loaded_frames.__next__()
            
            text_start_y = 90 # Initialize the starting Y position for drawing text
            text_step_y = 60  # Adjust this value if needed to fit your text size

            for traj_index, traj in enumerate(traj_pool):
                if traj[i] is not None:
                    for j in range(points_per_position):
                        x = round(traj[i][j * 2] * multiply_width + add)
                        y = round(traj[i][j * 2 + 1] * multiply_height)
                        cv2.circle(frame, (x, y), thickness, color_mapping[traj_index], -1)
                # Show the confidence and source of the trajectory, listing under the frame number
                if confidence_type == 'list':
                    cv2.putText(frame, f'T {traj_index}: {sum(traj.confidences[i])/3*100:.1f}%, {traj.match_types[i]}',
                                (10, text_start_y + traj_index*text_step_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, # Font scale
                                color_mapping[traj_index], # Color
                                1, # Thickness
                                cv2.LINE_AA)
                else:
                    cv2.putText(frame, f'T {traj_index}: {traj.confidences[i]*100:.1f}%, {traj.match_types[i]}',
                                (10, text_start_y + traj_index*text_step_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, # Font scale
                                color_mapping[traj_index], # Color
                                1, # Thickness
                                cv2.LINE_AA)
                cv2.putText(frame, f'{traj.sources[i]}, {pairing_timelines[traj_index][i]}',
                            (10, text_start_y + 20 + traj_index*text_step_y),  # Add 20 pixels for the second line
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, # Font scale
                            color_mapping[traj_index], # Color
                            1, # Thickness
                            cv2.LINE_AA)
                
            # Show the frame number at the top left corner
            cv2.putText(frame, f'Frame: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            out.write(frame)
            frame_index += 1

        out.release()
    except:
        out.release()
        raise ValueError("Error occurred during video writing.")

    return color_mapping

class ImagesSequenceVisualize(object):
    """
    A class for visualizing a sequence of video frames.
ss
    Attributes:
        array (numpy.ndarray): A 3D array of video frames, with shape (num_frames, height, width).
        total_frames (int): The total number of frames in the video.

    Methods:
        start(): Displays a widget for interacting with the video frames in the array.
    """

    def __init__(self, array):
        """
        Initializes a new instance of the ImagesSequenceVisualize class.

        Args:
            array (numpy.ndarray): A 3D array of video frames, with shape (num_frames, height, width).
        """
        # Only supports uint8 display
        if array.dtype != 'uint8':
            self.array = []
            for img in array:
                img = (img - img.min()) / (img.max() - img.min())
                img = (img * 255).astype('uint8')
                self.array.append(img)
            self.array = np.array(self.array)
        else:
            self.array = array
        # Reshape black white images
        if self.array.shape[-1] == 1:
            self.array = self.array.reshape(self.array.shape[:-1])
        self.total_frames = self.array.shape[0]

    def start(self):
        """
        Displays a widget for interacting with the video frames in the array.
        """
        # Define a function to display a single frame
        def display_frame(current_frame):
            """
            Displays a single video frame from the array.

            Args:
                current_frame (int): The index of the frame to display.

            Returns:
                A Jupyter widget displaying the selected frame.
            """
            frame = self.array[current_frame]
            image = Image.fromarray(frame)
            buffer = io.BytesIO()
            image.save(buffer, format='png')
            return widgets.Image(value=buffer.getvalue(), format='png')

        # Display the first frame and frame number
        slider = widgets.IntSlider(
            min=0, max=self.total_frames-1, step=1, value=0)
        interact(display_frame, current_frame=slider)
    
    def save_as_video(self, filename="output.mp4", fps=30):
        """
        Save the sequence of images as a video.

        Args:
            filename (str): The name of the file to save the video to.
            fps (int, optional): The frame rate of the video. Defaults to 30.
        """
        # Check if images are grayscale or RGB
        if len(self.array.shape) == 3:
            # Grayscale
            height, width = self.array.shape[1:]
            is_color = False
        elif len(self.array.shape) == 4:
            # RGB
            height, width, _ = self.array.shape[1:]
            is_color = True
        else:
            raise ValueError("Unsupported array shape for video: " + str(self.array.shape))
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is the codec for .mp4 files
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height), is_color)

        # Write each frame to the video file
        for i in range(self.total_frames):
            frame = self.array[i]
            if is_color:
                # Convert from RGB to BGR format for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

        # Release everything when job is finished
        out.release()

class ImagesSequenceVisualizeWithLabel(ImagesSequenceVisualize):

    def __init__(self, array, labels):
        """
        Initializes a new instance of the ImagesSequenceVisualize class.

        Args:
            array (numpy.ndarray): A 3D array of video frames, with shape (num_frames, height, width).
            labels (list): A list of labels (each is either 0 or 1) for each frame in the video.
        """
        super().__init__(array)
        self.labels = labels

    def start(self):
        """
        Displays a widget for interacting with the video frames in the array.
        Also displays the labels for each frame.
        """
        # Define a function to display a single frame
        def display_frame(current_frame):
            """
            Displays a single video frame from the array.

            Args:
                current_frame (int): The index of the frame to display.

            Returns:
                A Jupyter widget displaying the selected frame.
            """
            frame = self.array[current_frame]
            image = Image.fromarray(frame)
            buffer = io.BytesIO()
            image.save(buffer, format='png')
            return widgets.Image(value=buffer.getvalue(), format='png')

        # Define a function to display the label for a frame
        def display_label(current_frame):
            """
            Displays the label for a single video frame.

            Args:
                current_frame (int): The index of the frame to display.

            Returns:
                The label for the selected frame.
            """
            return self.labels[current_frame]

        # Display the first frame and frame number
        slider = widgets.IntSlider(
            min=0, max=self.total_frames-1, step=1, value=0)
        interact(display_frame, current_frame=slider)
        interact(display_label, current_frame=slider)


class VideoDisplay(object):
    """
    A class for visualizing a video.
    """

    def __init__(self, file, target_fps):
        """
        Initializes a new instance of the VideoDisplay class.
        Args:
            file (str): The path to the video file.
            target_fps (int): The frame rate to downsample the video to.
        """

        # Load a video clip
        self.video = VideoFileClip(file)

        # Get the frames of the video clip as a list, downsampled to the target frame rate
        frames = [frame for i, frame in enumerate(
            self.video.iter_frames(fps=target_fps))]

        # Convert the list of frames to a numpy array
        self.array = np.array(frames)
        self.total_frames = self.array.shape[0]

    def start(self):
        """
        Displays a widget for interacting with the video frames in the array.
        """
        # Define a function to display a single frame
        def display_frame(current_frame):
            frame = self.array[current_frame]
            image = Image.fromarray(frame)
            buffer = io.BytesIO()
            image.save(buffer, format='jpeg')
            return widgets.Image(value=buffer.getvalue(), format='jpeg')

        # Display the first frame and frame number
        slider = widgets.IntSlider(
            min=0, max=self.total_frames-1, step=1, value=0)
        interact(display_frame, current_frame=slider)