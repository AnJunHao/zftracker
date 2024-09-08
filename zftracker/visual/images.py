import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np
import cv2
from torch import zeros_like
import torch.nn.functional as F
import torch
from icecream import ic

from ..criterion.keypoint import OffsetKeypointEvaluator

def overlay_heatmaps_on_image(image, heatmaps, alpha=0.5, display=True, filename=None):
    """
    Overlays a set of heatmaps on an image and displays the result.
    Args:
        image (torch.Tensor): A 3D tensor representing the image with shape (channels, height, width).
        heatmaps (torch.Tensor): A 3D tensor representing the heatmaps with shape (channels, height, width).
        alpha (float, optional): The alpha value to use for blending the image and the heatmaps. Defaults to 0.5.
        display (bool, optional): Whether to display the overlay. Defaults to True.
        filename (str, optional): The name of the file to save the overlay to. Defaults to None.
    Returns:
        numpy.ndarray: A 3D array representing the blended image.
    """
    # Get the size of the image and the heatmaps
    _, height, width = image.shape
    if len(heatmaps.shape) == 2:
        heatmaps = heatmaps.unsqueeze(0)
    channels, smaller_height, smaller_width = heatmaps.shape

    ratio_height = height / smaller_height
    ratio_width = width / smaller_width

    # Resize the heatmaps to match the size of the image
    resized_heatmaps = F.interpolate(heatmaps.unsqueeze(0), scale_factor=(
        ratio_height, ratio_width), mode='bilinear').squeeze(0).cpu()

    # Normalize the heatmaps to [0, 1]
    resized_heatmaps = (resized_heatmaps - resized_heatmaps.min()) / \
        (resized_heatmaps.max() - resized_heatmaps.min())

    # Create an empty image with the same size as the original image but with as many channels as heatmaps
    heatmap_image = zeros_like(image)

    # For each heatmap, add it to the corresponding channel in the new image
    for c in range(channels):
        heatmap_image[c % 3] += resized_heatmaps[c]

    # Normalize the new image to [0, 1]
    heatmap_image = (heatmap_image - heatmap_image.min()) / \
        (heatmap_image.max() - heatmap_image.min())

    # Blend the original image and the heatmap overlay using the alpha parameter
    blended_image = alpha * heatmap_image + (1 - alpha) * image

    # To visualize the overlay, we need to convert the tensors to numpy arrays and transpose the dimensions
    blended_image_np = blended_image.permute(1, 2, 0).numpy()

    # Plot the blended image
    if display:
        plt.imshow(blended_image_np)
        plt.title("Heatmap Overlay")
        plt.show()

    # Save the blended image as a file
    if filename is not None:
        # OpenCV uses BGR, so we need to convert the image to BGR
        blended_image_bgr = cv2.cvtColor(blended_image_np, cv2.COLOR_RGB2BGR)

        # multiply the image by 255 to recover the original pixel values
        blended_image_bgr = (blended_image_bgr * 255).astype(np.uint8)

        # Save the image
        cv2.imwrite(filename, blended_image_bgr)

    return blended_image_np





def show_heatmaps(heatmap_tensor, channel_per_row, color_bar=True, filename=None):
    """
    Visualizes heatmaps in a grid.

    Args:
        heatmap_tensor (numpy.ndarray): A 3D numpy array representing the heatmaps.
            The shape should be (channel, height, width).
        channel_per_row (int): The number of channels to display per row in the grid.
    """
    assert len(
        heatmap_tensor.shape) == 3, "heatmap_tensor should be a 3D array (channel, height, width)"

    num_channels, height, width = heatmap_tensor.shape

    # Calculate the number of rows and columns needed for the subplot grid
    num_rows = num_channels // channel_per_row
    num_rows += 1 if num_channels % channel_per_row > 0 else 0
    num_cols = min(num_channels, channel_per_row)

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols*5, num_rows*5))

    for i in range(num_channels):
        ax = axes[i // channel_per_row, i % channel_per_row]

        # Plot the heatmap using the current channel
        cax = ax.matshow(heatmap_tensor[i, :, :])
        if color_bar:
            plt.colorbar(cax, ax=ax)
        ax.axis('off')

    # Hide any unused subplots
    for j in range(num_channels, num_rows * num_cols):
        axes[j // channel_per_row, j % channel_per_row].axis('off')

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, format='svg')  # Save the plot as a SVG file

    plt.show()


def visualize_offsets_from_heatmap(heatmap, offset_map, offset_norm_ratio=1, filename=None):
    # Create an instance of OffsetKeypointEvaluator
    helper = OffsetKeypointEvaluator()

    # Find original coordinates
    if len(heatmap.shape) == 2:
        original_coords = helper.find_local_peaks(heatmap.unsqueeze(0))
        heatmap = heatmap.cpu().numpy()
    elif len(heatmap.shape) == 3:
        if heatmap.shape[0] == 1:
            original_coords = helper.find_local_peaks(heatmap)
            heatmap = heatmap[0].cpu().numpy()
        elif heatmap.shape[0] == 2:
            original_coords = helper.find_local_peaks(heatmap[0].unsqueeze(0))
            heatmap = torch.max(heatmap[0], heatmap[1] / 4).cpu().numpy()
        else:
            raise ValueError("heatmap must have 1 or 2 channels")
    else:
        raise ValueError("heatmap must be 2D or 3D")

    if len(offset_map.shape) == 3:
        offset_map = offset_map.unsqueeze(0)

    # Get offset coordinates
    offset_coords = helper.apply_local_offsets(
        original_coords, offset_map * offset_norm_ratio, average_2x2=False)

    # Draw the heatmap
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')

    # Draw arrows from original to offset coordinates
    for (x_orig, y_orig), (x_offset, y_offset) in zip(original_coords[0], offset_coords[0]):
        dx = x_offset - x_orig
        dy = y_offset - y_orig
        plt.arrow(x_orig, y_orig, dx, dy, fc='#00ffff', ec='#00ffff')

    if filename is not None:
        plt.savefig(filename, format='svg')

    plt.show()

def show3d(heatmap):
    """
    Displays a 3D plot of a heatmap.

    Args:
        heatmap (numpy.ndarray): A 2D or 3D array representing the heatmap.

    Returns:
        None. Displays a Matplotlib 3D plot of the heatmap.
    """
    # Reshape heatmap into 2D
    if len(heatmap.shape) == 3:
        new_shape = heatmap.shape[:-1]
        heatmap = heatmap.reshape(new_shape)

    # Create X and Y coordinates for each pixel
    plt_x, plt_y = np.meshgrid(np.arange(heatmap.shape[1]),
                               np.arange(heatmap.shape[0]))
    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(plt_x, plt_y, heatmap, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Pixel Value')
    plt.show()


def interact3d(heatmap):
    """
    Displays an interactive 3D plot of a heatmap using Plotly.

    Args:
        heatmap (numpy.ndarray): A 2D or 3D array representing the heatmap.

    Returns:
        None. Displays an interactive 3D plot of the heatmap using Plotly.
    """
    # Reshape heatmap into 2D
    if len(heatmap.shape) == 3:
        new_shape = heatmap.shape[:-1]
        heatmap = heatmap.reshape(new_shape)

    # Create 3D surface plot
    fig = go.Figure(data=[go.Surface(z=heatmap)])

    # Set layout of the figure
    fig.update_layout(
        title='Heatmap 3D Plot',
        scene=dict(
            xaxis_title='Y_target',
            yaxis_title='Y_pred',
            zaxis_title='Factor'
        )
    )

    # Display the figure
    fig.show()


def draw_lines_on_image(image, lines, display=True):
    """
    Draws lines on the given image.

    Parameters:
    - image: A 3D NumPy array representing an image.
    - lines: A 2D NumPy array with shape (n, 4), where each row represents
             the coordinates (x1, y1, x2, y2) of a line to be drawn.
    """
    # Copy the image so that the original image is not altered
    img_with_lines = image.copy()

    # Iterate over each line
    for i, line in enumerate(lines):
        # Unpack line coordinates and round them to integers
        if len(line) == 4:
            x1, y1, x2, y2 = np.round(line * 4).astype(int)
        elif len(line) == 6:
            x1, y1, x2, y2, x3, y3 = np.round(line * 4).astype(int)
        # Draw the line on the image
        cv2.line(img_with_lines, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        if len(line) == 6:
            cv2.line(img_with_lines, (x2, y2), (x3, y3), color=(0, 255, 0), thickness=2)
        # Also draw the line number
        cv2.putText(img_with_lines, str(i), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

    if display:
        # Display the image with lines
        plt.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
        plt.axis('off') # Hide the axis
        plt.show()

    return img_with_lines



def draw_keypoints_on_image(kps_array, img_array, point_size=3, colormap='rainbow', display=True):
    """
    Draws keypoints on image.
    Args:
        kps_array (np.ndarray or list): A 2D array with shape (num_keypoints, 2), or a list containing 2D arrays.
        img_array (np.ndarray or torch.Tensor): A 3D array of image with shape (height, width, 3) or a 3D tensor of image with shape (channels, height, width).
        point_size (int, optional): The size of the keypoints to draw. Defaults to 3.
        colormap (str, optional): The name of the colormap to use. Defaults to 'rainbow'.
        display (bool, optional): Whether to display the image. Defaults to True.
    Returns:
        np.ndarray: A 3D array of image with keypoints drawn on them, with shape (height, width, 3).
    """

    # Convert tensors into numpy arrays and rearrange its dimensions
    if isinstance(img_array, torch.Tensor):
        img_array = img_array.permute(1, 2, 0).cpu().numpy()

    # Convert images to uint8 if they are not
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)

    if len(kps_array[0].shape) == 1:
        kps_array = [kps_array]

    # Round values to integers
    kps_array_int = [np.round(i).astype(int) for i in kps_array]

    img_copy = img_array.copy()

    # Draw keypoints on the image, for each group of keypoints, use different colors
    for i, kps in enumerate(kps_array_int):
        color = plt.cm.get_cmap(colormap)(i / len(kps_array_int))[:3]
        color = [int(i * 255) for i in color[::-1]]
        for kp in kps:
            cv2.circle(img_copy, tuple(kp), point_size, color, -1)

    if display:
        plt.imshow(img_copy)
        plt.show()

    return img_copy