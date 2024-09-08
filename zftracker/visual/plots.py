from ..util.str import format_scientific, format_percentage
from collections import Counter
import numpy as np
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb, to_hex
import colorsys
import cv2

from torchviz import make_dot


def visualize_model_structure(model, model_inputs):
    """
    Visualizes the computational graph of a PyTorch model using the torchviz library.
    """

    # Perform a forward pass (to generate the computational graph)
    output = model(*model_inputs)

    # Visualize the graph; note that 'params' is a dictionary of Parameter objects
    # that have been used when defining the model
    params = dict(model.named_parameters())

    # Make the visualization
    dot = make_dot(output, params=params)

    # Or to display it in a Jupyter notebook, you can use
    dot.view()

def display_color_mapping_image(color_mapping, filename=None):
    """
    Creates and displays an image that visualizes the color mapping for each trajectory.

    Args:
        color_mapping (list): List of colors for the trajectories.
        Note that colors are in BGR format.
        filename (str, optional): The name of the file to save the image to. Defaults to None.
    """
    # Calculate the dimensions of the legend
    legend_height = 20 * (len(color_mapping) + 2)  # 20 pixels per entry + title + bottom padding
    legend_width = 200  # Width of legend

    # Create an image for the legend
    legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)

    # Add title to the legend
    cv2.putText(legend, 'Color Mapping Legend:', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Add color mapping to the legend
    for idx, color in enumerate(color_mapping):
        text = f'Traj {idx}'
        cv2.putText(legend, text, (5, 35 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(legend, (150, 20 + idx * 20), (190, 40 + idx * 20), color, -1)

    # Display the legend image
    legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
    plt.axis('off')  # Hide axes
    plt.imshow(legend)

    # Save the legend image if a filename is specified
    if filename is not None:
        plt.savefig(filename, format='svg')
    
    return legend

def bar_plot_from_dict(*args, labels=None, log=False, title="", ylabel="", ylim=None, show_value=True):
    # Create a list of all keys, preserving order of first appearance
    all_keys = []
    for arg in args:
        for key in arg:
            if key not in all_keys:
                all_keys.append(key)

    # Count the frequency of each key
    key_counts = Counter(key for arg in args for key in arg)

    # Create a color map
    colors = plt.cm.get_cmap('tab20')

    # Create a mapping from keys to colors
    key_to_color = {key: colors(i % 20 // 10 + i % 10 * 2) for i, key in enumerate(all_keys)}

    # Create an index for each dictionary
    dict_indices = np.arange(len(args))

    # The width of the bars
    max_bars_in_group = max(len(arg) for arg in args)
    bar_width = 0.8 / max_bars_in_group

    plt.figure(figsize=(10, 6))

    # Determine if we use percentage format or scientific format
    if all(0.001 <= value <= 1 or value == 0 for arg in args for value in arg.values()):
        format_func = format_percentage
    elif any(value > 100000 or 0 < value < 0.001 for arg in args for value in arg.values()):
        format_func = format_scientific
    else:
        format_func = lambda x: str(round(x, 2))

    for i, (arg, dict_index) in enumerate(zip(args, dict_indices)):
        for j, key in enumerate(arg.keys()):
            # Get value for key
            value = arg.get(key, 0)

            # Create a bar for the dictionary
            bar = plt.bar(dict_index + j * bar_width, value, bar_width,
                          label=None if key_counts[key] == 1 else key,
                          color=key_to_color[key], log=log)

            # Add a text annotation for the bar
            if value > 0:
                if show_value:
                    plt.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height(),
                                ''.join([key, ": ", format_func(value)]) if key_counts[key] == 1 else format_func(value),
                                ha='center', va='bottom')
                else:
                    plt.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height(),
                                key if key_counts[key] == 1 else '',
                                ha='center', va='bottom')

    # Add labels, title, and legend
    plt.xlabel('Dictionaries')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(dict_indices + bar_width * max_bars_in_group / 2, labels)

    # Create legend
    legend_elements = [Patch(facecolor=key_to_color[key], edgecolor=key_to_color[key],
                             label=key) for key in key_to_color if key_counts[key] > 1]
    plt.legend(handles=legend_elements)

    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.show()

def darken_color(color, ratio):
    """
    Darkens a color by a given ratio.
    Args:
        color (str): A hexadecimal color string.
        ratio (float): The ratio by which to darken the color.
    Returns:
        str: A hexadecimal color string representing the darkened color.
    """
    # Convert color from hexadecimal to RGB
    color_rgb = to_rgb(color)

    # Convert RGB color to HLS (Hue, Lightness, Saturation)
    color_hls = colorsys.rgb_to_hls(*color_rgb)

    # Decrease the lightness by the given ratio to darken the color
    dark_color = colorsys.hls_to_rgb(
        color_hls[0], color_hls[1] * ratio, color_hls[2])

    # Convert the RGB color back to hexadecimal
    dark_color_hex = to_hex(dark_color)

    return dark_color_hex


def plot_metric(history, metrics, log="auto", show_optimum="auto", filename=None, ylim_lower='auto', ylim_upper='auto', alpha=0.5, format='auto'):
    """
    Plots a training and validation metric from a TensorFlow training history dictionary.

    Args:
        history (dict): A dictionary containing the training and validation metrics over the course of training.
        metric (tup): The names of the metric to plot in a tuple (e.g., 'loss', 'accuracy').
        log (bool, optional): Whether to use a logarithmic scale for the y-axis. Defaults to True.
        show_optimum (str, optional): Whether to show the optimum value for the metric. Can be 'min', 'max' or 'none'. Defaults to 'auto'.
        filename (str, optional): The name of the file to save the plot to. Defaults to None.
        ylim_lower (float, optional): The lower limit for the y-axis.
        ylim_upper (float, optional): The upper limit for the y-axis.
        alpha (float, optional): The alpha value that determines the transparency of the plot. Defaults to 0.5.
        format (str, optional): The format to use for the y-axis labels. Can be 'scientific' or 'percentage' or 'none'. Defaults to 'auto'.
    Returns:
        None. Displays a Matplotlib plot of the specified metric over the course of training.
    """
    # Plot the training and validation metric
    plt.figure(figsize=(8, 6))

    if ylim_lower == "auto":
        if "accuracy" in metrics[0] or "precision" in metrics[0] or "recall" in metrics[0] or "f1" in metrics[0]:
            ylim_lower = 0.0
        else:
            ylim_lower = None
    if ylim_upper == "auto":
        if "accuracy" in metrics[0] or "precision" in metrics[0] or "recall" in metrics[0] or "f1" in metrics[0]:
            ylim_upper = 1.0
        else:
            ylim_upper = None
    if log == "auto":
        if "loss" in metrics[0] or "error" in metrics[0] or 'mae' in metrics[0] or 'mse' in metrics[0]:
            log = True
        else:
            log = False
    if show_optimum == "auto":
        if "loss" in metrics[0] or "error" in metrics[0] or 'mae' in metrics[0] or 'mse' in metrics[0]:
            show_optimum = "min"
        elif "accuracy" in metrics[0] or "precision" in metrics[0] or "recall" in metrics[0] or "f1" in metrics[0]:
            show_optimum = "max"
        else:
            show_optimum = "none"
    if format == 'auto':
        if "loss" in metrics[0] or "error" in metrics[0] or 'mae' in metrics[0] or 'mse' in metrics[0]:
            format = 'scientific'
        elif "accuracy" in metrics[0] or "precision" in metrics[0] or "recall" in metrics[0] or "f1" in metrics[0]:
            format = 'percentage'
        else:
            format = 'none'

    # Log should be set before any plotting
    if log:
        plt.yscale("log")  # Use log scale if necessary

    # Choose the format function based on the format parameter
    if format == 'scientific':
        format_func = format_scientific
    elif format == 'percentage':
        format_func = format_percentage
    elif format == 'none':
        format_func = lambda x: str(x)
    else:
        raise ValueError(
            f"Param format should be 'scientific', 'percentage' or 'none', received {format}"
        )

    # Get the default color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    len_metrics = [len(history[metric]) for metric in metrics]
    if len(set(len_metrics)) == 1:
        duplicate = [False] * len(metrics)
    elif len(set(len_metrics)) == 2:
        min_len = min(len_metrics)
        max_len = max(len_metrics)
        if max_len % min_len == 0:
            factor = max_len // min_len
            duplicate = [factor
                         if len_metrics[i] == min_len
                         else False
                         for i in range(len(metrics))]

    for i, metric in enumerate(metrics):
        metric_history = history[metric]

        color = colors[i % len(colors)]  # Cycle through colors

        # Reduce alpha if plotting multiple metrics
        alpha = 0.5 if len(metrics) > 1 else 1.0

        if duplicate[i]:
            metric_history = list(np.repeat(metric_history, duplicate[i]))
        plt.plot(metric_history, label=metric.capitalize(),
                color=color, alpha=alpha)
        
        if show_optimum != "none":
            if show_optimum == "min":
                optimum = min(metric_history)
            elif show_optimum == "max":
                optimum = max(metric_history)
            else:
                raise ValueError(
                    f"Param show_optimum should be 'min', 'max' or 'none', received {show_optimum}"
                )

            # Darken the RGB color to improve visibility
            dark_color_hex = darken_color(color, 0.75)

            label = f'{show_optimum.capitalize()} {metric.capitalize()}'
            optimum_index = metric_history.index(optimum)
            plt.scatter(
                [optimum_index],
                [optimum],
                marker="o",
                s=20,
                label=label,
                color=dark_color_hex,
            )

            plt.annotate(
                f"{format_func(optimum)} | {optimum_index // duplicate[i] if duplicate[i] else optimum_index}",
                xy=(optimum_index, optimum),
                xytext=(optimum_index, optimum),
                horizontalalignment="center",
                verticalalignment="top",
                color=dark_color_hex,  # Use dark color here to distinguish between nearby points
            )

    plt.xlabel("Epochs")
    plt.ylabel(metric.capitalize())
    plt.legend()

    # Set the y-axis limits if specified
    if ylim_lower is not None:
        plt.ylim(bottom=ylim_lower)
    if ylim_upper is not None:
        plt.ylim(top=ylim_upper)

    if filename is not None:
        plt.savefig(filename, format='svg')  # Save the plot as a SVG file
    plt.show()