import numpy as np
import shapely.geometry as geom
from shapely.geometry import LineString, MultiLineString
from .evaluate_annotation_v1 import unpaired_classification
from icecream import ic
from ..util.tqdm import TQDM as tqdm
import csv
HEAD_LINE = ('文件名', '鱼ID (0~6)', '问题代码', '问题描述', '额外信息', '状态', '批注')
ERROR_REPRESENTATION = {'fn': '这两条鱼线相交，但是没有组成遮挡/被遮挡对，可能存在漏标',
                        'fp': ('标线未与其他标线交叉，但标注了遮挡，可能为错标或难以判断的情况', '与其他标线的最近距离：'),
                        'unpaired': '该鱼具有遮挡/被遮挡的标记，但图中没有其他对应被遮挡/遮挡的标记，图中一定有漏标或错标',
                        'missing': '鱼总数不足7条，存在漏标',
                        'length': ('鱼的长度大于或小于99.95%的其他鱼，可能存在错标', '该鱼的长度为：'),
                        'stats': '误报/已修正'}

def find_arc_center(A, B, C, return_radius=False):
    # Convert points to numpy arrays for easier calculations
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    
    # Calculate the midpoint of BC (M)
    M = (B + C) / 2

    # Calculate the slope of BC
    if C[0] == B[0]: # To handle the case of vertical line
        slope_BC = None
    else:
        slope_BC = (C[1] - B[1]) / (C[0] - B[0])
    
    # Calculate the slope of AB
    if B[0] == A[0]: # To handle the case of vertical line
        slope_AB = None
    else:
        slope_AB = (B[1] - A[1]) / (B[0] - A[0])
    
    # Calculate the slope of the radius at B
    if slope_AB is None:
        slope_radius_B = 0
    elif slope_AB == 0:
        slope_radius_B = None
    else:
        slope_radius_B = -1 / slope_AB
    
    # Calculate the slope of the line perpendicular to BC
    if slope_BC is None:
        slope_perpendicular_BC = 0
    elif slope_BC == 0:
        slope_perpendicular_BC = None
    else:
        slope_perpendicular_BC = -1 / slope_BC
    
    # Calculate the equations of the two lines in the format y = mx + c
    if slope_radius_B is not None:
        c_radius_B = B[1] - slope_radius_B * B[0]
    if slope_perpendicular_BC is not None:
        c_perpendicular_BC = M[1] - slope_perpendicular_BC * M[0]
    
    # Find the intersection of the two lines
    if slope_perpendicular_BC is None:
        # Vertical line through M
        x_center = M[0]
        y_center = slope_radius_B * x_center + c_radius_B
    elif slope_radius_B is None:
        # Vertical line through B
        x_center = B[0]
        y_center = slope_perpendicular_BC * x_center + c_perpendicular_BC
    else:
        # Solve for x and y
        x_center = (c_radius_B - c_perpendicular_BC) / (slope_perpendicular_BC - slope_radius_B)
        y_center = slope_perpendicular_BC * x_center + c_perpendicular_BC
    
    center = (x_center, y_center)
    if not return_radius:
        return center
    
    radius = np.linalg.norm(center - B)  # The radius is the distance from the center to B
    return center, radius

def calculate_absolute_radian(center, point):
    """Calculate the angle in radians between the x-axis and the line connecting center and point."""
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return np.arctan2(dy, dx) # Returns the angle in radians (-pi, pi]

def create_arc(center, radius, A, B, clockwise=False, numsegments=16):
    """
    Create an arc from point A to point B along a circle defined by center and radius.
    
    Parameters:
    center (tuple): The (x, y) coordinates of the center of the circle.
    radius (float): The radius of the circle.
    A (tuple): The (x, y) coordinates of point A on the circle.
    B (tuple): The (x, y) coordinates of point B on the circle.
    clockwise (bool): Whether the arc should be drawn clockwise. Defaults to False.
    numsegments (int): The number of segments for the arc approximation. Defaults to 8.
    
    Returns:
    geom.LineString: A Shapely LineString object representing the arc.
    """
    # Calculate the angles for points A and B
    start_angle = calculate_absolute_radian(center, A)
    end_angle = calculate_absolute_radian(center, B)
    
    # Normalize angles to be within [0, 2*pi)
    start_angle = start_angle % (2 * np.pi)
    end_angle = end_angle % (2 * np.pi)
    
    # Determine if we need to add or subtract 2*pi to make the arc go in the correct direction
    if clockwise and start_angle < end_angle:
        start_angle += 2 * np.pi
    elif not clockwise and start_angle > end_angle:
        end_angle += 2 * np.pi

    # Create the theta (angle) values for the arc
    theta = np.linspace(start_angle, end_angle, numsegments)

    # Calculate the x and y coordinates of the arc
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    # Create the arc as a Shapely LineString
    arc = geom.LineString(np.column_stack((x, y)))

    # Return the angle and the arc
    angle = end_angle - start_angle
    return angle, arc

def is_clockwise(A, B, C):
    return (B[0] - A[0]) * (C[1] - A[1]) < (B[1] - A[1]) * (C[0] - A[0])

def unify_midsection(A, B, distance=32):
    """
    Unify the distance between two points to a specified value.
    Args:
        A (tuple): The (x, y) coordinates of point A.
        B (tuple): The (x, y) coordinates of point B.
        distance (float): The distance between the two points. Defaults to 32.
    Returns:
        tuple: The new (x, y) coordinates of point B.
    """
    A = np.array(A)
    B = np.array(B)
    AB = B - A
    AB = AB / np.linalg.norm(AB) * distance
    return A + AB

def create_fish(A, B, C, numsegments=16, unify_distance=32, return_angle=False):
    B = unify_midsection(A, B, distance=unify_distance)
    center, radius = find_arc_center(A, B, C, return_radius=True)
    clockwise = is_clockwise(A, B, C)
    angle, arc_BC = create_arc(center, radius, B, C, clockwise, numsegments)
    # Combine line AB and the arc BC with line AB first
    fish = MultiLineString([LineString([A, B]), arc_BC])
    if return_angle:
        # Angle is in the range [0, 2*pi)
        return angle, fish
    else:
        return fish

def evaluate_annotations(data_dict, numsegments=16, min_distance=16, max_distance=2, upper_percentile=99.9, lower_percentile=0.1, unify_distance=32):
    """
    Evaluate the annotations in the data_dict.
    Args:
        data_dict (dict): A dictionary containing the annotations for each video.
        numsegments (int): The number of segments for the arc approximation. Defaults to 16.
        min_distance (int): The minimum distance between two fish for them to be considered separate. Defaults to 8.
        max_distance (int): The maximum distance between two fish for them to be considered intersecting. Defaults to 2.
    Returns:
        dict: A dictionary containing the false negatives and false positives.
    """
    errors = {'fn': [], 'fp': [], 'missing': [], 'tp': []}
    lengths = {}
    for key, values in tqdm(list(data_dict.items())):
        distances = {i: (np.inf, None) for i in range(0, 28, 4)}
        lengths[key] = []
        for i in range(0, 28, 4):
            if sum(values[i]) == 0:
                errors['missing'].append((key, i//4))
                continue
            fish_i = create_fish(values[i], values[i+1], values[i+2],
                                 numsegments=numsegments, unify_distance=unify_distance)
            lengths[key].append(fish_i.length)
            for j in range(i + 4, 28, 4):
                if sum(values[j]) == 0:
                    continue
                fish_j = create_fish(values[j], values[j+1], values[j+2],
                                     numsegments=numsegments, unify_distance=unify_distance)
                distance = fish_i.distance(fish_j)
                if distance < distances[i][0]:
                    distances[i] = (distance, j)
                if distance < distances[j][0]:
                    distances[j] = (distance, i)
                if distance <= max_distance:
                    if not (values[i+3][0] == values[j+3][1] == 1 or
                            values[i+3][1] == values[j+3][0] == 1 or
                            ((values[i+3][0] == -1 or values[i+3][1] == -1) and
                                (values[j+3][0] == -1 or values[j+3][1] == -1))):
                        errors['fn'].append((key, i//4, j//4))
        for i, (distance, j) in distances.items():
            if distance > min_distance:
                if values[i+3][0] == 1 or values[i+3][1] == 1:
                    errors['fp'].append((key, i//4, distance))
    
    all_lengths = np.array([length for lengths in lengths.values() for length in lengths])
    upper_bound = np.percentile(all_lengths, upper_percentile)
    lower_bound = np.percentile(all_lengths, lower_percentile)
    errors['length'] = []
    for key, values in lengths.items():
        for i, length in enumerate(values):
            if length < lower_bound or length > upper_bound:
                errors['length'].append((key, i, length))

    errors['unpaired'] = unpaired_classification(data_dict)
    
    return errors


def write_errors_to_csv(errors, filename):
    """
    Write the errors to a CSV file.
    Args:
        errors (dict): A dictionary containing the errors.
        filename (str): The name of the CSV file to write the errors to.
    """
    contents = []
    
    for error in errors['missing']:
        contents.append([error[0], None, 'missing', ERROR_REPRESENTATION['missing'], None, ERROR_REPRESENTATION['stats']])
    for error in errors['unpaired']:
        contents.append([error[0], ' & '.join(map(str, error[1:])), 'unpaired', ERROR_REPRESENTATION['unpaired'], None, ERROR_REPRESENTATION['stats']])
    for error in errors['fn']:
        contents.append([error[0], ' & '.join(map(str, error[1:])), 'fn', ERROR_REPRESENTATION['fn'], None, ERROR_REPRESENTATION['stats']])
    for error in errors['fp']:
        contents.append([error[0], error[1], 'fp', ERROR_REPRESENTATION['fp'][0], f"{ERROR_REPRESENTATION['fp'][1]}{error[2]:.2f}", ERROR_REPRESENTATION['stats']])
    for error in errors['length']:
        contents.append([error[0], error[1], 'length', ERROR_REPRESENTATION['length'][0], f"{ERROR_REPRESENTATION['length'][1]}{error[2]:.2f}", ERROR_REPRESENTATION['stats']])

    contents.sort(key=lambda x: x[0])
    with open(filename, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(HEAD_LINE)
        writer.writerows(contents)