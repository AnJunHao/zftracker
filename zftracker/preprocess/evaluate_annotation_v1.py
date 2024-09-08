import numpy as np
from icecream import ic
import csv

HEAD_LINE = ('文件名', '鱼ID (0~6)', '问题代码', '问题描述', '备注')
ERROR_REPRESENTATION = {'12_12_fn': ('头部标线与头部标线交叉，但未标注遮挡，大概率为漏标', '与这条鱼交叉：'),
                        '12_23_fn': ('头部标线与尾部标线交叉，但未标注遮挡，可能为漏标', '与这条鱼交叉：'),
                        '23_23_fn': ('尾部标线与尾部标线交叉，但未标注遮挡，可能为漏标', '与这条鱼交叉：'),
                        'fp': ('标线未与其他标线交叉，但标注了遮挡，可能为错标', '与其他标线的最近距离：'),
                        'unpaired': ('该鱼具有遮挡/被遮挡的标记，但图中没有其他对应被遮挡/遮挡的标记，图中一定有漏标或错标', '无')}

def unpaired_classification(dict_data):
    errors = []
    for key, values in dict_data.items():
        class_a = [values[i][0] for i in range(3, 28, 4) if values[i][0] >= 0]
        class_b = [values[i][1] for i in range(3, 28, 4) if values[i][1] >= 0]
        if sum(class_a) != sum(class_b):
            if sum(class_a) == 0 or sum(class_b) == 0:
                if sum(class_a) > 0:
                    # Find all indices of class_a with value greater than 0
                    indices = [i for i, v in enumerate(class_a) if v > 0]
                    errors.append((key, *indices))
                else:
                    indices = [i for i, v in enumerate(class_b) if v > 0]
                    errors.append((key, *indices))
    return errors

def ccw(A, B, C):
    """Check if points A, B, C are listed in counter-clockwise order"""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    """Check if line segments AB and CD intersect"""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def fish_intersect(a1, a2, a3, b1, b2, b3):
    if intersect(a1, a2, b1, b2):
        return '12_12'
    if intersect(a1, a2, b2, b3) or intersect(a2, a3, b1, b2):
        return '12_23'
    elif intersect(a2, a3, b2, b3):
        return '23_23'
    else:
        return False

# Python3 implementation of the approach 
from math import sqrt
 

def distance_point_to_segment(A, B, E):
    # Function to return the minimum distance 
    # between a line segment AB and a point E
    # Author: https://www.geeksforgeeks.org/minimum-distance-from-a-point-to-the-line-segment-using-vectors/
 
    # vector AB 
    AB = [None, None]
    AB[0] = B[0] - A[0]
    AB[1] = B[1] - A[1]
 
    # vector BP 
    BE = [None, None]
    BE[0] = E[0] - B[0]
    BE[1] = E[1] - B[1]
 
    # vector AP 
    AE = [None, None]
    AE[0] = E[0] - A[0]
    AE[1] = E[1] - A[1]
 
    # Variables to store dot product 
 
    # Calculating the dot product 
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1]
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1]
 
    # Minimum distance from 
    # point E to the line segment 
    reqAns = 0
 
    # Case 1 
    if (AB_BE > 0) :
 
        # Finding the magnitude 
        y = E[1] - B[1]
        x = E[0] - B[0]
        reqAns = sqrt(x * x + y * y)
 
    # Case 2 
    elif (AB_AE < 0) :
        y = E[1] - A[1]
        x = E[0] - A[0]
        reqAns = sqrt(x * x + y * y)
 
    # Case 3 
    else:
 
        # Finding the perpendicular distance 
        x1 = AB[0]
        y1 = AB[1]
        x2 = AE[0]
        y2 = AE[1]
        mod = sqrt(x1 * x1 + y1 * y1)
        reqAns = abs(x1 * y2 - y1 * x2) / mod
     
    return reqAns

def distance_between_line_segments(A, B, C, D):
    # Function to return the minimum distance 
    # between two line segments AB and CD
    distances = [distance_point_to_segment(A, B, C),
                 distance_point_to_segment(A, B, D),
                 distance_point_to_segment(C, D, A),
                 distance_point_to_segment(C, D, B)]
    return min(distances)

def distance_between_fish(A, B, C, D, E, F):
    # Function to return the minimum distance between two fish
    # Points ABC are the points of the first fish
    # Points DEF are the points of the second fish
    distances = [distance_between_line_segments(A, B, D, E),
                    distance_between_line_segments(A, B, E, F),
                    distance_between_line_segments(B, C, D, E),
                    distance_between_line_segments(B, C, E, F)]
    return min(distances)

def incorrectly_classified(dict_data, min_distance=32):
    errors = {'12_12_fn': [],
              '12_23_fn': [],
              '23_23_fn': [],
              'fp': []}
    for key, values in dict_data.items():
        is_intersected = {i: False for i in range(0, 28, 4)}
        for i in range(0, 28, 4):
            for j in range(i + 4, 28, 4):
                if sum(values[j]) > 0: # Check if it is a valid annotation
                    # Note: we pad zeros for annotations with fish number fewer than expected
                    # So we need to check if the sum is greater than 0
                        state = fish_intersect(values[i], values[i + 1], values[i + 2], 
                                               values[j], values[j + 1], values[j + 2])
                        if state:
                            is_intersected[i] = True
                            is_intersected[j] = True
                            if not (values[i+3][0] == values[j+3][1] == 1 or
                                    values[i+3][1] == values[j+3][0] == 1 or
                                    ((values[i+3][0] == -1 or values[i+3][1] == -1) and
                                     (values[j+3][0] == -1 or values[j+3][1] == -1))):
                                errors[state+'_fn'].append((key, i//4, j//4))
        fp_errors = []
        for i in range(0, 28, 4):
            
            if not is_intersected[i] and (values[i+3][0] == 1 or values[i+3][1] == 1):
                all_segments = [[values[j], values[j+1]]
                                for j in range(0, 28)
                                if not i <= j <= i+1 and sum(values[j]) > 0 and j % 4 <= 1]
                distances = ([distance_between_line_segments(segment[0], segment[1],
                                                             values[j], values[j+1])
                              for segment in all_segments
                              for j in range(i, i+2)])
                if all([d > min_distance for d in distances]):
                    fp_errors.append((key, i//4, min(distances)))
        if len(fp_errors) == 1:
            errors['fp'].append(fp_errors[0])
        elif len(fp_errors) > 1:
            # exclude the errors if their distance are close to each other
            pairs = []
            distances = []
            for i in range(len(fp_errors)):
                for j in range(i+1, len(fp_errors)):
                    pairs.append((i, j))
                    distances.append(distance_between_fish(values[fp_errors[i][1]*4], values[fp_errors[i][1]*4+1],
                                                           values[fp_errors[i][1]*4+2], values[fp_errors[j][1]*4],
                                                           values[fp_errors[j][1]*4+1], values[fp_errors[j][1]*4+2]))
            to_be_removed = set()
            for p, d in zip(pairs, distances):
                if d < min_distance * 4:
                    to_be_removed.add(p[0])
                    to_be_removed.add(p[1])
            for i in range(len(fp_errors)):
                if i not in to_be_removed:
                    errors['fp'].append(fp_errors[i])
    return errors

def write_errors_to_csv(dict_data, min_distance, filename):

    apparent_errors = unpaired_classification(dict_data)
    detailed_errors = incorrectly_classified(dict_data, min_distance)

    print(f'Number of apparent missing errors: {len(apparent_errors)}')
    for k, v in detailed_errors.items():
        print(f'Number of {k} errors: {len(v)}')

    all_errors = {}

    for key in ('23_23_fn', '12_23_fn', '12_12_fn'):
        for e in detailed_errors[key]:
            if e[0] not in all_errors:
                all_errors[e[0]] = {}
            
            all_errors[e[0]][e[1]] = [key, e[2]]

            all_errors[e[0]][e[2]] = [key, e[1]]

    for e in apparent_errors:
        for fish_id in e[1:]:
            if e[0] not in all_errors:
                all_errors[e[0]] = {}
            all_errors[e[0]][fish_id] = ['unpaired', '']

    for e in detailed_errors['fp']:
        if e[0] not in all_errors:
            all_errors[e[0]] = {}
        
        all_errors[e[0]][e[1]] = ['fp', e[2]]
        

    contents = []
    for key, values in all_errors.items():
        for k, v in values.items():
            contents.append([
                key, # image name
                k, # fish id
                v[0], # error code
                ERROR_REPRESENTATION[v[0]][0], # error description
                ERROR_REPRESENTATION[v[0]][1]+str(v[1]), # note
            ])
    
    contents.sort(key=lambda x: (x[0], x[1]))
    contents.insert(0, HEAD_LINE)

    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(contents)