import numpy as np
from .tqdm import TQDM as tqdm

def find_identity_switches(gt_traj, traj_pool, resolution=30, min_gap=120):

    current_id = -1
    switches = []

    for i in range(0, len(gt_traj), resolution):
        new_id = find_current_id(gt_traj[i], traj_pool, i)
        if new_id != current_id:
            if len(switches) and switches[-1][0] + min_gap > i:
                switches[-1] = (i, new_id)
                if switches[-1][1] == switches[-2][1]:
                    switches.pop()
            else:
                switches.append((i, new_id))
            current_id = new_id
    
    return switches

def find_current_id(gt_position, traj_pool, frame):
    distances = []
    for traj in traj_pool:
        distances.append(np.linalg.norm(traj[frame] - gt_position))
    return np.argmin(distances)

def get_geometric_data(traj_pool):

    x = []
    y = []

    for traj in traj_pool:
        diffs = np.diff([pos[4:6] for pos in traj], axis=0)
        accel = np.diff(diffs, axis=0)
        count = 0
        for i in range(3, len(traj)):
            dot_product = np.dot(accel[i-2], accel[i-3])
            if dot_product > -4 and traj.confidences[i] > 0.75:
                count += 1
            else:
                count = 0
            if count >= 32:
                x.append(traj[i-31:i])
                y.append(traj[i])

    x = np.array(x)
    y = np.array(y)

    return x, y
    
def find_fluctuating_points(traj_pool, threshold):

    results = {i: {'head': [],
                'midsec': [],
                'tail': []}
                for i in range(7)}
    pbar = tqdm(total=len(traj_pool) * (len(traj_pool[0]) - 2))
    for i, traj in enumerate(traj_pool):
        pbar.set_description(f'Processing Traj {i}')
        diffs = np.diff(traj, axis=0)
        accelerations = np.diff(diffs, axis=0)
        for frame in range(2, len(traj)):
            head_acc = np.linalg.norm(accelerations[frame - 2][0:2])
            midsec_acc = np.linalg.norm(accelerations[frame - 2][2:4])
            tail_acc = np.linalg.norm(accelerations[frame - 2][4:6])
            median_acc = np.median([head_acc, midsec_acc, tail_acc])
            if abs(head_acc - median_acc) > threshold:
                results[i]['head'].append((frame, abs(head_acc - median_acc)))
            if abs(midsec_acc - median_acc) > threshold:
                results[i]['midsec'].append((frame, abs(midsec_acc - median_acc)))
            if abs(tail_acc - median_acc) > threshold:
                results[i]['tail'].append((frame, abs(tail_acc - median_acc)))
            pbar.update(1)

    pbar.close()
    return results

def switch_id(traj_pool, switch_list, simple=False):
    """
    Switch the trajectory of the fish with the given id in the switch_list.
    The switch_list is a list of lists, where each list contains:
    1. The id of the first fish to switch
    2. The id of the second fish to switch
    3. The frame at which the switch should occur
    Example:
    switch_list = [[0, 1, 100], [2, 3, 200]]
    This will switch the trajectories of fish 0 and 1 at frame 100,
    and the trajectories of fish 2 and 3 at frame 200.
    """

    for index in range(len(switch_list)):
        
        a = switch_list[index][0]
        b = switch_list[index][1]

        for update_index in range(index+1, len(switch_list)):
            for i in range(2):
                if switch_list[update_index][i] == a:
                    switch_list[update_index][i] = b
                elif switch_list[update_index][i] == b:
                    switch_list[update_index][i] = a
        
        switch_frame = switch_list[index][2]

        a_traj_cut = traj_pool[a].trajectory[switch_frame:]
        a_conf_cut = traj_pool[a].confidences[switch_frame:]
        if not simple:
            a_type_cut = traj_pool[a].match_types[switch_frame:]
            a_src_cut = traj_pool[a].sources[switch_frame:]

        b_traj_cut = traj_pool[b].trajectory[switch_frame:]
        b_conf_cut = traj_pool[b].confidences[switch_frame:]
        if not simple:
            b_type_cut = traj_pool[b].match_types[switch_frame:]
            b_src_cut = traj_pool[b].sources[switch_frame:]

        traj_pool[a].trajectory[switch_frame:] = b_traj_cut
        traj_pool[a].confidences[switch_frame:] = b_conf_cut
        if not simple:
            traj_pool[a].match_types[switch_frame:] = b_type_cut
            traj_pool[a].sources[switch_frame:] = b_src_cut

        traj_pool[b].trajectory[switch_frame:] = a_traj_cut
        traj_pool[b].confidences[switch_frame:] = a_conf_cut
        if not simple:
            traj_pool[b].match_types[switch_frame:] = a_type_cut
            traj_pool[b].sources[switch_frame:] = a_src_cut

    return traj_pool