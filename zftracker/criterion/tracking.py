import numpy as np
from ..util.tqdm import TQDM as tqdm
import matplotlib.pyplot as plt

def mean_time_between_id_switch(coords, switch_list, resolution=1, bins = np.arange(0, 1001, 50), frame_rate=120, rescale=4):
    diffs = np.diff(coords, axis=0)
    velocities = [np.median([np.linalg.norm(d[i*2:i*2+2])
                        for i in range(len(d)//2)])
                for d in tqdm(diffs)]

    smoothed_velocities = [np.median(velocities[i-resolution*frame_rate//2:i+resolution*frame_rate//2])
                            for i in tqdm(range(len(velocities)))]
    
    further_smoothed_velocities = np.array([np.mean(smoothed_velocities[i*resolution*frame_rate:(i+1)*resolution*frame_rate])
                                    for i in tqdm(range(len(smoothed_velocities)//resolution*frame_rate))])*frame_rate*rescale
    
    plt.scatter([s[2]//resolution for s in switch_list], [further_smoothed_velocities[s[2]//resolution] for s in switch_list], c='r', label='Identity Switch')
    # Put the curve under the scatter plot
    plt.plot(further_smoothed_velocities, alpha=0.5, label='Velocity')
    plt.title('Velocity over time & Identity switches')
    plt.xlabel('Time (unit={resolution}seconds)')
    plt.ylabel('Average Velocity (pixels per second)')
    plt.legend()

    hist, bin_edges = np.histogram(further_smoothed_velocities, bins=bins)
    switch_hist, _ = np.histogram([further_smoothed_velocities[s[2]//resolution] for s in switch_list], bins=bins)

    return hist, switch_hist, bin_edges