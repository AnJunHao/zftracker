# zftracker

![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)
![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
![Development Status](https://img.shields.io/badge/status-PreAlpha-orange.svg)

`zftracker` is a cutting-edge Python library for tracking groups of zebrafish, designed to enhance research efficiency by providing a robust tool for behavioral analysis.

[Demo](https://youtu.be/9jiIeoRqJ-o)

## Features

- **Multi-Fish Tracking:** Accurately track multiple zebrafish simultaneously.
- **High Performance:** Achieves 10–20x performance improvements over previous systems, capable of tracking hours of zebrafish group movement in real time.
- **User-Friendly API:** Intuitive and easy to integrate into existing workflows.

## Benchmarks

Our system outperforms previous state-of-the-art models:

- Performance Summary

|                      Metric                      |    Our System     | Barreiros et al. (2021).[^1] |
| :----------------------------------------------: | :---------------: | :--------------------------: |
|                   **F1 Score**                   |    **0.9997**     |           ~0.9942            |
|                  **Precision**                   |        1.0        |             1.0              |
|                    **Recall**                    |    **0.9994**     |           ~0.9884            |
|              **Overlap Event CIR**               | **0.9991-0.9993** |        0.9375-0.9629         |
| **Average Interval Between Confusions (120fps)** |  **115 seconds**  |             N/A              |
| **Average Interval Between Confusions (30fps)**  |  **97 seconds**   |      8.40-9.18 seconds       |

[^1]: Barreiros, Marta de Oliveira, Diego de Oliveira Dantas, Luís Claudio de Oliveira Silva, et al. Zebrafish tracking using YOLOv2 and Kalman filter. *Scientific Reports*, 2021, 11(1): 3219.

## Installation

Install `zftracker` via pip:

```bash
pip install zftracker
```

## Quick Start Guide

```python
import zftracker as zf

# Specify the video file path
file = 'path/to/recorded_zebrafish.mp4'

# Initialize the tracking pipeline
pipeline = zf.inference.pipeline.PipelineV3(file, num_fish=7)

# Run the tracking process
traj_pool = pipeline.run()

# Save the tracking results
traj_pool.save_coords_numpy('path/to/trajectories.npy')
traj_pool.save_confs_numpy('path/to/confidences.npy')
```

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, open an issue on GitHub or contact:

- **Author:** Junhao An
- **Email:** anjunhao_23@163.com

## AI-Generated Code

Portions of this codebase were generated with the assistance of AI. While tested, further review is recommended.

## Development Status

The project is still under development. Data and models are currently not available to the public. We plan to release model files in the future.

## Update Log

- `0.0.7`: Initial release.

*Maintained by Junhao An.*