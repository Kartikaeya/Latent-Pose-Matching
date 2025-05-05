# Latent Pose Matching for Head Pose Estimation

This project implements a head pose estimation system using latent matching with DinoV2 features. The system estimates rough head rotation and keypoint positions by leveraging the powerful feature representations from the DinoV2 vision transformer model.

## Overview

The system works by:
1. Extracting DinoV2 features from input images
2. Using a latent matching algorithm to compare these features with a reference dataset
3. Estimating head pose (rotation) and keypoint positions based on the matched features

## Setup

1. Create the conda environment:
```bash
conda env create -f environment.yml
```

2. Activate the environment:
```bash
conda activate latent_pose_env
```

## Project Structure

- `dinov2_visualization.py`: Initial implementation for feature extraction and visualization
- (More components to be added for the complete pose estimation pipeline)

## Requirements

- Python 3.9
- PyTorch
- Transformers (for DinoV2)
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- Other dependencies as specified in `environment.yml`

## Future Development

- [ ] Implement reference dataset creation
- [ ] Develop latent matching algorithm
- [ ] Add head rotation estimation
- [ ] Implement keypoint detection
- [ ] Add evaluation metrics
- [ ] Create visualization tools for pose estimation results 