import argparse
import os
from latent_pca_video import load_model_and_processor, load_pca_models, process_image
from PIL import Image, ImageOps
import torch
import numpy as np
from scipy import ndimage
from BoW_template_retrieval import compute_template_descriptor
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F

def main(args, device):
    img_dir = args.img_dir
    output_dir = args.output_dir

    # Load model and processor
    print("Loading DinoV2 model and processor...")
    model, processor, device = load_model_and_processor()
    
    # Load PCA models
    print(f"Loading PCA models from {args.pca_model}")
    pca_mask, pca_rgb, kmeans = load_pca_models(args.pca_model)

    # load the template descriptors
    with open('./output/templates/data/template_descriptors.json', 'r') as f:
        json_data = json.load(f)
        template_descriptors = json_data['bag_of_words_descriptors']
        N = json_data['num_templates']
        cluster_occurrences = json_data['cluster_occurrences']

    # load template descriptors for the central poses
    template_descriptors = [
        template_descriptors['template_data_h-30_r0.pt'],
        template_descriptors['template_data_h0_r0.pt'],
        template_descriptors['template_data_h30_r0.pt']
    ]

    # Define distinct colors for each cluster
    cluster_colors = np.array([
        [1.0, 0.0, 0.0],    # Red
        [0.0, 1.0, 0.0],    # Green
        [0.0, 0.0, 1.0],    # Blue
        [1.0, 1.0, 0.0],    # Yellow
        [1.0, 0.0, 1.0],    # Magenta
        [0.0, 1.0, 1.0],    # Cyan
        [1.0, 0.5, 0.0],    # Orange
        [0.5, 0.0, 1.0],    # Purple
        [0.0, 1.0, 0.5],    # Spring Green
        [0.5, 1.0, 0.0],    # Lime Green
        [1.0, 0.0, 0.5]     # Pink
    ])

    # Load the images
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]

    bbox_dict = {}
    latent_dim = None
    img_files = sorted(img_files)
    for i, img_file in enumerate(img_files):
        print(f"Processing image {i} of {len(img_files)}")
        # if i > 50:
        #     break
        img_path = os.path.join(img_dir, img_file)
        
        # PIL read the image
        img = Image.open(img_path)
        # process the image
        latents, img, grid_shape = process_image(img, model, processor, device, use_custom_preprocessing=True, use_registers=True)

        # Project onto mask PCA
        mask_latents = pca_mask.transform(latents)
        
        # Reshape to grid dimensions
        grid_h, grid_w = grid_shape
        mask_grid = mask_latents.reshape(grid_h, grid_w)
        
        latent_dim = (grid_h, grid_w, latents.shape[1])
        binary_mask = (mask_grid < 0).astype(np.float32)
        
        # Find connected components
        labeled_array, num_features = ndimage.label(binary_mask)
        component_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background (0)
        largest_component = np.argmax(component_sizes) + 1  # +1 because we skipped background
        largest_mask = (labeled_array == largest_component).astype(np.float32)        
        filtered_latents = latents[largest_mask.reshape(-1) == 1]
         
        # Project filtered embeddings onto RGB PCA for visualization
        rgb_latents = pca_rgb.transform(filtered_latents)
        
        # Create RGB visualizations
        rgb_grid = np.zeros((grid_h, grid_w, 3))
        
        # Normalize RGB values to [0,1] range
        rgb_min = rgb_latents.min(axis=0)
        rgb_max = rgb_latents.max(axis=0)
        rgb_range = rgb_max - rgb_min
        rgb_range[rgb_range == 0] = 1  # Avoid division by zero
        normalized_rgb = (rgb_latents - rgb_min) / rgb_range
        mask_indices = np.where(largest_mask.reshape(-1) == 1)[0]

         # Apply GMM clustering
        cluster_labels = kmeans.predict(rgb_latents)
        cluster_grid = np.zeros((grid_h, grid_w, 3))

        largest_mask = np.zeros((grid_h, grid_w))
        for idx, rgb, label in zip(mask_indices, normalized_rgb, cluster_labels):
            row = idx // grid_w
            col = idx % grid_w
            rgb_grid[row, col] = rgb
            cluster_color_index = label % len(cluster_colors)
            if cluster_color_index != 4:
                cluster_grid[row, col] = cluster_colors[cluster_color_index]
                largest_mask[row, col] = 1

        # only preserve the largest connected component in largest_mask
        # Find connected components
        labeled_array, num_features = ndimage.label(largest_mask)
        component_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background (0)
        largest_component = np.argmax(component_sizes) + 1  # +1 because we skipped background
        largest_mask = (labeled_array == largest_component).astype(np.float32)

        # applu the largest mask to the cluster grid
        cluster_grid = cluster_grid * largest_mask.reshape(grid_h, grid_w, 1)

        if np.sum(largest_mask) == 0:
            print(f"No object found in {img_file}")
            bbox_dict[img_file] = None
            continue
        
        # construct a bounding box around the largest mask
        rows = np.any(largest_mask, axis=1)
        cols = np.any(largest_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = (cmin, rmin, cmax, rmax)
        
        # Scale the bbox coordinates to the original image size using float division
        scale_x = img.width / grid_w
        scale_y = img.height / grid_h

        scaled_bbox = (
            int(bbox[0] * scale_x),  # left
            int(bbox[1] * scale_y),  # top
            int(bbox[2] * scale_x),  # right
            int(bbox[3] * scale_y)   # bottom
        )

        # make the bbox square
        bbox_width = scaled_bbox[2] - scaled_bbox[0]
        bbox_height = scaled_bbox[3] - scaled_bbox[1]
        bbox_size = max(bbox_width, bbox_height) * 1.1
        # get the center of the bbox
        center_x = (scaled_bbox[0] + scaled_bbox[2]) // 2
        center_y = (scaled_bbox[1] + scaled_bbox[3]) // 2
        # get the top left and bottom right corners of the bbox
        top_left = (center_x - bbox_size // 2, center_y - bbox_size // 2)
        bottom_right = (center_x + bbox_size // 2, center_y + bbox_size // 2)
        scaled_bbox = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
        
        # Ensure bbox coordinates are within image bounds
        scaled_bbox = (
            max(0, top_left[0]),
            max(0, top_left[1]),
            min(img.width, bottom_right[0]),
            min(img.height, bottom_right[1])
        )
        
        # Check if bbox is valid (width and height > 0)
        bbox_width = scaled_bbox[2] - scaled_bbox[0]
        bbox_height = scaled_bbox[3] - scaled_bbox[1]
            
        if bbox_width <= 0 or bbox_height <= 0:
            print(f"Invalid bbox dimensions: {bbox_width}x{bbox_height} for {img_file}")
            bbox_dict[img_file] = None
            continue
        
        bbox_dict[img_file] = {
            'bbox': scaled_bbox,
            'center': (center_x, center_y),
            'size': bbox_size,
            'num_valid_pixels': np.sum(largest_mask),
            'latent_dim': latent_dim,
            'img': np.array(img),
            'cluster_grid': cluster_grid,
        }

    initial_bbox_size = np.mean([bbox_info['size'] for bbox_info in bbox_dict.values() if bbox_info is not None])
    initial_center = np.array([0, 0])
    # apply running average to box size and center
    for i, (img_file, bbox_info) in enumerate(bbox_dict.items()):
        print(f"Processing image {i} of {len(bbox_dict)}")
        img = Image.open(os.path.join(img_dir, img_file))

        if bbox_info is not None:
            img = Image.fromarray(bbox_info['img'])
            initial_bbox_size = 0.3 * np.array(bbox_info['size']) + 0.7 * initial_bbox_size
            initial_center = 0.3 * np.array(bbox_info['center']) + 0.7 * np.array(initial_center)

            new_bbox = (initial_center[0] - initial_bbox_size // 2, initial_center[1] - initial_bbox_size // 2, initial_center[0] + initial_bbox_size // 2, initial_center[1] + initial_bbox_size // 2)
            # crop the image using the new bbox
            new_cropped_img = img.crop(new_bbox)
            old_cropped_img = img.crop(bbox_info['bbox'])
        else:
            new_cropped_img = img
            old_cropped_img = img
        
        # save the old and new cropped images side by side with fixed axes sizes
        # also plot the bbox on the original image
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 4, 1)
        plt.imshow(img)
        if bbox_info is not None:
            plt.plot([bbox_info['bbox'][0], bbox_info['bbox'][2], bbox_info['bbox'][2], bbox_info['bbox'][0], bbox_info['bbox'][0]], [bbox_info['bbox'][1], bbox_info['bbox'][1], bbox_info['bbox'][3], bbox_info['bbox'][3], bbox_info['bbox'][1]], color='red', linewidth=2)
        plt.axis('off')
        plt.title("Original Image")
        plt.subplot(1, 4, 2)
        plt.imshow(img, alpha=0.5)
        if bbox_info is not None:
            cluster_grid = Image.fromarray((bbox_info['cluster_grid'] * 255).astype(np.uint8))
            cluster_grid = cluster_grid.resize((img.width, img.height), Image.Resampling.NEAREST)
            plt.imshow(cluster_grid, alpha=0.5)
            plt.title(f"Cluster Grid" if bbox_info is not None else "No object found")
        else:
            plt.imshow(np.zeros((37, 37, 3)))
            plt.title("No object found")
        plt.axis('off')
        plt.subplot(1, 4, 3)
        plt.imshow(old_cropped_img)
        plt.axis('off')
        plt.subplot(1, 4, 4)
        plt.imshow(new_cropped_img)
        plt.title(f"New Cropped Image" if bbox_info is not None else "No object found")
        plt.axis('off')
        plt.tight_layout()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # save the new_cropped_img with numbered filename for video creation
        frame_filename = f"frame_{i:06d}.png"
        plt.savefig(os.path.join(output_dir, frame_filename))
        plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help='Path to the directory containing the images')
    parser.add_argument("--output_dir", type=str, required=True, help='Path to the directory to save the cropped images')
    parser.add_argument("--pca_model", type=str, help='Path to the PCA model', default='./saved_pcas/DNEG_extreme_angles_pca.pt')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    main(args, device)