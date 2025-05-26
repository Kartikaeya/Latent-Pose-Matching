import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from tqdm import tqdm

def load_model_and_processor(use_registers=True):
    # Load DinoV2 base model and processor
    model_name = "facebook/dinov2-with-registers-base" if use_registers else "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Check for MPS (Metal Performance Shaders) availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    model = model.to(device)

    return model, processor, device

def pad_to_square(image):
    """Pad the image to make it square."""
    width, height = image.size
    max_dim = max(width, height)
    
    # Create a new black image with the square dimensions
    new_image = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
    
    # Calculate position to paste the original image (centered)
    paste_x = (max_dim - width) // 2
    paste_y = (max_dim - height) // 2
    
    # Paste the original image onto the new square image
    new_image.paste(image, (paste_x, paste_y))
    
    return new_image

def process_image(image_input, model, processor, device, use_custom_preprocessing=False, use_registers=False):
    """Process a single image and return its token embeddings.
    
    Args:
        image_input: Either a file path (str) or a PIL Image object
        model: DinoV2 model
        processor: DinoV2 processor
        device: Device to run inference on
        use_custom_preprocessing: Whether to use custom preprocessing
        use_registers: Whether to use DinoV2 with registers
    
    Returns:
        token_embeddings: Token embeddings from DinoV2
        processed_image: The processed PIL Image
        grid_shape: Tuple of (grid_h, grid_w)
    """
    # Load the image if a file path is provided
    if isinstance(image_input, str):
        original_image = Image.open(image_input).convert('RGB')
    else:
        original_image = image_input
    
    if use_custom_preprocessing:
        # Pad to square and resize to 518x518
        processed_image = pad_to_square(original_image)
        processed_image = processed_image.resize((518, 518), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and then to tensor
        image_array = np.array(processed_image)
        image_tensor = torch.from_numpy(image_array).float()
        
        # Convert from [H, W, C] to [C, H, W]
        image_tensor = image_tensor.permute(2, 0, 1)
        
        # Add batch dimension [B, C, H, W]
        image_tensor = image_tensor.unsqueeze(0)
        
        # Normalize using DinoV2's mean and std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor / 255.0 - mean) / std
        
        # Create dictionary with pixel_values
        inputs = {'pixel_values': image_tensor.to(device)}
    else:
        processed_image = original_image
        # Preprocess the image using the processor
        inputs = processor(images=processed_image, return_tensors="pt")
        # Move inputs to the correct device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the last hidden state (latent vectors)
    # Shape: [batch_size, seq_length, hidden_size]
    last_hidden_state = outputs.last_hidden_state.squeeze(0)
    
    # Get all token embeddings except CLS token and register tokens if using registers
    if use_registers:
        # Skip CLS token and 4 register tokens (skip first 5 tokens)
        token_embeddings = last_hidden_state[5:].cpu().numpy()
    else:
        # Skip only CLS token (skip first token)
        token_embeddings = last_hidden_state[1:].cpu().numpy()
    
    # Grid dimensions are always 37x37 for standard processing
    grid_h = grid_w = 37
    
    return token_embeddings, processed_image, (grid_h, grid_w)

def save_pca_models(pca_mask, pca_rgb, clustering_model, output_path):
    """Save PCA models and clustering model to a .pt file."""
    if isinstance(clustering_model, KMeans):
        clustering_data = {
            'cluster_centers_': clustering_model.cluster_centers_,
            'n_clusters': clustering_model.n_clusters_,
            'random_state': clustering_model.random_state
        }
    else:  # GaussianMixture
        clustering_data = {
            'means_': clustering_model.means_,
            'covariances_': clustering_model.covariances_,
            'weights_': clustering_model.weights_,
            'precisions_': clustering_model.precisions_,
            'precisions_cholesky_': clustering_model.precisions_cholesky_,
            'converged_': clustering_model.converged_,
            'n_iter_': clustering_model.n_iter_,
            'lower_bound_': clustering_model.lower_bound_,
            'n_components': clustering_model.n_components,
            'random_state': clustering_model.random_state,
            'covariance_type': clustering_model.covariance_type
        }
    
    torch.save({
        'pca_mask': {
            'n_components': pca_mask.n_components,
            'components_': pca_mask.components_,
            'mean_': pca_mask.mean_,
            'explained_variance_': pca_mask.explained_variance_,
            'explained_variance_ratio_': pca_mask.explained_variance_ratio_
        },
        'pca_rgb': {
            'n_components': pca_rgb.n_components,
            'components_': pca_rgb.components_,
            'mean_': pca_rgb.mean_,
            'explained_variance_': pca_rgb.explained_variance_,
            'explained_variance_ratio_': pca_rgb.explained_variance_ratio_
        },
        'clustering': {
            'method': 'gmm' if isinstance(clustering_model, GaussianMixture) else 'kmeans',
            **clustering_data
        }
    }, output_path)
    print(f"Saved PCA models and clustering model to {output_path}")

def load_pca_models(input_path):
    """Load PCA models from a .pt file."""
    # Load the model file
    pca_models = torch.load(input_path, map_location='cpu')
    
    # Reconstruct PCA models
    pca_mask = PCA(n_components=pca_models['pca_mask']['n_components'])
    pca_mask.components_ = pca_models['pca_mask']['components_']
    pca_mask.mean_ = pca_models['pca_mask']['mean_']
    pca_mask.explained_variance_ = pca_models['pca_mask']['explained_variance_']
    pca_mask.explained_variance_ratio_ = pca_models['pca_mask']['explained_variance_ratio_']
    
    pca_rgb = PCA(n_components=pca_models['pca_rgb']['n_components'])
    pca_rgb.components_ = pca_models['pca_rgb']['components_']
    pca_rgb.mean_ = pca_models['pca_rgb']['mean_']
    pca_rgb.explained_variance_ = pca_models['pca_rgb']['explained_variance_']
    pca_rgb.explained_variance_ratio_ = pca_models['pca_rgb']['explained_variance_ratio_']
    
    # Reconstruct clustering model
    clustering_data = pca_models['clustering']
    if clustering_data['method'] == 'kmeans':
        # KMeans
        kmeans = KMeans(
            n_clusters=clustering_data['n_clusters'],
            random_state=clustering_data['random_state'],
            n_init=1,  # Since we're loading pre-trained centers
            max_iter=1  # Since we're loading pre-trained centers
        )
        kmeans.cluster_centers_ = clustering_data['cluster_centers_']
    else:  # gmm
        gmm = GaussianMixture(
            n_components=clustering_data['n_components'],
            random_state=clustering_data['random_state'],
            covariance_type=clustering_data['covariance_type'],
            n_init=1,  # Since we're loading pre-trained parameters
            max_iter=1  # Since we're loading pre-trained parameters
        )
        gmm.means_ = clustering_data['means_']
        gmm.covariances_ = clustering_data['covariances_']
        gmm.weights_ = clustering_data['weights_']
        gmm.precisions_ = clustering_data['precisions_']
        gmm.precisions_cholesky_ = clustering_data['precisions_cholesky_']
        gmm.converged_ = clustering_data['converged_']
        gmm.n_iter_ = clustering_data['n_iter_']
        gmm.lower_bound_ = clustering_data['lower_bound_']
        kmeans = gmm
    
    return pca_mask, pca_rgb, kmeans

def train_pca_models(image_files, model, processor, device, use_custom_preprocessing=False, use_registers=False, 
                   pca_train_percent=10.0, threshold_greater_than=True, clustering_method='kmeans'):
    """Train PCA models on a subset of images."""
    # Select a subset of images for training
    num_train = int(len(image_files) * pca_train_percent / 100)
    train_files = np.random.choice(image_files, num_train, replace=False)
    print(f"Selected {num_train} frames ({pca_train_percent}%) for PCA training")
    
    # Process each training image and collect embeddings
    all_embeddings = []
    for image_path in tqdm(train_files, desc="Processing sample frames"):
        token_embeddings, _, _ = process_image(
            image_path, model, processor, device, 
            use_custom_preprocessing, use_registers
        )
        all_embeddings.append(token_embeddings)
    
    # Stack all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    print(f"Training PCA on {all_embeddings.shape[0]} embeddings")
    
    # Train PCA for mask
    print("Training PCA for mask...")
    pca_mask = PCA(n_components=1)
    pca_mask.fit(all_embeddings)
    
    # Filter embeddings using threshold
    print("Filtering embeddings...")
    mask_values = pca_mask.transform(all_embeddings)
    mask_values = mask_values.reshape(-1)  # Flatten to 1D array
    if threshold_greater_than:
        filtered_embeddings = all_embeddings[mask_values > 0]
    else:
        filtered_embeddings = all_embeddings[mask_values < 0]
    
    # Train PCA for RGB
    print("Training PCA for RGB...")
    pca_rgb = PCA(n_components=3)
    pca_rgb.fit(filtered_embeddings)
    
    # Train clustering model
    print(f"Training {clustering_method}...")
    rgb_latents = pca_rgb.transform(filtered_embeddings)
    if clustering_method == 'kmeans':
        kmeans = KMeans(n_clusters=9, random_state=0)
        kmeans.fit(rgb_latents)
        save_pca_models(pca_mask, pca_rgb, kmeans, "pca_models.pt")
        return pca_mask, pca_rgb, kmeans
    elif clustering_method == 'gmm':
        gmm = GaussianMixture(n_components=9, random_state=0, covariance_type='full')
        gmm.fit(rgb_latents)
        
        # Save models
        pca_models = {
            'pca_mask': {
                'n_components': pca_mask.n_components_,
                'components_': pca_mask.components_,
                'mean_': pca_mask.mean_,
                'explained_variance_': pca_mask.explained_variance_,
                'explained_variance_ratio_': pca_mask.explained_variance_ratio_
            },
            'pca_rgb': {
                'n_components': pca_rgb.n_components_,
                'components_': pca_rgb.components_,
                'mean_': pca_rgb.mean_,
                'explained_variance_': pca_rgb.explained_variance_,
                'explained_variance_ratio_': pca_rgb.explained_variance_ratio_
            },
            'gmm': {
                'n_components': gmm.n_components,
                'random_state': gmm.random_state,
                'covariance_type': gmm.covariance_type,
                'means_': gmm.means_,
                'covariances_': gmm.covariances_,
                'weights_': gmm.weights_,
                'precisions_': gmm.precisions_,
                'precisions_cholesky_': gmm.precisions_cholesky_,
                'converged_': gmm.converged_,
                'n_iter_': gmm.n_iter_,
                'lower_bound_': gmm.lower_bound_
            }
        }
        
        return pca_models
    
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")

def process_sequence_with_pca(image_files, model, processor, device, pca_mask, pca_rgb, kmeans,
                            output_dir, use_custom_preprocessing, use_registers, threshold_greater_than):
    """Process a sequence of images using pre-trained PCA models."""
    for frame_idx, image_path in enumerate(image_files):
        print(f"\nProcessing frame {frame_idx + 1}/{len(image_files)}: {image_path}")
        
        # Process image and get latent vector
        token_embeddings, processed_image, grid_shape = process_image(
            image_path, model, processor, device, 
            use_custom_preprocessing, use_registers
        )
        
        # Project onto learned PCA
        compressed_latents = pca_mask.transform(token_embeddings)
        
        # Reshape to grid_h x grid_w
        grid_h, grid_w = grid_shape
        latent_grid = compressed_latents.reshape(grid_h, grid_w)
        
        # Create binary mask based on threshold direction
        if threshold_greater_than:
            binary_mask = (latent_grid > 0).astype(np.float32)
            threshold_label = "Threshold (> 0)"
        else:
            binary_mask = (latent_grid < 0).astype(np.float32)
            threshold_label = "Threshold (< 0)"
        
        # Filter embeddings using mask
        filtered_embeddings = token_embeddings[binary_mask.reshape(-1) == 1]
        
        if len(filtered_embeddings) > 0:
            # Project onto RGB PCA
            rgb_latents = pca_rgb.transform(filtered_embeddings)
            
            # Define distinct colors for each cluster
            # Using a set of distinct colors that are easily distinguishable
            cluster_colors = np.array([
                [1.0, 0.0, 0.0],    # Red
                [0.0, 1.0, 0.0],    # Green
                [0.0, 0.0, 1.0],    # Blue
                [1.0, 1.0, 0.0],    # Yellow
                [1.0, 0.0, 1.0],    # Magenta
                [0.0, 1.0, 1.0],    # Cyan
                [1.0, 0.5, 0.0],    # Orange
                [0.5, 0.0, 1.0],    # Purple
                [0.0, 1.0, 0.5]     # Spring Green
            ])
            
            # Create RGB visualization
            rgb_grid = np.zeros((grid_h, grid_w, 3))
            mask_indices = np.where(binary_mask.reshape(-1) == 1)[0]
            
            # Apply clustering to determine cluster labels
            if isinstance(kmeans, KMeans):
                cluster_labels = kmeans.predict(rgb_latents)
            elif isinstance(kmeans, GaussianMixture):
                cluster_labels = kmeans.predict(rgb_latents)
            else:
                raise ValueError(f"Unknown clustering model type: {type(kmeans)}")
            
            # Create cluster label visualization
            cluster_grid = np.zeros((grid_h, grid_w, 3))
            cluster_label_grid = np.zeros((grid_h, grid_w))
            
            # Normalize RGB values to [0,1] range for visualization
            rgb_min = rgb_latents.min(axis=0)
            rgb_max = rgb_latents.max(axis=0)
            rgb_range = rgb_max - rgb_min
            rgb_range[rgb_range == 0] = 1  # Avoid division by zero
            normalized_rgb_latents = (rgb_latents - rgb_min) / rgb_range
            
            for idx, (rgb, label) in zip(mask_indices, zip(normalized_rgb_latents, cluster_labels)):
                row = idx // grid_w
                col = idx % grid_w
                rgb_grid[row, col] = rgb
                cluster_grid[row, col] = cluster_colors[label]
                cluster_label_grid[row, col] = label
        else:
            rgb_grid = np.zeros((grid_h, grid_w, 3))
        
        # Create visualization figure
        plt.figure(figsize=(25, 5))
        
        # Plot original image with cluster centroids
        plt.subplot(1, 5, 1)
        plt.imshow(processed_image)
        
        # Get processed image dimensions (width, height)
        processed_width, processed_height = processed_image.size
        
        # Calculate and plot centroids for each cluster
        for cluster_idx in range(9):
            # Get pixels belonging to this cluster
            cluster_pixels = np.argwhere(cluster_label_grid == cluster_idx)
            if len(cluster_pixels) > 0:
                # Calculate centroid in grid coordinates
                centroid_y = np.mean(cluster_pixels[:, 0])
                centroid_x = np.mean(cluster_pixels[:, 1])
                
                # Convert grid coordinates to processed image coordinates
                processed_centroid_x = centroid_x * (processed_width / grid_w)
                processed_centroid_y = centroid_y * (processed_height / grid_h)
                
                # Plot centroid on processed image
                plt.scatter(processed_centroid_x, processed_centroid_y, 
                          color=cluster_colors[cluster_idx],
                          marker='o', s=100, edgecolors='black', linewidth=2)
        
        plt.title(f'Frame {frame_idx + 1} with Cluster Centroids')
        plt.axis('off')
        
        # Plot histogram of compressed latent values
        plt.subplot(1, 5, 2)
        plt.hist(compressed_latents, bins=50)
        plt.axvline(x=0, color='r', linestyle='--', label=threshold_label)
        plt.title(f'Distribution of Compressed Latents')
        plt.legend()
        
        # Plot latent grid as image
        plt.subplot(1, 5, 3)
        plt.imshow(latent_grid, cmap='viridis')
        plt.colorbar()
        plt.title(f'Compressed Latent Grid ({grid_h}x{grid_w})')
        plt.axis('off')
        
        # Plot RGB visualization of filtered latents
        plt.subplot(1, 5, 4)
        plt.imshow(rgb_grid)
        plt.title('Filtered Latents (RGB)')
        plt.axis('off')
        
        # Normalize RGB values to [0,1] range
        rgb_min = rgb_latents.min(axis=0)
        rgb_max = rgb_latents.max(axis=0)
        rgb_range = rgb_max - rgb_min
        rgb_range[rgb_range == 0] = 1  # Avoid division by zero
        rgb_latents = (rgb_latents - rgb_min) / rgb_range
        
        # Plot cluster label visualization
        plt.subplot(1, 5, 5)
        plt.imshow(cluster_grid)
        plt.title('Cluster Labels')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(output_dir, f'frame_{frame_idx:06d}.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"Saved visualization to {output_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process video frames and perform PCA on latent vectors')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing video frames')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output visualizations')
    parser.add_argument('--use_custom_preprocessing', action='store_true', 
                      help='Use custom preprocessing (padding to square and resizing to 518x518)')

    parser.add_argument('--use_registers', action='store_true',
                      help='Use DinoV2 base with registers instead of DinoV2 base')
    parser.add_argument('--threshold_greater_than', action='store_true',
                      help='Threshold for values greater than 0 instead of less than 0')
    parser.add_argument('--pca_train_percent', type=float, default=10.0,
                      help='Percentage of frames to use for PCA training (default: 10.0)')
    parser.add_argument('--pca_models_path', type=str, default=None,
                      help='Path to load pre-trained PCA models. If not provided, will train new models.')
    parser.add_argument('--save_pca_models', type=str, default=None,
                      help='Path to save trained PCA models. Only used when training new models.')
    parser.add_argument('--clustering_method', type=str, default='kmeans',
                      choices=['kmeans', 'gmm'],
                      help='Clustering method to use: kmeans or gmm')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and processor
    print("Loading DinoV2 model and processor...")
    model, processor, device = load_model_and_processor(args.use_registers)
    
    # Get all image files in the directory
    image_files = sorted(glob.glob(os.path.join(args.input_dir, '*.png')) + \
                        glob.glob(os.path.join(args.input_dir, '*.jpg')) + \
                        glob.glob(os.path.join(args.input_dir, '*.jpeg')))
    
    if not image_files:
        print(f"No image files found in {args.input_dir}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    if args.pca_models_path:
        # Load pre-trained PCA models
        print(f"Loading PCA models from {args.pca_models_path}")
        pca_mask, pca_rgb, kmeans = load_pca_models(args.pca_models_path)
        process_sequence_with_pca(image_files, model, processor, device, pca_mask, pca_rgb, kmeans,
                                args.output_dir, args.use_custom_preprocessing, 
                                args.use_registers, args.threshold_greater_than)
    else:
        # Train PCA models and clustering
        print("\nTraining PCA models and clustering...")
        pca_models = train_pca_models(
            image_files, model, processor, device,
            use_custom_preprocessing=args.use_custom_preprocessing,
            use_registers=args.use_registers,
            pca_train_percent=args.pca_train_percent,
            threshold_greater_than=args.threshold_greater_than,
            clustering_method=args.clustering_method
        )
        
        # Extract models from the returned dictionary
        pca_mask = PCA()
        pca_mask.components_ = pca_models['pca_mask']['components_']
        pca_mask.mean_ = pca_models['pca_mask']['mean_']
        pca_mask.explained_variance_ = pca_models['pca_mask']['explained_variance_']
        pca_mask.explained_variance_ratio_ = pca_models['pca_mask']['explained_variance_ratio_']
        
        pca_rgb = PCA()
        pca_rgb.components_ = pca_models['pca_rgb']['components_']
        pca_rgb.mean_ = pca_models['pca_rgb']['mean_']
        pca_rgb.explained_variance_ = pca_models['pca_rgb']['explained_variance_']
        pca_rgb.explained_variance_ratio_ = pca_models['pca_rgb']['explained_variance_ratio_']
        
        if args.clustering_method == 'kmeans':
            kmeans = KMeans()
            kmeans.cluster_centers_ = pca_models['kmeans']['cluster_centers_']
            kmeans.n_clusters_ = pca_models['kmeans']['n_clusters']
            kmeans.random_state = pca_models['kmeans']['random_state']
        else:  # gmm
            gmm = GaussianMixture()
            gmm.means_ = pca_models['gmm']['means_']
            gmm.covariances_ = pca_models['gmm']['covariances_']
            gmm.weights_ = pca_models['gmm']['weights_']
            gmm.precisions_ = pca_models['gmm']['precisions_']
            gmm.precisions_cholesky_ = pca_models['gmm']['precisions_cholesky_']
            gmm.converged_ = pca_models['gmm']['converged_']
            gmm.n_iter_ = pca_models['gmm']['n_iter_']
            gmm.lower_bound_ = pca_models['gmm']['lower_bound_']
            gmm.n_components = pca_models['gmm']['n_components']
            gmm.random_state = pca_models['gmm']['random_state']
            gmm.covariance_type = pca_models['gmm']['covariance_type']
            kmeans = gmm
        
        # Save PCA models if requested
        if args.save_pca_models:
            # Save both PCA models and K-means
            save_pca_models(pca_mask, pca_rgb, kmeans, args.save_pca_models)
        
        # Process all frames using the learned PCA
        process_sequence_with_pca(image_files, model, processor, device, pca_mask, pca_rgb, kmeans,
                                args.output_dir, args.use_custom_preprocessing, 
                                args.use_registers, args.threshold_greater_than)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()