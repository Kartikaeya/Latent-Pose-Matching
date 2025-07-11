from latent_pca_video import load_model_and_processor, process_image, load_pca_models
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch
import torch.nn.functional as F
import argparse

def compute_template_descriptor(cluster_labels, kmeans, cluster_occurrences, N):
    if len(cluster_labels) > 0:
        # Count occurrences of each cluster in this template
        cluster_counts = Counter(cluster_labels)
        n_t = len(cluster_labels)  # total number of words in this template
        
        # Initialize descriptor vector
        template_descriptor = np.zeros(kmeans.n_components)
        
        # Calculate weighted word frequencies
        for cluster_id, count in cluster_counts.items():
            n_i_t = count  # number of occurrences of word i in template t
            n_i = cluster_occurrences[str(cluster_id)]  # number of occurrences of word i in all templates
            
            # Calculate TF-IDF like weight
            if n_t > 0 and n_i > 0:
                template_descriptor[cluster_id] = (n_i_t / n_t) * np.log(N / n_i)
    else:
        template_descriptor = np.zeros(kmeans.n_components)
    return template_descriptor

def get_closest_template_bow(query_image_path, debug=False):
    # read query image
    query_image = Image.open(query_image_path)

    # Load PCA models
    print("Loading PCA models...")
    pca_model_path = './saved_pcas/HH_PCA_GMM_v2.pt'

    pca_mask, pca_rgb, kmeans = load_pca_models(pca_model_path)

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
        [0.0, 1.0, 0.5]     # Spring Green
    ])

    # load dino model and processor
    model, processor, device = load_model_and_processor(use_registers=True)
    # Process through DinoV2
    token_embeddings, processed_image, grid_shape = process_image(
        query_image,
        model,
        processor,
        device, 
        use_custom_preprocessing=True,
        use_registers=True
    )
    
    # Project embeddings through PCA
    compressed_latents = pca_mask.transform(token_embeddings)
    latent_grid = compressed_latents.reshape(grid_shape)
    
    # Create binary mask based on threshold direction
    binary_mask = (latent_grid < 0).astype(np.float32)
    
    # Filter embeddings using mask
    filtered_embeddings = token_embeddings[binary_mask.reshape(-1) == 1]
    
    if len(filtered_embeddings) > 0:
        rgb_latents = pca_rgb.transform(filtered_embeddings)
        rgb_grid = np.zeros((grid_shape[0], grid_shape[1], 3))
        mask_indices = np.where(binary_mask.reshape(-1) == 1)[0]
        
        # Normalize RGB values to [0,1] range for visualization
        rgb_min = rgb_latents.min(axis=0)
        rgb_max = rgb_latents.max(axis=0)
        rgb_range = rgb_max - rgb_min
        rgb_range[rgb_range == 0] = 1
        normalized_rgb_latents = (rgb_latents - rgb_min) / rgb_range

        # Apply GMM clustering
        cluster_labels = kmeans.predict(rgb_latents)

        # Create cluster visualization
        cluster_grid = np.zeros((grid_shape[0], grid_shape[1], 3))
        
        for idx, (rgb, label) in zip(mask_indices, zip(normalized_rgb_latents, cluster_labels)):
            row = idx // grid_shape[1]
            col = idx % grid_shape[1]
            rgb_grid[row, col] = rgb
            cluster_grid[row, col] = cluster_colors[label % len(cluster_colors)]
    else:
        cluster_labels = np.array([])

    # load bag of words descriptors of all templates from json file
    with open("./output/templates//data/template_descriptors.json", "r") as f:
        json_data = json.load(f)
        template_descriptors = json_data['bag_of_words_descriptors']
        N = json_data['num_templates']
        cluster_occurrences = json_data['cluster_occurrences']
        template_descriptors = {k: v for k, v in template_descriptors.items() if 'h0' in k and '180' \
                                not in k and '150' not in k and '210' not in k}
    
    template_descriptor = compute_template_descriptor(cluster_labels, kmeans, cluster_occurrences, N)
    
    query_image_bag_of_words = template_descriptor

    # compute cosine similarity between query and template images
    cosine_similarities = []
    for template_key, template_descriptor in template_descriptors.items():
        cosine_similarities.append(F.cosine_similarity(torch.tensor(query_image_bag_of_words), torch.tensor(template_descriptor), dim=0))
    cosine_similarities = torch.tensor(cosine_similarities)
    # return top 4 template images with the highest cosine similarity
    top_4_indices = torch.topk(cosine_similarities, 4).indices

    # get the template keys based on indices
    top_4_template_keys = [list(template_descriptors.keys())[i] for i in top_4_indices]

    if debug:
        # visualize top 4 template images with the highest cosine similarity
        plt.figure(figsize=(15, 5))
        for i, template_key in enumerate(top_4_template_keys):
            plt.subplot(1, 4, i+1)
            template_data = torch.load(f"./output/templates/data/{template_key}")
            plt.imshow(template_data['rendered_image'].permute(1, 2, 0))
            plt.title(f'Template {i}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    return top_4_template_keys

if __name__ == "__main__":
    # argparse the query image path
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_image', type=str, default="./data/extreme_pose_right/frame_00001.png", help='Path to the query image')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    args = parser.parse_args()
    get_closest_template_bow(query_image_path=args.query_image, debug=args.debug)