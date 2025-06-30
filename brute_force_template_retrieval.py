from latent_pca_video import load_model_and_processor, load_pca_models
from find_pose import visualize_comparison
import argparse
import os
import numpy as np

def estimate_pose_from_templates(template_files, template_path, query_image_path, model, processor, device, pca_mask, pca_rgb,
                                 output_folder='./output/brute_force_retrieval_results/'):
    inlier_ratios = []
    num_inliers = []
    feature_matches = []
    num_feature_matches = []
    for i, template_file in enumerate(template_files):
        template_path = os.path.join(args.template_folder_path, template_file)
        print(f"Processing template: {template_path}")

        # TODO 2: figure out a better way to patchify images without losing resolution
        rvec, tvec, inlier_ratio, total_feature_matches, match_distances, total_inliers = visualize_comparison(
            image1_path=query_image_path,
            render_data_path=template_path,
            model=model,
            processor=processor,
            device=device,
            pca_mask=pca_mask,
            pca_rgb=pca_rgb,
            save_results=False,
            mesh_path='./data/DNEGSynFace_topology/generic_neutral_mesh.obj',
            output_folder=output_folder)
        if inlier_ratio is None:
            print(f"Skipping template {template_file} due to insufficient feature matches.")
            continue
        inlier_ratios.append(inlier_ratio)
        num_inliers.append(total_inliers)
        feature_matches.append({'num_feature_match': total_feature_matches, 'match_distances': np.mean(match_distances), 'index': i})
        num_feature_matches.append(total_feature_matches)
        
    threshold = np.array(num_feature_matches).mean()
    inlier_ratios = [inlier_ratios[i] if num_feature_matches[i] > threshold else 0 for i in range(len(inlier_ratios))]
    sorted_num_feature_matches = sorted(feature_matches, key=lambda x: x['num_feature_match'], reverse=True) 
    sorted_match_distances = sorted(feature_matches, key=lambda x: x['match_distances'])
    
    
    best_template_inlier_ratio = inlier_ratios.index(max(inlier_ratios))
    best_template_num_inliers = np.argmax(num_inliers)
    best_template_num_feature_matches = sorted_num_feature_matches[0]['index']
    best_template_match_distances = sorted_match_distances[0]['index']

    print(f"Best template file based on inliers ratio: {template_files[best_template_inlier_ratio]}")
    print(f"Best template file based on number of inliers: {template_files[best_template_num_inliers]}")
    print(f'Best template file based on number of feature matches: {template_files[best_template_num_feature_matches]}')
    print(f'Best template file based on match distances: {template_files[best_template_match_distances]}')
    query_image_file = os.path.basename(query_image_path)
    visualize_comparison(
        image1_path=query_image_path,
        render_data_path=os.path.join(args.template_folder_path, template_files[best_template_match_distances]),
        model=model,
        processor=processor,
        device=device,
        pca_mask=pca_mask,
        pca_rgb=pca_rgb,
        save_results=True,
        mesh_path='./data/DNEGSynFace_topology/generic_neutral_mesh.obj',
        filename=query_image_file[:-4],
        output_folder=output_folder,
        plot_title=f'PE based number of feature match distance (PnPRansac)'
    )
    return None, None, best_template_num_inliers

def get_closest_template(tvec, best_template_index, template_tvecs):
    prev_fit_tvec = tvec / np.linalg.norm(tvec)
    prev_template_tvec = template_tvecs[best_template_index]
    
    prev_fit_similarity = []
    prev_template_similarity = []
    for i, template_tvec in enumerate(template_tvecs):
        prev_fit_similarity.append(np.dot(prev_fit_tvec.squeeze(), template_tvec))
        prev_template_similarity.append(np.dot(prev_template_tvec.squeeze(), template_tvec))

    # get the indices of the top 4 highest prev_fit_similarities and prev_template_similarities
    prev_fit_indices = np.argsort(prev_fit_similarity)[-4:][::-1]
    prev_template_indices = np.argsort(prev_template_similarity)[-4:][::-1]
    # combine the indices and remove duplicates
    combined_indices = set(prev_fit_indices).union(set(prev_template_indices))
    # convert the set to a list and sort it
    combined_indices = sorted(list(combined_indices))
    # return the indices of the templates that are closest to the previous fit and template tvec
    return combined_indices

if __name__ == "__main__":
    # create a parser to get the query image path from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_images_dir', type=str, default="./data/metaphysic_cropped/", help='Path to the query images folder')
    parser.add_argument('--template_folder_path', type=str, default='./output/templates/data/', help='Path to the folder containing ' \
    'template data rendered from different poses')
    parser.add_argument('--pca_model', type=str, default='./saved_pcas/HH_PCA_GMM_v2.pt', help='Path to the folder containing PCA models for mask and RGB')
    parser.add_argument('--output_folder', type=str, default='./output/brute_force_retrieval_results/', help='Path to the output folder for results')
    args = parser.parse_args()
    
    # Load model and processor
    print("Loading DinoV2 model and processor...")
    model, processor, device = load_model_and_processor()
    
    # Load PCA models
    print(f"Loading PCA models from {args.pca_model}")
    pca_mask, pca_rgb, kmeans = load_pca_models(args.pca_model)
    template_files = [f for f in os.listdir(args.template_folder_path) if f.endswith('.pt')]
    template_files = sorted(template_files)

    template_files = ['template_data_h0_r240.pt', 'template_data_h0_r270.pt', 'template_data_h0_r300.pt', 'template_data_h0_r330.pt', 
                      'template_data_h0_r0.pt', 'template_data_h0_r30.pt', 'template_data_h0_r60.pt', 'template_data_h0_r90.pt', 'template_data_h0_r120.pt']
    
    # Check if query_images_dir is a file or directory
    if os.path.isfile(args.query_images_dir):
        query_image_files = [os.path.basename(args.query_images_dir)]
        query_images_dir = os.path.dirname(args.query_images_dir)
    else:
        query_images_dir = args.query_images_dir
        query_image_files = [f for f in os.listdir(query_images_dir) if f.endswith('.jpg') or f.endswith('.png')]
        query_image_files = sorted(query_image_files)
    
    if len(query_image_files) == 0:
        raise ValueError("No query images found in the specified directory.")
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    for i, query_image_file in enumerate(query_image_files):
        query_image_path = os.path.join(query_images_dir, query_image_file)
        print(f"Processing query image: {query_image_file}")
        rvec, tvec, best_template_index = estimate_pose_from_templates(template_files, args.template_folder_path, query_image_path, model, processor, device, pca_mask, 
                                        pca_rgb, args.output_folder)
        