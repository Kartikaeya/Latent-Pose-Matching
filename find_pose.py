import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import cv2
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader,
    TexturesVertex,
    PointLights,
    Materials,
    FoVPerspectiveCameras,
    BlendParams
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import so3_exp_map
from pytorch3d.utils import cameras_from_opencv_projection
from scipy import ndimage
from pathlib import Path
from PIL import Image
from scipy.spatial.distance import cdist
from latent_pca_video import (
    load_model_and_processor,
    process_image,
    load_pca_models
)

DEBUG = True

def find_feature_matches(latents1, latents2, mask1, mask2):
    """Find top K feature matches between two latent representations.
    Ensures one-to-one matches (no feature in image2 is matched to multiple features in image1)."""
    # Get coordinates of non-zero pixels in masks
    coords1 = np.argwhere(mask1.reshape(-1) == 1)
    coords2 = np.argwhere(mask2.reshape(-1) == 1)
    
    # Compute pairwise distances
    distances = cdist(latents1, latents2, metric='euclidean')
    
    # Sort matches by distance
    all_matches = []
    for i in range(len(latents1)):
        for j in range(len(latents2)):
            all_matches.append((i, j, distances[i, j]))
    
    # Sort matches by distance
    all_matches.sort(key=lambda x: x[2])
    
    # Initialize arrays to track used features for both images
    used_features1 = np.zeros(len(latents1), dtype=bool)
    used_features2 = np.zeros(len(latents2), dtype=bool)
    matches = []
    match_distances = []  # Store distances for each match
    
    # Take matches ensuring one-to-one correspondence for both images
    for i, j, dist in all_matches:
        if not used_features1[i] and not used_features2[j]:  # If neither feature has been used yet
            matches.append((coords1[i][0], coords2[j][0]))
            match_distances.append(dist)  # Store the distance for this match
            used_features1[i] = True  # Mark feature i as used
            used_features2[j] = True  # Mark feature j as used
    

    # preserve only the top 75% of the matches
    match_distances = match_distances[:(3*len(match_distances))//4]
    matches = [matches[i] for i in range(len(match_distances))]
    
    return matches, match_distances

def estimate_pose(real_points_2d, rendered_points_3d, camera_matrix, select_top_k_percent_matches=None):
    """Estimate pose using PnP-RANSAC algorithm.
    
    Args:
        real_points_2d: Nx2 array of 2D points from real image
        rendered_points_3d: Nx3 array of 3D points from rendered image
        camera_matrix: 3x3 camera matrix
        select_top_k_percent_matches: Percentage of top matches to select (optional)
    
    Returns:
        rvec: Rotation vector
        tvec: Translation vector
        inliers: Boolean array indicating which points are inliers
    """

    if select_top_k_percent_matches is not None:
        # select the top k percent of matches
        num_matches = len(real_points_2d)
        num_top_matches = int(num_matches * select_top_k_percent_matches)
        real_points_2d = real_points_2d[:num_top_matches]
        rendered_points_3d = rendered_points_3d[:num_top_matches]

    # assign a random color to each point
    colors = np.random.rand(len(real_points_2d), 3)

    # reflect the rendered points 2d about the yz-plane
    rendered_points_3d[:, 0] = -rendered_points_3d[:, 0]

    # Scale rendered 3D points to [0, 1] first, then to [0, 518]
    rendered_points_2d = rendered_points_3d[:, :2]  # Take only x,y coordinates
    min_coords = np.min(rendered_points_2d, axis=0)
    max_coords = np.max(rendered_points_2d, axis=0)
    range_coords = max_coords - min_coords
    range_coords[range_coords == 0] = 1  # Avoid division by zero
    
    # First normalize to [0,1]
    norm_rendered_points = (rendered_points_2d - min_coords) / range_coords
    # Then scale to [0,518]
    scaled_rendered_points = norm_rendered_points * 518

    if DEBUG:
        # Create a single plot
        plt.figure(figsize=(10, 5))
        
        # Plot points
        plt.scatter(real_points_2d[:, 0], 518 - real_points_2d[:, 1], c=colors, label='Real Image Points')
        plt.scatter(scaled_rendered_points[:, 0] + 600, scaled_rendered_points[:, 1], c=colors, label='Rendered Image Points')
        
        # Draw lines between corresponding points
        for p1, p2, color in zip(real_points_2d, scaled_rendered_points, colors):
            if p1[0] >200:
                plt.plot([p1[0], p2[0] + 600], [518-p1[1], p2[1]], 
                        color=color, alpha=0.7, linewidth=1)
        
        # Add vertical line to separate the two point sets
        plt.axvline(x=300, color='gray', linestyle='--', alpha=0.5)
        
        plt.legend()
        plt.title('Point Correspondence (Real vs Scaled Rendered)')
        plt.axis('equal')  # Ensure equal aspect ratio
        plt.tight_layout()
        plt.show()
    
    reprojection_error = 22
    # Use PnP-RANSAC with stricter parameters for more reliable matches
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        rendered_points_3d, 
        real_points_2d, 
        camera_matrix, 
        None,
        iterationsCount=50000,  # More iterations for better convergence
        reprojectionError=reprojection_error,  # Stricter reprojection error (was 12.0)
        confidence=0.95,  # Higher confidence threshold (was 0.95)
        flags=cv2.SOLVEPNP_EPNP,  # Using EPNP for better initialization
    )
    
    print("percentage of inliers", len(inliers) / len(real_points_2d))
    if not success:
        raise ValueError("PnP pose estimation failed")
    
    # Ensure rvec and tvec are in the correct format
    rvec = np.asarray(rvec, dtype=np.float32)
    tvec = np.asarray(tvec, dtype=np.float32)

    # Project 3D points using estimated pose
    projected_points, _ = cv2.projectPoints(rendered_points_3d, rvec, tvec, camera_matrix, None)
    projected_points = projected_points.reshape(-1, 2)

    # print the mean and max of the point distances
    point_distances = np.sqrt(np.sum((projected_points - real_points_2d)**2, axis=1))
    print(f"Mean reprojection error: {np.mean(point_distances):.2f} pixels")
    print(f"Max reprojection error: {np.max(point_distances):.2f} pixels")
    print(f"Number of RANSAC inliers: {len(inliers)}")
    print(f"Number of inliers within reprojection error: {np.sum(point_distances < reprojection_error)}")
    return rvec, tvec, inliers

def render_mesh_overlay(mesh_path, rvec, tvec, fov, image_size, original_image, camera_matrix, image_size_opencv_np):
    """Render a mesh using PyTorch3D and create an overlay with the original image.
    
    Args:
        mesh_path: Path to the OBJ file
        rvec: 3x1 rotation vector
        tvec: 3x1 translation vector
        fov: Field of view in degrees
        image_size: Size of the output image
        original_image: Original image to overlay with
        camera_matrix: Camera matrix
        image_size_opencv_np: Size of the output image in OpenCV format
    
    Returns:
        overlay: Rendered mesh overlaid on the original image
    """

    rvec_opencv_torch = torch.from_numpy(rvec).float().unsqueeze(0)[..., 0] # (1, 3)
    R_opencv_torch = so3_exp_map(rvec_opencv_torch) # (1, 3, 3)

    # 2. Convert tvec and camera_matrix to PyTorch tensors
    tvec_opencv_torch = torch.from_numpy(tvec).float().unsqueeze(0) # (1, 3)
    camera_matrix_opencv_torch = torch.from_numpy(camera_matrix).float().unsqueeze(0) # (1, 3, 3)
    image_size_torch = torch.tensor((image_size_opencv_np, image_size_opencv_np)).float().unsqueeze(0) # (1, 2)

    # 3. Use pytorch3d.utils.cameras_from_opencv_projection
    # This function handles the coordinate system conversion internally.
    cameras_pytorch3d = cameras_from_opencv_projection(
        R=R_opencv_torch,
        tvec=tvec_opencv_torch,
        camera_matrix=camera_matrix_opencv_torch,
        image_size=image_size_torch
    )
    # For PyTorch3D rendering, we'll use CPU as MPS has some issues
    render_device = torch.device("cpu")
    
    # Load the mesh
    verts, faces, aux = load_obj(
        mesh_path,
        load_textures=False
    )
    
    # Create gray vertex colors
    verts_rgb = torch.ones_like(verts.unsqueeze(0)) * 0.5  # Gray color with 0.5 intensity
    textures = TexturesVertex(verts_features=verts_rgb)
    
    # Create the mesh
    meshes = Meshes(
        verts=[verts.to(render_device)],
        faces=[faces.verts_idx.to(render_device)],
        textures=textures
    )
    
    # Set up camera using estimated pose
    cameras = FoVPerspectiveCameras(
        device=render_device,
        R=cameras_pytorch3d.R,
        T=cameras_pytorch3d.T[..., 0],
        znear=0.1,
        zfar=1000.0,
        fov=fov
    )
    
    # Set up rasterizer
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    
    # Set up lighting
    lights = PointLights(
        device=render_device,
        ambient_color=((0.8, 0.8, 0.8),),
        diffuse_color=((1.0, 1.0, 1.0),),
        specular_color=((0.3, 0.3, 0.3),),
        location=[[0, 1, 50]],  # Light slightly above camera
    )
    
    # Set up shader with transparency
    blend_params = BlendParams(
        sigma=1e-4,
        gamma=1e-4,
        background_color=(0.0, 0.0, 0.0)
    )
    
    shader = SoftPhongShader(
        device=render_device,
        cameras=cameras,
        lights=lights,
        blend_params=blend_params
    )
    
    # Set up renderer
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    
    # Render the mesh
    images = renderer(meshes)
    rendered_image = images[0, ..., :3].cpu().numpy()
    
    # # Debug: Save the rendered image without overlay
    # plt.figure(figsize=(10, 10))
    # plt.imshow(rendered_image)
    # plt.title('Debug: Rendered Mesh Only')
    # plt.axis('off')
    # plt.savefig('debug_rendered_mesh.png')
    # plt.close()
    
    # Create overlay with original image
    alpha = 0.5
    mask = (rendered_image.sum(axis=2) > 0).astype(np.float32)
    overlay = original_image.copy()
    for c in range(3):
        overlay[..., c] = (1 - mask * alpha) * original_image[..., c] + (mask * alpha) * (rendered_image[..., c] * 255)
    
    return overlay

def visualize_comparison(image1_path, render_data_path, output_folder,  model, processor, device, pca_mask, pca_rgb, filename = "pose_estimation", 
                        use_custom_preprocessing=False, use_registers=False, threshold_greater_than=False, mesh_path=None, fov=45, debug=False):
    """Process a real image and a rendered image from saved data and visualize their latent space comparison."""
    global DEBUG
    DEBUG = debug
    # Process real image
    latents1, img1, grid_shape = process_image(image1_path, model, processor, device, use_custom_preprocessing, use_registers)
    
    # Load rendered image from saved data
    print(f"Loading rendered data from {render_data_path}")
    render_data = torch.load(render_data_path)
    rendered_image = render_data['rendered_image']  # Shape: (3, H, W)
    patch_centers_3d = render_data['patch_centers_3d_coords']  # Shape: (37, 37, 3)
    
    # Convert rendered image tensor to PIL Image
    rendered_image_np = rendered_image.permute(1, 2, 0).cpu().numpy()
    rendered_image_np = np.clip(rendered_image_np, 0, 1)
    rendered_image_np = (rendered_image_np * 255).astype(np.uint8)
    rendered_image_pil = Image.fromarray(rendered_image_np)
    
    # Process rendered image using the same process_image function
    latents2, img2, grid_shape2 = process_image(rendered_image_pil, model, processor, device, use_custom_preprocessing, use_registers)
    
    # Project onto mask PCA
    mask_latents1 = pca_mask.transform(latents1)
    mask_latents2 = pca_mask.transform(latents2)
    
    # Reshape to grid dimensions
    grid_h, grid_w = grid_shape
    mask_grid1 = mask_latents1.reshape(grid_h, grid_w)
    mask_grid2 = mask_latents2.reshape(grid_h, grid_w)
    
    # Create binary masks
    if threshold_greater_than:
        binary_mask1 = (mask_grid1 > 0).astype(np.float32)
        binary_mask2 = (mask_grid2 > 0).astype(np.float32)
    else:
        binary_mask1 = (mask_grid1 < 0).astype(np.float32)
        binary_mask2 = (mask_grid2 < 0).astype(np.float32)
    
    # Find connected components
    labeled_array1, num_features1 = ndimage.label(binary_mask1)
    labeled_array2, num_features2 = ndimage.label(binary_mask2)
    
    # Get sizes of components
    component_sizes1 = np.bincount(labeled_array1.ravel())[1:]  # Skip background (0)
    component_sizes2 = np.bincount(labeled_array2.ravel())[1:]  # Skip background (0)
    
    # Find largest component
    largest_component1 = np.argmax(component_sizes1) + 1  # +1 because we skipped background
    largest_component2 = np.argmax(component_sizes2) + 1
    
    # Create masks for largest components
    largest_mask1 = (labeled_array1 == largest_component1).astype(np.float32)
    largest_mask2 = (labeled_array2 == largest_component2).astype(np.float32)
    
    # Filter embeddings using largest component masks
    filtered_latents1 = latents1[largest_mask1.reshape(-1) == 1]
    filtered_latents2 = latents2[largest_mask2.reshape(-1) == 1]
    
    # Project filtered embeddings onto RGB PCA for visualization
    rgb_latents1 = pca_rgb.transform(filtered_latents1)
    rgb_latents2 = pca_rgb.transform(filtered_latents2)
    
    # Create RGB visualizations
    rgb_grid1 = np.zeros((grid_h, grid_w, 3))
    rgb_grid2 = np.zeros((grid_h, grid_w, 3))
    
    # Normalize RGB values to [0,1] range
    rgb_min1 = rgb_latents1.min(axis=0)
    rgb_max1 = rgb_latents1.max(axis=0)
    rgb_range1 = rgb_max1 - rgb_min1
    rgb_range1[rgb_range1 == 0] = 1  # Avoid division by zero
    normalized_rgb1 = (rgb_latents1 - rgb_min1) / rgb_range1
    
    rgb_min2 = rgb_latents2.min(axis=0)
    rgb_max2 = rgb_latents2.max(axis=0)
    rgb_range2 = rgb_max2 - rgb_min2
    rgb_range2[rgb_range2 == 0] = 1  # Avoid division by zero
    normalized_rgb2 = (rgb_latents2 - rgb_min2) / rgb_range2
    
    # Fill RGB grids
    mask_indices1 = np.where(largest_mask1.reshape(-1) == 1)[0]
    mask_indices2 = np.where(largest_mask2.reshape(-1) == 1)[0]
    
    for idx, rgb in zip(mask_indices1, normalized_rgb1):
        row = idx // grid_w
        col = idx % grid_w
        rgb_grid1[row, col] = rgb
    
    for idx, rgb in zip(mask_indices2, normalized_rgb2):
        row = idx // grid_w
        col = idx % grid_w
        rgb_grid2[row, col] = rgb
    
    # Find all possible feature matches
    all_matches, match_distances = find_feature_matches(filtered_latents1, filtered_latents2, 
                                    largest_mask1, largest_mask2)
    
    
    print(f"\nTotal number of feature matches: {len(all_matches)}")
    
    
    # Extract 2D points from matches
    real_points_2d = []
    rendered_points_3d = []
    

    for idx1, idx2 in all_matches:
        # Get 2D coordinates in real image in latent space (37x37)
        y1, x1 = idx1 // grid_w, idx1 % grid_w
        real_points_2d.append([x1, y1])
        
        # Get 3D coordinates from rendered image
        y2, x2 = idx2 // grid_w, idx2 % grid_w
        rendered_points_3d.append(patch_centers_3d[y2, x2])
    
    real_points_2d = np.array(real_points_2d, dtype=np.float32)
    rendered_points_3d = np.array(rendered_points_3d, dtype=np.float32)
    
    # Filter out points with NaN values
    valid_mask = ~np.isnan(rendered_points_3d).any(axis=1)
    real_points_2d = real_points_2d[valid_mask]
    rendered_points_3d = rendered_points_3d[valid_mask]
    
    print(f"\nAfter filtering NaN values:")
    print(f"Number of valid point pairs: {len(real_points_2d)}")
    
    # Scale points from 37x37 to 518x518
    scale_factor = 518 / 37
    real_points_2d_scaled = real_points_2d * scale_factor
    
    # Create initial camera matrix for 518x518 resolution
    # Convert to PyTorch3D camera parameters
    image_size = 518
    
    # Calculate focal length from FOV
    # focal_length = (image_size/2) / tan(FOV/2)
    focal_length = (image_size/2) / np.tan(np.radians(fov/2))
    
    # Principal point at image center
    principal_point = torch.tensor([image_size/2, image_size/2], device=device)
    
    # Create camera matrix for OpenCV
    initial_camera_matrix = np.array([
        [focal_length, 0, principal_point[0].item()],
        [0, focal_length, principal_point[1].item()],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Estimate pose
    try:
        print("\nUsing PnP-RANSAC for pose estimation")
        rvec, tvec, inliers = estimate_pose(
            real_points_2d_scaled,
            rendered_points_3d, 
            initial_camera_matrix
        )
        R, _ = cv2.Rodrigues(rvec)
        camera_matrix = initial_camera_matrix
        
        # Save the results for use in render_ict-face.py
        results = {
            'rotation_matrix': R,  # 3x3 rotation matrix
            'translation_vector': tvec,  # 3x1 translation vector
            'camera_matrix': camera_matrix,  # 3x3 camera matrix
            'real_points_2d': real_points_2d_scaled,  # 2D points at 518x518 resolution
            'original_image': np.array(img1),  # Original image as numpy array
            'image_size': img1.size,  # Original image size
            'focal_length': focal_length,  # Focal length for PyTorch3D
            'principal_point': principal_point.cpu().numpy(),  # Principal point for PyTorch3D
            'fov': fov,  # Field of view in degrees
        }
        
        # Save results to a file
        output_path = Path(output_folder) / "pnp_ransac_results.pt"
        torch.save(results, output_path)
        print(f"\nSaved PnP RANSAC results to {output_path}")
        
        # Visualize the pose estimation
        plt.figure(figsize=(30, 10))
        
        # Left subplot: Original pose estimation visualization
        plt.subplot(131)
        plt.imshow(np.array(img1))
        plt.scatter(real_points_2d_scaled[:, 0], real_points_2d_scaled[:, 1], c='r', label='Real Image Points', alpha=0.3)
        plt.legend()
        plt.title('Real Image Points')
        plt.axis('off')
        
        # Middle subplot: Projected 3D points from mesh
        plt.subplot(132)
        plt.imshow(np.array(img1))
        
        # Project matched 3D points using estimated pose
        projected_points, _ = cv2.projectPoints(
            rendered_points_3d, rvec, tvec, camera_matrix, None
        )
        projected_points = projected_points.reshape(-1, 2)
        
        # Plot projected points
        plt.scatter(projected_points[:, 0], projected_points[:, 1], c='b', label='Projected Mesh Points', alpha=0.3)
        plt.legend()
        plt.title('Projected Mesh Points')
        plt.axis('off')
        
        # Right subplot: PyTorch3D rendered mesh
        if mesh_path is not None:
            overlay = render_mesh_overlay(
                mesh_path,
                rvec,
                tvec,
                fov,
                image_size,
                np.array(img1),
                camera_matrix,
                image_size
            )
            
            plt.subplot(133)
            plt.imshow(overlay)
            plt.title('PyTorch3D Rendered Mesh Overlay')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(Path(output_folder) / (str(filename) +'.png'))
        plt.close()
        
    except ValueError as e:
        print(f"Pose estimation failed: {e}")
    
    if DEBUG:
        # Create a new figure for feature matches in latent space
        plt.figure(figsize=(20, 10))
        
        # Create a single subplot that spans the full width
        plt.subplot(1, 1, 1)
        
        # Create a combined image by placing the two RGB grids side by side
        combined_grid = np.zeros((grid_h, grid_w * 2, 3))
        combined_grid[:, :grid_w] = rgb_grid1
        combined_grid[:, grid_w:] = rgb_grid2
        
        # Show the combined image
        plt.imshow(combined_grid)
        plt.title('RGB PCA Feature Matches\nLeft: Real Image, Right: Rendered Image')
        plt.axis('off')
        
        # Randomly sample 100 matches for visualization
        num_matches_to_show = min(100, len(all_matches))
        indices = np.random.choice(len(all_matches), num_matches_to_show, replace=False)
        sampled_matches = [all_matches[i] for i in indices]
        
        # Generate random colors for each line
        colors = np.random.rand(num_matches_to_show, 3)
        
        # Draw lines between matched features in latent space
        for match, color in zip(sampled_matches, colors):
            idx1, idx2 = match
            # Convert 1D indices to 2D coordinates
            y1, x1 = idx1 // grid_w, idx1 % grid_w
            y2, x2 = idx2 // grid_w, idx2 % grid_w
            
            # Draw line (x2 is offset by grid_w since it's in the second image)
            plt.plot([x1, x2 + grid_w], [y1, y2], 
                    color=color, alpha=0.9, linewidth=5)
        
        # Add vertical line to separate the two images
        plt.axvline(x=grid_w, color='white', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(Path(output_folder) / "feature_matches_comparison.png")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare a real image with a rendered image using DinoV2 and PCA projection')
    parser.add_argument('--image1', type=str, required=True, help='Path to real image')
    parser.add_argument('--pca_model', type=str, default='./saved_pcas/HH_PCA_GMM_v2.pt', help='Path to pre-trained PCA model')
    parser.add_argument('--use_custom_preprocessing', action='store_true', default=True,
                      help='Use custom preprocessing (padding to square and resizing to 518x518)')
    parser.add_argument('--use_registers', action='store_true', default=True,
                      help='Use DinoV2 base with registers instead of DinoV2 base')
    parser.add_argument('--threshold_greater_than', action='store_true', default=False,
                      help='Threshold for values greater than 0 instead of less than 0')
    parser.add_argument('--mesh', type=str, default='./data/FFHQ_UV_sample/hack_w_HIFI3D_UV.obj', help='Path to the mesh OBJ file')
    parser.add_argument('--fov', type=float, default=45, help='Field of view in degrees')
    parser.add_argument('--render_data_folder', type=str, required=True, help='Path to folder containing saved render_data.pt files')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    args = parser.parse_args()
    
    global DEBUG
    if args.debug:
        DEBUG = True
    else:
        DEBUG = False
    # Load model and processor
    print("Loading DinoV2 model and processor...")
    model, processor, device = load_model_and_processor(args.use_registers)
    
    # Load PCA models
    print(f"Loading PCA models from {args.pca_model}")
    pca_mask, pca_rgb, kmeans = load_pca_models(args.pca_model)
    
    render_data = Path(args.render_data_folder) / "template_data.pt"
    # Process and visualize images
    print("Processing and visualizing images...")
    visualize_comparison(args.image1, render_data, args.render_data_folder, model, processor, device, pca_mask, pca_rgb,
                        args.use_custom_preprocessing, args.use_registers, args.threshold_greater_than,
                        args.mesh, args.fov, args.debug)

if __name__ == "__main__":
    main() 