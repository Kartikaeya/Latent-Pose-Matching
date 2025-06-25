import torch
import numpy as np
import argparse
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    Materials,
    look_at_view_transform,
    FoVPerspectiveCameras,
    TexturesUV,
)
from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt
from pytorch3d.renderer import BlendParams
import torchvision.transforms as T
import torch.nn.functional as F
from latent_pca_video import load_model_and_processor, process_image, load_pca_models
from PIL import Image
from pathlib import Path
import json
from collections import Counter, defaultdict

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# Set blend parameters for transparent background
blend_params = BlendParams(
    sigma=5e-3,  # Higher sigma for smoother edges
    gamma=5e-3,  # Higher gamma for brighter blending
    background_color=(0.0, 0.0, 0.0),  # Black background (3 channels)
)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Render 3D face model with texture')
parser.add_argument('--obj', type=str, 
                    default='./data/DNEGSynFace_topology/generic_neutral_mesh.obj',
                    help='Path to the OBJ file')
parser.add_argument('--fov', type=float, default=12, 
                    help='Field of view for the camera')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Debug mode')
parser.add_argument('--output_dir', type=str, default='./output/templates',
                    help='Base output directory for all results')
args = parser.parse_args()

if args.debug:
    DEBUG = True
else:
    DEBUG = False

# Create output directory structure
output_dir = Path(args.output_dir) if not DEBUG else Path('./output/templates_debug')
viz_dir = output_dir / 'visualizations'
data_dir = output_dir / 'data'
viz_2d_dir = viz_dir / '2d'
viz_3d_dir = viz_dir / '3d'

# Create all necessary directories
for dir_path in [viz_2d_dir, viz_3d_dir, data_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Load the mesh
mesh = load_objs_as_meshes([args.obj], device=device)

# Define a Y-threshold in world coordinates to clip the mesh from below
y_threshold_world = -8.5  # You can adjust this value

# Get original vertices, faces, and textures
verts = mesh.verts_list()[0]
faces = mesh.faces_list()[0]
textures = mesh.textures

# Create a mask for vertices above the threshold
vert_mask = verts[:, 1] > y_threshold_world
new_verts = verts[vert_mask]

if new_verts.shape[0] > 0 and new_verts.shape[0] < verts.shape[0]:
    # Create a mapping from old vertex indices to new vertex indices
    old_to_new_vert_indices = -torch.ones(verts.shape[0], dtype=torch.long, device=device)
    old_to_new_vert_indices[vert_mask] = torch.arange(new_verts.shape[0], device=device)

    # Filter faces: keep only faces where all 3 vertices are in the new set of vertices
    face_vert_mask = vert_mask[faces]
    valid_faces_mask = face_vert_mask.all(dim=1)
    new_faces = old_to_new_vert_indices[faces[valid_faces_mask]]
    print(f"Number of faces kept: {new_faces.shape[0]} / {faces.shape[0]}")

    # Filter texture coordinates corresponding to the filtered faces
    faces_uvs = textures.faces_uvs_list()[0]
    new_faces_uvs = faces_uvs[valid_faces_mask]

    # Create a new TexturesUV object
    new_textures = TexturesUV(
        maps=textures.maps_padded(),
        verts_uvs=textures.verts_uvs_list(),
        faces_uvs=[new_faces_uvs]
    )

    # Create the new filtered mesh object
    meshes = Meshes(
        verts=[new_verts],
        faces=[new_faces],
        textures=new_textures
    )
elif new_verts.shape[0] == verts.shape[0]:
    print("No vertices were filtered out.")
    meshes = Meshes(
        verts=[mesh.verts_list()[0].to(device)],
        faces=[mesh.faces_list()[0].to(device)],
        textures=mesh.textures
    )
else:
    print("Warning: All vertices were filtered out. Rendering an empty mesh.")
    # Create an empty mesh to avoid crashing
    meshes = Meshes(
        verts=[torch.empty((0, 3), device=device)],
        faces=[torch.empty((0, 3), dtype=torch.long, device=device)],
        textures=None
    )

# Get material properties from the loaded mesh if available
if hasattr(mesh, 'materials'):
    print("Materials found")
    materials = mesh.materials
else:
    # Default material properties if not available from MTL
    materials = Materials(
        device=device,
        specular_color=((0.2, 0.2, 0.2),),
        shininess=10.0,
        ambient_color=((0.5, 0.5, 0.5),)
    )

# Define camera parameters
heights = [0, 30, -30, 45, -45]  # Heights in degrees
rotations = np.linspace(0, 330, 12)  # 12 rotations around Y axis
distance = 150  # Distance from origin

# Load DinoV2 model and processor
print("Loading DinoV2 model and processor...")
model, processor, pca_device = load_model_and_processor(use_registers=True)


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

# Initialize dictionary to store cluster labels for each template
template_cluster_labels = {}

# Process each viewpoint
for height in heights:
    for rotation in rotations:
        print(f"\nProcessing height: {height}°, rotation: {rotation}°")
        
        # Calculate camera position
        rad_height = np.radians(height)
        rad_rotation = np.radians(rotation)
        
        # Convert spherical to Cartesian coordinates
        x = distance * np.cos(rad_height) * np.sin(rad_rotation)
        y = distance * np.sin(rad_height)
        z = distance * np.cos(rad_height) * np.cos(rad_rotation)
        
        # Set up camera
        R, T = look_at_view_transform(
            eye=((x, y, z),),
            at=((0, 0, 0),),
            up=((0, 1, 0),),
            device=device
        )
        
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.1, zfar=1000.0, fov=args.fov)
        
        raster_settings = RasterizationSettings(
            image_size=518,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        
        lights = PointLights(
            device=device,
            ambient_color=((0.8, 0.8, 0.8),),
            diffuse_color=((0.7, 0.7, 0.7),),
            specular_color=((0.2, 0.2, 0.2),),
            location=[[0, 0, 100]],
        )
        
        shader = SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials,
            blend_params=blend_params
        )
        
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        
        # Render the mesh
        meshes = meshes.to(device)
        fragments = rasterizer(meshes_world=meshes)
        images = shader(fragments, meshes, lights=lights)
        
        # Extract Z-depth map and rendered image
        z_depth_map = fragments.zbuf[0, ..., 0].detach().clone()
        rendered_image = images[0, ..., :3].cpu().numpy()
        rendered_image = np.clip(rendered_image, 0, 1)
        
        # Calculate 3D coordinates for patch centers
        num_patches = 37
        image_render_size_h, image_render_size_w = 518, 518
        
        # Create grid of patch indices
        patch_indices_y, patch_indices_x = torch.meshgrid(
            torch.arange(num_patches, device=device),
            torch.arange(num_patches, device=device),
            indexing='ij'
        )
        
        # Calculate patch centers
        patch_size_h = image_render_size_h / num_patches
        patch_size_w = image_render_size_w / num_patches
        x_centers = (patch_indices_x.float() + 0.5) * patch_size_w
        y_centers = (patch_indices_y.float() + 0.5) * patch_size_h
        
        # Sample Z-depth and convert to 3D coordinates
        z_depth_map_for_sample = z_depth_map.unsqueeze(0).unsqueeze(0)
        norm_x = (x_centers / (image_render_size_w - 1.0)) * 2.0 - 1.0
        norm_y = (y_centers / (image_render_size_h - 1.0)) * 2.0 - 1.0
        sampling_grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)
        
        sampled_depths = F.grid_sample(
            z_depth_map_for_sample,
            sampling_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).squeeze()
        
        depth_mask = ~(torch.abs(sampled_depths - (-1)) < 1e-2)
        
        if DEBUG:
            plt.figure(figsize=(8, 8))
            plt.imshow(sampled_depths.cpu().numpy(), cmap='viridis')
            plt.colorbar(label='Z-depth')
            plt.title('Sampled Z-depths')
            plt.axis('off')
            plt.show()
        
        # Convert to 3D world coordinates
        ndc_x = (x_centers / (image_render_size_w - 1.0)) * 2.0 - 1.0
        ndc_y = 1.0 - (y_centers / (image_render_size_h - 1.0)) * 2.0
        
        xy_depth_flat = torch.stack((
            ndc_x.reshape(-1),
            ndc_y.reshape(-1),
            sampled_depths.reshape(-1)
        ), dim=-1)
        
        world_coords_flat = cameras.unproject_points(xy_depth_flat, world_coordinates=True)
        patch_centers_3d_coords = world_coords_flat.reshape(num_patches, num_patches, 3).detach().clone()
        
        # Rotate back the patch_centers_3d_coords based on camera rotation about Y-axis
        rad_rotation = np.radians(-2*rotation)
        rotation_matrix = torch.tensor([
            [np.cos(rad_rotation), 0, np.sin(rad_rotation)],
            [0, 1, 0],
            [-np.sin(rad_rotation), 0, np.cos(rad_rotation)]
        ], device=device, dtype=torch.float32)
        patch_centers_3d_coords = patch_centers_3d_coords @ rotation_matrix.T
        
        # Apply depth mask to patch centers
        patch_centers_3d_coords[~depth_mask] = float('nan')
        
        # flip x coordinates to match the original mesh orientation
        patch_centers_3d_coords[..., 0] *= -1
        
        # Convert rendered image to PIL Image for DinoV2 processing
        rendered_image_pil = Image.fromarray((rendered_image * 255).astype(np.uint8))
        
        # Process through DinoV2
        token_embeddings, processed_image, grid_shape = process_image(
            rendered_image_pil,
            model,
            processor,
            pca_device,
            use_custom_preprocessing=True,
            use_registers=True
        )
        
        # Project embeddings through PCA
        compressed_latents = pca_mask.transform(token_embeddings)
        latent_grid = compressed_latents.reshape(grid_shape)
        
        # Create binary mask based on threshold direction
        threshold_greater_than = False
        if threshold_greater_than:
            binary_mask = (latent_grid > 0).astype(np.float32)
        else:
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
            
            # Store cluster labels for this template
            template_key = f'template_data_h{height}_r{int(rotation)}.pt'
            template_cluster_labels[template_key] = cluster_labels
            
            # Create cluster visualization
            cluster_grid = np.zeros((grid_shape[0], grid_shape[1], 3))
            
            for idx, (rgb, label) in zip(mask_indices, zip(normalized_rgb_latents, cluster_labels)):
                row = idx // grid_shape[1]
                col = idx % grid_shape[1]
                rgb_grid[row, col] = rgb
                cluster_grid[row, col] = cluster_colors[label % len(cluster_colors)]
        else:
            rgb_grid = np.zeros((grid_shape[0], grid_shape[1], 3))
            cluster_grid = np.zeros((grid_shape[0], grid_shape[1], 3))
            # Store empty cluster labels for this template
            template_key = f'template_data_h{height}_r{int(rotation)}.pt'
            template_cluster_labels[template_key] = np.array([])
        
        # Create visualization figure
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(rendered_image)
        plt.title('Rendered Image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(rgb_grid)
        plt.title('PCA Latent (3-channel)')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(cluster_grid)
        plt.title('GMM Clusters')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save 2D visualization
        viz_2d_path = viz_2d_dir / f'template_viz_h{height}_r{int(rotation)}.png'
        plt.savefig(viz_2d_path)
        if DEBUG:
            plt.show()
        else:
            plt.close()
        
        # Create and save 3D visualization
        plt.figure(figsize=(8, 8))
        ax = plt.axes(projection='3d')
        
        x_coords = patch_centers_3d_coords[..., 0].cpu().numpy().flatten()
        y_coords = patch_centers_3d_coords[..., 1].cpu().numpy().flatten()
        z_coords = patch_centers_3d_coords[..., 2].cpu().numpy().flatten()
        
        valid_mask = ~np.isnan(x_coords)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        z_coords = z_coords[valid_mask]

        scatter = ax.scatter(
            x_coords,
            y_coords,
            z_coords,
            c=y_coords,
            cmap='viridis',
            s=2
        )
        
        ax.set_box_aspect([1, 1, 1])
        
        max_range = np.array([
            x_coords.max() - x_coords.min(),
            y_coords.max() - y_coords.min(),
            z_coords.max() - z_coords.min()
        ]).max() / 2.0
        
        mid_x = (x_coords.max() + x_coords.min()) * 0.5
        mid_y = (y_coords.max() + y_coords.min()) * 0.5
        mid_z = (z_coords.max() + z_coords.min()) * 0.5
        
        ax.set_xlim(-12, 12)
        ax.set_ylim(-8, 18)
        ax.set_zlim(-10, 15)
        
        plt.colorbar(scatter, label='Y coordinate')
        ax.set_title('3D Patch Centers')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Save 3D visualization
        viz_3d_path = viz_3d_dir / f'template_3d_h{height}_r{int(rotation)}.png'
        plt.savefig(viz_3d_path)
        if DEBUG:
            plt.show()
        else:
            plt.close()
        
        # Store data
        view_data = {
            'rendered_image': torch.from_numpy(rendered_image).permute(2, 0, 1),
            'patch_centers_3d_coords': patch_centers_3d_coords.cpu(),
            'patch_centers_z_depth': sampled_depths.cpu(),
            'dino_embeddings': torch.from_numpy(token_embeddings),
            'pca_mask_embeddings': torch.from_numpy(compressed_latents),
            'pca_rgb_embeddings': torch.from_numpy(rgb_latents) if len(filtered_embeddings) > 0 else None,
            'grid_shape': grid_shape,
            'binary_mask': torch.from_numpy(binary_mask),
            'camera_params': {
                'height': height,
                'rotation': rotation,
                'distance': distance
            }
        }
        
        # Save data
        data_path = data_dir / f'template_data_h{height}_r{int(rotation)}.pt'
        torch.save(view_data, data_path)
        
        print(f"Saved data for height {height}°, rotation {rotation}°")

# Calculate bag-of-words descriptors after processing all templates
print("\nCalculating bag-of-words descriptors...")
N = len(template_cluster_labels)  # total number of templates

# Calculate n_i (number of occurrences of each cluster across all templates)
cluster_occurrences = defaultdict(int)
for labels in template_cluster_labels.values():
    for label in labels:
        cluster_occurrences[label] += 1

# Calculate descriptors for each template
all_template_descriptors = {}
for template_key, labels in template_cluster_labels.items():
    if len(labels) > 0:
        # Count occurrences of each cluster in this template
        cluster_counts = Counter(labels)
        n_t = len(labels)  # total number of words in this template
        
        # Initialize descriptor vector
        template_descriptor = np.zeros(kmeans.n_components)
        
        # Calculate weighted word frequencies
        for cluster_id, count in cluster_counts.items():
            n_i_t = count  # number of occurrences of word i in template t
            n_i = cluster_occurrences[cluster_id]  # number of occurrences of word i in all templates
            
            # Calculate TF-IDF like weight
            if n_t > 0 and n_i > 0:
                template_descriptor[cluster_id] = (n_i_t / n_t) * np.log(N / n_i)
    else:
        template_descriptor = np.zeros(kmeans.n_components)
    
    all_template_descriptors[template_key] = template_descriptor

# Update the saved data with descriptors
for template_key, descriptor in all_template_descriptors.items():
    data_path = data_dir / template_key
    view_data = torch.load(data_path)
    view_data['bag_of_words_descriptor'] = descriptor
    torch.save(view_data, data_path)

# Save all template descriptors
descriptors_path = data_dir / 'template_descriptors.json'
serializable_descriptors = {}
with open(descriptors_path, 'w') as f:
    # Convert numpy arrays to lists for JSON serialization
    serializable_descriptors['bag_of_words_descriptors'] = {
        k: v.tolist() for k, v in all_template_descriptors.items()
    }
    serializable_descriptors['num_templates'] = int(N)
    serializable_descriptors['cluster_occurrences'] = {int(k): int(v) for k, v in cluster_occurrences.items()}
    json.dump(serializable_descriptors, f, indent=2)

print(f"Saved all template descriptors to {descriptors_path}")