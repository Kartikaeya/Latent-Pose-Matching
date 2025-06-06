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
args = parser.parse_args()

# Load the mesh
mesh = load_objs_as_meshes([args.obj], device=device)

# Define a Y-threshold in world coordinates to clip the mesh from below
y_threshold_world = -8.5  # You can adjust this value

# Get original vertices, faces, and textures
verts = mesh.verts_list()[0]
faces = mesh.faces_list()[0]
textures = mesh.textures

# Print vertex y-range for reference
print(f"Original Y-vertex range: [{verts[:, 1].min():.2f}, {verts[:, 1].max():.2f}]")
print(f"Filtering vertices with Y > {y_threshold_world}")

# Create a mask for vertices above the threshold
vert_mask = verts[:, 1] > y_threshold_world
new_verts = verts[vert_mask]
print(f"Number of vertices kept: {new_verts.shape[0]} / {verts.shape[0]}")

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

# Set up camera for right-facing view
R, T = look_at_view_transform(
    eye=((0, 0, 150),),  # Camera at (0, 0, 150) - front view
    at=((0, 0, 0),),    # Looking at origin
    up=((0, 1, 0),),    # Y-axis up
    device=device
)

cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.1, zfar=1000.0, fov=args.fov)

raster_settings = RasterizationSettings(
    image_size=518,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Set up lighting with adjusted parameters
lights = PointLights(
    device=device,
    ambient_color=((0.8, 0.8, 0.8),),  # Increased from 0.3 to 0.5
    diffuse_color=((0.7, 0.7, 0.7),),  # Kept the same
    specular_color=((0.2, 0.2, 0.2),),  # Kept the same
    location=[[0, 0, 100]],  # Light in front of the face
)

shader = SoftPhongShader(
    device=device,
    cameras=cameras,
    lights=lights,
    materials=materials,  # Use the materials from MTL
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
patch_size_h = image_render_size_h / num_patches        #14
patch_size_w = image_render_size_w / num_patches        #14
x_centers = (patch_indices_x.float() + 0.5) * patch_size_w        #7, 21, 35...
y_centers = (patch_indices_y.float() + 0.5) * patch_size_h        #7, 21, 35...

# Sample Z-depth and convert to 3D coordinates
z_depth_map_for_sample = z_depth_map.unsqueeze(0).unsqueeze(0)  # Use masked depth map
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

# Visualize sampled depths
plt.figure(figsize=(8, 8))
plt.imshow(sampled_depths.cpu().numpy(), cmap='viridis')
plt.colorbar(label='Z-depth')
plt.title('Sampled Z-depths')
plt.axis('off')
plt.show()

depth_mask = ~(torch.abs(sampled_depths - (-1)) < 1e-3)
print("valid depth patches:",depth_mask.sum())


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

# Apply depth mask to patch centers
patch_centers_3d_coords[~depth_mask] = float('nan')

# Create 2D visualization
plt.figure(figsize=(15, 5))

# 1. Rendered Image with Mask
plt.subplot(131)
plt.imshow(rendered_image)
plt.title('Rendered Image\n(Masked)')
plt.axis('off')

# 2. Z-depth Map
plt.subplot(132)
z_depth_vis = z_depth_map.cpu().numpy()
z_depth_vis[z_depth_vis > 0] = np.log(z_depth_vis[z_depth_vis > 0])
plt.imshow(z_depth_vis, cmap='viridis')
plt.colorbar(label='log(Z-depth)')
plt.title('Z-depth Map\n(Masked)')
plt.axis('off')

# 3. 3D Coordinates as RGB
plt.subplot(133)
# Normalize coordinates to [0,1] range for visualization
coords_rgb = patch_centers_3d_coords.cpu().numpy()
coords_rgb = np.nan_to_num(coords_rgb, nan=0)  # Replace NaN with 0

# Normalize each channel independently
for i in range(3):
    valid_mask = coords_rgb[..., i] != 0
    if valid_mask.any():
        min_val = coords_rgb[..., i][valid_mask].min()
        max_val = coords_rgb[..., i][valid_mask].max()
        if max_val > min_val:
            coords_rgb[..., i] = (coords_rgb[..., i] - min_val) / (max_val - min_val)

plt.imshow(coords_rgb)
plt.title('3D Coordinates\n(R=X, G=Y, B=Z)')
plt.axis('off')

plt.tight_layout()
plt.show()

# Create 3D visualization in a separate figure
plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')

# Get the coordinates
x_coords = patch_centers_3d_coords[..., 0].cpu().numpy().flatten()
y_coords = patch_centers_3d_coords[..., 1].cpu().numpy().flatten()
z_coords = patch_centers_3d_coords[..., 2].cpu().numpy().flatten()

# Remove NaN values
valid_mask = ~np.isnan(x_coords)

x_coords = x_coords[valid_mask]
y_coords = y_coords[valid_mask]
z_coords = z_coords[valid_mask]
print("valid 3d patches:",len(x_coords))
scatter = ax.scatter(
    x_coords,
    y_coords,
    z_coords,
    c=y_coords,
    cmap='viridis',
    s=2
)

# Set equal aspect ratio for all axes
ax.set_box_aspect([1, 1, 1])

# Set axis limits to be symmetric around the center
max_range = np.array([
    x_coords.max() - x_coords.min(),
    y_coords.max() - y_coords.min(),
    z_coords.max() - z_coords.min()
]).max() / 2.0

mid_x = (x_coords.max() + x_coords.min()) * 0.5
mid_y = (y_coords.max() + y_coords.min()) * 0.5
mid_z = (z_coords.max() + z_coords.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.colorbar(scatter, label='Y coordinate')
ax.set_title('3D Patch Centers')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Print coordinate ranges for debugging
print(f"X range: [{x_coords.min():.2f}, {x_coords.max():.2f}]")
print(f"Y range: [{y_coords.min():.2f}, {y_coords.max():.2f}]")
print(f"Z range: [{z_coords.min():.2f}, {z_coords.max():.2f}]")

plt.show()

# After viewing, save the visualizations
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(rendered_image)
plt.title('Rendered Image\n(Masked)')
plt.axis('off')

plt.subplot(132)
plt.imshow(z_depth_vis, cmap='viridis')
plt.colorbar(label='log(Z-depth)')
plt.title('Z-depth Map\n(Masked)')
plt.axis('off')

plt.subplot(133)
plt.imshow(coords_rgb)
plt.title('3D Coordinates\n(R=X, G=Y, B=Z)')
plt.axis('off')

plt.tight_layout()
plt.savefig('./output/single_image_fitting/left/template_viz.png')
plt.close()

# Save 3D visualization
plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
scatter = ax.scatter(
    x_coords,
    y_coords,
    z_coords,
    c=y_coords,
    cmap='viridis',
    s=2
)
ax.set_box_aspect([1, 1, 1])
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
plt.colorbar(scatter, label='Y coordinate')
ax.set_title('3D Patch Centers')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('./output/single_image_fitting/left/template_3d.png')
plt.close()

# Convert rendered image to PIL Image for DinoV2 processing
rendered_image_pil = Image.fromarray((rendered_image * 255).astype(np.uint8))

# Load DinoV2 model and processor with default settings from process_image_sequence
print("Loading DinoV2 model and processor...")
model, processor, device = load_model_and_processor(use_registers=True)

# Process the rendered image through DinoV2
print("Processing rendered image through DinoV2...")
token_embeddings, processed_image, grid_shape = process_image(
    rendered_image_pil,
    model,
    processor,
    device,
    use_custom_preprocessing=True,
    use_registers=True
)

# Load PCA models
print("Loading PCA models...")
pca_model_path = './saved_pcas/HH_PCA_GMM_v2.pt'  # Default from process_image_sequence
pca_mask, pca_rgb, kmeans = load_pca_models(pca_model_path)

# Project embeddings through PCA
print("Projecting embeddings through PCA...")
# First through mask PCA
compressed_latents = pca_mask.transform(token_embeddings)
# Reshape to grid shape
latent_grid = compressed_latents.reshape(grid_shape)

# Create binary mask based on threshold direction (using default from process_image_sequence)
threshold_greater_than = False
if threshold_greater_than:
    binary_mask = (latent_grid > 0).astype(np.float32)
else:
    binary_mask = (latent_grid < 0).astype(np.float32)

# Filter embeddings using mask
filtered_embeddings = token_embeddings[binary_mask.reshape(-1) == 1]

if len(filtered_embeddings) > 0:
    # Project onto RGB PCA
    rgb_latents = pca_rgb.transform(filtered_embeddings)
    
    # Create RGB visualization
    rgb_grid = np.zeros((grid_shape[0], grid_shape[1], 3))
    mask_indices = np.where(binary_mask.reshape(-1) == 1)[0]
    
    # Normalize RGB values to [0,1] range for visualization
    rgb_min = rgb_latents.min(axis=0)
    rgb_max = rgb_latents.max(axis=0)
    rgb_range = rgb_max - rgb_min
    rgb_range[rgb_range == 0] = 1  # Avoid division by zero
    normalized_rgb_latents = (rgb_latents - rgb_min) / rgb_range
    
    for idx, rgb in zip(mask_indices, normalized_rgb_latents):
        row = idx // grid_shape[1]
        col = idx % grid_shape[1]
        rgb_grid[row, col] = rgb
else:
    rgb_grid = np.zeros((grid_shape[0], grid_shape[1], 3))

# Create visualization figure for PCA results
plt.figure(figsize=(15, 5))

# 1. Original rendered image
plt.subplot(131)
plt.imshow(rendered_image)
plt.title('Rendered Image\n(Masked)')
plt.axis('off')

# 2. Latent grid visualization
plt.subplot(132)
plt.imshow(latent_grid, cmap='viridis')
plt.colorbar(label='Compressed Latent Value')
plt.title('Compressed Latent Grid')
plt.axis('off')

# 3. RGB PCA visualization
plt.subplot(133)
plt.imshow(rgb_grid)
plt.title('RGB PCA Projection')
plt.axis('off')

plt.tight_layout()
plt.show()

# Store data with both original and PCA embeddings
view_data = {
    'rendered_image': torch.from_numpy(rendered_image).permute(2, 0, 1),
    'patch_centers_3d_coords': patch_centers_3d_coords.cpu(),
    'patch_centers_z_depth': sampled_depths.cpu(),
    'dino_embeddings': torch.from_numpy(token_embeddings),  # Original high-dimensional embeddings
    'pca_mask_embeddings': torch.from_numpy(compressed_latents),  # 1D PCA embeddings
    'pca_rgb_embeddings': torch.from_numpy(rgb_latents) if len(filtered_embeddings) > 0 else None,  # 3D PCA embeddings
    'grid_shape': grid_shape,  # Store grid shape for reference
    'binary_mask': torch.from_numpy(binary_mask)  # Store the binary mask
}

# Save data
output_pt_file = './output/single_image_fitting/left/template_data.pt'
torch.save(view_data, output_pt_file)
print(f"\nData saved to {output_pt_file}")

# Print some information about the embeddings
print(f"\nDinoV2 Embeddings shape: {token_embeddings.shape}")
print(f"PCA Mask Embeddings shape: {compressed_latents.shape}")
if len(filtered_embeddings) > 0:
    print(f"PCA RGB Embeddings shape: {rgb_latents.shape}")
print(f"Grid shape: {grid_shape}")