import os
import torch
import numpy as np
import argparse
from PIL import Image
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    PointLights,
    Materials,
    look_at_view_transform,
    FoVPerspectiveCameras,
)
from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt
from pytorch3d.renderer import BlendParams
import torchvision.transforms as T
import torch.nn.functional as F

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
                    default=os.path.join("data", "FFHQ_UV_sample", "hack_w_HIFI3D_UV.obj"),
                    help='Path to the OBJ file (default: ICT-FaceKit/FaceXModel/generic_neutral_mesh.obj)')
parser.add_argument('--texture', type=str, default=os.path.join("data", "FFHQ_UV_sample", "000000.png"), 
                    help='Path to texture image (PNG)')
parser.add_argument('--fov', type=float, default=15, 
                    help='Field of view for the camera')
args = parser.parse_args()

# Load the 3D model
obj_file = args.obj

# Load the mesh
verts, faces, aux = load_obj(
    obj_file,
    load_textures=False  # We'll handle textures manually
)

# Create a mask for vertices above Y=-5
vertex_mask = verts[:, 1] > -9

# Filter vertices and create new vertex indices
valid_verts = verts[vertex_mask]
old_to_new_idx = torch.zeros(len(verts), dtype=torch.long)
old_to_new_idx[vertex_mask] = torch.arange(valid_verts.shape[0])

# Filter faces to keep only those where all vertices are above Y=-5
valid_faces = []
valid_faces_uvs = []
for face_idx, face in enumerate(faces.verts_idx):
    if all(vertex_mask[face]):
        # Update vertex indices to new numbering
        new_face = old_to_new_idx[face]
        valid_faces.append(new_face)
        if hasattr(faces, 'textures_idx'):
            # Get the UV face indices for this face
            uv_face = faces.textures_idx[face_idx]
            # Check if all UV vertices are valid
            if all(uv_face < len(aux.verts_uvs)):
                valid_faces_uvs.append(uv_face)

valid_faces = torch.stack(valid_faces) if valid_faces else torch.zeros((0, 3), dtype=torch.long)
valid_faces_uvs = torch.stack(valid_faces_uvs) if valid_faces_uvs else torch.zeros((0, 3), dtype=torch.long)

# Create textures if provided
if args.texture and os.path.exists(args.texture):
    # Load the texture image
    texture_image = Image.open(args.texture).convert('RGB')
    # Convert PIL image to numpy array and then to tensor
    texture_array = np.asarray(texture_image, dtype=np.float32) / 255.0
    texture_tensor = torch.from_numpy(texture_array).to(device)
    texture_tensor = texture_tensor.unsqueeze(0)  # Add batch dimension
    
    # Get UV coordinates for valid vertices
    if hasattr(aux, 'verts_uvs'):
        # Create a mask for valid UV vertices
        uv_vertex_mask = torch.zeros(len(aux.verts_uvs), dtype=torch.bool)
        for face_uv in valid_faces_uvs:
            uv_vertex_mask[face_uv] = True
        
        # Filter UV vertices and create new UV vertex indices
        valid_verts_uvs = aux.verts_uvs[uv_vertex_mask]
        old_to_new_uv_idx = torch.zeros(len(aux.verts_uvs), dtype=torch.long)
        old_to_new_uv_idx[uv_vertex_mask] = torch.arange(valid_verts_uvs.shape[0])
        
        # Update UV face indices to use new UV vertex numbering
        valid_faces_uvs = torch.stack([old_to_new_uv_idx[face_uv] for face_uv in valid_faces_uvs])
        
        textures = TexturesUV(
            maps=texture_tensor,
            faces_uvs=valid_faces_uvs.unsqueeze(0).to(device),
            verts_uvs=valid_verts_uvs.unsqueeze(0).to(device)
        )
    else:
        # If no UV coordinates, fall back to vertex colors
        verts_rgb = torch.ones_like(valid_verts.unsqueeze(0)) * 0.5  # Gray color with 0.5 intensity
        textures = TexturesVertex(verts_features=verts_rgb)
else:
    # Fallback to vertex colors if no texture
    verts_rgb = torch.ones_like(valid_verts.unsqueeze(0)) * 0.5  # Gray color with 0.5 intensity
    textures = TexturesVertex(verts_features=verts_rgb)

# Create the mesh with only vertices above Y=-5
meshes = Meshes(
    verts=[valid_verts.to(device)],
    faces=[valid_faces.to(device)],
    textures=textures
)

# Point light materials with maximum reflectivity
materials = Materials(
    device=device,
    specular_color=((0.2, 0.2, 0.2),),
    shininess=10.0,
    ambient_color=((0.5, 0.5, 0.5),)
)

# Set up camera for forward-facing view
R, T = look_at_view_transform(
    eye=((0, 0, 150),),  # Camera at (0, 0, 150)
    at=((0, 1.0, 0),),    # Looking at origin
    up=((0, 1, 0),),    # Y-axis up
    device=device
)

cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.1, zfar=1000.0, fov=args.fov)

raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Set up lighting
lights = PointLights(
    device=device,
    ambient_color=((0.8, 0.8, 0.8),),
    diffuse_color=((1.0, 1.0, 1.0),),
    specular_color=((0.3, 0.3, 0.3),),
    location=[[0, 1, 50]],  # Light slightly above camera
)

shader = SoftPhongShader(
    device=device,
    cameras=cameras,
    lights=lights,
    blend_params=blend_params
)

rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

# Render the mesh
meshes = meshes.to(device)
fragments = rasterizer(meshes_world=meshes)
images = shader(fragments, meshes, lights=lights)

# Extract Z-depth map
z_depth_map = fragments.zbuf[0, ..., 0].detach().clone()

# Convert rendered image to numpy for visualization
rendered_image = images[0, ..., :3].cpu().numpy()
rendered_image = np.clip(rendered_image, 0, 1)

# --- Calculate 3D coordinates for 37x37 patch centers ---
num_patches = 37
image_render_size_h, image_render_size_w = 512, 512

# Create grid of patch indices
patch_indices_y, patch_indices_x = torch.meshgrid(
    torch.arange(num_patches, device=device),
    torch.arange(num_patches, device=device),
    indexing='ij'
)

# Calculate patch centers in rendered image space
patch_size_h = image_render_size_h / num_patches
patch_size_w = image_render_size_w / num_patches
x_centers = (patch_indices_x.float() + 0.5) * patch_size_w
y_centers = (patch_indices_y.float() + 0.5) * patch_size_h

# Sample Z-depth at patch centers
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

# Unproject to 3D world coordinates
ndc_x = (x_centers / (image_render_size_w - 1.0)) * 2.0 - 1.0
ndc_y = 1.0 - (y_centers / (image_render_size_h - 1.0)) * 2.0

xy_depth_flat = torch.stack((
    ndc_x.reshape(-1),
    ndc_y.reshape(-1),
    sampled_depths.reshape(-1)
), dim=-1)

world_coords_flat = cameras.unproject_points(xy_depth_flat, world_coordinates=True)
patch_centers_3d_coords = world_coords_flat.reshape(num_patches, num_patches, 3).detach().clone()

# Handle invalid depths
invalid_depth_mask = (sampled_depths <= 0.0) | (sampled_depths >= cameras.zfar[0])
patch_centers_3d_coords[invalid_depth_mask] = torch.tensor([float('nan')], device=device)

# --- Visualize results ---
plt.figure(figsize=(15, 5))

# 1. Rendered Image
plt.subplot(131)
plt.imshow(rendered_image)
plt.title('Rendered Image\nForward View')
plt.axis('off')

# 2. Z-depth Map
plt.subplot(132)
z_depth_vis = z_depth_map.cpu().numpy()
z_depth_vis[z_depth_vis > 0] = np.log(z_depth_vis[z_depth_vis > 0])  # Log scale for better visualization
plt.imshow(z_depth_vis, cmap='viridis')
plt.colorbar(label='log(Z-depth)')
plt.title('Z-depth Map')
plt.axis('off')

# 3. 3D Coordinates Visualization (using X coordinate as example)
plt.subplot(133)
x_coords = patch_centers_3d_coords[..., 0].cpu().numpy()
x_coords[invalid_depth_mask.cpu()] = np.nan
plt.imshow(x_coords, cmap='coolwarm')
plt.colorbar(label='X coordinate')
plt.title('3D X Coordinates')
plt.axis('off')

plt.tight_layout()
plt.savefig('./output/single_image_fitting/forward/template_viz_forward.png')
plt.close()

# --- Store data ---
view_data = {
    'rendered_image': torch.from_numpy(rendered_image).permute(2, 0, 1),  # (3, H, W)
    'z_depth_map': z_depth_map.cpu(),  # (H, W)
    'patch_centers_3d_coords': patch_centers_3d_coords.cpu(),  # (37, 37, 3)
    'patch_centers_z_depth': sampled_depths.cpu()  # (37, 37)
}

# Save data
output_pt_file = './output/single_image_fitting/forward/template_data_forward.pt'
torch.save(view_data, output_pt_file)
print(f"\nData saved to {output_pt_file}")
