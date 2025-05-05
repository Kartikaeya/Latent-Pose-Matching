import os
import torch
import numpy as np
from pytorch3d.io import load_obj
from pytorch3d.renderer import (    
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

from pytorch3d.renderer import BlendParams

# Set blend parameters
blend_params = BlendParams(
    sigma=1e-4,
    gamma=1e-4,
    background_color=(1.0, 1.0, 1.0)
)

# Load the ICT face model
obj_file = os.path.join("ICT-FaceKit", "FaceXModel", "generic_neutral_mesh.obj")
verts, faces, _ = load_obj(obj_file)

# Create a batch of meshes
meshes = Meshes(
    verts=[verts],
    faces=[faces.verts_idx]
)

# Set up the renderer
# Camera positioned at (0, 0, 2.7) looking at origin (0, 0, 0)
R, T = look_at_view_transform(50, 0, 0, device=device)

cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.1, zfar=1000.0)

raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

lights = PointLights(location=[[0, 0, 1e5]], device=device,
                     ambient_color=((0.5, 0.5, 0.5),),
                     diffuse_color=((0.3, 0.3, 0.3),),
                     specular_color=((0.05, 0.05, 0.05),)
                     )

rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
shader = SoftPhongShader(
    device=device,
    cameras=cameras,
    lights=lights,
    blend_params=blend_params
)
renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

# Create textures
verts_rgb = torch.ones_like(verts)[None]  # White color
textures = TexturesVertex(verts_features=verts_rgb)

# Create materials
materials = Materials(
    device=device,
    specular_color=[[0.1, 0.1, 0.1]],
    shininess=5.0
)

# Add textures to the mesh
meshes = Meshes(
    verts=[verts],
    faces=[faces.verts_idx],
    textures=textures
)

# Render the mesh
meshes = meshes.to(device)
images = renderer(meshes)

# Convert to numpy and plot
image = images[0, ..., :3].cpu().numpy()
plt.imshow(image)
plt.axis('off')
plt.savefig('ict_face_render.png', bbox_inches='tight', pad_inches=0)
plt.close()
