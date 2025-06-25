import numpy as np
import scipy.io
from pathlib import Path

def update_obj_uv(obj_path, mat_path, output_path):
    # Load MAT file data
    mat_data = scipy.io.loadmat(mat_path)
    uv_vertices = mat_data['uv_idx_vt_idx']  # Using the 20480-indexed version
    uv_weights = mat_data['uv_idx_bw']
    
    # Validate dimensions
    assert uv_vertices.shape == (1024, 1024, 3), "Unexpected UV map shape"
    assert uv_weights.shape == (1024, 1024, 3), "Unexpected weights shape"
    
    # Create empty UV array (initialize to -1 for unmapped vertices)
    uv_map = -np.ones((20481, 2))  # +1 because vertex IDs start at 0
    
    # Convert UV grid to per-vertex coordinates
    for y in range(1024):
        for x in range(1024):
            # Convert pixel coordinates to [0,1] UV space
            u = x / 1023.0
            v = 1.0 - (y / 1023.0)  # Flip V to match OBJ convention
            
            for c in range(3):  # Process all 3 channels
                vertex_id = int(uv_vertices[y, x, c])
                weight = uv_weights[y, x, c]
                
                # Only update if weight is significant
                if weight > 1e-6 and vertex_id <= 20480:
                    if uv_map[vertex_id, 0] == -1:  # Not yet mapped
                        uv_map[vertex_id] = [u, v]
    
    # Read original OBJ
    vertices = []
    faces = []
    old_vts = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(line)
            elif line.startswith('vt '):
                old_vts.append(line)
            elif line.startswith('f '):
                faces.append(line)
    
    # Write new OBJ with updated UVs
    with open(output_path, 'w') as f:
        # Write vertices
        f.writelines(vertices)
        
        # Write new texture coordinates
        for i in range(20481):
            if uv_map[i, 0] != -1:  # Only write mapped UVs
                f.write(f"vt {uv_map[i, 0]:.6f} {uv_map[i, 1]:.6f}\n")
            else:
                f.write("vt 0.0 0.0\n")  # Default for unmapped vertices
        
        # Write faces (assuming they use sequential vertex/texture indices)
        f.writelines(faces)

# Usage
update_obj_uv(
    obj_path="./data/FFHQ_UV/hifi3dpp_mean_face.obj",
    mat_path="./data/FFHQ_UV/unwrap_1024_info.mat",
    output_path="./data/FFHQ_UV/hifi3dpp_with_uv.obj"
)