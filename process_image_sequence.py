import argparse
import os
from pathlib import Path
import torch
from find_pose import (
    load_model_and_processor,
    load_pca_models,
    visualize_comparison
)

def process_image_sequence(
    image_dir,
    render_data_folder,
    pca_model_path,
    mesh_path,
    output_dir,
    use_custom_preprocessing=True,
    use_registers=True,
    threshold_greater_than=False,
    fov=45,
    debug=False
):
    """Process a sequence of images using find_pose.py functionality.
    
    Args:
        image_dir: Directory containing input images
        render_data_folder: Path to folder containing saved render_data.pt files
        pca_model_path: Path to pre-trained PCA model
        mesh_path: Path to the mesh OBJ file
        output_dir: Directory to save results
        use_custom_preprocessing: Whether to use custom preprocessing
        use_registers: Whether to use DinoV2 base with registers
        threshold_greater_than: Threshold for values greater than 0
        fov: Field of view in degrees
        debug: Whether to enable debug mode
    """
    global DEBUG
    DEBUG = debug
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and processor
    print("Loading DinoV2 model and processor...")
    model, processor, device = load_model_and_processor(use_registers)
    
    # Load PCA models
    print(f"Loading PCA models from {pca_model_path}")
    pca_mask, pca_rgb, kmeans = load_pca_models(pca_model_path)
    
    # Get list of image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\nFound {len(image_files)} images to process")
    
    render_data_folder = Path(render_data_folder) / "template_data_forward.pt"
    # Process each image
    for i, image_file in enumerate(image_files, 1):
        print(f"\nProcessing image {i}/{len(image_files)}: {image_file}")
        
        # Create output subdirectory for this image
        image_name = Path(image_file).stem
        
        # Full path to the image
        image_path = os.path.join(image_dir, image_file)
        
        # try:
        # Process and visualize the image
        visualize_comparison(
            image_path,
            render_data_folder,
            output_dir,
            model,
            processor,
            device,
            pca_mask,
            pca_rgb,
            image_name,
            use_custom_preprocessing,
            use_registers,
            threshold_greater_than,
            mesh_path,
            fov,
            debug
        )
        print(f"Successfully processed {image_file}")
            
        # except Exception as e:
        #     print(f"Error processing {image_file}: {str(e)}")
        #     continue

def main():
    parser = argparse.ArgumentParser(description='Process a sequence of images using find_pose.py functionality')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--render_data_folder', type=str, required=True, help='Path to folder containing saved render_data.pt files')
    parser.add_argument('--pca_model', type=str, default='./saved_pcas/HH_PCA_GMM_v2.pt', help='Path to pre-trained PCA model')
    parser.add_argument('--mesh', type=str, default='./data/FFHQ_UV_sample/hack_w_HIFI3D_UV.obj', help='Path to the mesh OBJ file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--use_custom_preprocessing', action='store_true', default=True,
                      help='Use custom preprocessing (padding to square and resizing to 518x518)')
    parser.add_argument('--use_registers', action='store_true', default=True,
                      help='Use DinoV2 base with registers instead of DinoV2 base')
    parser.add_argument('--threshold_greater_than', action='store_true', default=False,
                      help='Threshold for values greater than 0 instead of less than 0')
    parser.add_argument('--fov', type=float, default=45, help='Field of view in degrees')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    
    args = parser.parse_args()
    
    process_image_sequence(
        args.image_dir,
        args.render_data_folder,
        args.pca_model,
        args.mesh,
        args.output_dir,
        args.use_custom_preprocessing,
        args.use_registers,
        args.threshold_greater_than,
        args.fov,
        args.debug
    )

if __name__ == "__main__":
    main() 