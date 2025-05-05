import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import face_alignment
from tqdm import tqdm

def process_images(input_dir, output_dir):
    """
    Process images in input_dir using 2DFAN and save landmark visualizations to output_dir
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize face alignment model
    fa = face_alignment.FaceAlignment('2D', device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get list of PNG files
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])
    if not image_files:
        print(f"No PNG files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} PNG files to process")
    
    # Process each image
    for img_file in tqdm(image_files, desc="Processing images"):
        # Load image
        img_path = os.path.join(input_dir, img_file)
        img = Image.open(img_path)
        img_np = np.array(img)
        
        # Get landmarks
        try:
            landmarks = fa.get_landmarks(img_np)
            if landmarks is None or len(landmarks) == 0:
                print(f"No face detected in {img_file}")
                # Save the original image (no annotations)
                output_path = os.path.join(output_dir, f"landmarks_{img_file}")
                img.save(output_path)
                continue
                
            # Use the first face detected
            landmarks = landmarks[0]
            
            # Create visualization
            plt.figure(figsize=(10, 10))
            plt.imshow(img_np)
            
            # Plot landmarks
            for point in landmarks:
                plt.gca().add_patch(Circle((point[0], point[1]), radius=2, color='red'))
            
            # Save visualization
            output_path = os.path.join(output_dir, f"landmarks_{img_file}")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Save landmarks as numpy array
            np.save(os.path.join(output_dir, f"landmarks_{os.path.splitext(img_file)[0]}.npy"), landmarks)
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser(description='Process facial landmarks using 2DFAN')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing input PNG images')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save landmark visualizations and data')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    process_images(args.input_dir, args.output_dir)
    print("Processing complete!")

if __name__ == "__main__":
    main() 