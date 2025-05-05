import os
import sys
from PIL import Image
import glob

def convert_and_rename(directory):
    # Create output directory if it doesn't exist
    output_dir = os.path.join(directory, 'converted')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all files in the directory
    files = sorted(glob.glob(os.path.join(directory, '*')))
    
    # Counter for sequential naming
    counter = 1
    
    for file_path in files:
        # Skip if it's a directory or zone identifier file
        if os.path.isdir(file_path) or 'Zone.Identifier' in file_path:
            continue
            
        try:
            # Open and convert image
            with Image.open(file_path) as img:
                # Convert to RGB if necessary (handles RGBA, etc.)
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create new filename
                new_filename = f'image_{counter:04d}.png'
                output_path = os.path.join(output_dir, new_filename)
                
                # Save as PNG
                img.save(output_path, 'PNG')
                print(f'Converted: {os.path.basename(file_path)} -> {new_filename}')
                counter += 1
                
        except Exception as e:
            print(f'Error processing {file_path}: {str(e)}')
            continue

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python convert_and_rename.py <directory_path>')
        sys.exit(1)
        
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f'Error: {directory} is not a valid directory')
        sys.exit(1)
        
    convert_and_rename(directory)
    print('Conversion and renaming completed!') 