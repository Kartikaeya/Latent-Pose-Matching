import os
import argparse
import random
import xml.etree.ElementTree as ET
from PIL import Image
import glob

def parse_xml(xml_path):
    """Parse XML file and return image filename and head bounding boxes."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image filename
        filename_elem = root.find('filename')
        if filename_elem is None:
            print(f"Warning: Missing filename in {xml_path}")
            return None, None, None, []
        filename = filename_elem.text
        
        # Get image size
        size = root.find('size')
        if size is None:
            print(f"Warning: Missing size element in {xml_path}")
            return None, None, None, []
            
        width_elem = size.find('width')
        height_elem = size.find('height')
        if width_elem is None or height_elem is None:
            print(f"Warning: Missing width or height in {xml_path}")
            return None, None, None, []
            
        width = int(float(width_elem.text))
        height = int(float(height_elem.text))
        
        # Get all head bounding boxes
        heads = []
        for obj in root.findall('object'):
            if obj.find('name') is None or obj.find('name').text != 'head':
                continue
                
            bndbox = obj.find('bndbox')
            if bndbox is None:
                print(f"Warning: Missing bndbox in object in {xml_path}")
                continue
                
            xmin_elem = bndbox.find('xmin')
            ymin_elem = bndbox.find('ymin')
            xmax_elem = bndbox.find('xmax')
            ymax_elem = bndbox.find('ymax')
            
            if any(elem is None for elem in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]):
                print(f"Warning: Missing coordinate in bndbox in {xml_path}")
                continue
                
            try:
                xmin = int(float(xmin_elem.text))
                ymin = int(float(ymin_elem.text))
                xmax = int(float(xmax_elem.text))
                ymax = int(float(ymax_elem.text))
                
                difficult_elem = obj.find('difficult')
                difficult = int(difficult_elem.text) if difficult_elem is not None else 0
                
                heads.append((xmin, ymin, xmax, ymax, difficult))
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid coordinate value in {xml_path}: {str(e)}")
                continue
        
        return filename, width, height, heads
        
    except ET.ParseError as e:
        print(f"Warning: XML parsing error in {xml_path}: {str(e)}")
        return None, None, None, []
    except Exception as e:
        print(f"Warning: Unexpected error parsing {xml_path}: {str(e)}")
        return None, None, None, []

def extract_heads(input_dir, output_dir, num_images):
    """Extract head crops from images based on XML annotations."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all XML files
    xml_files = glob.glob(os.path.join(input_dir, 'Annotations', '*.xml'))
    if not xml_files:
        print(f"No XML files found in {os.path.join(input_dir, 'Annotations')}")
        return
    
    # Shuffle XML files to get random selection
    random.shuffle(xml_files)
    
    processed_count = 0
    skipped_count = 0
    
    # Process files sequentially until we reach the target number
    for xml_path in xml_files:
        if processed_count >= num_images:
            break
            
        # Parse XML
        filename, width, height, heads = parse_xml(xml_path)
        if filename is None:  # Skip if parsing failed
            continue
            
        # Get corresponding image path
        image_path = os.path.join(input_dir, 'JPEGImages', filename)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        try:
            # Open image
            image = Image.open(image_path)
            
            # Process each head
            for xmin, ymin, xmax, ymax, difficult in heads:
                if processed_count >= num_images:
                    break
                    
                # Calculate head dimensions
                head_width = xmax - xmin
                head_height = ymax - ymin
                
                # Skip if either dimension is smaller than 224
                if head_width < 224 or head_height < 224:
                    skipped_count += 1
                    continue
                
                # Crop and save head
                head_crop = image.crop((xmin, ymin, xmax, ymax))
                output_filename = f"head_{processed_count:06d}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                head_crop.save(output_path)
                
                difficulty_str = "difficult" if difficult else "not difficult"
                print(f"Saved head crop {processed_count + 1}/{num_images}: {output_path} ({head_width}x{head_height}) - {difficulty_str}")
                processed_count += 1
                
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            continue
    
    print(f"\nExtracted {processed_count} head crops to {output_dir}")
    print(f"Skipped {skipped_count} heads that were smaller than 224x224 pixels")

def main():
    parser = argparse.ArgumentParser(description='Extract head crops from HollywoodHeads dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing Annotations and JPEGImages folders')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save extracted head crops')
    parser.add_argument('--num_images', type=int, required=True,
                      help='Number of head crops to extract')
    
    args = parser.parse_args()
    
    # Validate input directory structure
    if not os.path.exists(os.path.join(args.input_dir, 'Annotations')):
        print(f"Annotations directory not found in {args.input_dir}")
        return
    if not os.path.exists(os.path.join(args.input_dir, 'JPEGImages')):
        print(f"JPEGImages directory not found in {args.input_dir}")
        return
    
    extract_heads(args.input_dir, args.output_dir, args.num_images)

if __name__ == "__main__":
    main() 