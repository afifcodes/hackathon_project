import os
import tensorflow as tf
from PIL import Image

def check_images(directory):
    print(f"Scanning {directory} for corrupt images...")
    corrupt_count = 0
    checked_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                checked_count += 1
                try:
                    # 1. Check with PIL
                    with Image.open(file_path) as img:
                        img.verify()
                    
                    # 2. Check with TensorFlow decoding (stricter)
                    with open(file_path, 'rb') as f:
                        img_bytes = f.read()
                        # This mimics what the model loader does
                        tf.image.decode_image(img_bytes)
                        
                except Exception as e:
                    print(f"Found corrupt image: {file_path} - Error: {e}")
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                        corrupt_count += 1
                    except Exception as del_e:
                        print(f"Failed to delete {file_path}: {del_e}")

    print(f"\nScan complete. Checked {checked_count} images.")
    print(f"Found and removed {corrupt_count} corrupt images.")

if __name__ == "__main__":
    if os.path.exists('dataset'):
        check_images('dataset')
    else:
        print("Dataset directory not found.")
