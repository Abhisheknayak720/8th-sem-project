import os
from PIL import Image

# Set the desired size of images
size = 128

# Path to the base directory containing class folders
base_path = "D:/ml-test/Training_images"

# List all subdirectories (classes)
classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

# Iterate through each class
for class_name in classes:
    # Path to the current class directory
    class_path = os.path.join(base_path, class_name)

    # Create a separate folder for resized images
    resized_path = os.path.join(class_path, "resized")
    if not os.path.exists(resized_path):
        os.makedirs(resized_path)

    # List all files in the class directory
    files = os.listdir(class_path)

    # Iterate through each file
    for file_name in files:
        # Check if it's an image file
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Open the image
            with Image.open(os.path.join(class_path, file_name)) as img:
                # Resize the image
                resized_img = img.resize((size, size), Image.LANCZOS)

                # Generate output filename with class prefix
                output_file_name = os.path.join(resized_path, f"{class_name}_{file_name}")

                # Save the resized image
                resized_img.save(output_file_name, 'JPEG', quality=95)
                print(f"Resized: {file_name} (Class: {class_name})")
        else:
            print(f"Skipped: {file_name} (Not an image)")

print('All done!')
