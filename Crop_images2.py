import os
import logging
from rembg import remove
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def clear_directory(directory):
    """Delete all files in the specified directory."""
    try:
        if not os.path.exists(directory):
            logger.info(f"Directory does not exist, will be created: {directory}")
            return

        # List all files in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted file: {file_path}")
                # Skip directories
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")

        logger.info(f"Cleared all files in directory: {directory}")
    except Exception as e:
        logger.error(f"Failed to clear directory {directory}: {e}")


def process_fashion_images(input_dir, output_dir):
    """Process images in input_dir, remove backgrounds, and save as cropped_imageX.png with transparent background in output_dir."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Clear all files in output directory
    clear_directory(output_dir)

    # Check if input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return

    # Supported image formats
    supported_formats = ('.png', '.jpg', '.jpeg', '.webp')

    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]

    if not image_files:
        logger.warning(f"No supported image files found in {input_dir}")
        print("No supported image files found in the input directory.")
        return

    # Process each image with a sequential counter
    for index, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_dir, filename)
        output_filename = f"cropped_image{index}.png"
        output_path = os.path.join(output_dir, output_filename)

        try:
            # Open and process the image
            with open(input_path, 'rb') as input_file:
                input_data = input_file.read()

            # Remove background
            output_data = remove(input_data)

            # Convert output data to PIL Image
            output_image = Image.open(io.BytesIO(output_data))

            # Ensure image is in RGBA mode to preserve transparency
            if output_image.mode != 'RGBA':
                output_image = output_image.convert('RGBA')

            # Save the result as PNG to maintain transparency
            output_image.save(output_path, 'PNG')

            logger.info(f"Processed: {filename} -> {output_filename} with transparent background")
            print(f"Processed: {filename} -> {output_filename}")

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            print(f"Error processing {filename}: {e}")

    print(f"All images have been processed successfully!")
    print(f"All images have been saved in {output_dir} as cropped_image1.png, cropped_image2.png, etc.")


def main():
    input_dir = r"D:\Backup for AI Fashion Stylist\AI_Fashion_Stylist\Fashion Clothes"
    output_dir = r"D:\Backup for AI Fashion Stylist\AI_Fashion_Stylist\cropped images"

    process_fashion_images(input_dir, output_dir)


if __name__ == "__main__":
    main()