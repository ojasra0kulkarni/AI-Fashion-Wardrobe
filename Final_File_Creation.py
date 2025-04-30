import os
import shutil
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def read_combinations_file(file_path):
    """Read combinations_image_file_names.txt and return a list of image filenames for each day."""
    combinations = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Match lines like "1. cropped_image1.png - cropped_image2.png" or "1. None"
                match = re.match(r'(\d+)\.\s*(None|(?:[^\s]+\.png\s*-\s*)*[^\s]+\.png)?', line)
                if match:
                    day_num = int(match.group(1))
                    images_str = match.group(2)
                    if images_str == 'None' or not images_str:
                        images = []
                    else:
                        # Split images by " - "
                        images = [img.strip() for img in images_str.split('-')]
                    # Ensure day_num is 1 to 7
                    if 1 <= day_num <= 7:
                        combinations.append((day_num, images))
                    else:
                        logger.warning(f"Invalid day number in line: {line}")
                else:
                    logger.warning(f"Skipping malformed line: {line}")
        return combinations
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return []


def clear_day_directories(final_combinations_path):
    """Clear all files in Day-1 to Day-7 subdirectories."""
    try:
        for i in range(1, 8):
            day_path = os.path.join(final_combinations_path, f'Day-{i}')
            if os.path.exists(day_path):
                for filename in os.listdir(day_path):
                    file_path = os.path.join(day_path, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            logger.info(f"Deleted file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {e}")
            else:
                logger.warning(f"Directory not found: {day_path}")
    except Exception as e:
        logger.error(f"Error clearing day directories: {e}")


def create_directories(base_path):
    """Create Final_Combinations and Day-1 to Day-7 subdirectories, and clear existing files."""
    final_combinations_path = os.path.join(base_path, 'Final_Combinations')
    try:
        os.makedirs(final_combinations_path, exist_ok=True)
        logger.info(f"Created directory: {final_combinations_path}")
        for i in range(1, 8):
            day_path = os.path.join(final_combinations_path, f'Day-{i}')
            os.makedirs(day_path, exist_ok=True)
            logger.info(f"Created directory: {day_path}")
        # Clear existing files in Day-1 to Day-7
        clear_day_directories(final_combinations_path)
        return final_combinations_path
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        return None


def copy_images(combinations, source_dir, final_combinations_path):
    """Copy images to Day-X subdirectories based on combinations."""
    for day_num, images in combinations:
        dest_dir = os.path.join(final_combinations_path, f'Day-{day_num}')
        if not images:
            logger.info(f"No images to copy for Day-{day_num}")
            continue
        for image in images:
            src_path = os.path.join(source_dir, image)
            dest_path = os.path.join(dest_dir, image)
            try:
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dest_path)
                    logger.info(f"Copied {image} to {dest_dir}")
                else:
                    logger.warning(f"Image not found: {src_path}")
            except Exception as e:
                logger.error(f"Error copying {image} to {dest_dir}: {e}")


def main():
    base_path = r"D:\Backup for AI Fashion Stylist\AI_Fashion_Stylist"

    combinations_file = os.path.join(base_path, "combinations_image_file_names.txt")
    source_dir = os.path.join(base_path, "cropped images")

    # Check if source directory exists
    if not os.path.exists(source_dir):
        logger.error(f"Source directory not found: {source_dir}")
        return

    # Read combinations
    combinations = read_combinations_file(combinations_file)
    if not combinations:
        logger.error("No valid combinations found. Exiting.")
        return

    # Create directories and clear existing files
    final_combinations_path = create_directories(base_path)
    if not final_combinations_path:
        logger.error("Failed to create directories. Exiting.")
        return

    # Copy images
    copy_images(combinations, source_dir, final_combinations_path)
    logger.info("Image copying completed.")


if __name__ == "__main__":
    main()