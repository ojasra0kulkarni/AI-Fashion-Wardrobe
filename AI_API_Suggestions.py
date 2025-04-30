
import pandas as pd
import os
import logging
import google.generativeai as genai
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Gemini API settings
GEMINI_API_KEY = "haha"  # Replace with your Gemini API key
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-1.5-flash"

# Define clothing categories
TOP_WEAR = ['T-shirt', 'Long sleeve', 'Shirt', 'Kurta', 'Blouse', 'Top', 'Jacket', 'Outwear']
BOTTOM_WEAR = ['Pants', 'Skirt', 'Trousers', 'Shorts', 'Leggings']
DRESS = ['Dress']

def normalize_clothing_type(clothing_type):
    """Normalize clothing type for consistent matching."""
    if not clothing_type:
        return clothing_type
    clothing_type = ' '.join(clothing_type.strip().split()).title()
    type_map = {
        'Tshirt': 'T-shirt',
        'Longsleeve': 'Long sleeve',
        'Outwear': 'Outwear',
    }
    return type_map.get(clothing_type, clothing_type)

def get_season():
    """Determine the current season based on the current date (Indian weather)."""
    month = datetime.now().month
    if 3 <= month <= 5:
        return "Summer"
    elif 6 <= month <= 9:
        return "Monsoon"
    else:
        return "Winter"

def read_clothing_data(csv_path):
    """Read the CSV file and return the clothing data."""
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['Image', 'Clothing Types', 'Primary Color', 'Secondary Color']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"CSV file must contain columns: {required_columns}")
            return None
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        return None

def summarize_clothing_data(df):
    """Summarize clothing types and their colors from the DataFrame."""
    clothing_summary = []
    for _, row in df.iterrows():
        clothing_types_value = row['Clothing Types']
        if pd.notna(clothing_types_value) and isinstance(clothing_types_value, str) and clothing_types_value.lower().strip() != 'none':
            clothing_types = clothing_types_value.split(', ')
        else:
            clothing_types = []

        primary_color = row['Primary Color'] if pd.notna(row['Primary Color']) and row['Primary Color'].lower() != 'error' else 'Unknown'
        secondary_color = row['Secondary Color'] if pd.notna(row['Secondary Color']) and row['Secondary Color'].lower() != 'error' else 'None'

        for clothing in clothing_types:
            if clothing.lower().strip() not in ['error', 'not sure', '', 'person']:
                clothing_summary.append({
                    'type': normalize_clothing_type(clothing),
                    'primary_color': primary_color,
                    'secondary_color': secondary_color,
                    'image': row['Image']
                })

    seen = set()
    unique_clothing = []
    for item in clothing_summary:
        item_key = (item['type'], item['primary_color'], item['secondary_color'])
        if item_key not in seen:
            seen.add(item_key)
            unique_clothing.append(item)

    return unique_clothing

def generate_outfit_prompt(clothing_summary, season):
    """Generate a prompt for the AI to suggest outfits with image combinations."""
    if not clothing_summary:
        return "No valid clothing data available to generate outfit suggestions."

    prompt = (
        "You are a fashion stylist AI. Based on the following clothing items, suggest exactly 7 outfits, one for each day of the week, named 'Day 1' through 'Day 7'. "
        "Each outfit must be suitable for the given Indian season and include **both a top wear and a bottom wear** from the provided items, unless a dress is used, which counts as both top and bottom wear. "
        "Use the provided clothing items creatively and include additional items (e.g., shoes, shawls) as needed to complete the outfits. "
        "Provide a brief description of each outfit, why it suits the season (e.g., Summer: hot and dry, Monsoon: rainy and humid, Winter: cool and dry), and how the colors complement each other. "
        "If certain clothing types are not suitable for the season, explain why and suggest alternatives.\n\n"
    )

    prompt += f"**Season**: {season}\n\n"

    prompt += "**Available Clothing Items**:\n"
    for item in clothing_summary:
        prompt += f"- {item['type']} ({item['image']}): Primary Color = {item['primary_color']}, Secondary Color = {item['secondary_color']}\n"
    prompt += "\n"

    prompt += (
        "**Clothing Categories**:\n"
        f"- Top wear: {', '.join(TOP_WEAR)}\n"
        f"- Bottom wear: {', '.join(BOTTOM_WEAR)}\n"
        f"- Dress: {', '.join(DRESS)} (counts as both top and bottom wear)\n\n"
    )

    prompt += (
        "**Instructions**:\n"
        "1. Suggest exactly 7 complete outfits, named 'Day 1' to 'Day 7', using the available clothing items.\n"
        "2. **Each outfit must include a top wear and a bottom wear from the provided items**, or a dress that counts as both. For example, pair a T-shirt with Pants, or use a Dress with optional shoes.\n"
        "3. List the clothing items with their type, colors, and image filename (e.g., 'Top: T-shirt (vivid blue, vivid blue, cropped_image1.png)').\n"
        "4. Include shoes from the provided items if available; otherwise, suggest generic shoes if needed.\n"
        "5. Describe the combination of clothing types and colors, and explain why it is appropriate for the season (e.g., lightweight fabrics for Summer, waterproof or quick-drying items for Monsoon, warm layers for Winter).\n"
        "6. Highlight how the primary and secondary colors of the items complement each other or create a stylish contrast.\n"
        "7. If any clothing items are not suitable (e.g., heavy jackets in Summer), note this and suggest alternatives or exclude them.\n"
        "8. If accessories (e.g., handbag, backpack) or shoes are included, mention how they enhance the outfit.\n"
        "9. Keep the suggestions practical and fashionable, suitable for everyday wear in India.\n"
        "10. Ensure each outfit is distinct and suitable for a different day of the week.\n"
        "11. At the end of the response, include a section titled '**Image Combinations**' with exactly 7 lines, one for each day, listing the image filenames used in each outfit in the format: 'X. image1.png - image2.png - ...'. For outfits with a dress, include the dress image followed by the next available item (e.g., shoes). For example:\n"
        "    **Image Combinations**\n"
        "    1. cropped_image1.png - cropped_image5.png - cropped_image7.png\n"
        "    2. cropped_image9.png - cropped_image7.png\n"
        "    ...\n"
        "12. If no images are used for a day (e.g., only generic items), write 'X. None' for that day.\n"
        "13. Suggest what type of clothing to avoid during this season at the end, before the Image Combinations section.\n"
    )

    return prompt

def call_gemini_api(prompt):
    """Make a call to the Gemini API and save the response."""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 2000,
                "temperature": 0.7
            }
        )
        generated_text = response.text.strip()

        # Save response for debugging
        debug_path = r"D:\Backup for AI Fashion Stylist\AI_Fashion_Stylist\ai_response.txt"
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        logger.info(f"AI response saved to {debug_path}")

        return generated_text

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return f"Failed to get response from AI: {e}"

def parse_outfit_response(response, clothing_summary):
    """Parse AI response to extract outfits and image combinations."""
    outfits = []
    image_combinations = []
    current_outfit = None
    lines = response.split('\n')
    in_image_combinations = False

    for line in lines:
        line = line.strip()
        if line.startswith('**Image Combinations**'):
            in_image_combinations = True
            continue
        if in_image_combinations:
            # Match lines like "1. cropped_image10.png - cropped_image5.png - cropped_image7.png"
            # or "2. cropped_image9.png - cropped_image7.png" or "7. cropped_image9.png"
            match = re.match(r'(\d+)\.\s*([^\s]+\.png(?:\s*-\s*[^\s]+\.png)*|None)', line)
            if match:
                day_num = int(match.group(1))
                images_str = match.group(2)
                if images_str == 'None':
                    images = []
                else:
                    images = [img.strip() for img in images_str.split('-') if img.strip()]
                if 1 <= day_num <= 7:
                    image_combinations.append((day_num, images))
                    logger.info(f"Parsed image combination: Day {day_num} - {images}")
                else:
                    logger.warning(f"Invalid day number in image combinations: {line}")
            else:
                logger.warning(f"Failed to parse image combination line: {line}")
            continue
        if re.match(r'\*\*Day \d+\*\*', line):
            if current_outfit:
                outfits.append(current_outfit)
            current_outfit = {'header': line, 'items': [], 'description': []}
        elif re.match(r'-\s*(?:\w+:)?\s*(\w+(?:\s+\w+)?)\s*\(([^,]+),\s*([^,]+),\s*([^\s)]+\.png)\)', line) and current_outfit:
            match = re.match(r'-\s*(?:\w+:)?\s*(\w+(?:\s+\w+)?)\s*\(([^,]+),\s*([^,]+),\s*([^\s)]+\.png)\)', line)
            if match:
                item_name, primary_color, secondary_color, image_file = match.groups()
                normalized_item_name = normalize_clothing_type(item_name)
                logger.debug(f"Parsed item: {item_name} (normalized: {normalized_item_name}, image: {image_file})")
                for clothing in clothing_summary:
                    if clothing['image'] == image_file:
                        current_outfit['items'].append({
                            'type': clothing['type'],
                            'primary_color': clothing['primary_color'],
                            'secondary_color': clothing['secondary_color'],
                            'image': clothing['image']
                        })
                        logger.debug(f"Matched clothing_summary: {clothing['type']}")
                        break
                else:
                    current_outfit['items'].append({
                        'type': normalized_item_name,
                        'primary_color': primary_color,
                        'secondary_color': secondary_color,
                        'image': image_file
                    })
                    logger.debug(f"Used AI item: {normalized_item_name}")
        elif current_outfit and line:
            current_outfit['description'].append(line)

    if current_outfit:
        outfits.append(current_outfit)

    # Ensure exactly 7 image combinations
    if len(image_combinations) < 7:
        logger.warning(f"Only {len(image_combinations)} image combinations found. Filling missing days with 'None'.")
        existing_days = {day for day, _ in image_combinations}
        for i in range(1, 8):
            if i not in existing_days:
                image_combinations.append((i, []))
                logger.info(f"Added empty combination for Day {i}")
        image_combinations.sort(key=lambda x: x[0])

    return outfits, image_combinations

def write_outfit_file(outfits, output_path):
    """Write outfit suggestions with paired image filenames to a text file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Outfit Combinations for Seven Days of the Week\n")
            f.write("==============================================\n\n")
            for outfit in outfits:
                f.write(f"{outfit['header']}\n")
                f.write("Items:\n")
                for item in outfit['items']:
                    f.write(f"- {item['type']} (Primary: {item['primary_color']}, Secondary: {item['secondary_color']}) - Image: {item['image']}\n")
                paired_images = [item['image'] for item in outfit['items']]
                if paired_images:
                    f.write(f"Paired Images: {' - '.join(paired_images)}\n")
                else:
                    f.write("Paired Images: None\n")
                f.write("Description:\n")
                for desc_line in outfit['description']:
                    f.write(f"  {desc_line}\n")
                f.write("\n")
            f.write("==============================================\n")
        logger.info(f"Outfit combinations written to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write outfit file: {e}")

def write_image_combinations_file(image_combinations, output_path):
    """Write paired image filenames for each outfit to a text file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for day_num, images in image_combinations:
                if images:
                    f.write(f"{day_num}. {' - '.join(images)}\n")
                else:
                    f.write(f"{day_num}. None\n")
        logger.info(f"Image combinations written to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write image combinations file: {e}")

def main():
    csv_path = r"D:\Backup for AI Fashion Stylist\AI_Fashion_Stylist\cropped_clothing_analysis.csv"
    outfit_output_path = r"D:\Backup for AI Fashion Stylist\AI_Fashion_Stylist\outfit_combinations.txt"
    image_combinations_path = r"D:\Backup for AI Fashion Stylist\AI_Fashion_Stylist\combinations_image_file_names.txt"

    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return

    df = read_clothing_data(csv_path)
    if df is None:
        return

    clothing_summary = summarize_clothing_data(df)
    if not clothing_summary:
        logger.warning("No valid clothing items found in the CSV.")
        print("No valid clothing items found to generate outfit suggestions.")
        return

    has_top = any(item['type'] in TOP_WEAR or item['type'] in DRESS for item in clothing_summary)
    has_bottom = any(item['type'] in BOTTOM_WEAR or item['type'] in DRESS for item in clothing_summary)
    if not (has_top and has_bottom):
        logger.error("CSV lacks sufficient top wear or bottom wear items.")
        print("Error: CSV must contain both top wear and bottom wear.")
        return

    season = get_season()
    prompt = generate_outfit_prompt(clothing_summary, season)

    logger.info("Generated outfit suggestion prompt:")
    print("\n=== AI Outfit Suggestion Prompt ===")
    print(prompt)
    print("==================================")

    print("\n=== AI Outfit Suggestions ===")
    response = call_gemini_api(prompt)
    print(response)
    print("==============================")

    outfits, image_combinations = parse_outfit_response(response, clothing_summary)

    if outfits or image_combinations:
        write_outfit_file(outfits, outfit_output_path)
        write_image_combinations_file(image_combinations, image_combinations_path)
        print(f"\nOutfit combinations saved to {outfit_output_path}")
        print(f"Image combinations saved to {image_combinations_path}")
    else:
        logger.warning("No valid outfits or image combinations parsed.")
        print("No valid outfits found. Check ai_response.txt.")

if __name__ == "__main__":
    main()
