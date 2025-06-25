import os
import pandas as pd
import random
from pathlib import Path

# Define root of dataset
dataset_path = Path("data/images")
output_csv_path = "data/recipes.csv"
image_output_path = Path("data/images")
image_output_path.mkdir(parents=True, exist_ok=True)

# Map class folders to food names (you can customize this)
label_map = {
    "0": "Chicken Curry",
    "1": "Fried Rice",
    "2": "Grilled Salmon",
    "3": "Pasta Carbonara",
    "4": "Steamed Vegetables",
    "5": "Cheese Omelette",
    "6": "Beef Steak",
    "7": "Mushroom Soup",
    "8": "Tofu Stir Fry",
    "9": "Caesar Salad",
    "10": "Peanut Noodles"
}

ingredient_pool = {
    "Chicken Curry": ["chicken", "onion", "garlic", "turmeric", "oil", "chili"],
    "Fried Rice": ["rice", "egg", "carrot", "soy sauce", "peas", "onion"],
    "Grilled Salmon": ["salmon", "lemon", "olive oil", "pepper"],
    "Pasta Carbonara": ["pasta", "egg", "cheese", "bacon", "black pepper"],
    "Steamed Vegetables": ["broccoli", "carrot", "beans", "salt"],
    "Cheese Omelette": ["egg", "cheese", "milk", "butter"],
    "Beef Steak": ["beef", "salt", "pepper", "garlic"],
    "Mushroom Soup": ["mushroom", "cream", "onion", "garlic", "butter"],
    "Tofu Stir Fry": ["tofu", "soy sauce", "ginger", "garlic", "peppers"],
    "Caesar Salad": ["lettuce", "croutons", "cheese", "caesar dressing"],
    "Peanut Noodles": ["noodles", "peanut butter", "soy sauce", "lime"]
}

allergen_map = {
    "Cheese Omelette": "egg, dairy",
    "Peanut Noodles": "peanuts, gluten",
    "Pasta Carbonara": "gluten, egg",
    "Tofu Stir Fry": "soy",
    "Caesar Salad": "dairy",
}

# Collect data
rows = []
img_id = 1000

for label_folder in dataset_path.iterdir():
    label = label_folder.name
    food_name = label_map.get(label, f"Food {label}")
    ingredients = ", ".join(ingredient_pool.get(food_name, ["salt", "oil"]))
    allergens = allergen_map.get(food_name, "none")

    for img_file in label_folder.glob("*.jpg"):
        new_filename = f"{img_id}.jpg"
        dest = image_output_path / new_filename
        dest.write_bytes(img_file.read_bytes())  # copy image

        rows.append({
            "id": img_id,
            "name": food_name,
            "ingredients": ingredients,
            "calories": random.randint(200, 800),
            "allergens": allergens
        })
        img_id += 1

# Save CSV
df = pd.DataFrame(rows)
df.to_csv(output_csv_path, index=False)
print(f"Created {output_csv_path} with {len(df)} entries.")
