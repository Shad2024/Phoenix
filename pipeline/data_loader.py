from outputs import captured_images ,segmentation_summary
import csv
from collections import defaultdict
import os

# Define a consistent color map for materials
COLOR_MAP = {
    'rebar': '#B22222',    # Firebrick Red
    'wood': '#A0522D',     # Sienna Brown
    'pipe': '#FF8C00',     # Dark Orange
    'aluminum': '#A9A9A9', # Dark Gray
    'glass': '#ADD8E6',    # Light Blue
    'plastic': '#99FF99',  # Light Green
}

def load_and_structure_analysis_data(segmentation_summary, captured_images):
    """
    Loads data from the generated CSV and prepares it for the GUI.
    Returns: (list of image paths, list of analysis dictionaries)
    """
    grouped_data = defaultdict(list)
    
    try:
        with open(segmentation_summary.csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                next(reader) # Skip header
            except StopIteration:
                return [], [] # Empty file

            for row in reader:
                if len(row) < 3: continue
                
                # Strip spaces for robust parsing
                image_name, material, percentage_str = row[0].strip(), row[1].strip(), row[2].strip()
                
                try:
                    percentage = float(percentage_str)
                except ValueError:
                    continue

                grouped_data[image_name].append({
                    'material': material,
                    'percentage': percentage,
                    'color': COLOR_MAP.get(material.lower(), '#FFFFFF') # Default to white
                })
    except FileNotFoundError:
        print(f"Error: Analysis output file not found at {segmentation_summary.csv}")
        return [], []

    # Convert grouped data into ordered lists for GUI sequence
    image_names_sorted = sorted(grouped_data.keys())
    
    image_paths = []
    analysis_data_list = []
    
    for img_name in image_names_sorted:
        # Construct the full image path
        image_paths.append(os.path.join( captured_images , img_name))
        
        # Structure analysis data for the GUI
        structured_data = {}
        for item in grouped_data[img_name]:
            structured_data[item['material']] = {
                'percentage': item['percentage'],
                'color': item['color']
            }
        
        analysis_data_list.append(structured_data)
        
    return image_paths, analysis_data_list