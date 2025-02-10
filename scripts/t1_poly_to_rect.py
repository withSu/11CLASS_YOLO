import os
import shutil
import json
import math

# Function to calculate the slope of a polygon
def calculate_slope(points):
    if len(points) < 2:
        return 0
    x1, y1 = points[0]
    x2, y2 = points[1]
    if x2 - x1 == 0:  # Avoid division by zero
        return float('inf')
    return abs((y2 - y1) / (x2 - x1))

# Function to convert polygons to rectangles
def convert_polygons_to_rectangles(json_data):
    for shape in json_data.get("shapes", []):
        if shape.get("shape_type") == "polygon":
            slope = calculate_slope(shape["points"])
            if slope < 10:
                # Convert polygon to rectangle
                x_coords = [point[0] for point in shape["points"]]
                y_coords = [point[1] for point in shape["points"]]
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)
                shape["points"] = [[xmin, ymin], [xmax, ymax]]
                shape["shape_type"] = "rectangle"
    return json_data

# Function to process all files in a directory
def process_files_in_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                image_path = os.path.splitext(json_path)[0] + ".jpg"  # Assuming image extension is .jpg

                with open(json_path, "r") as json_file:
                    data = json.load(json_file)

                # Modify the JSON data
                modified_data = convert_polygons_to_rectangles(data)

                # Define new paths for output
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                output_json_path = os.path.join(output_subdir, file)
                output_image_path = os.path.join(output_subdir, os.path.basename(image_path))

                # Save the modified JSON file
                with open(output_json_path, "w") as output_json_file:
                    json.dump(modified_data, output_json_file, indent=4)

                # Copy the image file
                if os.path.exists(image_path):
                    shutil.copy(image_path, output_image_path)

# Specify input and output directories
input_directory = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/GT/kbs_poly_remains"
output_directory = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/GT/kbs_only_"

# Process all files
process_files_in_directory(input_directory, output_directory)

print(f"All files have been processed and saved to {output_directory}")
