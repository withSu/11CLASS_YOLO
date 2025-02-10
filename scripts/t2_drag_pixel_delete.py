import os
import cv2
import json
import base64
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from shutil import copy2

# Global variable to store drag regions
drag_regions = []

# Function to draw labels on the image
def draw_labels(image_path, json_data):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for shape in json_data['shapes']:
        points = shape['points']
        x1, y1 = points[0]
        x2, y2 = points[1]
        width = x2 - x1
        height = y2 - y1
        rect = Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    return fig, ax, image

# Callback functions for mouse interaction
def on_select(eclick, erelease):
    global drag_regions
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    drag_regions.append((x1, y1, x2, y2))
    print(f"Drag region added: {(x1, y1, x2, y2)}")

# Function to check if two rectangles overlap
def check_overlap(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x4 < x1 or x3 > x2 or y4 < y1 or y3 > y2)

# Function to process a single image and its corresponding JSON file
def process_image_and_json(image_path, json_path, output_dir):
    global drag_regions

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Process each drag region
    for drag_rect in drag_regions:
        x1, y1, x2, y2 = drag_rect

        # Modify the image
        image[y1:y2, x1:x2] = 0

        # Remove overlapping labels from JSON
        json_data['shapes'] = [
            shape for shape in json_data['shapes']
            if not check_overlap(
                (shape['points'][0][0], shape['points'][0][1], shape['points'][1][0], shape['points'][1][1]),
                drag_rect
            )
        ]

    # Save modified image and JSON
    os.makedirs(output_dir, exist_ok=True)
    output_image_path = os.path.join(output_dir, os.path.basename(image_path))
    output_json_path = os.path.join(output_dir, os.path.basename(json_path))
    cv2.imwrite(output_image_path, image)

    # Adjust JSON encoding for Labelme compatibility
    json_data['imagePath'] = os.path.basename(output_image_path)
    with open(output_image_path, 'rb') as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
    json_data['imageData'] = encoded_image

    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

# Main function
def main(input_image_folder, input_json_folder, output_folder):
    global drag_regions

    # Process each image and JSON pair in the input folders
    for filename in os.listdir(input_image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(input_image_folder, filename)
            json_path = os.path.join(input_json_folder, filename.replace('.jpg', '.json').replace('.png', '.json'))

            if os.path.exists(json_path):
                # Reset drag regions for each image
                drag_regions = []

                # Load image and JSON
                with open(json_path, 'r') as f:
                    json_data = json.load(f)

                fig, ax, image = draw_labels(image_path, json_data)

                # Add RectangleSelector for mouse interaction
                toggle_selector = RectangleSelector(ax, on_select,
                                                    interactive=True,  # Fixed the deprecated argument
                                                    button=[1], minspanx=5, minspany=5,
                                                    spancoords='pixels')

                plt.connect('key_press_event', lambda event: plt.close(fig) if event.key == 'enter' else None)
                plt.show()

                # Process the image and JSON after dragging
                process_image_and_json(image_path, json_path, output_folder)

# Example usage
input_image_folder = '/home/a/A_2024_selfcode/PCB/GT/GTday/errorone'
input_json_folder = '/home/a/A_2024_selfcode/PCB/GT/GTday/errorone'
output_folder = '/home/a/A_2024_selfcode/PCB/GT/GTday/errorone_after'

main(input_image_folder, input_json_folder, output_folder)