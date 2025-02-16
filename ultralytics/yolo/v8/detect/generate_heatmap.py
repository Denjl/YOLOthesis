import json
import cv2
import numpy as np
from pathlib import Path

def generate_heatmap(coords_file, output_file, img_shape, dot_size):
    # Load occlusion coordinates
    coords = []
    with open(coords_file, 'r') as f:
        for line in f:
            coords.extend(json.loads(line))

    # Create an empty heatmap
    heatmap = np.zeros(img_shape[:2], dtype=np.float32)

    # Increment heatmap values at occlusion coordinates
    for x, y in coords:
        #heatmap[y, x] += 1
        cv2.circle(heatmap, (x, y), dot_size, 1, thickness=-1)

    # Normalize the heatmap
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)

    # Apply a color map to the heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Draw 8x8 grid
    num_rows, num_cols = 8, 8
    row_height = img_shape[0] // num_rows
    col_width = img_shape[1] // num_cols

    for i in range(1, num_rows):
        y = i * row_height
        cv2.line(heatmap, (0, y), (img_shape[1], y), (255, 255, 255), 1)

    for j in range(1, num_cols):
        x = j * col_width
        cv2.line(heatmap, (x, 0), (x, img_shape[0]), (255, 255, 255), 1)


    # Save the heatmap
    cv2.imwrite(output_file, heatmap)

if __name__ == "__main__":
    coords_file = "C:/Users/Daniel/PycharmProjects/YOLOv8-DeepSORT-Object-Tracking/runs/detect/train22/occlusion_coords.json"
    output_file = "C:/Users/Daniel/PycharmProjects/YOLOv8-DeepSORT-Object-Tracking/runs/detect/train22/heatmap.png"
    img_shape = (720, 1280, 3)  # Replace with the shape of your video frames
    dot_size = 2

    generate_heatmap(coords_file, output_file, img_shape, dot_size)