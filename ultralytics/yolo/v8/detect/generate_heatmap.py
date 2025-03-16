import json
import cv2
import numpy as np
from pathlib import Path

def generate_heatmap(coords_file, first_appearance_file, movement_directions_file, output_file, img_shape, dot_size, grid_size, grid_output_file):
    # Load occlusion coordinates
    coords = []
    with open(coords_file, 'r') as f:
        for line in f:
            coords.extend(json.loads(line))

    # Load first appearance coordinates
    first_appearance_coords = []
    with open(first_appearance_file, 'r') as f:
        for line in f:
            first_appearance_coords.extend(json.loads(line))

    # Load movement directions
    movement_directions = {}
    with open(movement_directions_file, 'r') as f:
        for line in f:
            movement_directions.update(json.loads(line))

    # Create an empty heatmap
    heatmap = np.zeros(img_shape[:2], dtype=np.float32)

    # Increment heatmap values at occlusion coordinates
    for x, y in coords:
        cv2.circle(heatmap, (x, y), dot_size, 1, thickness=-1)

    # Normalize the heatmap
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)

    # Apply a color map to the heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Draw 16x16 grid
    num_rows, num_cols = grid_size, grid_size
    row_height = img_shape[0] // num_rows
    col_width = img_shape[1] // num_cols

    for i in range(1, num_rows):
        y = i * row_height
        cv2.line(heatmap, (0, y), (img_shape[1], y), (255, 255, 255), 1)

    for j in range(1, num_cols):
        x = j * col_width
        cv2.line(heatmap, (x, 0), (x, img_shape[0]), (255, 255, 255), 1)

    # Draw first appearance coordinates in a different color
    for x, y in first_appearance_coords:
        cv2.circle(heatmap, (x, y), dot_size, (0, 255, 0), thickness=-1)

    # Draw movement directions
    for cell, directions in movement_directions.items():
        row, col = eval(cell)
        x = col * col_width + col_width // 2
        y = row * row_height + row_height // 2
        direction = max(set(directions), key=directions.count)
        if direction == "north":
            cv2.arrowedLine(heatmap, (x, y), (x, y - row_height // 2), (255, 0, 0), 2)
        elif direction == "south":
            cv2.arrowedLine(heatmap, (x, y), (x, y + row_height // 2), (255, 0, 0), 2)
        elif direction == "east":
            cv2.arrowedLine(heatmap, (x, y), (x + col_width // 2, y), (255, 0, 0), 2)
        elif direction == "west":
            cv2.arrowedLine(heatmap, (x, y), (x - col_width // 2, y), (255, 0, 0), 2)

    # Save the heatmap
    cv2.imwrite(output_file, heatmap)

    # Count occlusions in each grid cell
    grid_counts = np.zeros((num_rows, num_cols), dtype=int)
    for x, y in coords:
        row = y // row_height
        col = x // col_width
        grid_counts[row, col] += 1

    # Save grid counts to a file
    with open(grid_output_file, 'w') as f:
        for row in grid_counts:
            json.dump(row.tolist(), f)
            f.write('\n')

if __name__ == "__main__":
    coords_file = "C:/Users/Daniel/PycharmProjects/YOLOv8-DeepSORT-Object-Tracking/runs/detect/train33/occlusion_coords.json"
    first_appearance_file = "C:/Users/Daniel/PycharmProjects/YOLOv8-DeepSORT-Object-Tracking/runs/detect/train33/first_appearance_coords.json"
    movement_directions_file = "C:/Users/Daniel/PycharmProjects/YOLOv8-DeepSORT-Object-Tracking/runs/detect/train33/movement_directions.json"
    output_file = "C:/Users/Daniel/PycharmProjects/YOLOv8-DeepSORT-Object-Tracking/runs/detect/train33/heatmap.png"
    grid_output_file = "C:/Users/Daniel/PycharmProjects/YOLOv8-DeepSORT-Object-Tracking/runs/detect/train33/grid_counts.json"
    img_shape = (720, 1280, 3)  # Replace with the shape of your video frames
    dot_size = 2
    grid_size = 16

    generate_heatmap(coords_file, first_appearance_file, movement_directions_file, output_file, img_shape, dot_size, grid_size, grid_output_file)