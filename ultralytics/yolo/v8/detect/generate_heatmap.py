import json
import cv2
import numpy as np
from pathlib import Path

def generate_heatmap(coords_file, first_appearance_file, movement_directions_file, output_file, output_file2, img_shape, dot_size, grid_size, grid_output_file, contaminated_output_file):
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

    # Create two heatmaps
    heatmap = np.zeros(img_shape[:2], dtype=np.float32)  # For first appearance, occlusion, and movement directions
    heatmap2 = np.full((img_shape[0], img_shape[1], 3), (255, 0, 0), dtype=np.uint8)  # Blue background for contaminated squares

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
        cv2.line(heatmap2, (0, y), (img_shape[1], y), (255, 255, 255), 1)

    for j in range(1, num_cols):
        x = j * col_width
        cv2.line(heatmap, (x, 0), (x, img_shape[0]), (255, 255, 255), 1)
        cv2.line(heatmap2, (x, 0), (x, img_shape[0]), (255, 255, 255), 1)

    # Count occlusions and first appearances in each grid cell
    occlusion_counts = np.zeros((num_rows, num_cols), dtype=int)
    first_appearance_counts = np.zeros((num_rows, num_cols), dtype=int)

    for x, y in coords:
        row = y // row_height
        col = x // col_width
        occlusion_counts[row, col] += 1

    for x, y in first_appearance_coords:
        row = y // row_height
        col = x // col_width
        first_appearance_counts[row, col] += 1

    # Draw first appearance coordinates on heatmap
    for x, y in first_appearance_coords:
        cv2.circle(heatmap, (x, y), dot_size, (0, 255, 0), thickness=-1)  # Green for first appearance

    # Find and mark contaminated squares
    contaminated_squares = []
    for row in range(num_rows):
        for col in range(num_cols):
            if occlusion_counts[row, col] > 1 and first_appearance_counts[row, col] == 0:
                # Get the movement direction for this square
                direction = movement_directions.get(str((row, col)), [])
                if not direction:
                    continue
                direction = max(set(direction), key=direction.count)  # Most common direction

                # Search in the movement direction
                current_row, current_col = row, col
                path_squares = [(row, col)]  # Include the starting square in the path
                found = False
                while True:
                    if direction == "north":
                        current_row -= 1
                    elif direction == "south":
                        current_row += 1
                    elif direction == "east":
                        current_col += 1
                    elif direction == "west":
                        current_col -= 1

                    # Stop if out of bounds
                    if current_row < 0 or current_row >= num_rows or current_col < 0 or current_col >= num_cols:
                        break

                    # Add the square to the path
                    path_squares.append((current_row, current_col))

                    # Check if the square has more than 1 first appearance coordinate
                    if first_appearance_counts[current_row, current_col] > 1:
                        found = True
                        break

                # If a valid path is found, mark all squares in the path as contaminated
                if found:
                    contaminated_squares.extend(path_squares)  # Include all squares in the path
                    contaminated_squares.append((current_row, current_col))  # Include the ending square

    # Draw contaminated squares on heatmap2
    for r, c in contaminated_squares:
        x1, y1 = c * col_width, r * row_height
        x2, y2 = x1 + col_width, y1 + row_height
        cv2.rectangle(heatmap2, (x1, y1), (x2, y2), (0, 255, 255), -1)  # Yellow color

    # Draw movement directions on heatmap
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

    # Save the heatmaps
    cv2.imwrite(output_file, heatmap)  # Save heatmap with first appearance, occlusion, and movement directions
    cv2.imwrite(output_file2, heatmap2)  # Save heatmap with contaminated squares only

    # Save contaminated squares to a file
    with open(contaminated_output_file, 'w') as f:
        json.dump(contaminated_squares, f)

    # Save grid counts to a file
    with open(grid_output_file, 'w') as f:
        for row in occlusion_counts:
            json.dump(row.tolist(), f)
            f.write('\n')


def get_squares_between(row1, col1, row2, col2):
    """Get all squares between two grid squares."""
    squares = []
    if row1 == row2:  # Horizontal movement
        for c in range(min(col1, col2) + 1, max(col1, col2)):
            squares.append((row1, c))
    elif col1 == col2:  # Vertical movement
        for r in range(min(row1, row2) + 1, max(row1, row2)):
            squares.append((r, col1))
    return squares


if __name__ == "__main__":
    coords_file = "C:/Users/Daniel/PycharmProjects/YOLOv8-DeepSORT-Object-Tracking/runs/detect/train33/occlusion_coords.json"
    first_appearance_file = "C:/Users/Daniel/PycharmProjects/YOLOv8-DeepSORT-Object-Tracking/runs/detect/train33/first_appearance_coords.json"
    movement_directions_file = "C:/Users/Daniel/PycharmProjects/YOLOv8-DeepSORT-Object-Tracking/runs/detect/train33/movement_directions.json"
    output_file = "C:/Users/Daniel/PycharmProjects/YOLOv8-DeepSORT-Object-Tracking/runs/detect/train33/heatmap.png"
    output_file2 = "C:/Users/Daniel/PycharmProjects/YOLOv8-DeepSORT-Object-Tracking/runs/detect/train33/heatmap2.png"
    grid_output_file = "C:/Users/Daniel/PycharmProjects/YOLOv8-DeepSORT-Object-Tracking/runs/detect/train33/grid_counts.json"
    contaminated_output_file = "C:/Users/Daniel/PycharmProjects/YOLOv8-DeepSORT-Object-Tracking/runs/detect/train33/contaminated_squares.json"
    img_shape = (720, 1280, 3)  # Replace with the shape of your video frames
    dot_size = 2
    grid_size = 16

    generate_heatmap(coords_file, first_appearance_file, movement_directions_file, output_file, output_file2, img_shape, dot_size, grid_size, grid_output_file, contaminated_output_file)