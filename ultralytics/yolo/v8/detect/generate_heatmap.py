import json
import cv2
import numpy as np
from pathlib import Path

def generate_heatmap(coords_file, first_appearance_file, movement_directions_file, output_file, output_file2, img_shape, dot_size, grid_size, grid_output_file, contaminated_output_file):
    # Načítanie súradníc oklúzií z JSON súboru
    coords = []
    with open(coords_file, 'r') as f:
        for line in f:
            coords.extend(json.loads(line))

    # Načítanie súradníc prvého výskytu objektov z JSON súboru
    first_appearance_coords = []
    with open(first_appearance_file, 'r') as f:
        for line in f:
            first_appearance_coords.extend(json.loads(line))

    # Načítanie smerov pohybu z JSON súboru
    movement_directions = {}
    with open(movement_directions_file, 'r') as f:
        for line in f:
            movement_directions.update(json.loads(line))

    # Vytvorenie dvoch tepelných máp
    heatmap = np.zeros(img_shape[:2], dtype=np.float32)  # Pre prvý výskyt, oklúziu a smery pohybu
    heatmap2 = np.full((img_shape[0], img_shape[1], 3), (255, 0, 0), dtype=np.uint8)  # Modré pozadie pre kontaminované štvorce

     # Zvýšenie hodnôt tepelnej mapy na súradniciach oklúzií
    for x, y in coords:
        cv2.circle(heatmap, (x, y), dot_size, 1, thickness=-1)

    # Normalizácia tepelnej mapy
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)

    # Aplikácia farebnej mapy na tepelnú mapu
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Mriežka 16x16
    num_rows, num_cols = grid_size, grid_size
    row_height = img_shape[0] // num_rows
    col_width = img_shape[1] // num_cols

    # Vytvorenie mriežky
    for i in range(1, num_rows):
        y = i * row_height
        cv2.line(heatmap, (0, y), (img_shape[1], y), (255, 255, 255), 1)
        cv2.line(heatmap2, (0, y), (img_shape[1], y), (255, 255, 255), 1)

    for j in range(1, num_cols):
        x = j * col_width
        cv2.line(heatmap, (x, 0), (x, img_shape[0]), (255, 255, 255), 1)
        cv2.line(heatmap2, (x, 0), (x, img_shape[0]), (255, 255, 255), 1)

    # Počítanie oklúzií a prvých výskytov v každej bunke mriežky
    occlusion_counts = np.zeros((num_rows, num_cols), dtype=int)
    first_appearance_counts = np.zeros((num_rows, num_cols), dtype=int)

    # Počítanie oklúzií v každej bunke
    for x, y in coords:
        row = y // row_height
        col = x // col_width
        occlusion_counts[row, col] += 1

     # Počítanie prvých výskytov v každej bunke
    for x, y in first_appearance_coords:
        row = y // row_height
        col = x // col_width
        first_appearance_counts[row, col] += 1

    # Kreslenie súradníc prvého výskytu na tepelnej mape
    for x, y in first_appearance_coords:
        cv2.circle(heatmap, (x, y), dot_size, (0, 255, 0), thickness=-1)  # Zelená farba pre prvý výskyt

    # Hľadanie a označenie kontaminovaných štvorcov
    contaminated_squares = []
    for row in range(num_rows):
        for col in range(num_cols):
            # Štvorec je potenciálne kontaminovaný, ak má viac ako 1
            if occlusion_counts[row, col] > 1 and first_appearance_counts[row, col] == 0:
                 # Získanie smeru pohybu pre tento štvorec
                direction = movement_directions.get(str((row, col)), [])
                if not direction:
                    continue
                # Určenie najčastejšieho smeru pohybu    
                direction = max(set(direction), key=direction.count)  # Najčastejší smer

                # Hľadanie v smere pohybu
                current_row, current_col = row, col
                path_squares = [(row, col)]  # Zahrnutie počiatočného štvorca do cesty
                found = False
                while True:
                    # Aktualizácia pozície podľa smeru pohybu
                    if direction == "north":
                        current_row -= 1
                    elif direction == "south":
                        current_row += 1
                    elif direction == "east":
                        current_col += 1
                    elif direction == "west":
                        current_col -= 1

                    # Zastavenie, ak sme mimo hraníc
                    if current_row < 0 or current_row >= num_rows or current_col < 0 or current_col >= num_cols:
                        break

                    # Pridanie štvorca do cesty
                    path_squares.append((current_row, current_col))

                     # Kontrola, či má štvorec viac ako 1 súradnicu prvého výskytu
                    if first_appearance_counts[current_row, current_col] > 1:
                        found = True
                        break

                 # Ak je nájdená platná cesta, označíme všetky štvorce v ceste ako kontaminované
                if found:
                    contaminated_squares.extend(path_squares)  # Zahrnutie všetkých štvorcov v ceste
                    contaminated_squares.append((current_row, current_col))  # Zahrnutie koncového štvorca

    # Kreslenie kontaminovaných štvorcov na heatmap2
    for r, c in contaminated_squares:
        x1, y1 = c * col_width, r * row_height
        x2, y2 = x1 + col_width, y1 + row_height
        cv2.rectangle(heatmap2, (x1, y1), (x2, y2), (0, 255, 255), -1)  # Žltá farba pre kontaminované štvorce

    # Kreslenie smerov pohybu na tepelnej mape pomocou šípok
    for cell, directions in movement_directions.items():
        row, col = eval(cell)
        x = col * col_width + col_width // 2 # Stred bunky x
        y = row * row_height + row_height // 2 # Stred bunky y
        direction = max(set(directions), key=directions.count) # Najčastejší smer
        # Kreslenie šípky podľa smeru
        if direction == "north":
            cv2.arrowedLine(heatmap, (x, y), (x, y - row_height // 2), (255, 0, 0), 2)
        elif direction == "south":
            cv2.arrowedLine(heatmap, (x, y), (x, y + row_height // 2), (255, 0, 0), 2)
        elif direction == "east":
            cv2.arrowedLine(heatmap, (x, y), (x + col_width // 2, y), (255, 0, 0), 2)
        elif direction == "west":
            cv2.arrowedLine(heatmap, (x, y), (x - col_width // 2, y), (255, 0, 0), 2)

     # Uloženie tepelných máp
    cv2.imwrite(output_file, heatmap)  # Uloženie tepelnej mapy s prvým výskytom, oklúziou a smermi pohybu
    cv2.imwrite(output_file2, heatmap2)  # Uloženie tepelnej mapy len s kontaminovanými štvorcami

     # Uloženie kontaminovaných štvorcov do súboru
    with open(contaminated_output_file, 'w') as f:
        json.dump(contaminated_squares, f)

    # Uloženie počtov v mriežke do súboru
    with open(grid_output_file, 'w') as f:
        for row in occlusion_counts:
            json.dump(row.tolist(), f)
            f.write('\n')

# Získa všetky štvorce medzi dvoma mriežkovými štvorcami
def get_squares_between(row1, col1, row2, col2):
    """Get all squares between two grid squares."""
    squares = []
    if row1 == row2:  # Horizontálny pohyb
        for c in range(min(col1, col2) + 1, max(col1, col2)):
            squares.append((row1, c))
    elif col1 == col2:  # Vertikálny pohyb
        for r in range(min(row1, row2) + 1, max(row1, row2)):
            squares.append((r, col1))
    return squares

# Spustenie ako samostatný skript
if __name__ == "__main__":
    import argparse
    
    # Nastavenie parsera argumentov príkazového riadku
    parser = argparse.ArgumentParser(description="Generate heatmaps from tracking data")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory containing input files and where to save outputs")
    parser.add_argument("--img_shape", type=tuple, default=(720, 1280, 3), help="Video frame dimensions")
    parser.add_argument("--dot_size", type=int, default=2, help="Size of dots on heatmap")
    parser.add_argument("--grid_size", type=int, default=16, help="Number of grid divisions")
    
    args = parser.parse_args()
    
    # Vytvorenie cesty k adresáru
    save_dir = Path(args.save_dir)

    # Volanie funkcie na generovanie tepelnej mapy
    generate_heatmap(
        coords_file=str(save_dir / "occlusion_coords.json"),
        first_appearance_file=str(save_dir / "first_appearance_coords.json"),
        movement_directions_file=str(save_dir / "movement_directions.json"),
        output_file=str(save_dir / "heatmap.png"),
        output_file2=str(save_dir / "heatmap2.png"),
        img_shape=args.img_shape,
        dot_size=args.dot_size,
        grid_size=args.grid_size,
        grid_output_file=str(save_dir / "grid_counts.json"),
        contaminated_output_file=str(save_dir / "contaminated_squares.json")
    )