# Ultralytics YOLO 🚀, GPL-3.0 license
import json

# Importy potrebné pre rôzne funkcie, ako je spracovanie obrázkov, sledovanie objektov a generovanie heatmapy
import hydra
import torch
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from generate_heatmap import generate_heatmap

import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np

# Paleta farieb pre vizualizáciu
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

deepsort = None

# Inicializácia sledovača DeepSort
def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    # Nastavenie parametrov DeepSort sledovača
    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
# Funkcia na konverziu súradníc z formátu xyxy na xywh
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h
# Funkcia na konverziu súradníc z formátu xyxy na tlwh
def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

# Funkcia na výpočet farby pre rôzne triedy objektov
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

# Funkcia na vykreslenie okrajov okolo textu
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

# Funkcia na vykreslenie jedného ohraničujúceho rámca na obrázku
def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# Funkcia na vykreslenie boxov a vizualizáciu objektov
def draw_boxes(img, bbox, names, object_id, identities=None, confidences=None, offset=(0, 0)):
    height, width, _ = img.shape

    # Odstránenie ID, ktoré už nie sú sledované
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    # Kreslenie mriežky 16x16
    num_rows, num_cols = 16, 16
    row_height = height // num_rows
    col_width = width // num_cols

    for i in range(1, num_rows):
        y = i * row_height
        cv2.line(img, (0, y), (width, y), (255, 255, 255), 1)

    for j in range(1, num_cols):
        x = j * col_width
        cv2.line(img, (x, 0), (x, height), (255, 255, 255), 1)        

    # Kreslenie boxov pre každý detegovaný objekt
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # Výpočet stredu spodného okraja boxu
        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))

        # Získanie ID objektu
        id = int(identities[i]) if identities is not None else 0

        # Vytvorenie nového bufferu pre nový objekt
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        conf = confidences[i] if confidences is not None else 0
        label = '{}{:d}'.format("", id) + ":" + '%s %.2f' % (obj_name, conf)

        # Pridanie stredu do bufferu
        data_deque[id].appendleft(center)
        UI_box(box, img, label=label, color=color, line_thickness=2)

        # Kreslenie trajektórií
        for i in range(1, len(data_deque[id])):          
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # Výpočet hrúbky pre trajektóriu
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            # Kreslenie trajektórie
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)

        # Zobrazenie statusu "occluded"
        occlusion_status = deepsort.get_occlusion_status(id)
        if occlusion_status:
            cv2.putText(img, "Occluded", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    return img

# Trieda na predikciu detekcií
class DetectionPredictor(BasePredictor):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.first_appearance = {}  # Inicializácia slovníka pre prvý výskyt objektov
        self.previous_positions = {}  # Inicializácia slovníka pre predchádzajúce pozície objektov

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    # Predspracovanie obrázkov
    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    # Postprocesing obrázkov
    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    # Funkcia na spracovanie predikcií a ukladanie výsledkov
    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        occluded_objects = []  # Zoznam pre prekryté objekty
        occlusion_coords = []  # Zoznam pre súradnice oklúzií
        first_appearance_coords = []  # Zoznam pre súradnice prvého výskytu
        movement_directions = {}  # Slovník pre smery pohybu v každej bunke mriežky

        if len(im.shape) == 3:
            im = im[None]  
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)  # cesta k uloženiu obrázka
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # výpis rozmerov
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        all_outputs.append(det)
        if len(det) == 0:
            return log_string

        # Počítanie detekcií podľa tried    
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # počet detekcií pre triedu
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # Príprava dát pre DeepSort
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] 
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        # Spracovanie každej detekcie
        for *xyxy, conf, cls in reversed(det):
            if int(cls) != 2:  # Filter iba pre triedu "car" (ID 2)
                continue
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append(conf.item())
            oids.append(int(cls))
        if len(xywh_bboxs) > 0:  # Kontrola, či existujú detekcie
            xywhs = torch.Tensor(xywh_bboxs)
            confss = torch.Tensor(confs)

            # Aktualizácia DeepSort sledovača
            outputs = deepsort.update(xywhs, confss, oids, im0)
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -2]
                object_id = outputs[:, -1]

                # Vytvorenie slovníka pre mapovanie ID na skóre dôveryhodnosti
                track_confidences = {track_id: conf for track_id, conf in zip(identities, confs)}

                # Zarovnanie dôveryhodností s ID objektov
                aligned_confs = [track_confidences.get(track_id, 0) for track_id in identities]

                # Vykreslenie boxov a trajektórií
                draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities, confidences=aligned_confs)

                # Zaznamenávanie prekrytých objektov a súradníc prvého výskytu
                for i, track_id in enumerate(identities):
                    if deepsort.get_occlusion_status(track_id):
                        # Zaznamenanie prekrytého objektu
                        occluded_objects.append({
                            'track_id': int(track_id),  
                            'bbox': [int(coord) for coord in bbox_xyxy[i]],  
                            'class': self.model.names[int(object_id[i])] 
                        })
                        # Zaznamenanie stredu ohraničujúceho rámca ako súradnice oklúzie
                        x_center = (bbox_xyxy[i][0] + bbox_xyxy[i][2]) // 2
                        y_center = (bbox_xyxy[i][1] + bbox_xyxy[i][3]) // 2
                        occlusion_coords.append((x_center, y_center))
                    # Zaznamenanie súradníc prvého výskytu objektu
                    if track_id not in self.first_appearance:
                        self.first_appearance[track_id] = (bbox_xyxy[i][0] + bbox_xyxy[i][2]) // 2, (bbox_xyxy[i][1] + bbox_xyxy[i][3]) // 2
                        first_appearance_coords.append(self.first_appearance[track_id])

                    # Sledovanie smeru pohybu objektu
                    current_position = (bbox_xyxy[i][0] + bbox_xyxy[i][2]) // 2, (bbox_xyxy[i][1] + bbox_xyxy[i][3]) // 2
                    if track_id in self.previous_positions:
                        prev_position = self.previous_positions[track_id]
                        dx = current_position[0] - prev_position[0]
                        dy = current_position[1] - prev_position[1]
                        direction = self.calculate_direction(dx, dy)
                        grid_cell = (current_position[1] // (im0.shape[0] // 16), current_position[0] // (im0.shape[1] // 16))
                        if grid_cell not in movement_directions:
                            movement_directions[grid_cell] = []
                        movement_directions[grid_cell].append(direction)
                    self.previous_positions[track_id] = current_position

        # Uloženie dát o prekrytých objektoch do súboru
        if occluded_objects:
            with open(self.save_dir / 'occluded_objects.json', 'a') as f:
                json.dump(occluded_objects, f)
                f.write('\n')

        # Uloženie súradníc oklúzie do súboru
        if occlusion_coords:
            with open(self.save_dir / 'occlusion_coords.json', 'a') as f:
                json.dump([(int(x), int(y)) for x, y in occlusion_coords], f)
                f.write('\n')

        # Uloženie súradníc prvého výskytu do súboru
        if first_appearance_coords:
            with open(self.save_dir / 'first_appearance_coords.json', 'a') as f:
                json.dump([(int(x), int(y)) for x, y in first_appearance_coords], f)
                f.write('\n')

        # Uloženie smerov pohybu do súboru
        if movement_directions:
            with open(self.save_dir / 'movement_directions.json', 'a') as f:
                json.dump({str(k): v for k, v in movement_directions.items()}, f)
                f.write('\n')

        return log_string

    def calculate_direction(self, dx, dy):
        if abs(dx) > abs(dy):
            if dx > 0:
                return "east" # východ
            else:
                return "west" # západ
        else:
            if dy > 0:
                return "south" # juh
            else:
                return "north" # sever

    def run_generate_heatmap(self):
    
        save_dir = str(self.save_dir)
        coords_file = str(Path(save_dir) / "occlusion_coords.json")
        first_appearance_file = str(Path(save_dir) / "first_appearance_coords.json")
        movement_directions_file = str(Path(save_dir) / "movement_directions.json")
        output_file = str(Path(save_dir) / "heatmap.png")
        output_file2 = str(Path(save_dir) / "heatmap2.png")
        grid_output_file = str(Path(save_dir) / "grid_counts.json")
        contaminated_output_file = str(Path(save_dir) / "contaminated_squares.json")
    
    # Priame volanie funkcie generate_heatmap namiesto použitia subprocess
    
        generate_heatmap(
            coords_file=coords_file,
            first_appearance_file=first_appearance_file,
            movement_directions_file=movement_directions_file,
            output_file=output_file,
            output_file2=output_file2,
            img_shape=(720, 1280, 3),  # Prispôsobte rozmerom vášho videa
            dot_size=2,
            grid_size=16,
            grid_output_file=grid_output_file,
            contaminated_output_file=contaminated_output_file
        )            

# Hlavná funkcia, ktorá spúšťa predikciu pomocou Hydra frameworku
@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker() # Inicializácia DeepSort sledovača
    cfg.model = cfg.model or "yolov8n.pt" # Použitie predvoleného modelu, ak nie je špecifikovaný
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # Kontrola veľkosti obrázka
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets" # Určenie zdroja dát
    predictor = DetectionPredictor(cfg) # Vytvorenie prediktora
    predictor() # Spustenie predikcie
    predictor.run_generate_heatmap()  # Generovanie tepelných máp po dokončení predikcie

# Spustenie hlavnej funkcie, ak je skript spustený priamo
if __name__ == "__main__":
    predict()
