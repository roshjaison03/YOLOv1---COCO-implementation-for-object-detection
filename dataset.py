"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image
import json
from collections import defaultdict

class COCODataset(torch.utils.data.Dataset):
    """Dataset loader for COCO-format annotations (JSON).

    Produces the same (image, label_matrix) outputs as `VOCDataset` so
    the rest of the training code can remain unchanged.
    """

    def __init__(self, annotation_file, img_dir, S=7, B=2, C=80, transform=None):
        with open(annotation_file, "r") as f:
            ann = json.load(f)

        self.img_dir = img_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

        # images list and mapping
        self.images = ann.get("images", [])
        self.id_to_info = {img["id"]: img for img in self.images}

        # annotations grouped by image_id
        self.ann_by_image = defaultdict(list)
        for a in ann.get("annotations", []):
            self.ann_by_image[a["image_id"]].append(a)

        # category id mapping: COCO category ids may be non-contiguous
        categories = ann.get("categories", [])
        if categories:
            orig_ids = sorted([c["id"] for c in categories])
            self.cat_to_idx = {orig: i for i, orig in enumerate(orig_ids)}
            # allow user-specified C to override, but keep mapping size
            if C != len(orig_ids):
                self.C = len(orig_ids)
        else:
            # fallback: derive from annotations
            cat_ids = sorted({a["category_id"] for a in ann.get("annotations", [])})
            self.cat_to_idx = {orig: i for i, orig in enumerate(cat_ids)}
            if C != len(cat_ids):
                self.C = len(cat_ids)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_info = self.images[index]
        image_id = img_info["id"]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        img_w, img_h = image.size

        boxes = []
        for ann in self.ann_by_image.get(image_id, []):
            # COCO bbox: [x_min, y_min, width, height] in pixels
            x_min, y_min, w_px, h_px = ann["bbox"]
            x_center = (x_min + w_px / 2.0) / img_w
            y_center = (y_min + h_px / 2.0) / img_h
            w = w_px / img_w
            h = h_px / img_h
            class_label = self.cat_to_idx[ann["category_id"]]
            boxes.append([class_label, x_center, y_center, w, h])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 5))
        else:
            boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = width * self.S, height * self.S

            obj_idx = self.C
            box_start = self.C + 1
            if label_matrix[i, j, obj_idx] == 0:
                label_matrix[i, j, obj_idx] = 1
                label_matrix[i, j, box_start: box_start + 4] = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
    
    
#class VOCDataset(torch.utils.data.Dataset):
#     def __init__(
#         self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
#     ):
#         self.annotations = pd.read_csv(csv_file)
#         self.img_dir = img_dir
#         self.label_dir = label_dir
#         self.transform = transform
#         self.S = S
#         self.B = B
#         self.C = C

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, index):
#         label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
#         boxes = []
#         with open(label_path) as f:
#             for label in f.readlines():
#                 class_label, x, y, width, height = [
#                     float(x) if float(x) != int(float(x)) else int(x)
#                     for x in label.replace("\n", "").split()
#                 ]

#                 boxes.append([class_label, x, y, width, height])

#         img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
#         image = Image.open(img_path)
#         boxes = torch.tensor(boxes)

#         if self.transform:
#             # image = self.transform(image)
#             image, boxes = self.transform(image, boxes)

#         # Convert To Cells
#         label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
#         for box in boxes:
#             class_label, x, y, width, height = box.tolist()
#             class_label = int(class_label)

#             # i,j represents the cell row and cell column
#             i, j = int(self.S * y), int(self.S * x)
#             x_cell, y_cell = self.S * x - j, self.S * y - i

#             """
#             Calculating the width and height of cell of bounding box,
#             relative to the cell is done by the following, with
#             width as the example:
            
#             width_pixels = (width*self.image_width)
#             cell_pixels = (self.image_width)
            
#             Then to find the width relative to the cell is simply:
#             width_pixels/cell_pixels, simplification leads to the
#             formulas below.
#             """
#             width_cell, height_cell = (
#                 width * self.S,
#                 height * self.S,
#             )

#             # If no object already found for specific cell i,j
#             # Note: This means we restrict to ONE object per cell
#             obj_idx = self.C
#             box_start = self.C + 1
#             if label_matrix[i, j, obj_idx] == 0:
#                 # Set that there exists an object
#                 label_matrix[i, j, obj_idx] = 1

#                 # Box coordinates
#                 box_coordinates = torch.tensor(
#                     [x_cell, y_cell, width_cell, height_cell]
#                 )

#                 label_matrix[i, j, box_start: box_start + 4] = box_coordinates

#                 # Set one hot encoding for class_label
#                 label_matrix[i, j, class_label] = 1

#         return image, label_matrix