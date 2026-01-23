import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import os
from PIL import Image
import torchvision
import torchvision.transforms as transforms # Import torchvision.transforms

class COCODataset(Dataset):
    def __init__(self, annotation_path, image_dir, S, B, C=1, transform=None):
        self.coco = COCO(annotation_path)
        self.image_dir = image_dir
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform # This will now be a torchvision.transforms.Compose
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def extract_boxes(self, image_id, img_width, img_height):
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']

            # Convert to x1, y1, x2, y2 and normalize
            x1 = x / img_width
            y1 = y / img_height
            x2 = (x + w) / img_width
            y2 = (y + h) / img_height

            boxes.append([x1, y1, x2, y2])

        return torch.tensor(boxes) if boxes else torch.zeros((0, 4))

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Get image info
        img_info = self.coco.loadImgs([image_id])[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])

        # Load image (cv2 reads as BGR, convert to RGB, then to PIL Image)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image) # Convert to PIL Image for torchvision transforms

        # Extract bounding boxes in [x1, y1, x2, y2] normalized format
        raw_boxes_xyxy = self.extract_boxes(
            image_id=image_id,
            img_width=img_info['width'],
            img_height=img_info['height']
        )

        # Get raw labels (category_ids)
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ann_ids)
        # category_ids in COCO are 1-indexed, adjust to 0-indexed for class_label
        raw_labels = torch.tensor([ann['category_id'] - 1 for ann in anns], dtype=torch.long)

        # Apply image transformations. Bounding box adjustments will be handled for geometric transforms if needed,
        # but for typical YOLO transforms (Resize, ToTensor), we can apply them directly to the image.
        # Bounding box coordinates extracted by extract_boxes are already normalized.
        if self.transform:
            # Removed: Incorrect scaling of raw_boxes_xyxy based on image resize.
            # Normalized bounding box coordinates (0-1) should not change with image pixel resizing.
            image = self.transform(image) # Apply all image transforms

        # Now, construct the YOLO label matrix (S, S, C + B*5)
        # Format: [class_prob (C elements), conf1, x1, y1, w1, h1, conf2, x2, y2, w2, h2]
        target_matrix = torch.zeros((self.S, self.S, self.C + self.B * 5))

        if raw_boxes_xyxy.numel() > 0: # If there are any boxes
            for i, bbox_xyxy in enumerate(raw_boxes_xyxy):
                x1, y1, x2, y2 = bbox_xyxy.tolist()

                # Convert [x1, y1, x2, y2] to [x_center, y_center, width, height] normalized
                x_center_norm = (x1 + x2) / 2.0
                y_center_norm = (y1 + y2) / 2.0
                width_norm = x2 - x1
                height_norm = y2 - y1

                class_id = raw_labels[i].item() # 0-indexed class id

                # Determine which grid cell the center of the object falls into
                cell_x = int(self.S * x_center_norm)
                cell_y = int(self.S * y_center_norm)

                # Calculate cell-relative coordinates (0 to 1 within the cell)
                x_cell = self.S * x_center_norm - cell_x
                y_cell = self.S * y_center_norm - cell_y

                # Calculate width and height relative to the entire image, NOT scaled by S
                w_cell = width_norm
                h_cell = height_norm

                # Ensure cell indices are within bounds
                if cell_x < 0 or cell_x >= self.S or cell_y < 0 or cell_y >= self.S:
                    # This should ideally not happen if bounding boxes are valid and normalized
                    continue

                # Assign values to the target matrix for the first box predictor (B=0)
                # Check if objectness score for this cell's first box is 0
                # This prevents overwriting if multiple boxes fall into the same cell
                if target_matrix[cell_y, cell_x, self.C] == 0: # C is the index for objectness of first box
                    target_matrix[cell_y, cell_x, self.C] = 1.0 # Objectness score for box 1
                    # Coords for box 1 (x, y, w, h)
                    target_matrix[cell_y, cell_x, self.C + 1:self.C + 5] = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                    # Class probability (one-hot encoding) - at index 'class_id'
                    target_matrix[cell_y, cell_x, class_id] = 1.0

        return image, target_matrix