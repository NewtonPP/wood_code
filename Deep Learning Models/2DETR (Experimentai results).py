# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 13:00:20 2025

@author: ae1028
"""



from torchvision.ops import nms
import glob
import csv
import torch
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning
import os
import cv2
import shutil
import json
import random
import numpy as np
import supervision as sv
from coco_eval import CocoEvaluator
from tqdm import tqdm
import transformers
from torchvision.transforms import functional as F
from transformers import DetrForObjectDetection, DetrImageProcessor
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from coco_eval import CocoEvaluator
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
torch.set_float32_matmul_precision('high')  # ✅ Optimizes Tensor Core usage

# Step 1: Paths
data_dir = r"E:\MSU\1Summer 2024\Wood chip\Generative model for wood chip size\Literature Review\data\woodchip_data\Annotated Labelme\Drax\full"
output_dir = r"F:\3Spring 2025\Woodchip\Report22_DETR\output\checkpoint\New folder"
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

# Step 2: Create Output Directories
for subset in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(subset, "images"), exist_ok=True)
    os.makedirs(os.path.join(subset, "annotations"), exist_ok=True)

# Step 3: Split Dataset
all_images = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
all_annotations = [f for f in os.listdir(data_dir) if f.endswith(".json")]
all_data = [(img, img.replace(".jpg", ".json")) for img in all_images if img.replace(".jpg", ".json") in all_annotations]

# 42, 13, 69, 9,32
random.seed(42)
random.shuffle(all_data)
train_split, val_split = int(0.7 * len(all_data)), int(0.85 * len(all_data))
train_data, val_data, test_data = all_data[:train_split], all_data[train_split:val_split], all_data[val_split:]

# Step 4: Move Files
def move_files(data, subset_dir):
    for img, ann in data:
        shutil.copy(os.path.join(data_dir, img), os.path.join(subset_dir, "images", img))
        shutil.copy(os.path.join(data_dir, ann), os.path.join(subset_dir, "annotations", ann))

move_files(train_data, train_dir)
move_files(val_data, val_dir)
move_files(test_data, test_dir)

print("Dataset split completed!")

# Step 5: Convert Annotations to COCO Format
def convert_to_coco_format(data_dir, output_file):
    coco_format = {
        "images": [], 
        "annotations": [], 
        "categories": [{"id": 0, "name": "woodchip"}]  
    }

    annotation_id = 0
    for img_idx, img_file in enumerate(os.listdir(os.path.join(data_dir, "images"))):
        if not img_file.endswith(".jpg"): 
            continue
        
        annotation_file = os.path.join(data_dir, "annotations", img_file.replace(".jpg", ".json"))
        with open(annotation_file, "r") as f:
            annotations = json.load(f)
        
        coco_format["images"].append({
            "id": img_idx, 
            "file_name": img_file,
            "width": annotations.get("imageWidth", 1920),
            "height": annotations.get("imageHeight", 1080),
        })

        for shape in annotations["shapes"]:
            points = shape["points"]
            x_coords, y_coords = [p[0] for p in points], [p[1] for p in points]
            bbox = [min(x_coords), min(y_coords), max(x_coords)-min(x_coords), max(y_coords)-min(y_coords)]
            segmentation = [coord for point in points for coord in point]
            
            coco_format["annotations"].append({
                "id": annotation_id, 
                "image_id": img_idx,
                "category_id": 0,  
                "bbox": bbox,
                "area": bbox[2] * bbox[3], 
                "iscrowd": 0,
                "segmentation": [segmentation],
            })
            annotation_id += 1

    with open(output_file, "w") as f:
        json.dump(coco_format, f)

    print(f"COCO annotation file saved: {output_file}")  # ✅ Added confirmation message


convert_to_coco_format(train_dir, os.path.join(train_dir, "annotations", "annotations.coco.json"))
convert_to_coco_format(val_dir, os.path.join(val_dir, "annotations", "annotations.coco.json"))
convert_to_coco_format(test_dir, os.path.join(test_dir, "annotations", "annotations.coco.json"))

print("COCO conversion completed!")



# Load the saved COCO annotation file for the train dataset
with open(os.path.join(train_dir, "annotations", "annotations.coco.json"), "r") as f:
    coco_annotations = json.load(f)

print("Number of Images:", len(coco_annotations["images"]))
print("Number of Annotations:", len(coco_annotations["annotations"]))

# Check first few annotations
for ann in coco_annotations["annotations"][:5]:
    print(f"Image ID: {ann['image_id']}, BBox: {ann['bbox']}, Category: {ann['category_id']}")





# Load the saved COCO annotation file for the train dataset
with open(os.path.join(val_dir, "annotations", "annotations.coco.json"), "r") as f:
    coco_annotations = json.load(f)

print("Number of Images:", len(coco_annotations["images"]))
print("Number of Annotations:", len(coco_annotations["annotations"]))

# Check first few annotations
for ann in coco_annotations["annotations"][:5]:
    print(f"Image ID: {ann['image_id']}, BBox: {ann['bbox']}, Category: {ann['category_id']}")






# Load the saved COCO annotation file for the train dataset
with open(os.path.join(test_dir, "annotations", "annotations.coco.json"), "r") as f:
    coco_annotations = json.load(f)

print("Number of Images:", len(coco_annotations["images"]))
print("Number of Annotations:", len(coco_annotations["annotations"]))

# Check first few annotations
for ann in coco_annotations["annotations"][:5]:
    print(f"Image ID: {ann['image_id']}, BBox: {ann['bbox']}, Category: {ann['category_id']}")






# Define the annotation file name
ANNOTATION_FILE_NAME = "annotations.coco.json"

# Directories for train, val, and test sets
TRAIN_DIRECTORY = os.path.join(train_dir)
VAL_DIRECTORY = os.path.join(val_dir)
TEST_DIRECTORY = os.path.join(test_dir)

# Load the image processor for DETR
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")

# Define Albumentations augmentations
# Define Albumentations augmentations (Ensure Pascal VOC format)
train_augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
#    A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # NEW: Blur
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # NEW: Noise
    ToTensorV2()
], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))



# Custom CocoDetection class
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_directory_path, image_processor, train=True, augmentations=None):
        annotation_file_path = os.path.join(image_directory_path, "annotations", "annotations.coco.json")
        super(CocoDetection, self).__init__(os.path.join(image_directory_path, "images"), annotation_file_path)
        self.image_processor = image_processor
        self.augmentations = augmentations if train else None  

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        images_np = np.array(images)

        # Convert bounding boxes from COCO format to Pascal VOC format
        bboxes = []
        labels = []
        for ann in annotations:
            x_min, y_min, width, height = ann["bbox"]
            x_max = x_min + width
            y_max = y_min + height
            bboxes.append([x_min, y_min, x_max, y_max])  
            labels.append(ann["category_id"])  


        original_bboxes = np.array(bboxes, dtype=np.float32).tolist()  

        # Visualize before augmentation
#        self.visualize_boxes(images_np, original_bboxes, title="Before Augmentation (Ground Truth)")

        # Apply augmentation (if enabled)
        if self.augmentations and len(bboxes) > 0:
            augmented = self.augmentations(image=images_np, bboxes=bboxes, labels=labels)
            images_np = augmented["image"]
            bboxes = augmented["bboxes"]

            # Visualize after augmentation
#            self.visualize_boxes(images_np, bboxes, title="After Augmentation")

        # Convert augmented bounding boxes back to COCO format
        bboxes_coco = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            bboxes_coco.append([x_min, y_min, width, height])  


        annotations_copy = [ann.copy() for ann in annotations]  

        for i, ann in enumerate(annotations_copy):
            ann["bbox"] = bboxes_coco[i]  

        annotations = {'image_id': image_id, 'annotations': annotations_copy}

        # Apply DETR Image Processor
        encoding = self.image_processor(images=images_np, annotations=annotations, return_tensors="pt")

        pixel_values = encoding["pixel_values"].squeeze()  
        target = encoding["labels"][0]  

        return pixel_values, target

    def visualize_boxes(self, image, boxes, title="Bounding Boxes"):
        """
        Helper function to visualize bounding boxes on an image.
        Matplotlib expects (H, W, C), but PyTorch tensors are (C, H, W),
        so we need to convert format before plotting.
        """
        if isinstance(image, torch.Tensor):  # If it's a tensor, convert it
            image = image.permute(1, 2, 0).cpu().numpy()
        elif image.shape[0] == 3:  # If it's (3, H, W), convert it to (H, W, 3)
            image = np.transpose(image, (1, 2, 0))
    
        # Ensure values are between 0-255
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
    
        for bbox in boxes:
            x_min, y_min, x_max, y_max = bbox  # Pascal VOC format
    
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)
    
        plt.title(title)
        plt.axis("off")
        plt.show()

# Load train, validation, and test datasets

TRAIN_DATASET = CocoDetection(TRAIN_DIRECTORY, image_processor, train=True, augmentations=train_augmentations)
VAL_DATASET = CocoDetection(VAL_DIRECTORY, image_processor, train=False)
TEST_DATASET = CocoDetection(TEST_DIRECTORY, image_processor, train=False)




def collate_fn(batch):
    """
    Collate function for DataLoader to pad images and create a mask.
    DETR requires padding due to varying image sizes in the dataset.
    """
    pixel_values = [item[0] for item in batch]  # Extract images
    encoding = image_processor.pad(pixel_values, return_tensors="pt")  # Pad images
    labels = [item[1] for item in batch]  # Extract labels (targets)

    return {
        'pixel_values': encoding['pixel_values'],  # Padded image tensors
        'pixel_mask': encoding['pixel_mask'],  # Padding mask
        'labels': labels  # Targets for the model
    }

# Create DataLoaders for training, validation, and testing
TRAIN_DATALOADER = DataLoader(
    dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=4, shuffle=True)

VAL_DATALOADER = DataLoader(
    dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=4)

TEST_DATALOADER = DataLoader(
    dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=4)

# Confirm dataloader creation
print("DataLoaders successfully created!")




# Define DETR Model with PyTorch Lightning
class DetrModel(pl.LightningModule):
    def __init__(self, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4):
        super().__init__()

        # Define DETR model
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-101",
            num_labels=1,  
            ignore_mismatched_sizes=True
        )

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        # ✅ Loss storage (for smooth plotting)
        self.training_losses = []
        self.validation_losses = []
        self.current_training_losses = []  # Stores per batch, reset every epoch
        self.current_validation_losses = []

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        
        # Convert labels to correct device
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        # Forward pass
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
    
        # ✅ Store per-batch loss for averaging
        self.current_training_losses.append(loss.item())
    
        # ✅ Explicitly set batch size to avoid warning
        batch_size = batch["pixel_values"].size(0)
        
        # ✅ Log training loss
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v.item(), prog_bar=False, logger=True, batch_size=batch_size)
    
        return loss


    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
    
        # ✅ Store per-batch validation loss for averaging
        self.current_validation_losses.append(loss.item())
    
        # ✅ Explicitly set batch size
        batch_size = batch["pixel_values"].size(0)
        
        # ✅ Log validation loss
        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log(f"val_{k}", v.item(), prog_bar=False, logger=True, batch_size=batch_size)
    
        return loss


    def on_train_epoch_end(self):
        """Store the averaged loss for the entire epoch"""
        if self.current_training_losses:
            avg_loss = np.mean(self.current_training_losses)
            self.training_losses.append(avg_loss)
            self.current_training_losses = []  # Reset for next epoch

    def on_validation_epoch_end(self):
        """Store the averaged validation loss for the entire epoch"""
        if self.current_validation_losses:
            avg_loss = np.mean(self.current_validation_losses)
            self.validation_losses.append(avg_loss)
            self.current_validation_losses = []  # Reset for next epoch

    def configure_optimizers(self):
        """Optimizer with different learning rate for the backbone."""
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]
            },
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def train_dataloader(self):
        return TRAIN_DATALOADER

    def val_dataloader(self):
        return VAL_DATALOADER

    def on_train_end(self):
        """✅ Plot training and validation loss curves at the end of training"""
        self.plot_losses()

    def plot_losses(self):
        """✅ Improved loss curve visualization (epoch-wise)"""
        if len(self.training_losses) == 0 or len(self.validation_losses) == 0:
            print("No loss data to plot. Ensure training and validation steps are running.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.training_losses, label="Training Loss", color="blue", linewidth=2, marker="o")
        plt.plot(self.validation_losses, label="Validation Loss", color="red", linewidth=2, marker="o")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Curves (Smoothed)")
        plt.legend()
        plt.grid()

        # ✅ Save plot
        plot_path = os.path.join(os.getcwd(), "loss_curve.png")
        plt.savefig(plot_path)
        print(f"Loss curve saved at {plot_path}")
        plt.show()



# Initialize the model
model = DetrModel()

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath= r"F:\3Spring 2025\Woodchip\Report22_DETR\output\checkpoint\New folder",  # Folder to save checkpoints
    filename="best-detr",  # Checkpoint filename
    monitor="val_loss",  # Save based on validation loss
    mode="min",  # Save when val_loss is minimized
    save_top_k=1,  # Keep only the best model
    verbose=True
)
# Set up PyTorch Lightning Trainer
trainer = Trainer(
    max_epochs=150,  # Change as needed
    gradient_clip_val=0.1,
    accumulate_grad_batches=8,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",  
#    log_every_n_steps=10,
    check_val_every_n_epoch=1,
    precision="16-mixed" if torch.cuda.is_available() else 32,  # Use mixed precision if available
    callbacks=[checkpoint_callback],
)


# Start training
trainer.fit(model)


checkpoint_dir = r"F:\3Spring 2025\Woodchip\Report22_DETR\output\checkpoint\New folder"
best_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "best-detr*.ckpt")))

if best_checkpoints:
    best_model_path = best_checkpoints[-1]  # Get the most recent best checkpoint
    print(f"✅ Loading best model from: {best_model_path}")
    model = DetrModel.load_from_checkpoint(best_model_path)

else:
    print(f"⚠️ No checkpoint found in {checkpoint_dir}. Ensure training completed successfully.")




#  Define device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.70  # Adjust confidence threshold if needed
model.to(DEVICE)
model.eval()  # Set to evaluation mode

#  Select a random test image
image_ids = TEST_DATASET.coco.getImgIds()
image_id = random.choice(image_ids)
print(f"🔹 Selected Image ID: {image_id}")

#  Load image metadata and annotations
image_info = TEST_DATASET.coco.loadImgs(image_id)[0]
annotations = TEST_DATASET.coco.imgToAnns[image_id]
image_path = os.path.join(TEST_DATASET.root, image_info["file_name"])

#  Load and preprocess the image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from OpenCV BGR to RGB

# ✅ Define Box Annotator for Visualization
box_annotator = sv.BoxAnnotator()

# ✅ Run Inference
with torch.no_grad():
    inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)

    print("🔹 Model Outputs:", outputs.keys())  # Print model output keys

    if "logits" in outputs and "pred_boxes" in outputs:
        target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
        results = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=CONFIDENCE_THRESHOLD,
            target_sizes=target_sizes
        )[0]
    else:
        print("❌ Model output missing required keys!")


    # Extract confidence scores manually
    confidence_scores = results["scores"].cpu().numpy()

# ✅ Convert model detections to Supervision format & Apply NMS
detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=0.5)

# ✅ Only display confidence scores (no label name)
labels = [f"{conf:.2f}" for conf in detections.confidence]

# ✅ Visualize detections
detected_image = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

print("🔹 Model Detections")
plt.figure(figsize=(12, 12), dpi=600)  # Increase figure size & DPI
plt.imshow(detected_image)
plt.axis("off")
plt.show()


print(detections)



# ✅ Define thresholds for confidence & NMS
CONFIDENCE_THRESHOLD = 0.50  # Adjust based on model performance
NMS_THRESHOLD = 0.40  # Adjust for best duplicate removal
IOU_THRESHOLD = 0.50  # Minimum IoU to consider a match

def compute_iou(box1, box2):
    """Compute IoU (Intersection over Union) between two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    # Compute intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    # Compute union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def filter_predictions(predictions, gt_annotations, iou_threshold=IOU_THRESHOLD):
    """Filter out predicted boxes that do not match any ground truth boxes."""
    filtered_predictions = []
    
    for pred in predictions:
        image_id = pred["image_id"]
        pred_box = pred["bbox"]  # COCO format: (x_min, y_min, width, height)

        # Get ground truth boxes for the same image
        gt_boxes = [ann["bbox"] for ann in gt_annotations if ann["image_id"] == image_id]

        # Check if this prediction has any GT match
        has_match = any(compute_iou(pred_box, gt_box) >= iou_threshold for gt_box in gt_boxes)

        if has_match:
            filtered_predictions.append(pred)  # Keep only matched predictions

    return filtered_predictions


# ✅ Convert bounding boxes to COCO format (xywh)
def convert_to_xywh(boxes):
    """Convert bounding boxes from xyxy (x_min, y_min, x_max, y_max) to COCO format (x_min, y_min, width, height)."""
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


# ✅ Apply Non-Maximum Suppression (NMS) to remove duplicate detections
def apply_nms(predictions, iou_threshold=0.5):
    """Apply NMS to remove overlapping duplicate bounding boxes."""
    filtered_predictions = {}
    for image_id, prediction in predictions.items():
        if len(prediction["boxes"]) == 0:
            continue
        
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["labels"]

        # ✅ Perform NMS
        keep_indices = nms(boxes, scores, iou_threshold)
        
        # ✅ Keep only selected boxes
        filtered_predictions[image_id] = {
            "boxes": boxes[keep_indices],
            "scores": scores[keep_indices],
            "labels": labels[keep_indices],
        }
    
    return filtered_predictions


# ✅ Prepare model predictions for COCO evaluation with confidence threshold
def prepare_for_coco_detection(predictions):
    """Convert model detections to COCO format while applying confidence filtering."""
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction["boxes"]) == 0:
            continue
        
        boxes = convert_to_xywh(prediction["boxes"]).tolist()  # Convert bbox format
        scores = prediction["scores"].tolist()  # Confidence scores
        labels = prediction["labels"].tolist()  # Class IDs

        # ✅ Apply Confidence Filtering
        filtered_results = []
        for k, box in enumerate(boxes):
            if scores[k] >= CONFIDENCE_THRESHOLD:  # Filter by confidence
                filtered_results.append({
                    "image_id": original_id,
                    "category_id": 0,  # ✅ Always 0 (Single-class: "woodchip")
                    "bbox": box,
                    "score": scores[k],
                })

        coco_results.extend(filtered_results)
    
    return coco_results


# ✅ Initialize COCO Evaluator
evaluator = CocoEvaluator(coco_gt=TEST_DATASET.coco, iou_types=["bbox"])

print("🔹 Running evaluation on test dataset...")

# ✅ Iterate over test dataset
unique_image_ids = set()
for idx, batch in enumerate(tqdm(TEST_DATALOADER)):
    pixel_values = batch["pixel_values"].to(DEVICE)
    pixel_mask = batch["pixel_mask"].to(DEVICE)
    labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    # ✅ Extract target image sizes for post-processing
    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = image_processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes)

    # ✅ Prepare predictions as dictionary
    predictions = {target["image_id"].item(): output for target, output in zip(labels, results)}

    # ✅ Collect unique image IDs (for debugging)
    batch_image_ids = list(predictions.keys())
    unique_image_ids.update(batch_image_ids)
    print(f"🔹 Processed Image IDs in Batch {idx+1}: {batch_image_ids}")

    # ✅ Apply Non-Maximum Suppression (NMS)
    predictions = apply_nms(predictions, iou_threshold=NMS_THRESHOLD)

    # ✅ Convert predictions for COCO format with confidence filtering
    predictions = prepare_for_coco_detection(predictions)

    # ✅ Fetch ground truth annotations
    gt_annotations = TEST_DATASET.coco.dataset["annotations"]

    # ✅ Filter out extra predictions that do not match any ground truth
    filtered_predictions = filter_predictions(predictions, gt_annotations)

    # ✅ Debug: Check Predictions Before COCO Evaluation
    for pred in filtered_predictions[:3]:  # Only print a few samples
        print(f"🔹 Image ID: {pred['image_id']}")
        print(f"  - Category ID: {pred['category_id']}")
        print(f"  - BBox (COCO xywh format): {pred['bbox']}")
        print(f"  - Confidence Score: {pred['score']:.3f}\n")

    evaluator.update(filtered_predictions)  # Use filtered predictions for evaluation

# ✅ Print total unique images processed
print(f"🔹 Total Unique Image IDs Processed: {len(unique_image_ids)} / {len(TEST_DATASET)}")

# ✅ Compute and display final COCO evaluation metrics
evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()





coco_eval = evaluator.coco_eval['bbox']  # 'bbox' since we're evaluating object detection
# Extract AR at IoU=0.50, size=all, maxDets=100
iou_threshold_index = 0  # IoU=0.50
size_category_index = 0  # 'all' size category
max_dets_index = 2  # maxDets=100

mAR50 = np.mean(coco_eval.eval['recall'][iou_threshold_index, :, size_category_index, max_dets_index])

print(f"🔹 mAR50 (Mean Average Recall @ IoU 0.50, maxDets=100): {mAR50:.4f}")


# Print the full AR matrix
print("🔹 Full Recall Matrix:\n", coco_eval.eval['recall'])


iou_thresholds = coco_eval.params.iouThrs  # Get list of IoU values

print("\n🔹 Average Recall (AR) for Different IoUs:\n")
for i, iou in enumerate(iou_thresholds):
    ar_value = np.mean(coco_eval.eval['recall'][i, :, size_category_index, max_dets_index])
    print(f"IoU={iou:.2f} -> AR: {ar_value:.4f}")





def visualize_gt_vs_predictions(image_id):
    """Visualize ground truth (GT) vs. model predictions for a given image ID."""
    # ✅ Load image metadata and annotations
    image_info = TEST_DATASET.coco.loadImgs(image_id)[0]
    annotations = TEST_DATASET.coco.imgToAnns[image_id]
    image_path = os.path.join(TEST_DATASET.root, image_info["file_name"])
    
    # ✅ Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # ✅ Run Model Inference
    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
        results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    # ✅ Convert results to Detections format
    detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=0.5)

    # ✅ Initialize plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)

    # ✅ Draw GT Boxes (Red)
    for ann in annotations:
        x_min, y_min, width, height = ann['bbox']
        rect = patches.Rectangle(
            (x_min, y_min), width, height, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

    # ✅ Draw Predicted Boxes (Blue)
    for box in detections.xyxy:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor="blue", facecolor="none"
        )
        ax.add_patch(rect)

    plt.title("Red: GT Bounding Boxes | Blue: Predictions")
    plt.axis("off")
    plt.show()


# ✅ Now visualize GT vs. Predictions
for i in range(5):  
    image_id = random.choice(TEST_DATASET.coco.getImgIds())
    print(f"🔹 Visualizing Image ID: {image_id}")
    visualize_gt_vs_predictions(image_id)
    
    
    
    
# Define discard threshold (adjust if needed)
discard_threshold = 0.1  # 5% margin from edges

def is_near_edge(x_min, y_min, x_max, y_max, img_width, img_height):

    is_near = (
        x_min <= img_width * discard_threshold or
        x_max >= img_width * (1 - discard_threshold) or
        y_min <= img_height * discard_threshold or
        y_max >= img_height * (1 - discard_threshold)
    )
#    if is_near:
#        print(f"❌ Discarding box near edge: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
    return is_near


def calculate_predicted_dimensions_from_detr(results, img_shape):

    dimensions = []
    img_height, img_width = img_shape[:2]

    # Extract bounding boxes
    boxes = results["boxes"].cpu().numpy()  # DETR stores boxes as a tensor

    for x_min, y_min, x_max, y_max in boxes:
        # Ensure bounding box is within image bounds
        x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(img_width, x_max), min(img_height, y_max)

        # 🚀 **Filter out near-edge bounding boxes**
        if is_near_edge(x_min, y_min, x_max, y_max, img_width, img_height):
            continue  # Skip this detection

        # Calculate dimensions
        length = abs(x_max - x_min)
        width = abs(y_max - y_min)
        diameter = (length ** 2 + width ** 2) ** 0.5  # Pythagorean theorem

        dimensions.append((length, width, diameter))

    return dimensions




def calculate_ground_truth_dimensions_from_coco(image_id, coco_dataset):

    dimensions = []
    annotations = coco_dataset.imgToAnns[image_id]

    for ann in annotations:
        x_min, y_min, width, height = ann['bbox']  # COCO format (x_min, y_min, width, height)
        x_max, y_max = x_min + width, y_min + height

        # Calculate dimensions
        length = abs(x_max - x_min)
        width = abs(y_max - y_min)
        diameter = (length ** 2 + width ** 2) ** 0.5

        dimensions.append((length, width, diameter))

    return dimensions


predicted_dimensions = []
truth_dimensions = []

# Create CSV file to store results
csv_file_path = os.path.join(output_dir, 'filtered_predicted_vs_truth_dimensions.csv')
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Image File', 'Length', 'Width', 'Diameter', 'Type'])

    # Iterate over test dataset
    for image_id in TEST_DATASET.coco.getImgIds():
        image_info = TEST_DATASET.coco.loadImgs(image_id)[0]
        image_path = os.path.join(TEST_DATASET.root, image_info["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # ✅ Get ground truth dimensions
        truth_dims = calculate_ground_truth_dimensions_from_coco(image_id, TEST_DATASET.coco)
        truth_dimensions.extend(truth_dims)

        # ✅ Run inference with DETR
        with torch.no_grad():
            inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
            results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

        # ✅ Extract **filtered** predicted dimensions
        pred_dims = calculate_predicted_dimensions_from_detr(results, image.shape)

        predicted_dimensions.extend(pred_dims)

        # ✅ Save results to CSV
        for length, width, diameter in pred_dims:
            csv_writer.writerow([image_info["file_name"], length, width, diameter, "Predicted"])
        for length, width, diameter in truth_dims:
            csv_writer.writerow([image_info["file_name"], length, width, diameter, "Ground Truth"])

print("✅ Filtered dimension extraction completed. Results saved to CSV.")




# Convert to numpy arrays
predicted_lengths = [d[0] for d in predicted_dimensions]
truth_lengths = [d[0] for d in truth_dimensions]

predicted_widths = [d[1] for d in predicted_dimensions]
truth_widths = [d[1] for d in truth_dimensions]

predicted_diameters = [d[2] for d in predicted_dimensions]
truth_diameters = [d[2] for d in truth_dimensions]

# Plot histograms
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.hist(predicted_lengths, bins=30, color='blue', edgecolor='black', alpha=0.5, label='Predicted')
plt.hist(truth_lengths, bins=30, color='orange', edgecolor='black', alpha=0.5, label='Actual')
plt.title('Histogram of Wood Chip Lengths (Filtered)')
plt.xlabel('Length (pixels)')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(predicted_widths, bins=30, color='green', edgecolor='black', alpha=0.5, label='Predicted')
plt.hist(truth_widths, bins=30, color='orange', edgecolor='black', alpha=0.5, label='Actual')
plt.title('Histogram of Wood Chip Widths (Filtered)')
plt.xlabel('Width (pixels)')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(predicted_diameters, bins=30, color='red', edgecolor='black', alpha=0.5, label='Predicted')
plt.hist(truth_diameters, bins=30, color='orange', edgecolor='black', alpha=0.5, label='Actual')
plt.title('Histogram of Wood Chip Diameters (Filtered)')
plt.xlabel('Diameter (pixels)')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'filtered_comparison_histograms.png'))
plt.show()

print("✅ Filtered histograms generated and saved!")

# Print new counts
print(f"Total GT Chips: {len(truth_dimensions)}")
print(f"Total Predicted Chips (After Filtering): {len(predicted_dimensions)}")







# The Kullback-Leibler (KL) divergence

from scipy.stats import entropy

def compute_kl_divergence(actual, predicted, bins=30):
    """
    Computes the Kullback-Leibler (KL) divergence between two distributions.
    """
    hist_actual, bin_edges = np.histogram(actual, bins=bins, density=True)
    hist_predicted, _ = np.histogram(predicted, bins=bin_edges, density=True)

    # Normalize to probability distributions
    hist_actual += 1e-10  # Avoid zero probabilities (prevents log(0))
    hist_predicted += 1e-10  # Avoid zero probabilities

    return entropy(hist_predicted, hist_actual)  # KL(P || Q)

# Compute KL divergence for each distribution
kl_length = compute_kl_divergence(truth_lengths, predicted_lengths)
kl_width = compute_kl_divergence(truth_widths, predicted_widths)
kl_diameter = compute_kl_divergence(truth_diameters, predicted_diameters)

# Plot histograms with KL divergence values
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.hist(predicted_lengths, bins=30, color='blue', edgecolor='black', alpha=0.5, label='Predicted')
plt.hist(truth_lengths, bins=30, color='orange', edgecolor='black', alpha=0.5, label='Actual')
plt.title(f'Histogram of Wood Chip Lengths\nKL Divergence: {kl_length:.4f}')
plt.xlabel('Length (pixels)')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(predicted_widths, bins=30, color='green', edgecolor='black', alpha=0.5, label='Predicted')
plt.hist(truth_widths, bins=30, color='orange', edgecolor='black', alpha=0.5, label='Actual')
plt.title(f'Histogram of Wood Chip Widths\nKL Divergence: {kl_width:.4f}')
plt.xlabel('Width (pixels)')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(predicted_diameters, bins=30, color='red', edgecolor='black', alpha=0.5, label='Predicted')
plt.hist(truth_diameters, bins=30, color='orange', edgecolor='black', alpha=0.5, label='Actual')
plt.title(f'Histogram of Wood Chip Diameters\nKL Divergence: {kl_diameter:.4f}')
plt.xlabel('Diameter (pixels)')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'filtered_comparison_histograms_with_kl.png'))
plt.show()

# Print KL divergence values
print(f"KL Divergence (Length): {kl_length:.4f}")
print(f"KL Divergence (Width): {kl_width:.4f}")
print(f"KL Divergence (Diameter): {kl_diameter:.4f}")
