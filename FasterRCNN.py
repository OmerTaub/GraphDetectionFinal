import os
import torch
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import VisionDataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import math
from collections import defaultdict


class CustomDataset(VisionDataset):
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms=transforms)
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labels"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "labels", self.labels[idx])
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # Initialize boxes as an empty tensor for the case of no objects
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        try:
            with open(label_path) as f:
                lines = f.readlines()
                if lines:
                    boxes_list = []
                    for line in lines:
                        class_id, x_center, y_center, w, h = map(float, line.split())
                        class_id += 1
                        x_min = (x_center - w / 2) * width
                        y_min = (y_center - h / 2) * height
                        x_max = (x_center + w / 2) * width
                        y_max = (y_center + h / 2) * height
                        boxes_list.append([x_min, y_min, x_max, y_max])
                    boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
        except FileNotFoundError:
            print(f"Label file not found for the image: {img_path}. Assuming no objects.")
            # No change needed here, boxes are already initialized as empty for this case

        num_objs = len(boxes)
        # Here, you might need to adjust according to your actual classes. Assuming class 0 for all objects for simplicity
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {"boxes": boxes, "labels": labels, "image_id": image_id}

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return int(len(self.imgs))


import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def get_model(pretrained=True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


def compute_iou(boxes1, boxes2):

    boxes1.to('cpu')
    boxes2.to('cpu')

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou_matrix = inter / union

    return iou_matrix


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves. """
    recall = torch.cat((torch.tensor([0]), recall, torch.tensor([1])))
    precision = torch.cat((torch.tensor([0]), precision, torch.tensor([0])))

    # Compute the precision envelope
    for i in range(precision.size(0) - 2, -1, -1):
        precision[i] = torch.max(precision[i], precision[i + 1])

    # Integrate area under curve
    i = torch.where(recall[1:] != recall[:-1])[0]
    ap = torch.sum((recall[i + 1] - recall[i]) * precision[i + 1])
    return ap


def compute_area(bbox):
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    elif len(bbox) == 3:
        point, width, height = bbox
        return width * height
    else:
        raise ValueError("Unexpected bounding box format!")


def evaluate_faster_model(model, data_loader, iou_threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    detections = defaultdict(list)
    ground_truths = defaultdict(list)

    with torch.no_grad():
        for images, annotations in data_loader:
            images = [img.to(device) for img in images]
            predictions = model(images)

            for pred, anno in zip(predictions, annotations):
                for label in range(2):
                    detections[label].append(pred['boxes'][pred['labels'] == label].cpu())
                    ground_truths[label].append(anno['boxes'][anno['labels'] == label].cpu())

    average_precisions = {}

    for label in range(8):
        # Sort detections by descending confidence
        valid_detections = [det for det in detections[label] if det.numel() > 0]
        sorted_detections = sorted(valid_detections, key=lambda x: x[-1][-1].item(), reverse=True)

        true_positives = torch.zeros(len(sorted_detections))
        false_positives = torch.zeros(len(sorted_detections))
        n_ground_truths = sum([len(boxes) for boxes in ground_truths[label]])

        for i, boxes_tensor in enumerate(sorted_detections):
            for box in boxes_tensor:
                assigned_gt = False
                for gt_box in ground_truths[label]:
                    if box.numel() > 0 and gt_box.numel() > 0:
                        iou = compute_iou(box, gt_box)
                    else:
                        iou = 0.0

                    if iou > iou_threshold:
                        assigned_gt = True
                        break
                if assigned_gt:
                    true_positives[i] = 1
                else:
                    false_positives[i] = 1

        tp_cumsum = torch.cumsum(true_positives, dim=0)
        fp_cumsum = torch.cumsum(false_positives, dim=0)

        recalls = tp_cumsum / n_ground_truths
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        # Compute AP
        average_precision = torch.trapz(precisions, recalls)
        average_precisions[label] = average_precision.item()

    valid_values = [v for v in average_precisions.values() if not math.isnan(v)]
    mAP = sum(valid_values) / len(valid_values)
    return mAP


import torch


def get_losses(model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_classification_loss = 0.0
    total_box_regression_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, annotations in data_loader:
            images = list(img.to(device) for img in images)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            # Get model predictions
            model.train()  # Temporarily set model to training mode to compute losses
            loss_dict = model(images, annotations)
            model.eval()  # Set it back to evaluation mode

            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            total_classification_loss += loss_dict['loss_classifier'].item()
            total_box_regression_loss += loss_dict['loss_box_reg'].item()
            num_batches += 1

    # After all data has been processed, compute average losses
    avg_loss = total_loss / num_batches
    avg_classification_loss = total_classification_loss / num_batches
    avg_box_regression_loss = total_box_regression_loss / num_batches

    return avg_loss, avg_classification_loss, avg_box_regression_loss


def evaluate_and_get_losses(model, data_loader, iou_threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    detections = defaultdict(list)
    ground_truths = defaultdict(list)
    total_loss = 0.0
    total_classification_loss = 0.0
    total_box_regression_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, annotations in tqdm(data_loader):
            images = [img.to(device) for img in images]
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            predictions = model(images)
            model.train()  # Switch to training mode once before loss computation
            loss_dict = model(images, annotations)
            total_loss += sum(loss for loss in loss_dict.values()).item()
            total_classification_loss += loss_dict['loss_classifier'].item()
            total_box_regression_loss += loss_dict['loss_box_reg'].item()
            num_batches += 1
            model.eval()  # Switch back to evaluation mode after loss computation

            for pred, anno in zip(predictions, annotations):
                for label in range(2):  # Assuming label range for detections
                    detections[label].append(pred['boxes'][pred['labels'] == label])
                    ground_truths[label].append(anno['boxes'][anno['labels'] == label])

    model.to('cpu')  # Move model back to CPU to free GPU resources if needed

    average_precisions = {}
    for label in range(2):
        detections_label = torch.cat(detections[label], dim=0)
        ground_truths_label = torch.cat(ground_truths[label], dim=0)
        if len(ground_truths_label) == 0:
            average_precisions[label] = float('nan')
            continue

        # Vectorized IOU computation and TP/FP calculation
        ious = compute_iou(detections_label, ground_truths_label)
        matched_gt_idxs = set()
        true_positives = torch.zeros(detections_label.shape[0], device=device)
        false_positives = torch.zeros(detections_label.shape[0], device=device)

        for i, max_iou in enumerate(ious.max(1).values):
            if max_iou >= iou_threshold:
                matched_idx = ious[i].argmax().item()
                if matched_idx not in matched_gt_idxs:
                    true_positives[i] = 1
                    matched_gt_idxs.add(matched_idx)
                else:
                    false_positives[i] = 1
            else:
                false_positives[i] = 1

        true_positives = torch.cumsum(true_positives, dim=0)
        false_positives = torch.cumsum(false_positives, dim=0)
        recall = true_positives / len(ground_truths_label)
        precision = true_positives / (true_positives + false_positives)
        average_precisions[label] = compute_ap(recall.to('cpu'),
                                               precision.to('cpu'))  # Move computations to CPU at the end

    valid_values = [v for v in average_precisions.values() if not math.isnan(v)]
    mAP = sum(valid_values) / len(valid_values) if valid_values else 0

    avg_loss = total_loss / num_batches
    avg_classification_loss = total_classification_loss / num_batches
    avg_box_regression_loss = total_box_regression_loss / num_batches

    return mAP, avg_loss, avg_classification_loss, avg_box_regression_loss


def visualize_predictions(image, model, device, threshold=0.5):
    # Convert image to tensor and add batch dimension
    img_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    # Ensure the model is in evaluation mode
    model.eval()
    with torch.no_grad():
        predictions = model(img_tensor)

    # The predictions variable contains a dict for each input image
    pred = predictions[0]

    # Convert the image to a NumPy array for visualization
    np_image = np.array(image)
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(np_image)

    # Process each detected object in the image
    for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
        if score > threshold:
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f'{label.item()}: {score.item():.2f}', color='white', fontsize=8,
                    bbox=dict(facecolor='red', alpha=0.5, pad=2, edgecolor='none'))

    plt.axis('off')
    plt.show()
