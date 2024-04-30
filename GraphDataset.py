from torch.utils.data import Dataset
import os
from PIL import ImageFile, Image
from torchvision import transforms
import torch

from torch.utils.data._utils.collate import default_collate



from torch.utils.data import Dataset, DataLoader

class GraphDataset(Dataset):
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path
        self.image_name_list = os.listdir(self.image_path)
        self.label_name_list = os.listdir(self.label_path)
        # load your dataset (image paths and annotations)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_path, f'{self.image_name_list[idx]}'))
        box_strs = [box for box in
                    open(os.path.join(self.label_path, f'{self.label_name_list[idx]}')).read().split('\n') if
                    box != '' and box != ' ']
        boxes = [list(map(float, box.split(' '))) for box in box_strs]
        labels = [int(box.split(' ')[0]) for box in box_strs]

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        image = transforms.ToTensor()(image)
        print(image.shape, target["boxes"].shape, target["labels"].shape)
        # Note: The custom_collate_fn seems to be designed for batching, not for individual samples
        # So it might not be necessary or appropriate to call it here unless you're making modifications for batching
        # target = custom_collate_fn([{'labels': labels, 'boxes': target['boxes']}])


        print(f"Image shape: {image.shape}, Boxes shape: {target['boxes'].shape}, Labels shape: {target['labels'].shape}")
        return image, target

    def __len__(self):
        return len(os.listdir(self.image_path))





def custom_collate_fn(batch):
    batch_images = [item[0] for item in batch]  # item[0] is the image
    batch_boxes = [item[1]["boxes"] for item in batch]  # item[1]["boxes"] for boxes
    batch_labels = [item[1]["labels"] for item in batch]  # item[1]["labels"] for labels

    # Collate images using default_collate
    collated_images = default_collate(batch_images)

    # Find the max number of boxes in any image to pad all others to match
    max_num_boxes = max([boxes.shape[0] for boxes in batch_boxes])

    # Pad boxes so all have the same number of boxes, ensuring equal tensor sizes for stacking
    padded_boxes = [torch.nn.functional.pad(boxes, pad=(0, 0, 0, max_num_boxes - boxes.shape[0]), mode='constant', value=0)
                    for boxes in batch_boxes]
    collated_boxes = torch.stack(padded_boxes)

    # Collate labels using default_collate
    collated_labels = default_collate(batch_labels)

    targets = {"boxes": collated_boxes, "labels": collated_labels}
    return collated_images, targets




