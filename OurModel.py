import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch import optim


class BrojectDetect:
    def __init__(self, num_classes):
        """
        Initializes the object detection model with a ResNet50 backbone.

        Parameters:
        - num_classes: int, the number of classes including the background.
        """
        self.num_classes = num_classes
        self.backbone = self._get_resnet50_backbone(pretrained=True)
        self.model = self._create_detection_model()

    def _get_resnet50_backbone(self, pretrained=True):
        """
        Prepares the ResNet50 backbone by removing the average pooling
        and fully connected layers, adapting it for object detection.
        """
        model = resnet50(pretrained=pretrained)
        modules = list(model.children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048
        return backbone

    def _get_rpn_and_roi_pool(self):
        """
        Creates the anchor generator for the Region Proposal Network (RPN)
        and the RoI align layer for the heads.
        """
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        return anchor_generator, roi_pooler

    def _create_detection_model(self):
        """
        Integrates the ResNet50 backbone with the Faster R-CNN architecture,
        returning the complete model for object detection.
        """
        anchor_generator, roi_pooler = self._get_rpn_and_roi_pool()
        model = FasterRCNN(backbone=self.backbone,
                           num_classes=self.num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)
        return model

    def get_model(self):
        """
        Returns the initialized object detection model.
        """
        return self.model

    def parameters(self):
        """
        Returns the parameters of the model.
        """
        return self.model.parameters()

    def train(self, train_loader, val_loader, num_epochs=10, learning_rate=0.005):

        # Move model to the appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # Parameters to optimize and optimizer setup
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)

        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0

            for images, targets in train_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                total_train_loss += losses.item()

            # Validation loop
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    total_val_loss += losses.item()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
