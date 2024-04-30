from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image, ImageDraw
import random
import numpy as np
import os


from PIL import Image


def get_bbox_from_label(label_path, image_size=(640, 640)):
    """Reads bounding box data from a label file and converts it to pixel coordinates.

    Parameters:
    - label_path: The file path to the label file.
    - image_size: A tuple of (width, height) of the image.

    Returns:
    - A list of bounding boxes in the format [x_min, y_min, x_max, y_max].
    """
    with open(label_path, 'r') as file:
        lines = file.readlines()

    bboxes = []
    for line in lines:
        # Extract the normalized bbox values (center_x, center_y, width, height)
        _, x_center, y_center, width, height = map(float, line.strip().split())

        # Convert normalized values to pixel coordinates
        x_center, y_center = x_center * image_size[0], y_center * image_size[1]
        width, height = width * image_size[0], height * image_size[1]

        # Calculate the min and max x and y values
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        # Append the bbox to the list
        bboxes.append([x_min, y_min, x_max, y_max])

    return bboxes


def create_bbox_mask(image_path, bboxes):
    """Creates a mask for the given image with bounding boxes.
    Everything inside the bboxes is black, and everything outside is white."""
    # Load the image to get its size
    image = Image.open(image_path)

    # Create a white mask of the same size as the image
    mask = Image.new("RGB", image.size, "white")
    draw = ImageDraw.Draw(mask)

    # For each bbox, fill the area with black
    for bbox in bboxes:
        # The bbox format is assumed to be [x_min, y_min, x_max, y_max]
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], fill="black")

    return mask





# Check if CUDA is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# Initialize the pipeline with the specified device
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    variant="fp16",  # Use fp16 variant for less memory usage on GPU
    torch_dtype=torch.float32,
    use_auth_token=True  # If you're using a model from Hugging Face, you might need your token
).to(device)

# Example usage, assuming you have the `get_bbox_from_label` function from the provided context
bboxes = get_bbox_from_label('training_data/labels/random_image_20240219202256802520.txt')

mask = create_bbox_mask('training_data/images/random_image_20240219202256802520.jpg', bboxes)
#show mask
mask.show()
# Load the image and mask
image = Image.open('training_data/images/random_image_20240219202256802520.jpg')
prompt = "biology images objects and some text around."

# Run the pipeline
image2 = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
image2.show()


# Save the result
image2.save('output.png')

