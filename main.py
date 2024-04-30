import PIL.Image
import torch



def transform_size_image(image, size):
    """
    Resize the image to the specified size
    :param image: PIL image
    :param size: tuple (width, height)
    :return: PIL image
    """
    image = image.resize(size, PIL.Image.BILINEAR)
    print(image.size)
    return image


def main():
    for i in range(1, 11):
        image = PIL.Image.open(f"real_images/images/image_{i}.png")
        image = transform_size_image(image, (640, 640))
        image.save(f"real_images/images/image_{i}.png")


if __name__ == '__main__':
    main()