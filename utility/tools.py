import os
import io
import base64
import re
from typing import List
from PIL import Image


def load_images_from_folder(folder: str):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
            img_path = os.path.join(folder, filename)
            images.append(img_path)

    return images


def image_to_base64(image_path: str):
    try:
        if not os.path.exists(image_path):
            return "Error: File does not exist."

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    except Exception as e:
        return f"Error: {str(e)}"


def batch_image_to_base64(image_paths: List[str]):
    encoded_images = []
    for image_path in image_paths:
        encoded_image = image_to_base64(image_path)
        encoded_images.append(encoded_image)
    return encoded_images


def PIL_2_base64(image: Image.Image) -> str:
    """Convert a PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    base64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return base64_img


def save_base64_image_2_local(base64_string: str, save_path: str):
    """Save a base64 image to local file. Don't include the "data:image" part """
    try:
        image_data = base64.b64decode(base64_string)
        with open(save_path, "wb") as file:
            file.write(image_data)
        return f"Image has been saved to {save_path}"
    except Exception as e:
        return f"Error: {str(e)}"


def is_PIL_image(image)-> bool:
    """Check if the input image is a PIL image object"""
    try:
        from PIL import Image
        return isinstance(image, Image.Image)
    except ImportError:
        return False

def is_base64_image(base64_string: str) -> bool:
    """Check if the input string is a base64 image"""
    try:
        if isinstance(base64_string, str):
            return base64_string.startswith("data:image") and ";base64," in base64_string
        else:
            return False
    except Exception as e:
        return False


def is_base64(string):
    # 检查长度为4的倍数
    if len(string) % 4 != 0:
        return False

    # 使用正则匹配
    base64_pattern = r'^[A-Za-z0-9+/]*={0,2}$'
    return bool(re.match(base64_pattern, string))

if __name__ == "__main__":
    image = "../model/image_dataset/avatar.jpg"
    encoded_string = image_to_base64(image)
    print(encoded_string)
