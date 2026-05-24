import cv2
import numpy as np
from rembg import remove
from PIL import Image

def remove_background(image):
    # Chuyển từ PIL sang dạng byte để rembg xử lý
    img_byte_arr = np.array(image)
    result = remove(img_byte_arr)
    return Image.fromarray(result)

def resize_with_padding(image, size=(28, 28)):
    # Resize về đúng kích thước mô hình AI yêu cầu
    image = image.resize(size)
    return image.convert('L') # Chuyển sang ảnh xám
