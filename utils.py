import cv2
import numpy as np

def resize_with_padding(img_gray, target_size=(28, 28)):
    """
    Hàm này nhận vào ảnh xám, thêm viền đen để thành hình vuông, 
    sau đó resize về 28x28 mà không làm biến dạng vật thể.
    """
    h, w = img_gray.shape
    size = max(h, w)
    
    # Tạo nền đen vuông
    pad_img = np.zeros((size, size), dtype=np.uint8)
    
    # Căn giữa ảnh vào nền đen
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    pad_img[y_off:y_off+h, x_off:x_off+w] = img_gray
    
    # Resize về 28x28
    return cv2.resize(pad_img, target_size)