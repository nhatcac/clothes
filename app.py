import cv2
import numpy as np
import gradio as gr
from keras.models import load_model

# 1. Load bộ não AI
model = load_model('model_data/model_quanao.h5')
class_names = ['Áo thun', 'Quần dài', 'Áo len', 'Váy', 'Áo khoác', 'Sandal', 'Áo sơ mi', 'Giày', 'Túi', 'Ủng']

# 2. Bộ từ điển gợi ý phối đồ
fashion_tips = {
    'Áo thun': 'Nên phối cùng quần jean hoặc quần short.',
    'Quần dài': 'Kết hợp với áo sơ mi cho vẻ lịch sự.',
    'Áo len': 'Phối cùng quần dài vào ngày lạnh.',
    'Váy': 'Kết hợp cùng sandal hoặc giày thể thao.',
    'Áo khoác': 'Khoác ngoài áo thun rất phong cách.',
    'Áo sơ mi': 'Kết hợp với quần tây.',
    'Sandal': 'Phù hợp với váy hoặc quần short.',
    'Giày': 'Kết hợp hoàn hảo với quần dài.',
    'Túi': 'Phụ kiện điểm nhấn.',
    'Ủng': 'Kết hợp với quần skinny hoặc váy dài.'
}

# 3. Hàm xử lý ảnh với kỹ thuật PADDING (Giữ tỉ lệ chuẩn)
def resize_with_padding(img_gray, target_size=(28, 28)):
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

def predict_clothing(img):
    # Chuyển ảnh màu sang xám
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    # Dùng kỹ thuật Padding để không làm méo ảnh
    img_processed = resize_with_padding(img_gray)
    
    # Nghịch đảo màu (Vì Fashion MNIST là nền đen)
    img_inv = 255 - img_processed
    img_input = img_inv.reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    # Dự đoán
    prediction = model.predict(img_input)[0]
    conf = np.max(prediction) * 100
    label = class_names[np.argmax(prediction)]
    
    # Cảnh báo "Lạ"
    if conf < 40:
        return "AI chưa từng thấy món đồ này, thử lại với ảnh khác nhé!", "Không có gợi ý."
    
    # Gợi ý phối đồ
    advice = fashion_tips.get(label, "Hãy tự tin mặc món đồ bạn thích!")
    
    return f"AI đoán: {label} ({conf:.1f}%)", advice

# 4. Giao diện ổn định
interface = gr.Interface(
    fn=predict_clothing, 
    inputs=gr.Image(type="pil", label="Tải ảnh quần áo"), 
    outputs=[gr.Textbox(label="Kết quả nhận diện"), gr.Textbox(label="Gợi ý phối đồ")],
    title="📦 AI Logistics & Stylist",
    description="Nhận diện quần áo thông minh (Có Padding để tăng độ chính xác)."
)

if __name__ == "__main__":
    interface.launch()