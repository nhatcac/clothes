import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# 1. Load bộ não AI (Dùng cache để app chạy nhanh hơn)
@st.cache_resource
def get_model():
    return load_model('model_data/model_quanao.h5')

model = get_model()
class_names = ['Áo thun', 'Quần dài', 'Áo len', 'Váy', 'Áo khoác', 'Sandal', 'Áo sơ mi', 'Giày', 'Túi', 'Ủng']

# 2. Từ điển gợi ý
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

# 3. Hàm xử lý ảnh (Giữ nguyên logic của bạn)
def resize_with_padding(img_gray, target_size=(28, 28)):
    h, w = img_gray.shape
    size = max(h, w)
    pad_img = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    pad_img[y_off:y_off+h, x_off:x_off+w] = img_gray
    return cv2.resize(pad_img, target_size)

# 4. Giao diện Streamlit
st.title("📦 AI Logistics & Stylist")
st.write("Tải ảnh quần áo lên để AI nhận diện và tư vấn phối đồ.")

uploaded_file = st.file_uploader("Chọn ảnh (jpg, png, webp)...", type=["jpg", "png", "webp"])

if uploaded_file is not None:
    # Đọc ảnh
    img_pil = Image.open(uploaded_file)
    st.image(img_pil, caption="Ảnh của bạn", use_column_width=True)
    
    if st.button("Phân tích ngay"):
        with st.spinner('AI đang suy nghĩ...'):
            # Xử lý ảnh sang xám
            img_gray = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2GRAY)
            
            # Padding
            img_processed = resize_with_padding(img_gray)
            
            # Chuẩn hóa
            img_inv = 255 - img_processed
            img_input = img_inv.reshape(1, 28, 28, 1).astype('float32') / 255.0
            
            # Dự đoán
            prediction = model.predict(img_input)[0]
            conf = np.max(prediction) * 100
            label = class_names[np.argmax(prediction)]
            
            # Kết quả
            if conf < 40:
                st.warning("AI chưa tự tin lắm, thử lại với ảnh rõ nét hơn nhé!")
            else:
                st.success(f"### AI đoán: {label} ({conf:.1f}%)")
                st.info(fashion_tips.get(label, "Hãy tự tin mặc món đồ bạn thích!"))
                
                # Biểu đồ độ tin cậy
                st.bar_chart(dict(zip(class_names, prediction)))
