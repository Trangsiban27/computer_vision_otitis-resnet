import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from src.model import build_model
from src.label_mapping import CLASS_NAMES, map_5class_to_binary

DISEASE_TRANSLATION = {
    "Acute Otitis Media": "Viêm tai giữa cấp tính",
    "Chronic Otitis Media": "Viêm tai giữa mạn tính",
    "Cerumen Impaction": "Nút ráy tai",
    "Myringosclerosis": "Xơ hóa màng nhĩ",
    "Normal": "Màng nhĩ bình thường"
}

st.set_page_config(
    page_title="Hệ thống AI Chẩn đoán Nội soi Tai",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Hệ thống Sàng lọc Bệnh lý Màng Nhĩ Sử Dụng ResNet")
st.markdown("Hệ thống sử dụng mạng ResNet-50 để phân tích ảnh nội soi tai.")

# # ==========================================
# # 2. LOAD MÔ HÌNH (SỬ DỤNG CACHE ĐỂ KHÔNG BỊ LAG)
# # ==========================================
# # @st.cache_resource là CỰC KỲ QUAN TRỌNG giúp model chỉ load 1 lần duy nhất vào RAM
@st.cache_resource 
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    # Khởi tạo khung mô hình (Giống hệt lúc train)
    model = build_model(num_classes=5, freeze_backbone=True, unfreeze_last_layers=1, se_reduction=16)
    
    # Nạp trọng số
    checkpoint_path = "checkpoints/best_model.pth" # Sửa lại nếu đường dẫn khác
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    model.to(device)
    model.eval() # Bật chế độ dự đoán
    return model, device

with st.spinner("Đang khởi động lõi AI..."):
    model, device = load_model()

# # ==========================================
# # 3. TIỀN XỬ LÝ ẢNH (GIỐNG TẬP TEST)
# # ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# # ==========================================
# # 4. THIẾT KẾ GIAO DIỆN CHÍNH
# # ==========================================
# # Khu vực tải ảnh lên
uploaded_file = st.file_uploader("📁 Chọn một bức ảnh nội soi tai (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mở ảnh bằng PIL
    image = Image.open(uploaded_file).convert('RGB')
    
    # Chia giao diện làm 2 cột
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Ảnh Nội Soi Gốc")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Kết Quả Phân Tích từ hệ thống")
        
        # Tiền xử lý và đưa vào mô hình
        img_tensor = transform(image).unsqueeze(0).to(device) # Thêm chiều Batch [1, 3, 224, 224]
        
        with torch.no_grad():
            outputs = model(img_tensor)

            # Áp dụng Softmax để biến output thành % xác suất
            probabilities = F.softmax(outputs, dim=1)[0] * 100 
            
            # Lấy ra class có % cao nhất
            predicted_class_idx = torch.argmax(probabilities).item()
            predicted_class_name = CLASS_NAMES[predicted_class_idx]
            predicted_class_name_vi = DISEASE_TRANSLATION.get(predicted_class_name, "")
            confidence = probabilities[predicted_class_idx].item()
            
            # Quy đổi ra Nhị phân (Bệnh vs Khỏe)
            binary_result = map_5class_to_binary([predicted_class_idx])[0]
            binary_name = "BỆNH LÝ (Cần can thiệp)" if binary_result == 1 else "BÌNH THƯỜNG / XƠ HÓA NHẸ"
            binary_color = "red" if binary_result == 1 else "green"

#         # Hiển thị Kết luận tổng quát (Nhị phân)
        st.markdown(f"### Sàng lọc: <span style='color:{binary_color}'>{binary_name}</span>", unsafe_allow_html=True)
        
        # Hiển thị Chi tiết 5 lớp
        st.markdown(f"**Chẩn đoán chi tiết:** {predicted_class_name} - {predicted_class_name_vi}")
        st.markdown(f"**Xác suất dự đoán đúng:** {confidence:.2f}%")
        st.progress(int(confidence)) # Thanh tiến trình trực quan
        
        st.divider()
        st.markdown("**Xác suất các bệnh lý khác:**")
        
        # In ra các xác suất còn lại
        for i, class_name in enumerate(CLASS_NAMES):
            if i != predicted_class_idx:
                class_name_vi = DISEASE_TRANSLATION.get(class_name, "")
                st.write(f"- **{class_name}** *({class_name_vi})*: **{probabilities[i].item():.2f}%**")