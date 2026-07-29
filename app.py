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
    page_title="Hệ thống AI Chẩn đoán Viêm tai giữa",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Hệ thống Sàng lọc Viêm tai giữa sử Dụng ResNet")
st.markdown("Hệ thống sử dụng mạng ResNet-50 và SE-ResNet50 (cải tiến) để phân tích ảnh nội soi tai.")

@st.cache_resource 
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    model = build_model(num_classes=5, freeze_backbone=True, unfreeze_last_layers=1, se_reduction=8)
    se_model = build_model(num_classes=5, resnet_version="se_resnet50", freeze_backbone=True, unfreeze_last_layers=1, se_reduction=8)
    
    checkpoint_path = "checkpoints/best_model.pth" 
    se_checkpoint_path = "checkpoints/se_best_model.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    se_model.load_state_dict(torch.load(se_checkpoint_path, map_location=device))
    
    model.to(device)
    model.eval()

    se_model.to(device)
    se_model.eval()
    return model, se_model, device

with st.spinner("Đang khởi động lõi AI..."):
    model, se_model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("📁 Chọn một bức ảnh nội soi tai (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mở ảnh bằng PIL
    image = Image.open(uploaded_file).convert('RGB')
    
    # Chia giao diện làm 2 cột
    col1, col2 = st.columns([0.8, 1])
    
    with col1:
        st.subheader("Ảnh Nội Soi Gốc")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Kết Quả Phân Tích từ hệ thống")
        
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            se_outputs = se_model(img_tensor)

            probabilities = F.softmax(outputs, dim=1)[0] * 100 
            se_probabilities = F.softmax(se_outputs, dim=1)[0] * 100
            
            predicted_class_idx = torch.argmax(probabilities).item()
            predicted_class_name = CLASS_NAMES[predicted_class_idx]
            predicted_class_name_vi = DISEASE_TRANSLATION.get(predicted_class_name, "")
            confidence = probabilities[predicted_class_idx].item()

            se_predicted_class_idx = torch.argmax(se_probabilities).item()
            se_predicted_class_name = CLASS_NAMES[se_predicted_class_idx]
            se_predicted_class_name_vi = DISEASE_TRANSLATION.get(se_predicted_class_name, "")
            se_confidence = se_probabilities[se_predicted_class_idx].item()

            binary_result = map_5class_to_binary([predicted_class_idx])[0]
            binary_name = "BỆNH LÝ (Cần can thiệp)" if binary_result == 1 else "BÌNH THƯỜNG / XƠ HÓA NHẸ"
            binary_color = "red" if binary_result == 1 else "green"

        st.markdown(f"### Sàng lọc: <span style='color:{binary_color}'>{binary_name}</span>", unsafe_allow_html=True)
        
        st.markdown(f"**Chẩn đoán chi tiết:** {predicted_class_name} - {predicted_class_name_vi}")
        st.markdown(f"**Độ tự tin của ResNet-50:** {confidence:.2f}%")

        st.divider()

        st.markdown(f"**Chẩn đoán chi tiết từ SE-ResNet50:** {se_predicted_class_name} - {se_predicted_class_name_vi}")
        st.markdown(f"**Độ tự tin của SE-ResNet50:** {se_confidence:.2f}%")
        
        st.divider()
        st.markdown("**Xác suất các bệnh lý khác:**")
        
        for i, class_name in enumerate(CLASS_NAMES):
            if i != predicted_class_idx:
                class_name_vi = DISEASE_TRANSLATION.get(class_name, "")
                st.write(f"- **{class_name}** *({class_name_vi})*: **{probabilities[i].item():.2f}%**")