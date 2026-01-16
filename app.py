import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="AI วินิจฉัยปอดอักเสบ", layout="centered")
st.title("🫁 ระบบวินิจฉัยปอดอักเสบด้วย AI")

@st.cache_resource
def load_my_model():
    model = models.vgg16(pretrained=False)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 2)
    # ชื่อไฟล์ต้องตรงกับที่น้องอัปโหลดขึ้น GitHub
    model.load_state_dict(torch.load('pneumonia_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_my_model()

uploaded_file = st.file_uploader("เลือกภาพเอกซเรย์ปอด...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='ภาพที่อัปโหลด', use_column_width=True)
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(image).unsqueeze(0)

    if st.button('กดเพื่อวินิจฉัย'):
        with st.spinner('กำลังประมวลผล...'):
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            if predicted.item() == 1:
                st.error("⚠️ ผลวินิจฉัย: ตรวจพบภาวะปอดอักเสบ")
            else:
                st.success("✅ ผลวินิจฉัย: ปอดปกติ")
