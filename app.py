import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import xml.etree.ElementTree as ET
import os
from model import SimpleLeafNet

# --- 1. SET UP THE WEB PAGE ---
st.set_page_config(page_title="AgriGuard Offline AI", page_icon="🌱", layout="centered")
st.title("🌱 AgriGuard: Crop Disease Diagnosis")
st.write("Upload a picture of a leaf, and our Federated AI will instantly diagnose it and fetch remedies from our offline XML document database.")

# --- 2. LOAD THE BRAIN (CACHED FOR SPEED) ---
@st.cache_resource
def load_model():
    model = SimpleLeafNet()
    try:
        model.load_state_dict(torch.load("agriguard_model.pth"))
        model.eval()
        return model
    except:
        return None

model = load_model()

# --- 3. CALCULATE NETWORK ACCURACY (CACHED) ---
@st.cache_data
def calculate_accuracy():
    if model is None:
        return 0, 0
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    GLOBAL_CLASSES = {'tomato_early_blight': 0, 'potato_early_blight': 1, 'healthy': 2}
    folders = ["data/client_abhishek", "data/client_diwakar", "data/client_abid"]
    
    correct, total = 0, 0
    with torch.no_grad():
        for folder in folders:
            if os.path.exists(folder):
                try:
                    dataset = ImageFolder(root=folder, transform=transform)
                    for img, local_label in dataset:
                        class_name = dataset.classes[local_label]
                        true_id = GLOBAL_CLASSES.get(class_name, -1)
                        
                        output = model(img.unsqueeze(0))
                        _, pred = torch.max(output, 1)
                        
                        if pred.item() == true_id:
                            correct += 1
                        total += 1
                except Exception:
                    pass
    
    if total == 0:
        return 0, 0
    return (correct / total) * 100, total

# --- 4. SIDEBAR DASHBOARD ---
st.sidebar.title("📊 Network Stats")
st.sidebar.write("Federated Learning MVP")

if model:
    with st.sidebar.spinner("Calculating accuracy..."):
        acc_score, total_imgs = calculate_accuracy()
    
    st.sidebar.metric(label="Global AI Accuracy", value=f"{acc_score:.1f}%")
    st.sidebar.write(f"Tested on **{total_imgs}** edge images.")
    st.sidebar.success("Global Brain Active")
else:
    st.sidebar.error("Model not found! Run training first.")

# --- 5. MAIN APP: IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Choose a leaf image to test...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf', use_container_width=True)
    
    st.divider()
    st.write("🔍 **Analyzing using Edge AI...**")

    # Prepare Image & Predict
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted_id = torch.max(outputs, 1)
        disease_id = str(predicted_id.item())

    # --- 6. FETCH CURE FROM XML DOCUMENT DATABASE ---
    try:
        tree = ET.parse('remedies.xml')
        root = tree.getroot()

        found = False
        for disease in root.findall('Disease'):
            if disease.get('id') == disease_id:
                found = True
                name = disease.find('Name').text
                organic = disease.find('Remedy/Organic').text
                chemical = disease.find('Remedy/Chemical').text
                
                # Display Results
                st.success(f"🚨 **DISEASE DETECTED:** {name}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"🌱 **Organic Treatment:**\n\n{organic}")
                with col2:
                    st.warning(f"🧪 **Chemical Treatment:**\n\n{chemical}")
                break

        if not found:
            st.error("Disease ID not found in the XML document database.")
    except Exception as e:
        st.error(f"Error reading the XML document database: {e}")