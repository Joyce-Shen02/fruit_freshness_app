#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import os
os.environ["HOME"] = os.getcwd()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: fresh vs spoiled
model.load_state_dict(torch.load("/Users/joyce/Desktop/bootcamp/virtual_intern/week5,6/resnet50_fruit.pth", map_location=device))
model.to(device)
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# In[2]:


st.title("Fruit Freshness Classifier")

uploaded_file = st.file_uploader("Upload an image of a fruit", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        confidence = probs[0][predicted.item()].item() * 100
        label = "Fresh" if predicted.item() == 0 else "Spoiled"

    st.write(f"### Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}%")







