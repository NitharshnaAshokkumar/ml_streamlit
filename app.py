import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import pywt
import urllib.request
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

st.set_page_config(page_title="Adversarial Shield App 🛡️", layout="wide")
st.title(" Adversarial Shield Learning Simulator")

LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
LABELS = urllib.request.urlopen(LABELS_URL).read().decode("utf-8").split("\n")

def get_label(idx):
    return LABELS[idx] if 0 <= idx < len(LABELS) else f"Class {idx}"

@st.cache_resource
def load_model(name):
    model = models.__dict__[name](pretrained=True)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- DEFENSES ----------------
def jpeg_compression(img, quality=50):
    _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, 1)

def gaussian_denoising(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def bit_depth_reduction(img, bits=5):
    shift = 8 - bits
    reduced = np.floor(img / (2 ** shift)) * (2 ** shift)
    return np.clip(reduced, 0, 255).astype(np.uint8)

def wavelet_denoising(img):
    result = []
    for c in cv2.split(img):
        coeffs2 = pywt.dwt2(c, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        HH *= 0
        recon = pywt.idwt2((LL, (LH, HL, HH)), 'bior1.3')
        result.append(np.clip(recon, 0, 255).astype(np.uint8))
    return cv2.merge(result)

def non_local_means(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# ---------------- ATTACKS ----------------
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    return torch.clamp(image + epsilon * sign_data_grad, 0, 1)

def pgd_attack(model, images, labels, epsilon, alpha, iters):
    ori_images = images.clone().detach()
    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = F.nll_loss(F.log_softmax(outputs, dim=1), labels)
        grad = torch.autograd.grad(loss, images)[0]
        adv_images = images + alpha * grad.sign()
        eta = torch.clamp(adv_images - ori_images, -epsilon, epsilon)
        images = torch.clamp(ori_images + eta, 0, 1).detach()
    return images

def cw_attack(image, model, label, c=1e-3, iters=50):
    delta = torch.zeros_like(image, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=0.01)
    for _ in range(iters):
        output = model(image + delta)
        real = output[0, label]
        other = torch.max(output[0, torch.arange(len(output[0])) != label])
        loss = torch.clamp(other - real + c, min=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return image + delta.detach()

# ---------------- GradCAM Helper ----------------
def get_last_conv_layer(model):
    name = model.__class__.__name__.lower()
    if "resnet" in name:
        return "layer4"
    elif "vgg" in name or "densenet" in name:
        return "features"
    return None

def generate_gradcam(model, img_tensor, target_class):
    target_layer = get_last_conv_layer(model)
    cam_extractor = GradCAM(model, target_layer=target_layer)
    out = model(img_tensor.unsqueeze(0))
    activation_map = cam_extractor(target_class, out)
    result = overlay_mask(
        transforms.ToPILImage()(img_tensor),
        transforms.ToPILImage()(activation_map[0]),
        alpha=0.6
    )
    return np.array(result)

# ---------------- Confidence Plot ----------------
def plot_confidence_bar(output, title="Prediction Confidence"):
    probs = torch.nn.functional.softmax(output, dim=1)[0]
    top5_prob, top5_catid = torch.topk(probs, 5)
    fig, ax = plt.subplots(figsize=(4, 2))
    y_labels = [get_label(c) for c in top5_catid]
    ax.barh(y_labels, top5_prob.detach().numpy(), color="skyblue")
    ax.set_xlabel("Probability")
    ax.set_title(title)
    ax.invert_yaxis()
    st.pyplot(fig)

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Control Panel")
model_choice = st.sidebar.selectbox("Model", ["resnet18", "vgg16", "densenet121"])
attack_type = st.sidebar.selectbox("Attack Type", ["FGSM", "PGD", "CW"])
shield_type = st.sidebar.selectbox("Defense Shield", [
    "JPEG Compression", "Gaussian Denoising", "Bit Depth Reduction",
    "Wavelet Denoising", "Non-local Means"
])
epsilon = st.sidebar.slider("Epsilon (Attack Strength)", 0.001, 0.3, 0.05, step=0.01)
uploaded_files = st.file_uploader("📸 Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# ---------------- Main Logic ----------------
if uploaded_files:
    model = load_model(model_choice)
    results = []

    for file in uploaded_files:
        st.markdown(f"## 🖼️ {file.name}")
        img = Image.open(file).convert("RGB")
        img_cv = np.array(img)
        img_tensor = transform(img).unsqueeze(0)

        # Original prediction
        output = model(img_tensor)
        orig_pred = torch.argmax(output, 1).item()
        orig_label = get_label(orig_pred)

        # Attack
        if attack_type == "FGSM":
            img_tensor.requires_grad = True
            output = model(img_tensor)
            loss = F.nll_loss(F.log_softmax(output, dim=1), torch.tensor([orig_pred]))
            model.zero_grad()
            loss.backward()
            data_grad = img_tensor.grad.data
            adv_tensor = fgsm_attack(img_tensor, epsilon, data_grad)
        elif attack_type == "PGD":
            adv_tensor = pgd_attack(model, img_tensor, torch.tensor([orig_pred]), epsilon, alpha=0.01, iters=10)
        else:
            adv_tensor = cw_attack(img_tensor, model, orig_pred)

        adv_img = (adv_tensor.squeeze().permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)

        # Adversarial prediction
        adv_output = model(adv_tensor)
        adv_pred = torch.argmax(adv_output, 1).item()
        adv_label = get_label(adv_pred)

        # Shield defense
        if shield_type == "JPEG Compression":
            shield_img = jpeg_compression(adv_img)
        elif shield_type == "Gaussian Denoising":
            shield_img = gaussian_denoising(adv_img)
        elif shield_type == "Bit Depth Reduction":
            shield_img = bit_depth_reduction(adv_img)
        elif shield_type == "Wavelet Denoising":
            shield_img = wavelet_denoising(adv_img)
        elif shield_type == "Non-local Means":
            shield_img = non_local_means(adv_img)
        else:
            shield_img = adv_img

        shield_tensor = transform(Image.fromarray(shield_img)).unsqueeze(0)
        shield_output = model(shield_tensor)
        shield_pred = torch.argmax(shield_output, 1).item()
        shield_label = get_label(shield_pred)

        # GradCAMs
        gradcam_orig = generate_gradcam(model, img_tensor.squeeze(), orig_pred)
        gradcam_adv = generate_gradcam(model, adv_tensor.squeeze(), adv_pred)

        # Columns layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img, caption=f"🟢 Original: {orig_label}", use_container_width=True)
            st.image(gradcam_orig, caption="GradCAM (Original)")
            plot_confidence_bar(output, "Original Confidence")

        with col2:
            st.image(adv_img, caption=f"⚡ Adversarial ({attack_type}): {adv_label}", use_container_width=True)
            st.image(gradcam_adv, caption="GradCAM (Adversarial)")
            plot_confidence_bar(adv_output, "Adversarial Confidence")

        with col3:
            st.image(shield_img, caption=f"🛡️ Shield: {shield_type} → {shield_label}", use_container_width=True)
            plot_confidence_bar(shield_output, "Shielded Confidence")

        results.append({
            "Image": file.name,
            "Model": model_choice,
            "Attack": attack_type,
            "Defense": shield_type,
            "Epsilon": epsilon,
            "Original Prediction": orig_label,
            "Adversarial Prediction": adv_label,
            "Recovered Prediction": shield_label
        })

    st.dataframe(pd.DataFrame(results))

else:
    st.info("👆 Upload images to start your adversarial defense experiment.")

with st.expander("📘 Learn More"):
    st.markdown("""
    - **FGSM**: Fast Gradient Sign Method  
    - **PGD**: Projected Gradient Descent (iterative)  
    - **CW**: Carlini & Wagner optimization-based attack  
    - **JPEG Compression**: Removes high-frequency perturbations  
    - **Wavelet Denoising**: Channel-wise wavelet noise reduction  
    - **GradCAM**: Highlights focus regions  
    - **Confidence Graphs**: Show how attacks & shields affect model certainty
    """)

st.success("✅ App ready with GradCAM + Confidence Visualization!")
