# adversarial_shield.py

import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import foolbox as fb
import numpy as np
import pywt
from PIL import Image, ImageFilter
import torch.nn.functional as F
import cv2

# ==========================================================
# Normalized Model Wrapper
# ==========================================================
class NormalizedModel(nn.Module):
    """Wrap a base model and apply ImageNet normalization inside forward."""
    def __init__(self, base_model, mean=None, std=None):
        super().__init__()
        self.base = base_model
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        # register buffers so they move with .to(device)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        # x expected in [0,1]
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return self.base(x)

# ==========================================================
# Main Adversarial Shield Class
# ==========================================================
class AdversarialShield:
    def __init__(self, device=None):
        # Device selection
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Load base models and wrap with normalization
        base_models = {
            'resnet18': models.resnet18(pretrained=True),
            'vgg16': models.vgg16(pretrained=True),
            'densenet121': models.densenet121(pretrained=True)
        }

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        self.models = {}
        for name, base in base_models.items():
            base.eval()
            wrapped = NormalizedModel(base, mean=mean, std=std).to(self.device).eval()
            self.models[name] = wrapped

        # Foolbox model cache
        self.fmodels = {}

        # Transform pipelines
        self.to_tensor = T.Compose([T.Resize((224,224)), T.ToTensor()])
        self.to_pil = T.ToPILImage()

    # ======================================================
    # Helper: Foolbox wrapper
    # ======================================================
    def _get_fmodel(self, model_name):
        if model_name not in self.fmodels:
            fmodel = fb.PyTorchModel(self.models[model_name], bounds=(0,1), device=self.device)
            self.fmodels[model_name] = fmodel
        return self.fmodels[model_name]

    # ======================================================
    # Prediction
    # ======================================================
    def predict(self, image, model_name):
        """
        image: PIL.Image (RGB) or tensor (1,C,H,W) in [0,1]
        returns: (pred_index, confidence, input_tensor)
        """
        if isinstance(image, Image.Image):
            img_t = self.to_tensor(image).unsqueeze(0)  # 1,C,H,W
        else:
            img_t = image.unsqueeze(0) if image.dim() == 3 else image

        img_t = img_t.to(self.device)
        model = self.models[model_name]
        with torch.no_grad():
            logits = model(img_t)
            probs = F.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        return int(idx.item()), float(conf.item()), img_t

    # ======================================================
    # Generate Adversarial Image
    # ======================================================
    def generate_adversary(self, image_tensor, model_name, attack_type, epsilon):
        """
        image_tensor: torch tensor (1,C,H,W)
        model_name: str
        attack_type: 'FGSM' | 'PGD' | 'CW'
        epsilon: attack strength
        returns: adversarial tensor (1,C,H,W) in [0,1]
        """
        device = self.device
        model = self.models[model_name]
        fmodel = self._get_fmodel(model_name)

        img = image_tensor.to(device).clamp(0, 1)

        with torch.no_grad():
            logits = model(img)
            label = torch.argmax(logits, dim=1)

        attack_type = attack_type.upper()
        if attack_type == 'FGSM':
            attack = fb.attacks.LinfFastGradientAttack()
            epsilons = [epsilon]
        elif attack_type == 'PGD':
            attack = fb.attacks.LinfPGD(steps=40, abs_stepsize=epsilon / 3)
            epsilons = [epsilon]
        elif attack_type == 'CW':
            attack = fb.attacks.L2CarliniWagnerAttack(steps=100)
            epsilons = [epsilon * 3]
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        advs, _, _ = attack(fmodel, img, label, epsilons=epsilons)
        adv = advs[0]
        adv_t = torch.from_numpy(adv).float().cpu() if isinstance(adv, np.ndarray) else adv.detach().cpu().float()
        adv_t = adv_t.clamp(0, 1)
        return adv_t

    # ======================================================
    # Defense Shields / Purifiers
    # ======================================================
    def apply_shield(self, adv_tensor, defense_type="JPEG Compression"):
        """
        adv_tensor: tensor (1,C,H,W) in [0,1]
        defense_type: one of ['JPEG Compression', 'Gaussian Denoising', 
                              'Autoencoder Purifier', 'Bit Depth Reduction',
                              'Wavelet Denoising', 'Non-local Means']
        returns: purified tensor (1,C,H,W)
        """
        # Convert tensor to OpenCV image for operations
        img = adv_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        defense = defense_type.lower()

        if "jpeg" in defense:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            _, enc = cv2.imencode('.jpg', img, encode_param)
            dec = cv2.imdecode(enc, 1)
            purified = dec
        elif "gaussian" in defense:
            purified = cv2.GaussianBlur(img, (5, 5), 0)
        elif "autoencoder" in defense:
            pil_img = Image.fromarray(img)
            pil_tmp = pil_img.filter(ImageFilter.MedianFilter(size=3))
            pil_final = pil_tmp.filter(ImageFilter.GaussianBlur(radius=0.6))
            purified = np.array(pil_final)
        elif "bit depth" in defense:
            shift = 8 - 5
            purified = np.floor(img / (2 ** shift)) * (2 ** shift)
        elif "wavelet" in defense:
            coeffs2 = pywt.dwt2(img, 'bior1.3')
            LL, (LH, HL, HH) = coeffs2
            HH.fill(0)
            purified = pywt.idwt2((LL, (LH, HL, HH)), 'bior1.3')
        elif "non-local" in defense:
            purified = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        else:
            purified = img

        purified = np.clip(purified, 0, 255).astype(np.uint8)
        purified_tensor = T.ToTensor()(Image.fromarray(purified)).unsqueeze(0).float().cpu().clamp(0, 1)
        return purified_tensor
