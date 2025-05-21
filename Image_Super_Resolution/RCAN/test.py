
import torch
from torchvision.transforms.functional import to_tensor
from torch import _to_functional_tensor, nn, optim
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.transforms import Compose
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
import os
from rcan import RCAN, ResidualInResidual, ResidualGroup, ResidualChannelAttentionBlock, DF2KDataLoader, DF2KDataset, RandomRotation
from rcan import Network
from rcan_runner import load_rcan_model,rcan_sr, rcan_single_img
import torchvision.transforms.functional as TF

# Đường dẫn ảnh
hr_img_path = "D:/HUST/CV/Project/ComVis2025.2/Restora_Vision/Image_Super_Resolution/RCAN/test_image/Set5/HR/baby.png"
lr_img_path = "D:/HUST/CV/Project/ComVis2025.2/Restora_Vision/Image_Super_Resolution/RCAN/test_image/Set5/LR/baby.png"
lr_img = Image.open(lr_img_path)
# Load model
model = load_rcan_model()
sr_images = rcan_single_img(lr_img,model)
sr_image = sr_images.squeeze(0)
TF.to_pil_image(sr_image).show()
