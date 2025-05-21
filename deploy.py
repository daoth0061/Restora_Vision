import sys
import os
sys.path.append(os.path.abspath("."))


import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL.Image as pil_image
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as data
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, ToPILImage
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.transforms.functional import rgb_to_grayscale


from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from skimage.metrics import structural_similarity as ssim  # Consider renaming to avoid conflict

#from basicsr.utils import img2tensor


# RCAN-related imports
from Image_Super_Resolution.RCAN.rcan import (
    RCAN,
    ResidualInResidual,
    ResidualGroup,
    ResidualChannelAttentionBlock,
    DF2KDataLoader,
    DF2KDataset,
    RandomRotation,
    Network
)
from Image_Super_Resolution.RCAN.rcan_runner import load_rcan_model, rcan_sr, rcan_single_img

# Import model here
from Image_Super_Resolution.SwinIR.models.network_swinir import SwinIR

from Image_Super_Resolution.SRGAN.model.model import Generator as SRGAN

from Image_Super_Resolution.DiffIR_SR.DiffIR.archs.S2_arch import DiffIRS2 as DiffIRS2SR
from Image_Super_Resolution.DiffIR_SR.infer import pad_test_sr

from Image_Deblurring.DiffIR_Deblur.DiffIR.archs.S2_arch import DiffIRS2 as DiffIRS2DB
from Image_Deblurring.DiffIR_Deblur.infer import pad_test_db

from Colorization.DDColor.ddcolor_model import DDColor
from Colorization.DDColor.basicsr.archs.ddcolor_arch_utils.unet import Hook, CustomPixelShuffle_ICNR,  UnetBlockWide, NormType, custom_conv_layer




model_swinir = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.,
               depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
               mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')

model_srgan = SRGAN(in_channels=3)
model_rcan = load_rcan_model()
model_diffIR_SR = DiffIRS2SR(n_encoder_res= 9, dim= 64, scale=4,num_blocks= [13,1,1,1],
                                   num_refinement_blocks= 13,heads= [1,2,4,8], ffn_expansion_factor= 2.2,LayerNorm_type= "BiasFree")
model_diffIR_DB = DiffIRS2DB(n_encoder_res = 5, dim = 48, num_blocks = [3,5,6,6], 
                           num_refinement_blocks = 4, heads = [1,2,4,8], ffn_expansion_factor = 2,LayerNorm_type= "WithBias")
model_nafnet = model_swinir
model_ddcolor = DDColor(
            encoder_name='convnext-l',
            decoder_name='MultiScaleColorDecoder',
            input_size=[256, 256],
            num_output_channels=2,
            last_norm='Spectral',
            do_normalize=False,
            num_queries=100,
            num_scales=3,
            dec_layers=9,
        )


# model_path = 'model_zoo/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'
model_path_swinir = 'Image_Super_Resolution/SwinIR/model_zoo/model_weight_X4_swinir.pth'
model_path_srgan = 'Image_Super_Resolution/SRGAN/checkpoint/gen.pth.tar'
model_path_diffIR_SR = "Image_Super_Resolution/DiffIR_SR/DiffIR/weights/RealworldSR-DiffIRS2x4.pth"
model_path_diffIR_DB = "Image_Deblurring/DiffIR_Deblur/DiffIR/weights/Deblurring-DiffIRS2.pth"
model_path_nafnet = model_path_swinir
model_path_ddcolor = 'Colorization/DDColor/weights/pytorch_model.pt'


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, model, device):
    param_key_g = "params"
    pretrained_dict = torch.load(model_path)
    if param_key_g not in pretrained_dict.keys():
        param_key_g = "state_dict"
    model.load_state_dict(pretrained_dict[param_key_g] if param_key_g in pretrained_dict.keys() else pretrained_dict, strict=True)
    model.to(device)
    model.eval()
    
def load_color_model(model_path, model, device):
    param_key_g = "params"
    pretrained_dict = torch.load(model_path)
    model.load_state_dict(pretrained_dict[param_key_g] 
                          if param_key_g in pretrained_dict.keys() 
                          else pretrained_dict, strict=False) 
    # Change strict to False please, add it as a parameter from load_model maybe
    model.to(device)
    model.eval()


load_model(model_path_swinir, model_swinir, DEVICE)
model_srgan = model_swinir
# load_model(model_path_srgan, model_srgan, DEVICE)
load_model(model_path_diffIR_SR, model_diffIR_SR, DEVICE)
load_model(model_path_diffIR_DB, model_diffIR_DB, DEVICE)
load_model(model_path_nafnet, model_nafnet, DEVICE)
load_color_model(model_path_ddcolor, model_ddcolor, DEVICE)

def preprocess_image(image, device):
    """
    Preprocess an image for inference.
    Args:
        image_path (str): Path to the image.
        device (torch.device): Device to process the image.
        scale_factor (int): Factor to upscale the image.
    Returns:
        lr_image (torch.Tensor): Preprocessed low-resolution image tensor.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to [0, 1] range
    ])
    lr_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    return lr_image

def postprocess_image(output_tensor):
    """
    Post-process the model output tensor into an image.
    Args:
        output_tensor (torch.Tensor): Output tensor from the model.
    Returns:
        np.ndarray: Output image in RGB format.
    """
    output_image = output_tensor.squeeze(0).clamp(0, 1).cpu().numpy()
    output_image = np.transpose(output_image, (1, 2, 0))  # Convert to HWC format
    output_image = (output_image * 255).astype(np.uint8)  # Convert to [0, 255]
    return output_image

def preprocess_image_diffIR_sr(im, device):
    lq,mod_pad_h,mod_pad_w= pad_test_sr(im, 4)
    return lq,mod_pad_h,mod_pad_w

def postprocess_image_diffIR_sr(sr, mod_pad_h, mod_pad_w):
    _, _, h, w = sr.size()
    sr = sr[:, :, 0:h - mod_pad_h * 4, 0:w - mod_pad_w * 4]
    output_image = sr.squeeze(0).clamp(0, 1).cpu().numpy()
    output_image = np.transpose(output_image, (1, 2, 0))  # Convert to HWC format
    output_image = (output_image * 255).astype(np.uint8)  # Convert to [0, 255]
    return output_image

def preprocess_image_diffIR_db(im, device):
    lq,mod_pad_h,mod_pad_w= pad_test_db(im, 4)
    return lq,mod_pad_h,mod_pad_w

def postprocess_image_diffIR_db(sr, mod_pad_h, mod_pad_w):
    _, _, h, w = sr.size()
    sr = sr[:, :, 0:h - mod_pad_h * 4, 0:w - mod_pad_w * 4]
    output_image = sr.squeeze(0).clamp(0, 1).cpu().numpy()
    output_image = np.transpose(output_image, (1, 2, 0))  # Convert to HWC format
    output_image = (output_image * 255).astype(np.uint8)  # Convert to [0, 255]
    return output_image

def generate_grayscale_rgb(img_rgb: np.ndarray, size=(256, 256)) -> np.ndarray:
    """
    Tạo ảnh RGB xám (giả màu) từ ảnh RGB gốc bằng cách giữ kênh L, gán kênh a và b = 0.
    
    Args:
        img_rgb (np.ndarray): Ảnh RGB gốc, giá trị pixel [0, 1].
        size (tuple): Kích thước resize để đưa vào mô hình.

    Returns:
        np.ndarray: Ảnh RGB đã được chuyển sang xám (giữ nguyên L).
    """
    img_resized = cv2.resize(img_rgb, size)
    img_lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2Lab)
    img_l = img_lab[:, :, :1]
    img_gray_lab = np.concatenate([img_l, np.zeros_like(img_l), np.zeros_like(img_l)], axis=-1)
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_Lab2RGB)
    return img_gray_rgb

def colorize_image(uploaded_image: Image.Image, model_ddcolor, device="cuda") -> np.ndarray:
    """
    Màu hóa ảnh grayscale bằng mô hình đã huấn luyện.
    
    Args:
        uploaded_image (PIL.Image): Ảnh đầu vào (RGB hoặc grayscale).
        model_ddcolor (torch.nn.Module): Mô hình màu hóa đã load.
        device (str): Thiết bị tính toán ("cuda" hoặc "cpu").

    Returns:
        np.ndarray: Ảnh đã được tô màu (dạng uint8, RGB).
    """
    # Chuyển ảnh sang RGB và chuẩn hóa về [0, 1]
    img = uploaded_image.convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    height, width = img_np.shape[:2]

    # Lấy kênh L gốc (full size)
    orig_l = cv2.cvtColor(img_np, cv2.COLOR_RGB2Lab)[:, :, :1]

    # Tạo ảnh RGB xám (256x256) dùng để đưa vào model
    img_gray_rgb = generate_grayscale_rgb(img_np)

    # Chuẩn bị tensor input
    tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        output_ab = model_ddcolor(tensor_gray_rgb).cpu()

    # Resize output và ghép lại với kênh L ban đầu
    output_ab_resized = F.interpolate(output_ab, size=(height, width))[0].numpy().transpose(1, 2, 0)
    output_lab = np.concatenate((orig_l, output_ab_resized), axis=-1)
    output_rgb = cv2.cvtColor(output_lab, cv2.COLOR_Lab2RGB)

    # Trả về ảnh uint8
    return (output_rgb * 255.0).round().astype(np.uint8)





# Streamlit app
st.title("Image Processing with RestoraVision")

# Task selection
task = st.selectbox("Select a Task", ("Resolution Enhancement", "Deblurring","Colorization"))

# Model selection based on task
if task == "Resolution Enhancement":
    model_choice = st.selectbox("Select a Model", ("Swin IR", "SRGAN", "RCAN", "DiffIR"))
    if model_choice == "Swin IR":
        model = model_swinir
    elif model_choice == "SRGAN":
        model = model_srgan # Replace by SR_GAN model
    elif model_choice == "DiffIR":
        model = model_diffIR_SR
    else: 
        model = model_rcan
elif task == "Deblurring":
    model_choice = st.selectbox("Select a Model", ("DiffIR"))
    model = model_diffIR_DB 
else:
    model_choice = st.selectbox("Select a Model",("DDColor"))
    model = model_ddcolor

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    before = time.time()
    input_image = Image.open(uploaded_file).convert("RGB")
    if task == "Colorization":
        gray_img_rgb = generate_grayscale_rgb(input_image)  # ảnh RGB xám
        input_image = Image.fromarray((gray_img_rgb * 255).astype(np.uint8))  # chuyển lại thành PIL.Image

    # Hiển thị ảnh (gốc hoặc đã xám hóa tùy task)
    st.image(input_image, caption="Uploaded Image", use_column_width=False)

    # Process image
    if st.button("Process Image"):
        with st.spinner("Processing..."):
            lr_torch = preprocess_image(input_image, DEVICE)
            if task == "Resolution Enhancement":
                if model_choice == "DiffIR":
                    lr_torch, mod_pad_h, mod_pad_w = preprocess_image_diffIR_sr(lr_torch, DEVICE)
                    with torch.no_grad():
                        sr_torch = model(lr_torch)
                        sr_image = postprocess_image_diffIR_sr(sr_torch, mod_pad_h, mod_pad_w)
                
                elif model_choice == "RCAN":
                    sr_image = rcan_single_img( input_image,model)
                    
                else:
 
                    with torch.no_grad():
                        sr_torch = model(lr_torch)
                        sr_image = postprocess_image(sr_torch)
            elif model_choice == "Deblurring":
                if model_choice == "DiffIR":
                    lr_torch, mod_pad_h, mod_pad_w = preprocess_image_diffIR_db(lr_torch, DEVICE)
                    with torch.no_grad():
                        sr_torch = model(lr_torch)
                        sr_image = postprocess_image_diffIR_db(sr_torch, mod_pad_h, mod_pad_w)
            else:
                # Load ảnh và tô màu
                input_img = Image.open("input.jpg")
                sr_image = colorize_image(input_img, model_ddcolor, device="cuda")

                

            
        st.success(f"Processing complete in {time.time() - before:.4f}s!")
        
        # Display the result
        st.image(sr_image, caption="Processed Image", use_column_width=False)