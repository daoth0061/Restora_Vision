import torch
import torch.nn as nn 
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import streamlit as st 
from torchvision.transforms import ToTensor, ToPILImage
from basicsr.utils import img2tensor
import PIL.Image as pil_image
import time
from torch.utils import data as data
import cv2
from Image_Super_Resolution.FSRCNN.utils import preprocess

# Import model here
from Image_Super_Resolution.SwinIR.models.network_swinir import SwinIR

from Image_Super_Resolution.DiffIR_SR.DiffIR.archs.S2_arch import DiffIRS2 as DiffIRS2SR
from Image_Super_Resolution.DiffIR_SR.infer import pad_test_sr

from Image_Deblurring.DiffIR_Deblur.DiffIR.archs.S2_arch import DiffIRS2 as DiffIRS2DB
from Image_Deblurring.DiffIR_Deblur.infer import pad_test_db

from Image_Super_Resolution.FSRCNN.models import FSRCNN


model_swinir = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.,
               depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
               mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')

model_srgan = model_swinir
model_fsrcnn = FSRCNN(scale_factor= 4)
model_diffIR_SR = DiffIRS2SR(n_encoder_res= 9, dim= 64, scale=4,num_blocks= [13,1,1,1],
                                   num_refinement_blocks= 13,heads= [1,2,4,8], ffn_expansion_factor= 2.2,LayerNorm_type= "BiasFree")
model_diffIR_DB = DiffIRS2DB(n_encoder_res = 5, dim = 48, num_blocks = [3,5,6,6], 
                           num_refinement_blocks = 4, heads = [1,2,4,8], ffn_expansion_factor = 2,LayerNorm_type= "WithBias")
model_nafnet = model_swinir


# model_path = 'model_zoo/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'
model_path_swinir = 'Image_Super_Resolution/SwinIR/model_zoo/model_weight_X4_swinir.pth'
model_path_srgan = model_path_swinir
model_path_fsrcnn = 'Restora_Vision\Image_Super_Resolution\FSRCNN\model\fsrcnn_x4.pth'
model_path_diffIR_SR = "Image_Super_Resolution/DiffIR_SR/DiffIR/weights/RealworldSR-DiffIRS2x4.pth"
model_path_diffIR_DB = "Image_Deblurring/DiffIR_Deblur/DiffIR/weights/Deblurring-DiffIRS2.pth"
model_path_nafnet = model_path_swinir


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, model, device):
    param_key_g = "params"
    pretrained_dict = torch.load(model_path)
    model.load_state_dict(pretrained_dict[param_key_g] if param_key_g in pretrained_dict.keys() else pretrained_dict, strict=True)
    model.to(device)
    model.eval()

load_model(model_path_swinir, model_swinir, DEVICE)
load_model(model_path_srgan, model_srgan, DEVICE)
load_model(model_path_fsrcnn, model_fsrcnn, DEVICE)
load_model(model_path_diffIR_SR, model_diffIR_SR, DEVICE)
load_model(model_path_diffIR_DB, model_diffIR_DB, DEVICE)
load_model(model_path_nafnet, model_nafnet, DEVICE)



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

def fsrcnn_predict(model, lr_image):
    """
    Predict a super-resolved image using the FSRCNN model.

    Args:
        model (torch.nn.Module): Pre-trained FSRCNN model.
        lr_image (PIL.Image.Image): Low-resolution input image (PIL format).

    Returns:
        sr_image_cv (np.ndarray): Super-resolved image in OpenCV format (BGR).
        sr_rgb_image (PIL.Image.Image): Super-resolved image in PIL format (RGB).
    """
    # Convert LR image to tensor
    lr_ycbcr = lr_image.convert('YCbCr')
    lr_y, lr_cb, lr_cr = lr_ycbcr.split()
    lr_tensor = torch.from_numpy(np.array(lr_y)).float().div(255.0).unsqueeze(0).unsqueeze(0).to(next(model.parameters()).device)

    # Perform inference
    with torch.no_grad():
        preds = model(lr_tensor).clamp(0.0, 1.0)

    # Convert SR tensor to image (Y channel)
    sr_y_channel = preds.mul(255.0).byte().cpu().numpy().squeeze(0).squeeze(0)

    # Resize Cb, Cr to match SR image dimensions
    lr_cb_resized = lr_cb.resize((sr_y_channel.shape[1], sr_y_channel.shape[0]), resample=pil_image.BICUBIC)
    lr_cr_resized = lr_cr.resize((sr_y_channel.shape[1], sr_y_channel.shape[0]), resample=pil_image.BICUBIC)

    # Merge SR Y channel with resized Cb, Cr channels
    sr_ycbcr = pil_image.merge('YCbCr', (
        pil_image.fromarray(sr_y_channel, mode='L'),
        lr_cb_resized,
        lr_cr_resized
    ))

    # Convert YCbCr to RGB
    sr_rgb_image = sr_ycbcr.convert('RGB')

    # Convert to OpenCV format for visualization
    # sr_image_cv = cv2.cvtColor(np.array(sr_rgb_image), cv2.COLOR_RGB2BGR)

    return sr_rgb_image

# Streamlit app
st.title("Image Processing with RestoraVision")

# Task selection
task = st.selectbox("Select a Task", ("Resolution Enhancement", "Deblurring"))

# Model selection based on task
if task == "Resolution Enhancement":
    model_choice = st.selectbox("Select a Model", ("Swin IR", "SRGAN", "FSRCNN", "DiffIR"))
    if model_choice == "Swin IR":
        model = model_swinir
    elif model_choice == "SRGAN":
        model = model_srgan # Replace by SR_GAN model
    elif model_choice == "DiffIR":
        model = model_diffIR_SR
    else: 
        model = model_fsrcnn
        
else:
    model_choice = st.selectbox("Select a Model", ("DiffIR", "NAF Net"))
    model = model_diffIR_DB if model_choice == "DiffIR" else model_nafnet

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    before = time.time()
    input_image = Image.open(uploaded_file).convert("RGB")
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
                else:
 
                    with torch.no_grad():
                        sr_torch = model(lr_torch)
                        sr_image = postprocess_image(sr_torch)
            else:
                if model_choice == "DiffIR":
                    lr_torch, mod_pad_h, mod_pad_w = preprocess_image_diffIR_db(lr_torch, DEVICE)
                    with torch.no_grad():
                        sr_torch = model(lr_torch)
                        sr_image = postprocess_image_diffIR_db(sr_torch, mod_pad_h, mod_pad_w)
                else:
                    
                    with torch.no_grad():
                        sr_torch = model(lr_torch)
                        sr_image = postprocess_image(sr_torch)

            
        st.success(f"Processing complete in {time.time() - before:.4f}s!")
        
        # Display the result
        st.image(sr_image, caption="Processed Image", use_column_width=False)
