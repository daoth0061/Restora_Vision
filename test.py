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




model_swinir = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.,
               depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
               mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')

model_srgan = SRGAN(in_channels=3)
# model_rcan = load_rcan_model()
model_diffIR_SR = DiffIRS2SR(n_encoder_res= 9, dim= 64, scale=4,num_blocks= [13,1,1,1],
                                   num_refinement_blocks= 13,heads= [1,2,4,8], ffn_expansion_factor= 2.2,LayerNorm_type= "BiasFree")
model_diffIR_DB = DiffIRS2DB(n_encoder_res = 5, dim = 48, num_blocks = [3,5,6,6], 
                           num_refinement_blocks = 4, heads = [1,2,4,8], ffn_expansion_factor = 2,LayerNorm_type= "WithBias")
model_nafnet = model_swinir


# model_path = 'model_zoo/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'
model_path_swinir = 'Image_Super_Resolution/SwinIR/model_zoo/model_weight_X4_swinir.pth'
model_path_srgan = 'Image_Super_Resolution/SRGAN/checkpoint/gen.pth.tar'
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

load_model(model_path_swinir, model_swinir, DEVICE)
# load_model(model_path_srgan, model_srgan, DEVICE)
checkpoint = torch.load(model_path_srgan)
print(checkpoint.keys())

#load_model(model_path_diffIR_SR, model_diffIR_SR, DEVICE)
#load_model(model_path_diffIR_DB, model_diffIR_DB, DEVICE)
#load_model(model_path_nafnet, model_nafnet, DEVICE)