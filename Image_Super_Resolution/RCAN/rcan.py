import gc
import os
import random
import time
import wandb
from tqdm import tqdm

import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.transforms import Compose
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

# ------------------------------ HYPERPARAMS ------------------------------ #
NUM_RESIDUAL_GROUPS = 8
NUM_RESIDUAL_BLOCKS = 16
KERNEL_SIZE = 3
REDUCTION_RATIO = 16
NUM_CHANNELS = 64
UPSCALE_FACTOR = 4
    
class DF2KDataset(Dataset):
    def __init__(self, hr_images_path, random_crop:bool=True, transforms:Compose=None, output_size:tuple=(256, 256), seed:int=1989):
        super(DF2KDataset, self).__init__()
        
        self.hr_images_path = hr_images_path
        
        hr_images_list = os.listdir(hr_images_path)
        
        self.hr_images_list = [hr_images_path + image_name for image_name in hr_images_list]
        
        self.random_crop = random_crop
        self.transforms = transforms
        self.output_size = output_size
        self.seed = seed
    
    def __getitem__(self, idx):
        hr_image_path = self.hr_images_list[idx]
        lr_image_path = hr_image_path.replace('.png', 'x4.png').replace('HR', 'LR_bicubic/X4')
        
        hr_image = Image.open(hr_image_path)
        lr_image = Image.open(lr_image_path)
        
        # -------------------- RANDOM CROP -------------------- #
        if self.random_crop:
            w, h = hr_image.size
            th, tw = self.output_size
            random.seed(self.seed + idx)
            i = random.choice(list(range(0, h-th+1, 4)))
            j = random.choice(list(range(0, w-tw+1, 4)))

            hr_image = transforms.functional.crop(hr_image, i, j, th, tw)
            lr_image = transforms.functional.crop(lr_image, i//4, j//4, th//4, tw//4)
        
        # -------------------- OTHER TRANSFORMATIONS -------------------- #
        if self.transforms:
            torch.manual_seed(seed=self.seed)
            hr_image = self.transforms(hr_image)
            torch.manual_seed(seed=self.seed)
            lr_image = self.transforms(lr_image)
            
        # -------------------- TO TENSOR -------------------- #
        hr_image = transforms.functional.to_tensor(hr_image)
        lr_image = transforms.functional.to_tensor(lr_image)
        
        return hr_image, lr_image
        
    def __len__(self):
        return len(self.hr_images_list)
    
class DF2KDataLoader(DataLoader):
    def __init__(self, hr_images_path, 
                 random_crop:bool=True, transforms:Compose=None, output_size:tuple=(256, 256), seed:int=1989,
                 batch_size:int=1, shuffle:bool=True, num_crops:int=1):
        """
        - hr_images_path: str
            Path to the high-resolution images
        - random_crop: bool
            Whether to crop the images. Default is True. You might want to crop on the training set, but not on the validation set.
        - transforms: Compose
            A Compose of transformations, applied similarly on both high-res and low-res images.
            >> NOTE: Some transformations might worsen the quality of image, so consider carefully.
        - output_size: tuple
            Size of the random crop applied to an image. Default is 256x256.
            >> NOTE: Only meaningful when `random_crop` is True.
        - seed: int
            A random number, meant to keep all transformations the same for high-res and low-res images
        - num_crops: int
            The number of random crops applied to an image. Size of dataset is multiplied accordingly. Default is 1.
            >> NOTE: Only meaningful when `random_crop` is True.
        """
        if random_crop and num_crops > 1:
            random.seed(1989)
            sub_datasets = [DF2KDataset(hr_images_path, random_crop=random_crop, transforms=transforms, output_size=output_size, seed=seed+int(random.random()*10)) 
                            for _ in range(num_crops)]
            self.dataset = ConcatDataset(sub_datasets)
        else:
            self.dataset = DF2KDataset(hr_images_path, random_crop=random_crop, transforms=transforms, output_size=output_size, seed=seed)
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle)

class RandomRotation(object):
    def __call__(self, img):
        rotation_angle = torch.randint(0, 3, (1,)).item() * 90

        return transforms.functional.rotate(img, rotation_angle)

TRANSFORMS = Compose([
    transforms.RandomHorizontalFlip(),
    RandomRotation()
])


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, num_channels=NUM_CHANNELS, reduction_ratio=REDUCTION_RATIO, kernel_size=KERNEL_SIZE):
        
        super(ResidualChannelAttentionBlock, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, num_channels//reduction_ratio, kernel_size=1, stride=1),
            nn.ReLU(),
            # nn.BatchNorm2d(num_channels//reduction_ratio),
            nn.Conv2d(num_channels//reduction_ratio, num_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        block_input = x.clone()

        residual = self.feature_extractor(x) # Feature extraction
        rescale = self.channel_attention(residual) # Rescaling vector
        
        block_output = block_input + (residual * rescale)
        
        return block_output

class ResidualGroup(nn.Module):
    def __init__(self, num_residual_blocks=NUM_RESIDUAL_BLOCKS,
                 num_channels=NUM_CHANNELS, reduction_ratio=REDUCTION_RATIO, kernel_size=KERNEL_SIZE):
        
        super(ResidualGroup, self).__init__()

        self.residual_blocks = nn.Sequential(
            *[ResidualChannelAttentionBlock(num_channels=num_channels, reduction_ratio=reduction_ratio, kernel_size=kernel_size) 
              for _ in range(num_residual_blocks)]
        )

        self.final_conv = nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    def forward(self, x):
        group_input = x.clone()

        residual = self.residual_blocks(x) # Residual blocks
        residual = self.final_conv(residual) # Final convolution

        group_output = group_input + residual

        return group_output

class ResidualInResidual(nn.Module):
    def __init__(self, num_residual_groups=NUM_RESIDUAL_GROUPS, num_residual_blocks=NUM_RESIDUAL_BLOCKS,
                 num_channels=NUM_CHANNELS, reduction_ratio=REDUCTION_RATIO, kernel_size=KERNEL_SIZE):
        
        super(ResidualInResidual, self).__init__()

        self.residual_groups = nn.Sequential(
            *[ResidualGroup(num_residual_blocks=num_residual_blocks,
                            num_channels=num_channels, reduction_ratio=reduction_ratio, kernel_size=kernel_size) 
              for _ in range(num_residual_groups)]
        )

        self.final_conv = nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    def forward(self, x):
        shallow_feature = x.clone()
        
        residual = self.residual_groups(x) # Residual groups
        residual = self.final_conv(residual) # Final convolution

        deep_feature = shallow_feature + residual

        return deep_feature

class RCAN(nn.Module):
    
    def __init__(self, num_residual_groups=NUM_RESIDUAL_GROUPS, num_residual_blocks=NUM_RESIDUAL_BLOCKS,
                 num_channels=NUM_CHANNELS, reduction_ratio=REDUCTION_RATIO, kernel_size=KERNEL_SIZE):
        
        super(RCAN, self).__init__()

        self.shallow_conv = nn.Conv2d(3, num_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.residual_in_residual = ResidualInResidual(num_residual_groups=num_residual_groups, num_residual_blocks=num_residual_blocks,
                                                       num_channels=num_channels, reduction_ratio=reduction_ratio, kernel_size=kernel_size)
        self.upscaling_module = nn.PixelShuffle(upscale_factor=UPSCALE_FACTOR)
        self.reconstruction_conv = nn.Conv2d(num_channels // (UPSCALE_FACTOR ** 2), 3, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    def forward(self, x):
        shallow_feature = self.shallow_conv(x) # Initial convolution
        deep_feature = self.residual_in_residual(shallow_feature) # Residual in Residual
        upscaled_image = self.upscaling_module(deep_feature) # Upscaling module
        reconstructed_image = self.reconstruction_conv(upscaled_image) # Reconstruction

        return reconstructed_image

class Network:
    def __init__(self, for_inference=True, device='cuda', run_id=None):        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.for_inference = for_inference
        self.model = RCAN().to(device)
        self.device = device
    
    def save_network(self, epoch, train_loss, valid_loss, checkpoint_path):
        checkpoint = {
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learning_rate_scheduler': self.scheduler.state_dict(),
            'network': self
        }
        torch.save(checkpoint, checkpoint_path)

    def load_network(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        if not self.for_inference:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['learning_rate_scheduler'])
        return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['valid_loss']
    
    def inference(self, lr_image, hr_image=None):
        """
        - lr_image: torch.Tensor
            3D Tensor (C, H, W)
        - hr_image: torch.Tesnor
            3D Tensor (C, H, W). This parameter is optional, for comparing the model output and the 
            ground-truth high-res image. If used solely for inference, skip this. Default is None/
        """
        lr_image = lr_image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            sr_image = self.model(lr_image)
        
        lr_image = lr_image.squeeze(0)
        sr_image = sr_image.squeeze(0)
        
        return sr_image    
