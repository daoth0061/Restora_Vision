# rcan_runner.py

import torch
from torchvision.transforms.functional import to_tensor
from torch import _to_functional_tensor, nn, optim
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.transforms import Compose
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
import os
from .rcan import RCAN, ResidualInResidual, ResidualGroup, ResidualChannelAttentionBlock, DF2KDataLoader, DF2KDataset, RandomRotation
from .rcan import Network

# ------------------------------ HYPERPARAMS ------------------------------ #
NUM_RESIDUAL_GROUPS = 8
NUM_RESIDUAL_BLOCKS = 16
KERNEL_SIZE = 3
REDUCTION_RATIO = 16
NUM_CHANNELS = 64
UPSCALE_FACTOR = 4

class TestDataset(Dataset):
    def __init__(self, hr_images_path):
        super(TestDataset, self).__init__()
        hr_images_list = os.listdir(hr_images_path)
        self.hr_images_list = [hr_images_path + image_name for image_name in hr_images_list]
    
    def __getitem__(self, idx):
        hr_image_path = self.hr_images_list[idx]
        lr_image_path = hr_image_path.replace('HR', 'LR')
        
        hr_image = Image.open(hr_image_path)
        lr_image = Image.open(lr_image_path)
        
        hr_image = transforms.functional.to_tensor(hr_image)
        lr_image = transforms.functional.to_tensor(lr_image)
        
        return hr_image, lr_image
        
    def __len__(self):
        return len(self.hr_images_list)
    
def test(network, hr_images, lr_images, show_image=True, idx=0):
    """
    - network: Network
        Model to be tested.
    - hr_images: torch.Tensor
        4D Tensor (B, C, H, W)
    - lr_images: torch.Tensor
        4D Tensor (B, C, H, W)
    - show_image: bool
        Whether to add an inference step and show lr-sr-hr images side by side.
    - idx: int
        Index of image in batch, chosen to be shown. Default is 0.
        NOTE: Please be careful to keep idx within the batch size.
    """
    hr_images = hr_images.cuda()
    lr_images = lr_images.cuda()
    
    def calculate_psnr(original_images, reconstructed_images):

        if original_images.shape != reconstructed_images.shape:
            raise ValueError("Images must have the same shape")

        mse = F.mse_loss(original_images, reconstructed_images, reduction='mean')
        max_pixel_value = 1.0
        psnr_value = 10 * torch.log10((max_pixel_value ** 2) / mse)
        return psnr_value.item()

    def calculate_ssim(original_images, reconstructed_images):

        if original_images.shape != reconstructed_images.shape:
            raise ValueError("Images in the batch must have the same shape")

        ssim_value = ssim(original_images, reconstructed_images) 
        return ssim_value.item()
        
    network.model.eval()
    with torch.no_grad():
        sr_images = network.model(lr_images)
    
    loss_func = nn.L1Loss(reduction='mean')
    l1_loss = loss_func(hr_images.cpu(), sr_images.cpu()).item()
    psnr_value = calculate_psnr(hr_images.cpu(), sr_images.cpu())
    ssim_value = calculate_ssim(hr_images.cpu(), sr_images.cpu())
    
    if show_image:
        network.inference(lr_images[idx], hr_images[idx])
    
    return l1_loss, psnr_value, ssim_value,sr_images

def test_on_dataset(model, dataset, IDX):
    l1_batch_loss = 0
    psnr_batch_value = 0
    ssim_batch_value = 0
    for i in range(len(dataset)):
        hr_image, lr_image = dataset[i]
        if hr_image.shape[0] == 4:  # Nếu có 4 kênh (RGBA)
            hr_image = hr_image[:3, :, :]  # Lấy 3 kênh đầu tiên (RGB)
        if hr_image.shape[0] != 3:
            hr_image = hr_image.expand(3, -1, -1)
            lr_image = lr_image.expand(3, -1, -1)
        hr_image = hr_image.unsqueeze(0)
        lr_image = lr_image.unsqueeze(0)

        if i in IDX:
            l1_loss, psnr_value, ssim_value = test(model, hr_image, lr_image, show_image=True)
        else:
            l1_loss, psnr_value, ssim_value = test(model, hr_image, lr_image, show_image=False)

        l1_batch_loss += l1_loss
        psnr_batch_value += psnr_value
        ssim_batch_value += ssim_value

    print("Metrics across " + str(len(dataset)) + " images:")
    print(f"> L1 Loss: {l1_batch_loss/len(dataset):.4f}")
    print(f"> PSNR: {psnr_batch_value/len(dataset):.4f}")
    print(f"> SSIM: {ssim_batch_value/len(dataset):.4f}")
    
def load_rcan_model(model_path='D:/HUST/CV/Project/ComVis2025.2/Restora_Vision/Image_Super_Resolution/RCAN/model/final_network.pth'):
    """
    Load mô hình RCAN từ đường dẫn checkpoint.
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_network = torch.load(model_path, map_location=DEVICE)

    rcan_model = Network(for_inference=True)
    rcan_model.model.load_state_dict(trained_network.model.state_dict())
    rcan_model.model.eval()
    print("✅ Model loaded successfully.")

    return rcan_model

def rcan_single_img(lr_image, model):
    lr_tensor = to_tensor(lr_image) if isinstance(lr_image, Image.Image) else lr_image
    if lr_tensor.ndim == 3:
        lr_tensor = lr_tensor.unsqueeze(0)
        
    device = next(model.model.parameters()).device
    lr_tensor = lr_tensor.to(device)       
    with torch.no_grad():
        sr_image = model.model(lr_tensor)      
        
    sr_image = sr_image.detach().cpu()
        
    if sr_image.dim() == 4:
        sr_image = sr_image.squeeze(0)

    # Bước 3: Chuyển về numpy array [H, W, C]
    sr_image = sr_image.permute(1, 2, 0).numpy()

    # Bước 4: Scale về [0, 255] nếu đang là [0, 1] hoặc [-1, 1]
    sr_image = np.clip(sr_image * 255.0, 0, 255).astype(np.uint8)

    # Bước 5: Chuyển thành PIL image nếu muốn
    sr_pil_image = Image.fromarray(sr_image)
    return sr_pil_image

def rcan_sr(hr_image, lr_image, model):
    """
    Test một ảnh duy nhất với mô hình RCAN đã huấn luyện.

    Params:
    - hr_image: PIL.Image hoặc torch.Tensor
    - lr_image: PIL.Image hoặc torch.Tensor
    - model: Network object (từ load_rcan_model)
    
    Returns:
    - sr_image: torch.Tensor (1 ảnh đã được SR)
    """
    # Chuyển về tensor nếu là ảnh PIL
    hr_tensor = to_tensor(hr_image) if isinstance(hr_image, Image.Image) else hr_image
    lr_tensor = to_tensor(lr_image) if isinstance(lr_image, Image.Image) else lr_image

    # Đảm bảo shape (1, 3, H, W)
    if hr_tensor.ndim == 3:
        hr_tensor = hr_tensor.unsqueeze(0)
    if lr_tensor.ndim == 3:
        lr_tensor = lr_tensor.unsqueeze(0)

    # Chỉ lấy 3 kênh đầu tiên nếu ảnh có RGBA
    if hr_tensor.shape[1] == 4:
        hr_tensor = hr_tensor[:, :3, :, :]
        lr_tensor = lr_tensor[:, :3, :, :]
    if hr_tensor.shape[1] != 3:
        hr_tensor = hr_tensor.expand(-1, 3, -1, -1)
        lr_tensor = lr_tensor.expand(-1, 3, -1, -1)

    # Gọi test
    l1_loss, psnr_value, ssim_value, sr_images = test(model, hr_tensor, lr_tensor, show_image=True, idx=0)

    print("---- Single Image Test Result ----")
    print(f"L1 Loss: {l1_loss:.4f}")
    print(f"PSNR: {psnr_value:.4f}")
    print(f"SSIM: {ssim_value:.4f}")

    return sr_images[0].cpu()    
if __name__ == "__main__":
    hr_img = Image.open("D:/HUST/CV/Project/RestoraVision/RestoraVision/Image Super Resolution/RCAN/test_image/Set5/HR/butterfly.png").convert('RGB')
    lr_img = Image.open("D:/HUST/CV/Project/RestoraVision/RestoraVision/Image Super Resolution/RCAN/test_image/Set5/LR/butterfly.png").convert('RGB')
    
    sr_image = rcan_sr(hr_img, lr_img)
    sr_pil = to_pil_image(sr_image.clamp(0, 1))
    sr_pil.show(title="Super-Resolved Image")