import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import glob
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import logging
import matplotlib
matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_and_filter_dataset(image_paths, mask_paths, img_size=256):
    valid_image_paths = []
    valid_mask_paths = []

    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), desc="Validating files"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"corrupted image: {img_path}")
                continue
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"corrupted mask: {mask_path}")
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (img_size, img_size))
            mask_resized = cv2.resize(mask, (img_size, img_size))
            valid_image_paths.append(img_path)
            valid_mask_paths.append(mask_path)
            
        except Exception as e:
            logger.warning(f"problematic pair - Image: {img_path}, Error: {str(e)}")
            continue
    
    logger.info(f"validation complete. Valid pairs: {len(valid_image_paths)}/{len(image_paths)}")
    return valid_image_paths, valid_mask_paths

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, img_size=256):
        self.image_paths, self.mask_paths = validate_and_filter_dataset(image_paths, mask_paths, img_size)
        self.transform = transform
        self.img_size = img_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))\
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
    
        if isinstance(mask, torch.Tensor):
            mask = (mask > 0).float().unsqueeze(0)
        else:
            mask = (mask > 0).astype(np.float32)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return image, mask

def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

def test_dataloader(dataloader, device, num_batches=2):
    try:
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_batches:
                break
            images, masks = images.to(device), masks.to(device)
            logger.info(f"Batch {i}: Images shape: {images.shape}, Masks shape: {masks.shape}")
        logger.info("Dataloader test passed!")
        return True
    except Exception as e:
        logger.error(f"test failed: {str(e)}")
        return False

# U-Net Model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                   diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        self.down4 = DownSample(512, 1024)
        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)

def dice_coefficient(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def dice_loss(pred, target, smooth=1e-6):
    return 1 - dice_coefficient(pred, target, smooth)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss_val = dice_loss(pred, target)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss_val

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    train_losses = []
    val_losses = []
    val_dice_scores = []
    best_dice = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        #validating
        model.eval()
        val_loss = 0.0
        total_dice = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                dice_score = dice_coefficient(outputs, masks)
                total_dice += dice_score.item()
                val_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Dice': f'{dice_score.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        avg_dice = total_dice / len(val_loader)
        val_losses.append(avg_val_loss)
        val_dice_scores.append(avg_dice)
        scheduler.step()
        
        logger.info(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_dice:.4f}')
        
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), 'best_unet_model.pth')
            logger.info(f'Dice score: {best_dice:.4f}')
    
    return train_losses, val_losses, val_dice_scores

def calculate_iou(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def visualize_results(image, true_mask, pred_mask, idx):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    device = image.device
    
    #denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    image = image * std + mean
    image = torch.clamp(image, 0, 1)
    
    #to cpu
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    true_mask_np = true_mask.detach().cpu().squeeze().numpy()
    pred_mask_np = pred_mask.detach().cpu().squeeze().numpy()
    
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(true_mask_np, cmap='gray')
    axes[1].set_title('True Mask')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask_np, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'result_{idx}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved visualization: result_{idx}.png")


def test_model(model_path, test_loader, device):
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    results = []
    
    logger.info("testing...")
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predictions = (outputs > 0.5).float()
            dice_score = dice_coefficient(predictions, masks)
            iou_score = calculate_iou(predictions, masks)
            total_dice += dice_score.item()
            total_iou += iou_score.item()
            if i < 10:
                visualize_results(images[0], masks[0], predictions[0], i)
            
            results.append({
                "batch": i, 
                "dice_score": dice_score.item(), 
                "iou_score": iou_score.item()
            })
    
    avg_dice = total_dice / len(results)
    avg_iou = total_iou / len(results)
    
    logger.info(f"Test Results: Avg Dice: {avg_dice:.4f}, Avg IoU: {avg_iou:.4f}")
    return avg_dice, avg_iou, results

def main():
    IMG_SIZE = 256
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    IMAGE_DIR = "/home/atharv/Desktop/projects/img_segmentation_using_u-net/dataset/Image"
    MASK_DIR = "/home/atharv/Desktop/projects/img_segmentation_using_u-net/dataset/Mask"
    image_extensions = ['*.jpg',]
    mask_extensions = ['*.png',]
    
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext.upper())))
    
    mask_paths = []
    for ext in mask_extensions:
        mask_paths.extend(glob.glob(os.path.join(MASK_DIR, ext)))
        mask_paths.extend(glob.glob(os.path.join(MASK_DIR, ext.upper())))
    
    image_paths = sorted(image_paths)
    mask_paths = sorted(mask_paths)
    
    logger.info(f"Found {len(image_paths)} images and {len(mask_paths)} masks")
    
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    # Create datasets
    try:
        train_dataset = SegmentationDataset(
            train_imgs, train_masks, 
            transform=get_transforms(is_train=True),
            img_size=IMG_SIZE
        )
        
        val_dataset = SegmentationDataset(
            val_imgs, val_masks, 
            transform=get_transforms(is_train=False),
            img_size=IMG_SIZE
        )
    except ValueError as e:
        logger.error(f"Dataset creation failed: {str(e)}")
        return
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    if not test_dataloader(train_loader, DEVICE) or not test_dataloader(val_loader, DEVICE):
        logger.error("Dataloader test failed. Exiting.")
        return
    
    #model
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    criterion = CombinedLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    logger.info(f"Starting training on {DEVICE}")
    
    # Train model
    train_losses, val_losses, val_dice_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, DEVICE
    )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_dice_scores, label='Val Dice Score')
    plt.title('Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    test_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    avg_dice, avg_iou, test_results = test_model('best_unet_model.pth', test_loader, DEVICE)
    
    results = {
        'configuration': {
            'img_size': IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'device': str(DEVICE)
        },
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_dice_scores': val_dice_scores
        },
        'test_results': {
            'avg_dice_score': avg_dice,
            'avg_iou_score': avg_iou,
            'batch_results': test_results
        },
        'dataset_info': {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'total_batches_tested': len(test_results)
        }
    }
    
    with open('unet_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info("Training done")

if __name__ == '__main__':
    main()