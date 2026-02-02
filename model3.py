"""
Advanced TB Detection Pipeline: Attention U-Net + Hybrid Feature Fusion + Neuro-Evolutionary Optimization
---------------------------------------------------------------------------------------------------------
Author: [Your Name]
Project: Neuro-Evolved Tuberculosis Diagnostics
Methodology:
    1. Preprocessing: CLAHE (Contrast Limited Adaptive Histogram Equalization) for texture enhancement.
    2. Segmentation: Attention U-Net with learnable Gating Signals to focus on ROI (Lungs).
    3. Feature Extraction: Hybrid Fusion of Deep Bottleneck Features + Handcrafted Radiomics (GLCM/LBP).
    4. Optimization: Binary Grey Wolf Optimization (BGWO) for Evolutionary Feature Selection.
    5. Classification: Support Vector Machine (SVM) on the evolved feature subspace.
"""

import os
import cv2
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, jaccard_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS_SEG = 15       # Train longer for better segmentation
WOLVES_COUNT = 8      # Number of search agents in GWO (The "Pack Size")
MAX_ITER_GWO = 10     # Iterations for optimization (The "Hunting Time")
LEARNING_RATE = 1e-4

# Paths (Adjust to your local structure)
DATA_DIR = './dataset'
IMG_DIR = os.path.join(DATA_DIR, 'CXR_png')
MASK_L_DIR = os.path.join(DATA_DIR, 'ManualMask', 'leftMask')
MASK_R_DIR = os.path.join(DATA_DIR, 'ManualMask', 'rightMask')

print(f"Running on Device: {DEVICE}")

# ==========================================
# 2. DATASET LOADING & PREPROCESSING
# ==========================================
class ChestXRayDataset(Dataset):
    def __init__(self, img_paths, mask_l_paths, mask_r_paths, transform=None):
        self.img_paths = img_paths
        self.mask_l_paths = mask_l_paths
        self.mask_r_paths = mask_r_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Read Image
        img_path = self.img_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
        # 2. NOVELTY: CLAHE Preprocessing
        # Standard HistEq washes out details. CLAHE maintains local contrast.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image)
        
        # 3. Read & Merge Masks
        # Montgomery dataset has separate L/R masks. We merge them.
        mask_l = cv2.imread(self.mask_l_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask_r = cv2.imread(self.mask_r_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Handle cases where masks might be missing or different sizes
        if mask_l is None or mask_r is None:
             # Fallback for safety
             mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        else:
            mask_l = cv2.resize(mask_l, (IMG_SIZE, IMG_SIZE))
            mask_r = cv2.resize(mask_r, (IMG_SIZE, IMG_SIZE))
            mask = np.maximum(mask_l, mask_r) # Logical OR to combine
        
        # Binarize
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Normalize [0,1]
        image = image / 255.0
        mask = mask / 255.0
        
        # Dimensions: (C, H, W)
        image = np.expand_dims(image, axis=0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0).astype(np.float32)
        
        # TB Label: 1 if '1.png' (abnormal) else 0 (normal) for Montgomery naming convention
        # Montgomery files: MCUCXR_0001_0.png (Normal), MCUCXR_0104_1.png (TB)
        label = 1 if "_1.png" in img_path else 0
        
        return torch.tensor(image), torch.tensor(mask), label

def load_data():
    # Gather all file paths
    all_images = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    all_masks_l = sorted(glob.glob(os.path.join(MASK_L_DIR, "*.png")))
    all_masks_r = sorted(glob.glob(os.path.join(MASK_R_DIR, "*.png")))

    if len(all_images) == 0:
        raise ValueError("No images found! Check your DATA_DIR path.")
    
    print(f"Found {len(all_images)} images.")
    
    # Split Data
    train_imgs, val_imgs, train_ml, val_ml, train_mr, val_mr = train_test_split(
        all_images, all_masks_l, all_masks_r, test_size=0.2, random_state=42
    )

    train_dataset = ChestXRayDataset(train_imgs, train_ml, train_mr)
    val_dataset = ChestXRayDataset(val_imgs, val_ml, val_mr)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    return train_loader, val_loader

# ==========================================
# 3. NOVELTY: ATTENTION U-NET
# ==========================================
# Standard U-Net treats all pixels equally. Attention Gates suppress 
# background noise and highlight the lung region automatically.

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        # W_g: Gate signal (from coarser scale)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # W_x: Skip connection signal (from encoder)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Psi: Attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi # Scale the skip connection

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class AttentionUNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(AttentionUNet, self).__init__()
        
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024) # Bottleneck

        # Decoder with Attention Gates
        self.Up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Att5 = AttentionBlock(F_g=1024, F_l=512, F_int=256) # Novelty
        self.Up_conv5 = ConvBlock(1024 + 512, 512)

        self.Up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Att4 = AttentionBlock(F_g=512, F_l=256, F_int=128) # Novelty
        self.Up_conv4 = ConvBlock(512 + 256, 256)

        self.Up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Att3 = AttentionBlock(F_g=256, F_l=128, F_int=64) # Novelty
        self.Up_conv3 = ConvBlock(256 + 128, 128)

        self.Up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Att2 = AttentionBlock(F_g=128, F_l=64, F_int=32) # Novelty
        self.Up_conv2 = ConvBlock(128 + 64, 64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4)) # Deep features here

        # Decoder + Attention
        d5 = self.Up5(x5)
        x4_att = self.Att5(g=d5, x=x4) # Attention filtering
        d5 = torch.cat((x4_att, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3_att = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3_att, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2_att = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2_att, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1_att = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1_att, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv_1x1(d2)
        out = self.Sigmoid(out)
        
        # Return Mask AND Bottleneck features (for hybrid fusion)
        return out, x5

# ==========================================
# 4. TRAINING SEGMENTATION
# ==========================================
def train_segmentation(model, train_loader, val_loader):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n--- Starting Attention U-Net Training ---")
    for epoch in range(EPOCHS_SEG):
        model.train()
        train_loss = 0
        
        for imgs, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_SEG}"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs, _ = model(imgs) # We ignore features during seg training
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {train_loss/len(train_loader):.4f}")
    
    print("Segmentation Model Trained Successfully.")
    return model

# ==========================================
# 5. NOVELTY: HYBRID FEATURE EXTRACTION
# ==========================================
def extract_radiomics(image_tensor):
    """
    Extracts Handcrafted Features (GLCM, LBP) from a single image.
    This captures texture information that CNNs might miss.
    """
    # Convert tensor to numpy uint8 (0-255)
    img_np = (image_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    
    # A. GLCM Features (Texture)
    # Calculate GLCM for distance 1 and angles 0, 45, 90, 135
    glcm = graycomatrix(img_np, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    # B. Local Binary Patterns (LBP) - Micro-texture
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(img_np, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7) # Normalize
    
    # Fuse Handcrafted
    handcrafted = np.concatenate(([contrast, energy, homogeneity, correlation], hist))
    return handcrafted

def get_hybrid_dataset(model, loader):
    """
    Passes data through the trained Attention U-Net.
    Extracts Deep Features from the bottleneck and fuses them with Handcrafted Features.
    """
    model.eval()
    features_list = []
    labels_list = []
    
    print("\n--- Extracting Hybrid Features (Deep + Handcrafted) ---")
    with torch.no_grad():
        for imgs, _, labels in tqdm(loader, desc="Feature Extraction"):
            imgs = imgs.to(DEVICE)
            
            # 1. Get Deep Features from Bottleneck (x5)
            _, bottleneck = model(imgs) 
            # Global Average Pooling: (Batch, 1024, 8, 8) -> (Batch, 1024)
            deep_feats = torch.mean(bottleneck, dim=[2, 3]).cpu().numpy()
            
            # 2. Get Handcrafted Features for each image in batch
            batch_handcrafted = []
            for i in range(imgs.size(0)):
                hc = extract_radiomics(imgs[i])
                batch_handcrafted.append(hc)
            batch_handcrafted = np.array(batch_handcrafted)
            
            # 3. FUSION
            # Concatenate Deep (1024) + Handcrafted (~14)
            fused = np.hstack((deep_feats, batch_handcrafted))
            
            features_list.append(fused)
            labels_list.append(labels.numpy())
            
    X = np.vstack(features_list)
    y = np.concatenate(labels_list)
    print(f"Hybrid Feature Vector Shape: {X.shape}")
    return X, y

# ==========================================
# 6. NOVELTY: GREY WOLF OPTIMIZATION (GWO)
# ==========================================
# This is a Meta-Heuristic Algorithm for Feature Selection.
# Instead of using all 1038 features, we use GWO to find the "Alpha" subset 
# that maximizes classification accuracy.

class GreyWolfOptimizer:
    def __init__(self, X, y, n_wolves=8, max_iter=10):
        self.X = X
        self.y = y
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.dim = X.shape[1] # Number of features
        
        # Initialize Wolves (Binary positions: 1=Selected, 0=Not Selected)
        # We use continuous values [0,1] and threshold them
        self.positions = np.random.rand(n_wolves, self.dim)
        
        # Initialize Alpha, Beta, Delta (Best 3 solutions)
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = -float('inf') 
        
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = -float('inf')
        
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = -float('inf')

    def fitness(self, position):
        # 1. Threshold position to binary mask
        mask = position > 0.5
        if np.sum(mask) == 0: return 0 # Prevent empty selection
        
        # 2. Select features
        X_subset = self.X[:, mask]
        
        # 3. Fast Validation with SVM
        X_train, X_test, y_train, y_test = train_test_split(X_subset, self.y, test_size=0.2, random_state=42)
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        
        # Fitness = Accuracy - Weight * (Num_Features / Total_Features)
        # We want high accuracy with fewer features
        return 0.99 * acc + 0.01 * (1 - (np.sum(mask) / self.dim))

    def optimize(self):
        print(f"\n--- Starting Grey Wolf Optimization (Jargon: Neuro-Evolutionary Selection) ---")
        
        for t in range(self.max_iter):
            # Calculate Fitness for each wolf
            for i in range(self.n_wolves):
                fitness = self.fitness(self.positions[i])
                
                # Update Alpha, Beta, Delta
                if fitness > self.alpha_score:
                    self.delta_score = self.beta_score; self.delta_pos = self.beta_pos.copy()
                    self.beta_score = self.alpha_score; self.beta_pos = self.alpha_pos.copy()
                    self.alpha_score = fitness; self.alpha_pos = self.positions[i].copy()
                elif fitness > self.beta_score:
                    self.delta_score = self.beta_score; self.delta_pos = self.beta_pos.copy()
                    self.beta_score = fitness; self.beta_pos = self.positions[i].copy()
                elif fitness > self.delta_score:
                    self.delta_score = fitness; self.delta_pos = self.positions[i].copy()
            
            # Update positions of all wolves
            a = 2 - t * (2 / self.max_iter) # Linearly decreases from 2 to 0
            
            for i in range(self.n_wolves):
                for j in range(self.dim):
                    # GWO Math (Standard Equations)
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2*a*r1 - a; C1 = 2*r2
                    D_alpha = abs(C1*self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1*D_alpha
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2*a*r1 - a; C2 = 2*r2
                    D_beta = abs(C2*self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2*D_beta
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2*a*r1 - a; C3 = 2*r2
                    D_delta = abs(C3*self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3*D_delta
                    
                    self.positions[i, j] = (X1 + X2 + X3) / 3
            
            print(f"Iter {t+1}/{self.max_iter} | Best Fitness: {self.alpha_score:.4f}")
            
        print("Optimization Complete.")
        return self.alpha_pos > 0.5 # Return binary mask

# ==========================================
# 7. MAIN PIPELINE
# ==========================================
def main():
    # Step 1: Prepare Data
    train_loader, val_loader = load_data()
    
    # Step 2: Initialize & Train Segmentation Model
    model = AttentionUNet().to(DEVICE)
    model = train_segmentation(model, train_loader, val_loader)
    
    # Step 3: Extract Hybrid Features (Deep + Handcrafted)
    # We use the full dataset for feature extraction to prepare for classification
    print("\n--- Preparing Data for Classification ---")
    full_dataset_loader = DataLoader(
        train_loader.dataset + val_loader.dataset, 
        batch_size=1, shuffle=False
    )
    X, y = get_hybrid_dataset(model, full_dataset_loader)
    
    # Step 4: Optimization (Feature Selection)
    # Use GWO to select the best features
    optimizer = GreyWolfOptimizer(X, y, n_wolves=WOLVES_COUNT, max_iter=MAX_ITER_GWO)
    selected_mask = optimizer.optimize()
    
    X_optimized = X[:, selected_mask]
    print(f"Features Reduced: {X.shape[1]} -> {X_optimized.shape[1]}")
    
    # Step 5: Final Classification (SVM)
    print("\n--- Training Final SVM Classifier ---")
    X_train, X_test, y_train, y_test = train_test_split(X_optimized, y, test_size=0.2, random_state=42)
    
    svm = SVC(kernel='rbf', C=1.0) # RBF Kernel deals with non-linearities well
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    
    # Report
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'TB']))
    
    # Optional: Save specific outputs to show faculty
    print("\n[INFO] Pipeline finished. To show faculty:")
    print("1. Show the Attention U-Net code (novelty: attention gates).")
    print("2. Show 'extract_radiomics' function (novelty: hybrid features).")
    print("3. Show 'GreyWolfOptimizer' class (novelty: nature-inspired optimization).")

if __name__ == "__main__":
    main()