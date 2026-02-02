import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern, hog
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, jaccard_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==============================================================================
# 1. NOVEL PREPROCESSING: CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ==============================================================================
# WHY: Standard equalization washes out details. CLAHE enhances local contrast, 
# making subtle TB infiltrates in the lung fields more visible for the model.
# ==============================================================================
def apply_clahe(image_path, size=(256, 256)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    
    # Create CLAHE object (Arguments: clipLimit=2.0, tileGridSize=(8,8))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    
    # Normalize to 0-1 range
    img_normalized = img_clahe / 255.0
    return img_normalized.astype(np.float32)

# ==============================================================================
# 2. NOVEL SEGMENTATION: Attention Gate U-Net
# ==============================================================================
# WHY: Standard U-Net treats all pixels equally. Attention Gates (AG) allow the 
# model to suppress irrelevant regions (background, bones) and highlight salient 
# features (lung fields) automatically during the skip connection phase.
# ==============================================================================

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: Gating signal (from lower layer)
        # x: Skip connection signal (from encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # Scale skip connection by attention map

class AdaptiveAttentionUNet(nn.Module):
    def __init__(self):
        super(AdaptiveAttentionUNet, self).__init__()
        
        # Encoder (Contracting Path)
        self.enc1 = self.conv_block(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck (Deep Features reside here)
        self.bottleneck = self.conv_block(256, 512)
        
        # Attention Gates & Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att3 = AttentionBlock(F_g=512, F_l=256, F_int=128) # NOVELTY: Attention Gate
        self.dec3 = self.conv_block(512 + 256, 256) # Concatenation size adjustment

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att2 = AttentionBlock(F_g=256, F_l=128, F_int=64)  # NOVELTY: Attention Gate
        self.dec2 = self.conv_block(256 + 128, 128)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att1 = AttentionBlock(F_g=128, F_l=64, F_int=32)   # NOVELTY: Attention Gate
        self.dec1 = self.conv_block(128 + 64, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder with Attention
        # Notice we pass 'b' (gating) and 'e3' (skip connection) to Attention Block
        d3 = self.up3(b)
        x3 = self.att3(g=d3, x=e3) 
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=e2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=e1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.final(d1)), b  # Return mask AND bottleneck features

# ==============================================================================
# 3. NOVEL OPTIMIZATION: Teamwork Optimization Algorithm (TOA)
# ==============================================================================
# WHY: Most models use a static threshold (0.5) to convert probability maps to 
# binary masks. TOA is a meta-heuristic that treats this threshold as a variable 
# to be optimized by a "team" of agents. 
# It maximizes the Dice Coefficient on the Validation Set.
# ==============================================================================

class TeamworkOptimizer:
    def __init__(self, predictions, ground_truth, n_agents=10, iterations=15):
        self.preds = predictions.detach().cpu().numpy()
        self.gt = ground_truth.detach().cpu().numpy()
        self.n_agents = n_agents
        self.iterations = iterations
        self.agents = np.random.uniform(0.1, 0.9, n_agents) # Initial thresholds
        self.best_score = 0
        self.best_threshold = 0.5

    def dice_coefficient(self, threshold):
        # Calculate Dice for a given threshold
        bin_preds = (self.preds > threshold).astype(np.float32)
        intersection = np.sum(bin_preds * self.gt)
        return (2. * intersection) / (np.sum(bin_preds) + np.sum(self.gt) + 1e-6)

    def optimize(self):
        print("\n--- Starting TOA Optimization for Segmentation Threshold ---")
        for t in range(self.iterations):
            # 1. Evaluation
            scores = np.array([self.dice_coefficient(th) for th in self.agents])
            
            # 2. Find Global Best (Pg)
            current_best_idx = np.argmax(scores)
            if scores[current_best_idx] > self.best_score:
                self.best_score = scores[current_best_idx]
                self.best_threshold = self.agents[current_best_idx]
            
            # 3. Update Positions (Simplified TOA Equation)
            # Xt+1 = Xt + alpha * (Interaction) + beta * (Guidance from Best)
            alpha = 0.1 * np.random.rand()
            beta = 0.2 * np.random.rand()
            
            for i in range(self.n_agents):
                # Interaction: Move towards a random partner
                partner_idx = np.random.randint(0, self.n_agents)
                interaction = self.agents[partner_idx] - self.agents[i]
                
                # Guidance: Move towards the best agent
                guidance = self.best_threshold - self.agents[i]
                
                # Update
                self.agents[i] = self.agents[i] + alpha * interaction + beta * guidance
                
                # Clamp values
                self.agents[i] = np.clip(self.agents[i], 0.05, 0.95)
                
            print(f"TOA Iteration {t+1}/{self.iterations}: Best Dice = {self.best_score:.4f} at Threshold = {self.best_threshold:.4f}")
            
        return self.best_threshold

# ==============================================================================
# 4. HYBRID FEATURE EXTRACTION & FUSION
# ==============================================================================
# WHY: Deep Learning is great at semantic features, but Handcrafted features 
# (LBP, HOG) are mathematically robust for texture (important for TB infiltrates).
# Fusing them provides a richer representation than either alone.
# ==============================================================================

def extract_handcrafted_features(image_np):
    # Image needs to be uint8 0-255
    img = (image_np * 255).astype(np.uint8)
    
    # 1. LBP (Texture)
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10 + 3), range=(0, 10 + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6) # Normalize
    
    # 2. HOG (Shape/Gradient)
    # Resize to smaller block for HOG speed
    img_small = cv2.resize(img, (64, 64))
    hog_feats = hog(img_small, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=False)
    
    # Fusion
    return np.concatenate([lbp_hist, hog_feats])

def get_hybrid_vectors(model, data_loader, device):
    model.eval()
    deep_features_list = []
    handcrafted_features_list = []
    labels_list = []
    
    print("\n--- Extracting Hybrid Features ---")
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            inputs_np = inputs.cpu().numpy().squeeze() # Remove batch dim if batch=1
            
            # A. Get Deep Features from Bottleneck
            # Pass through encoder parts manually or change forward return
            # Here we assume model.forward returns (mask, bottleneck)
            _, bottleneck = model(inputs) 
            
            # Global Average Pooling on Bottleneck (512, 32, 32) -> (512)
            deep_feat = torch.mean(bottleneck, dim=[2, 3])
            deep_features_list.append(deep_feat.cpu().numpy())
            
            # B. Get Handcrafted Features
            # Loop over batch
            curr_handcrafted = []
            if len(inputs_np.shape) == 2: # Single image batch
                curr_handcrafted.append(extract_handcrafted_features(inputs_np))
            else:
                for img_idx in range(inputs_np.shape[0]):
                    curr_handcrafted.append(extract_handcrafted_features(inputs_np[img_idx]))
            
            handcrafted_features_list.append(np.array(curr_handcrafted))
            labels_list.append(labels.numpy())

    # Concatenate all
    D = np.vstack(deep_features_list)
    H = np.vstack(handcrafted_features_list)
    Y = np.concatenate(labels_list)
    
    # Early fusion
    X_fused = np.hstack((D, H))
    print(f"Deep Feature Shape: {D.shape}")
    print(f"Handcrafted Feature Shape: {H.shape}")
    print(f"Fused Feature Shape: {X_fused.shape}")
    
    return X_fused, Y

# ==============================================================================
# 5. MAIN PIPELINE EXECUTION
# ==============================================================================

def run_project_pipeline():
    # A. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdaptiveAttentionUNet().to(device)
    
    # [Dummy Data Generator for Codebase demonstration]
    # In real use, use your Datasets
    print("Initializing Dummy Data...")
    X_dummy = torch.rand(20, 1, 256, 256) # 20 images
    Y_mask_dummy = torch.randint(0, 2, (20, 1, 256, 256)).float() # 20 masks
    Y_cls_dummy = torch.randint(0, 2, (20,)) # 20 Class labels (TB/Normal)
    
    dataset = torch.utils.data.TensorDataset(X_dummy, Y_cls_dummy) # For Classification
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # B. Train Segmentation (Simplified)
    # ... (Standard training loop would go here) ...
    print("Segmentation Training Completed (Simulated).")
    
    # C. Perform TOA Optimization
    # We use the model's output on validation set to find best threshold
    sample_img = X_dummy[0:5].to(device)
    sample_gt = Y_mask_dummy[0:5].to(device)
    preds, _ = model(sample_img)
    
    optimizer_toa = TeamworkOptimizer(preds, sample_gt, n_agents=5, iterations=5)
    optimal_threshold = optimizer_toa.optimize()
    
    print(f"\n>>> FINAL OPTIMIZED THRESHOLD: {optimal_threshold:.4f} <<<")
    print("(This threshold will be used to generate binary masks for radiologist visual support)")

    # D. Feature Extraction & Hybrid Fusion
    X_fused, Y_labels = get_hybrid_vectors(model, dataloader, device)
    
    # E. Feature Selection via PCA (Novelty: Dimensionality Reduction)
    # We reduce the massive fused vector to the most informative components
    print("\n--- Applying PCA for Feature Selection ---")
    pca = PCA(n_components=0.95) # Keep 95% variance
    X_pca = pca.fit_transform(X_fused)
    print(f"Features reduced from {X_fused.shape[1]} to {X_pca.shape[1]} components.")
    
    # F. Final Classification (SVM)
    # SVM is excellent for high-dimensional feature spaces created by hybrid fusion
    print("\n--- Training SVM Classifier ---")
    X_train, X_test, y_train, y_test = train_test_split(X_pca, Y_labels, test_size=0.2)
    clf = SVC(kernel='rbf', C=1.0, probability=True)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n>>> FINAL CLASSIFICATION ACCURACY: {acc*100:.2f}% <<<")

if __name__ == "__main__":
    run_project_pipeline()