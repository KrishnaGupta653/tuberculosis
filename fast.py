import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import os
import copy
from torch.cuda.amp import GradScaler, autocast # For Mixed Precision (Speed)

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
DATA_DIR = './dataset'
MODEL_SAVE_PATH = 'best_hybrid_tb_model.pth'
BATCH_SIZE = 16 
LEARNING_RATE = 1e-4
EPOCHS = 15
NUM_WORKERS = 4      # Optimized for parallel loading
PIN_MEMORY = True    # Faster Host-to-GPU transfer
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 2. HYBRID MODEL ARCHITECTURE (DenseNet + ViT)
# ---------------------------------------------------------
class DenseNetViTHybrid(nn.Module):
    def __init__(self, num_classes=2, embed_dim=1024, num_heads=8, num_layers=4):
        super(DenseNetViTHybrid, self).__init__()
        
        # A. CNN Backbone (DenseNet-121)
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features  # (B, 1024, 7, 7)
        
        # B. Transformer Components
        self.flatten = nn.Flatten(2)       # (B, 1024, 49)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, 49))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # C. Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)           # CNN Features
        x = x.flatten(2) + self.pos_embedding # Flatten & Add Position
        x = x.permute(0, 2, 1)         # Swap for Transformer: (B, 49, 1024)
        x = self.transformer(x)        # Global Context Attention
        x = x.permute(0, 2, 1)         # Swap back
        x = self.classifier(x)         # Classification
        return x

# ---------------------------------------------------------
# 3. ROBUST DATA PIPELINE
# ---------------------------------------------------------
# Wrapper to apply transforms strictly *after* splitting
class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def get_dataloaders(data_dir):
    # [cite_start]Train: Heavy Augmentation (Robustness) [cite: 94]
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Val: No Augmentation (Honest Evaluation)
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load raw dataset
    base_dataset = datasets.ImageFolder(data_dir)
    
    # Split
    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size
    train_raw, val_raw = random_split(base_dataset, [train_size, val_size])
    
    # Apply correct transforms
    train_dataset = TransformedSubset(train_raw, transform_train)
    val_dataset = TransformedSubset(val_raw, transform_val)

    # Fast Loaders with num_workers & pin_memory
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    return train_loader, val_loader, base_dataset.classes

# ---------------------------------------------------------
# 4. FAST TRAINING LOOP (AMP ENABLED)
# ---------------------------------------------------------
def train_model():
    print(f"Initializing Hybrid Model on {DEVICE} with {NUM_WORKERS} workers...")
    train_loader, val_loader, class_names = get_dataloaders(DATA_DIR)
    
    model = DenseNetViTHybrid().to(DEVICE)
    
    # AdamW + Cosine Schedule for "Best" Convergence
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    # Scalar for Mixed Precision Speedup
    scaler = GradScaler() 

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(DEVICE, non_blocking=True) # non_blocking for speed
                labels = labels.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()

                # Mixed Precision Context
                with torch.set_grad_enabled(phase == 'train'):
                    with autocast(enabled=(DEVICE.type == 'cuda')): 
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        # Scale loss to handle float16 precision
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"--> Best Model Saved! ({best_acc:.4f})")

    print(f'\nFinal Best Validation Accuracy: {best_acc:.4f}')

if __name__ == '__main__':
    # Fix for Windows multiprocessing
    torch.multiprocessing.freeze_support()
    train_model()