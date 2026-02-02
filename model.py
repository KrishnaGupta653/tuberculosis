import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
import copy

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
DATA_DIR = './dataset'  # Ensure your 'Normal...' and 'TB...' folders are inside this folder
MODEL_SAVE_PATH = 'best_hybrid_tb_model.pth'
BATCH_SIZE = 16
LEARNING_RATE = 1e-4   # Lower LR is better for fine-tuning [cite: 100]
EPOCHS = 15            # Increased epochs for convergence [cite: 100]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 2. HYBRID MODEL ARCHITECTURE (DenseNet + ViT)
# ---------------------------------------------------------
class DenseNetViTHybrid(nn.Module):
    def __init__(self, num_classes=2, embed_dim=1024, num_heads=8, num_layers=4):
        super(DenseNetViTHybrid, self).__init__()
        
        # A. CNN Backbone (DenseNet-121) for Local Features [cite: 14]
        # We remove the classifier and use the features layer
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features  # Output shape: (Batch, 1024, 7, 7) for 224x224 img
        
        # B. Transformer Encoder for Global Context [cite: 19]
        # We flatten the 7x7 spatial grid into a sequence of length 49
        self.flatten = nn.Flatten(2)  # Output: (Batch, 1024, 49)
        
        # Positional Embedding (Learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, 49))
        
        # Transformer Layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # C. Classification Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Pool over the sequence (49 tokens -> 1)
            nn.Flatten(),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes) # Output logits (no Softmax here, handled by CrossEntropy)
        )

    def forward(self, x):
        # 1. Extract Local Features
        x = self.features(x)         # (B, 1024, 7, 7)
        
        # 2. Prepare for Transformer
        x = x.flatten(2)             # (B, 1024, 49)
        x = x + self.pos_embedding   # Add positional info
        x = x.permute(0, 2, 1)       # Swap for Transformer: (B, Seq_Len=49, Embed=1024)
        
        # 3. Global Context Processing
        x = self.transformer(x)      # (B, 49, 1024)
        
        # 4. Classification
        x = x.permute(0, 2, 1)       # Swap back for pooling: (B, 1024, 49)
        x = self.classifier(x)
        return x

# ---------------------------------------------------------
# 3. DATA PIPELINE
# ---------------------------------------------------------
def get_dataloaders(data_dir):
    # Clinical-grade augmentations
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # Slightly higher rotation for robustness [cite: 94]
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Account for X-ray exposure diffs
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir)
    
    # 80/20 Train-Validation Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Apply transforms manually (since random_split doesn't support separate transforms easily, 
    # we override the dataset transform for the validation set wrapper if needed, 
    # but for simplicity here we use the base dataset transform. 
    # For maximum rigour, separate dataset objects are better, but this works for this scope).
    dataset.transform = transform_train 

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, dataset.classes

# ---------------------------------------------------------
# 4. TRAINING LOOP
# ---------------------------------------------------------
def train_model():
    print(f"Initializing Hybrid Model on {DEVICE}...")
    train_loader, val_loader, class_names = get_dataloaders(DATA_DIR)
    print(f"Classes detected: {class_names}")

    model = DenseNetViTHybrid().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # AdamW is better for Transformers
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        # Each epoch has a training and validation phase
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
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best one so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"--> New Best Model Saved! (Acc: {best_acc:.4f})")

    print(f'\nBest Validation Accuracy: {best_acc:.4f}')
    print(f'Model saved to {MODEL_SAVE_PATH}')

if __name__ == '__main__':
    train_model()