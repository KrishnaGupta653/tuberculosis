"""
================================================================================
PRODUCTION-READY TB DETECTION SYSTEM WITH MODEL PERSISTENCE
================================================================================
CRITICAL IMPROVEMENT: This version saves ALL trained components for deployment

WHAT GETS SAVED:
────────────────
1. Trained ensemble models (SVM, RF, GB)
2. Feature scaler (normalization parameters)
3. Selected feature indices (from quantum optimization)
4. Model metadata (accuracy, feature names, etc.)

WHY THIS MATTERS:
─────────────────
✓ Train once, use forever
✓ Deploy to production
✓ Real-time predictions on new patients
✓ No need to retrain for every new image

USAGE:
──────
Training: python tb_detection_complete.py --mode train
Prediction: python tb_detection_complete.py --mode predict --image path/to/xray.png
================================================================================
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, accuracy_score, 
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor
import pywt
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import warnings
import pickle
import json
from datetime import datetime
import argparse
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    """Central configuration for reproducibility and easy tuning"""
    
    # Dataset
    DATA_DIR = './dataset'
    CATEGORIES = ['Normal', 'TB']
    IMG_SIZE = 224
    
    # Model Saving
    MODEL_DIR = './saved_models'  # ← NEW: Where to save trained models
    MODEL_NAME = 'tb_detector_v1'
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Segmentation
    ENTROPY_WINDOW = 15
    MORPHOLOGY_ITERATIONS = 3
    MIN_CONTOUR_AREA = 1000
    
    # Feature Extraction
    GLCM_DISTANCES = [1, 3, 5]
    GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    WAVELET_FAMILY = 'db4'
    WAVELET_LEVELS = 3
    FRACTAL_BOXES = [2, 4, 8, 16, 32]
    
    # Quantum Optimization
    QUANTUM_POPULATION = 30
    QUANTUM_GENERATIONS = 15
    ROTATION_ANGLE = 0.05 * np.pi
    
    # Model Training
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    DROPOUT_RATE = 0.3
    MC_SAMPLES = 20
    
    # Validation
    CROSS_VAL_FOLDS = 5
    TEST_SIZE = 0.2
    RANDOM_SEED = 42

# Set seeds
np.random.seed(Config.RANDOM_SEED)
torch.manual_seed(Config.RANDOM_SEED)
random.seed(Config.RANDOM_SEED)

# Create model directory
os.makedirs(Config.MODEL_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# SEGMENTATION (Same as before)
# ═══════════════════════════════════════════════════════════════════════════

class EntropyGuidedSegmentor:
    """Entropy-based lung segmentation (no changes needed)"""
    
    def __init__(self):
        self.entropy_window = Config.ENTROPY_WINDOW
        
    def compute_local_entropy(self, image):
        from skimage.filters.rank import entropy as rank_entropy
        from skimage.morphology import disk
        img_uint8 = (image / image.max() * 255).astype(np.uint8)
        entr = rank_entropy(img_uint8, disk(self.entropy_window // 2))
        return entr
    
    def multi_scale_clahe(self, image):
        scales = [8, 16, 32]
        enhanced_stack = []
        for grid_size in scales:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(grid_size, grid_size))
            enhanced = clahe.apply(image)
            enhanced_stack.append(enhanced)
        weights = [0.2, 0.5, 0.3]
        fused = np.zeros_like(image, dtype=np.float32)
        for w, img in zip(weights, enhanced_stack):
            fused += w * img.astype(np.float32)
        return fused.astype(np.uint8)
    
    def attention_weighted_morph(self, binary_mask, attention_map):
        attention_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        kernel_base = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        result = np.zeros_like(binary_mask)
        block_size = 32
        h, w = binary_mask.shape
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = binary_mask[i:i+block_size, j:j+block_size]
                attn_block = attention_norm[i:i+block_size, j:j+block_size]
                mean_attn = np.mean(attn_block)
                iterations = int(1 + mean_attn * Config.MORPHOLOGY_ITERATIONS)
                processed_block = cv2.morphologyEx(block, cv2.MORPH_CLOSE, 
                                                   kernel_base, iterations=iterations)
                result[i:i+block_size, j:j+block_size] = processed_block
        return result
    
    def segment_lung_roi(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, None, None
        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = self.multi_scale_clahe(gray)
        entropy_map = self.compute_local_entropy(enhanced)
        sensitivity = 0.5
        threshold_val = np.mean(entropy_map) + sensitivity * np.std(entropy_map)
        _, binary_entropy = cv2.threshold(entropy_map.astype(np.uint8), 
                                          int(threshold_val), 255, cv2.THRESH_BINARY)
        _, binary_otsu = cv2.threshold(enhanced, 0, 255, 
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_combined = cv2.bitwise_and(binary_entropy, binary_otsu)
        refined_mask = self.attention_weighted_morph(binary_combined, entropy_map)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, 
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(gray)
        if contours:
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in sorted_contours:
                area = cv2.contourArea(cnt)
                if area > Config.MIN_CONTOUR_AREA:
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        if circularity > 0.3:
                            cv2.drawContours(final_mask, [cnt], -1, 255, -1)
        segmented_img = cv2.bitwise_and(img, img, mask=final_mask)
        return segmented_img, final_mask, entropy_map


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (Same as before)
# ═══════════════════════════════════════════════════════════════════════════

class FractalWaveletExtractor:
    """Fractal + Wavelet radiomics (no changes needed)"""
    
    def compute_fractal_dimension(self, image):
        threshold = np.mean(image)
        binary = (image > threshold).astype(np.uint8)
        box_sizes = Config.FRACTAL_BOXES
        counts = []
        for box_size in box_sizes:
            try:
                shrunk = cv2.resize(binary, 
                                   (binary.shape[1] // box_size, 
                                    binary.shape[0] // box_size),
                                   interpolation=cv2.INTER_NEAREST)
                count = np.sum(shrunk > 0)
                counts.append(count)
            except:
                counts.append(0)
        counts = np.array(counts) + 1
        box_sizes = np.array(box_sizes)
        log_boxes = np.log(1.0 / box_sizes)
        log_counts = np.log(counts)
        coeffs = np.polyfit(log_boxes, log_counts, 1)
        fractal_dim = coeffs[0]
        return fractal_dim
    
    def extract_wavelet_features(self, image):
        features = []
        coeffs = pywt.wavedec2(image, Config.WAVELET_FAMILY, level=Config.WAVELET_LEVELS)
        for i, (cH, cV, cD) in enumerate(coeffs[1:], 1):
            for coeff, name in zip([cH, cV, cD], ['Horizontal', 'Vertical', 'Diagonal']):
                features.extend([
                    np.mean(coeff),
                    np.std(coeff),
                    np.max(np.abs(coeff)),
                    np.percentile(coeff, 75) - np.percentile(coeff, 25)
                ])
        return np.array(features)
    
    def extract_gabor_features(self, image):
        features = []
        frequencies = [0.1, 0.3, 0.5]
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        for freq in frequencies:
            for theta in orientations:
                real, imag = gabor(image, frequency=freq, theta=theta)
                features.extend([
                    np.mean(real),
                    np.std(real),
                    np.mean(np.abs(real))
                ])
        return np.array(features)
    
    def extract_advanced_glcm(self, image):
        features = []
        image_norm = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
        glcm = graycomatrix(image_norm, 
                           distances=Config.GLCM_DISTANCES, 
                           angles=Config.GLCM_ANGLES, 
                           levels=256, 
                           symmetric=True, 
                           normed=True)
        properties = ['contrast', 'dissimilarity', 'homogeneity', 
                     'energy', 'correlation', 'ASM']
        for prop in properties:
            try:
                values = graycoprops(glcm, prop).ravel()
                features.extend([
                    np.mean(values),
                    np.std(values),
                    np.max(values),
                    np.min(values)
                ])
            except:
                features.extend([0, 0, 0, 0])
        return np.array(features)
    
    def extract_all_features(self, segmented_img, mask):
        if len(segmented_img.shape) == 3:
            gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = segmented_img
        masked_region = cv2.bitwise_and(gray, gray, mask=mask)
        try:
            fractal_dim = self.compute_fractal_dimension(masked_region)
            wavelet_feats = self.extract_wavelet_features(masked_region)
            gabor_feats = self.extract_gabor_features(masked_region)
            glcm_feats = self.extract_advanced_glcm(masked_region)
            combined = np.concatenate([
                [fractal_dim],
                wavelet_feats,
                gabor_feats,
                glcm_feats
            ])
            return combined
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════
# DEEP LEARNING (Same as before)
# ═══════════════════════════════════════════════════════════════════════════

class UncertaintyAwareCNN(nn.Module):
    """DenseNet feature extractor (no changes needed)"""
    
    def __init__(self, num_classes=2):
        super(UncertaintyAwareCNN, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=Config.DROPOUT_RATE)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, num_classes)
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x, return_uncertainty=False):
        feats = self.features(x)
        feats = self.global_pool(feats)
        feats = torch.flatten(feats, 1)
        logits = self.classifier(feats)
        if return_uncertainty:
            log_var = self.uncertainty_head(feats)
            return logits, log_var
        return feats
    
    def extract_features(self, x):
        self.eval()
        with torch.no_grad():
            feats = self.features(x)
            feats = self.global_pool(feats)
            feats = torch.flatten(feats, 1)
        return feats.cpu().numpy()


class HybridFeatureFusion:
    """Combines deep + radiomic features"""
    
    def __init__(self):
        self.cnn_model = UncertaintyAwareCNN()
        self.cnn_model.to(Config.DEVICE)
        self.cnn_model.eval()
        self.radiomic_extractor = FractalWaveletExtractor()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_hybrid(self, segmented_img, mask):
        img_tensor = self.transform(segmented_img).unsqueeze(0).to(Config.DEVICE)
        deep_features = self.cnn_model.extract_features(img_tensor).flatten()
        radiomic_features = self.radiomic_extractor.extract_all_features(segmented_img, mask)
        if radiomic_features is None:
            return None
        hybrid_vector = np.concatenate([deep_features, radiomic_features])
        return hybrid_vector


# ═══════════════════════════════════════════════════════════════════════════
# QUANTUM OPTIMIZATION (Same as before)
# ═══════════════════════════════════════════════════════════════════════════

class QuantumFeatureSelector:
    """Quantum-inspired feature selection"""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.population_size = Config.QUANTUM_POPULATION
        self.generations = Config.QUANTUM_GENERATIONS
        self.quantum_pop = self._initialize_quantum_population()
        
    def _initialize_quantum_population(self):
        quantum_pop = np.zeros((self.population_size, self.n_features, 2))
        quantum_pop[:, :, 0] = 1.0 / np.sqrt(2)
        quantum_pop[:, :, 1] = 1.0 / np.sqrt(2)
        return quantum_pop
    
    def _measure_quantum_state(self, quantum_individual):
        binary = np.zeros(self.n_features, dtype=int)
        for i in range(self.n_features):
            alpha = quantum_individual[i, 0]
            prob_select = alpha ** 2
            if np.random.rand() < prob_select:
                binary[i] = 1
        return binary
    
    def _fitness_function(self, binary_mask):
        selected_indices = np.where(binary_mask == 1)[0]
        if len(selected_indices) < 5:
            return 0.0
        X_subset = self.X[:, selected_indices]
        X_train, X_val, y_train, y_val = train_test_split(
            X_subset, self.y, test_size=0.25, random_state=42, stratify=self.y
        )
        clf = SVC(kernel='rbf', C=10.0, gamma='scale')
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_val, y_val)
        lambda_penalty = 0.01
        penalty = lambda_penalty * (len(selected_indices) / self.n_features)
        fitness = accuracy - penalty
        return fitness
    
    def _quantum_rotation_gate(self, quantum_individual, best_individual, fitness, best_fitness):
        rotation_angle = Config.ROTATION_ANGLE
        for i in range(self.n_features):
            alpha, beta = quantum_individual[i]
            best_bit = best_individual[i]
            if fitness < best_fitness:
                if best_bit == 1:
                    delta_theta = rotation_angle
                else:
                    delta_theta = -rotation_angle
            else:
                delta_theta = rotation_angle * 0.1 * (np.random.rand() - 0.5)
            cos_theta = np.cos(delta_theta)
            sin_theta = np.sin(delta_theta)
            new_alpha = cos_theta * alpha - sin_theta * beta
            new_beta = sin_theta * alpha + cos_theta * beta
            norm = np.sqrt(new_alpha**2 + new_beta**2)
            quantum_individual[i, 0] = new_alpha / norm
            quantum_individual[i, 1] = new_beta / norm
        return quantum_individual
    
    def optimize(self):
        print(f"\n{'='*80}")
        print("QUANTUM-INSPIRED FEATURE OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Feature Space Dimension: {self.n_features}")
        print(f"Quantum Population Size: {self.population_size}")
        print(f"Generations: {self.generations}")
        print(f"{'='*80}\n")
        
        global_best_binary = None
        global_best_fitness = -np.inf
        history = []
        
        for generation in range(self.generations):
            classical_pop = []
            for q_individual in self.quantum_pop:
                binary = self._measure_quantum_state(q_individual)
                classical_pop.append(binary)
            
            fitnesses = []
            for binary in classical_pop:
                fit = self._fitness_function(binary)
                fitnesses.append(fit)
            
            fitnesses = np.array(fitnesses)
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > global_best_fitness:
                global_best_fitness = fitnesses[gen_best_idx]
                global_best_binary = classical_pop[gen_best_idx].copy()
            
            for i, q_individual in enumerate(self.quantum_pop):
                self.quantum_pop[i] = self._quantum_rotation_gate(
                    q_individual, global_best_binary, fitnesses[i], global_best_fitness
                )
            
            num_selected = np.sum(global_best_binary)
            print(f"Generation {generation+1}/{self.generations} | "
                  f"Best Fitness: {global_best_fitness:.4f} | "
                  f"Features Selected: {num_selected}/{self.n_features}")
            
            history.append({
                'generation': generation + 1,
                'best_fitness': global_best_fitness,
                'mean_fitness': np.mean(fitnesses),
                'features_selected': num_selected
            })
        
        print(f"\n{'='*80}")
        print("QUANTUM OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"Final Feature Count: {np.sum(global_best_binary)}/{self.n_features}")
        print(f"Reduction: {100*(1 - np.sum(global_best_binary)/self.n_features):.1f}%")
        print(f"{'='*80}\n")
        
        selected_indices = np.where(global_best_binary == 1)[0]
        return selected_indices, history


# ═══════════════════════════════════════════════════════════════════════════
# 🔥 NEW: ENSEMBLE WITH SAVE/LOAD FUNCTIONALITY
# ═══════════════════════════════════════════════════════════════════════════

class EnsembleClassifier:
    """
    CRITICAL IMPROVEMENT: Added save() and load() methods
    """
    
    def __init__(self):
        self.models = {
            'SVM': SVC(kernel='rbf', C=10.0, gamma='scale', probability=True),
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        self.weights = {}
        self.scaler = StandardScaler()
        self.is_trained = False  # Track training status
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train all models and determine ensemble weights"""
        print(f"\n{'='*80}")
        print("TRAINING ENSEMBLE CLASSIFIERS")
        print(f"{'='*80}\n")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            val_acc = model.score(X_val_scaled, y_val)
            self.weights[name] = val_acc
            print(f"  → Validation Accuracy: {val_acc*100:.2f}%")
        
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        self.is_trained = True
        
        print(f"\nEnsemble Weights: {self.weights}")
        print(f"{'='*80}\n")
    
    def predict_proba(self, X):
        """Weighted ensemble prediction"""
        if not self.is_trained:
            raise RuntimeError("Model not trained! Load a trained model or train first.")
        
        X_scaled = self.scaler.transform(X)
        ensemble_probs = np.zeros((X.shape[0], 2))
        
        for name, model in self.models.items():
            probs = model.predict_proba(X_scaled)
            ensemble_probs += self.weights[name] * probs
        
        return ensemble_probs
    
    def predict(self, X):
        """Final class prediction"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    # 🔥 NEW: SAVE METHOD
    def save(self, filepath):
        """
        Save trained ensemble to disk
        
        SAVES:
        ──────
        - All 3 trained models (SVM, RF, GB)
        - Ensemble weights
        - Feature scaler (normalization parameters)
        - Training metadata
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model!")
        
        model_data = {
            'models': self.models,
            'weights': self.weights,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'categories': Config.CATEGORIES,
                'img_size': Config.IMG_SIZE
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Ensemble models saved to: {filepath}")
    
    # 🔥 NEW: LOAD METHOD
    @classmethod
    def load(cls, filepath):
        """
        Load trained ensemble from disk
        
        RETURNS:
        ────────
        EnsembleClassifier instance ready for predictions
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls()
        instance.models = model_data['models']
        instance.weights = model_data['weights']
        instance.scaler = model_data['scaler']
        instance.is_trained = model_data['is_trained']
        
        print(f"✓ Ensemble models loaded from: {filepath}")
        print(f"  Trained on: {model_data['timestamp']}")
        
        return instance


# ═══════════════════════════════════════════════════════════════════════════
# 🔥 NEW: MODEL MANAGER - Saves EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════

class TBDetectorModelManager:
    """
    Manages saving and loading of ALL system components
    
    WHAT GETS SAVED:
    ────────────────
    1. Trained ensemble (SVM, RF, GB)
    2. Feature scaler
    3. Selected feature indices (from quantum optimization)
    4. Training performance metrics
    5. Model metadata
    """
    
    def __init__(self, model_name=None):
        self.model_name = model_name or Config.MODEL_NAME
        self.model_dir = Config.MODEL_DIR
        
    def save_complete_model(self, ensemble, selected_indices, scaler, 
                           training_metrics, quantum_history):
        """
        Save complete trained model with all components
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(self.model_dir, f"{self.model_name}_{timestamp}")
        
        print(f"\n{'='*80}")
        print("SAVING COMPLETE MODEL PACKAGE")
        print(f"{'='*80}\n")
        
        # 1. Save ensemble models
        ensemble_path = f"{base_path}_ensemble.pkl"
        ensemble.save(ensemble_path)
        
        # 2. Save feature selection data
        feature_data = {
            'selected_indices': selected_indices,
            'scaler': scaler,
            'original_feature_count': scaler.n_features_in_,
            'selected_feature_count': len(selected_indices),
            'reduction_ratio': 1 - (len(selected_indices) / scaler.n_features_in_)
        }
        feature_path = f"{base_path}_features.pkl"
        with open(feature_path, 'wb') as f:
            pickle.dump(feature_data, f)
        print(f"✓ Feature selector saved to: {feature_path}")
        
        # 3. Save training metrics and metadata
        metadata = {
            'model_name': self.model_name,
            'timestamp': timestamp,
            'training_metrics': training_metrics,
            'quantum_history': quantum_history,
            'config': {
                'data_dir': Config.DATA_DIR,
                'categories': Config.CATEGORIES,
                'img_size': Config.IMG_SIZE,
                'quantum_population': Config.QUANTUM_POPULATION,
                'quantum_generations': Config.QUANTUM_GENERATIONS,
            },
            'file_paths': {
                'ensemble': ensemble_path,
                'features': feature_path,
                'metadata': f"{base_path}_metadata.json"
            }
        }
        
        metadata_path = f"{base_path}_metadata.json"
        with open(metadata_path, 'w') as f:
            # Convert numpy types to native Python for JSON serialization
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(metadata, f, indent=2, default=convert)
        print(f"✓ Metadata saved to: {metadata_path}")
        
        # 4. Create a "latest" symlink for easy loading
        latest_info = {
            'latest_model_base': base_path,
            'timestamp': timestamp
        }
        latest_path = os.path.join(self.model_dir, 'latest_model.json')
        with open(latest_path, 'w') as f:
            json.dump(latest_info, f, indent=2)
        
        print(f"\n{'='*80}")
        print("MODEL PACKAGE SAVED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Base path: {base_path}")
        print(f"Files created:")
        print(f"  - {ensemble_path}")
        print(f"  - {feature_path}")
        print(f"  - {metadata_path}")
        print(f"{'='*80}\n")
        
        return base_path
    
    def load_complete_model(self, model_path=None):
        """
        Load complete trained model
        
        PARAMETERS:
        ───────────
        model_path: str (optional)
            If None, loads the latest trained model
            If provided, loads specific model
        
        RETURNS:
        ────────
        dict with keys:
            - 'ensemble': Trained EnsembleClassifier
            - 'selected_indices': Feature selection indices
            - 'scaler': Feature scaler
            - 'metadata': Training information
        """
        # If no path provided, load latest
        if model_path is None:
            latest_path = os.path.join(self.model_dir, 'latest_model.json')
            if not os.path.exists(latest_path):
                raise FileNotFoundError(
                    f"No trained model found in {self.model_dir}. "
                    "Please train a model first!"
                )
            with open(latest_path, 'r') as f:
                latest_info = json.load(f)
            model_path = latest_info['latest_model_base']
        
        print(f"\n{'='*80}")
        print("LOADING COMPLETE MODEL PACKAGE")
        print(f"{'='*80}\n")
        
        # Load ensemble
        ensemble_path = f"{model_path}_ensemble.pkl"
        ensemble = EnsembleClassifier.load(ensemble_path)
        
        # Load feature data
        feature_path = f"{model_path}_features.pkl"
        with open(feature_path, 'rb') as f:
            feature_data = pickle.load(f)
        print(f"✓ Feature selector loaded from: {feature_path}")
        
        # Load metadata
        metadata_path = f"{model_path}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Metadata loaded from: {metadata_path}")
        
        print(f"\n{'='*80}")
        print("MODEL LOADED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Model trained on: {metadata['timestamp']}")
        print(f"Test accuracy: {metadata['training_metrics']['test_accuracy']*100:.2f}%")
        print(f"Features: {feature_data['selected_feature_count']}/{feature_data['original_feature_count']}")
        print(f"{'='*80}\n")
        
        return {
            'ensemble': ensemble,
            'selected_indices': feature_data['selected_indices'],
            'scaler': feature_data['scaler'],
            'metadata': metadata
        }


# ═══════════════════════════════════════════════════════════════════════════
# 🔥 NEW: PREDICTION PIPELINE FOR NEW IMAGES
# ═══════════════════════════════════════════════════════════════════════════

class TBPredictor:
    """
    Production-ready predictor for new X-ray images
    
    USAGE:
    ──────
    predictor = TBPredictor()
    predictor.load_model()  # Loads latest trained model
    result = predictor.predict('patient_xray.png')
    """
    
    def __init__(self):
        self.model_manager = TBDetectorModelManager()
        self.segmentor = EntropyGuidedSegmentor()
        self.feature_fusion = HybridFeatureFusion()
        self.model_loaded = False
        
    def load_model(self, model_path=None):
        """Load trained model (latest by default)"""
        model_package = self.model_manager.load_complete_model(model_path)
        
        self.ensemble = model_package['ensemble']
        self.selected_indices = model_package['selected_indices']
        self.scaler = model_package['scaler']
        self.metadata = model_package['metadata']
        self.model_loaded = True
        
        print("✓ Predictor ready for inference")
    
    def predict(self, image_path, return_details=True):
        """
        Predict TB status for a new X-ray image
        
        PARAMETERS:
        ───────────
        image_path: str
            Path to chest X-ray image
        return_details: bool
            If True, returns detailed results including confidence scores
        
        RETURNS:
        ────────
        dict with keys:
            - 'prediction': 'Normal' or 'TB'
            - 'confidence': Probability (0-1)
            - 'probabilities': {'Normal': X, 'TB': Y}
            - 'risk_level': 'Low', 'Medium', or 'High'
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded! Call load_model() first.")
        
        print(f"\nProcessing: {image_path}")
        print("-" * 60)
        
        # 1. Segmentation
        print("Step 1/4: Segmenting lung regions...")
        segmented_img, mask, entropy_map = self.segmentor.segment_lung_roi(image_path)
        
        if segmented_img is None:
            raise ValueError(f"Failed to segment image: {image_path}")
        
        # 2. Feature extraction
        print("Step 2/4: Extracting hybrid features...")
        features = self.feature_fusion.extract_hybrid(segmented_img, mask)
        
        if features is None:
            raise ValueError(f"Failed to extract features from: {image_path}")
        
        # 3. Feature selection and normalization
        print("Step 3/4: Applying feature selection...")
        features_normalized = self.scaler.transform([features])
        features_selected = features_normalized[:, self.selected_indices]
        
        # 4. Prediction
        print("Step 4/4: Generating prediction...")
        prediction_idx = self.ensemble.predict(features_selected)[0]
        probabilities = self.ensemble.predict_proba(features_selected)[0]
        
        # Format results
        prediction_label = Config.CATEGORIES[prediction_idx]
        confidence = probabilities[prediction_idx]
        
        # Risk stratification
        if confidence > 0.9:
            risk_level = "High"
        elif confidence > 0.7:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        print("-" * 60)
        print(f"✓ Analysis complete")
        print(f"  Prediction: {prediction_label}")
        print(f"  Confidence: {confidence*100:.1f}%")
        print(f"  Risk Level: {risk_level}")
        print("-" * 60)
        
        if return_details:
            return {
                'prediction': prediction_label,
                'confidence': float(confidence),
                'probabilities': {
                    'Normal': float(probabilities[0]),
                    'TB': float(probabilities[1])
                },
                'risk_level': risk_level,
                'image_path': image_path
            }
        else:
            return prediction_label


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION (Same as before)
# ═══════════════════════════════════════════════════════════════════════════

class ResultVisualizer:
    """Generates publication-quality plots"""
    
    @staticmethod
    def plot_segmentation_pipeline(image_path, segmentor, save_path='segmentation_pipeline.png'):
        original = cv2.imread(image_path)
        original = cv2.resize(original, (Config.IMG_SIZE, Config.IMG_SIZE))
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        segmented, mask, entropy = segmentor.segment_lung_roi(image_path)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('A) Original X-Ray', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(original_gray, cmap='gray')
        axes[0, 1].set_title('B) Grayscale', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(entropy, cmap='jet')
        axes[0, 2].set_title('C) Entropy Map (Attention)', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('D) Binary Mask', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('E) Segmented ROI', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        overlay = original.copy()
        overlay[mask == 0] = overlay[mask == 0] * 0.3
        axes[1, 2].imshow(cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('F) Overlay Visualization', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.suptitle('Multi-Scale Attention-Gated Segmentation Pipeline', 
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Segmentation visualization saved to {save_path}")
    
    @staticmethod
    def plot_quantum_evolution(history, save_path='quantum_evolution.png'):
        df = pd.DataFrame(history)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(df['generation'], df['best_fitness'], 'b-o', linewidth=2, label='Best Fitness')
        axes[0].plot(df['generation'], df['mean_fitness'], 'r--', linewidth=2, label='Mean Fitness')
        axes[0].set_xlabel('Generation', fontsize=12)
        axes[0].set_ylabel('Fitness Score', fontsize=12)
        axes[0].set_title('Quantum Optimization Convergence', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(df['generation'], df['features_selected'], 'g-s', linewidth=2)
        axes[1].set_xlabel('Generation', fontsize=12)
        axes[1].set_ylabel('Number of Selected Features', fontsize=12)
        axes[1].set_title('Feature Space Reduction', fontsize=13, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Quantum evolution plot saved to {save_path}")
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion matrix saved to {save_path}")
    
    @staticmethod
    def plot_roc_curve(y_true, y_probs, classes, save_path='roc_curve.png'):
        plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(classes):
            y_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ ROC curve saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def train_model():
    """Complete training pipeline with model saving"""
    
    print("\n" + "="*80)
    print(" "*20 + "TB DETECTION SYSTEM - TRAINING MODE")
    print("="*80 + "\n")
    
    # Initialize components
    segmentor = EntropyGuidedSegmentor()
    feature_fusion = HybridFeatureFusion()
    model_manager = TBDetectorModelManager()
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: DATA LOADING & PREPROCESSING
    # ─────────────────────────────────────────────────────────────────────────
    
    print("PHASE 1: Data Loading and Preprocessing")
    print("-" * 80)
    
    X_features = []
    y_labels = []
    image_paths = []
    
    for label_idx, category in enumerate(Config.CATEGORIES):
        category_path = os.path.join(Config.DATA_DIR, category)
        
        if not os.path.exists(category_path):
            print(f"ERROR: Directory not found: {category_path}")
            continue
        
        print(f"\nProcessing class: {category}")
        images = [f for f in os.listdir(category_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Found {len(images)} images")
        
        for img_name in tqdm(images, desc=f"  Extracting features"):
            img_path = os.path.join(category_path, img_name)
            
            try:
                segmented_img, mask, entropy_map = segmentor.segment_lung_roi(img_path)
                if segmented_img is None or mask is None:
                    continue
                
                feature_vector = feature_fusion.extract_hybrid(segmented_img, mask)
                if feature_vector is not None:
                    X_features.append(feature_vector)
                    y_labels.append(label_idx)
                    image_paths.append(img_path)
                    
            except Exception as e:
                print(f"    Skipped {img_name}: {e}")
    
    X = np.array(X_features)
    y = np.array(y_labels)
    
    print(f"\n{'='*80}")
    print("FEATURE EXTRACTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples processed: {len(X)}")
    print(f"Feature vector dimension: {X.shape[1]}")
    print(f"Class distribution: Normal={np.sum(y==0)}, TB={np.sum(y==1)}")
    print(f"{'='*80}\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: QUANTUM OPTIMIZATION
    # ─────────────────────────────────────────────────────────────────────────
    
    print("PHASE 2: Quantum-Inspired Feature Selection")
    print("-" * 80)
    
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    quantum_selector = QuantumFeatureSelector(X_normalized, y)
    selected_indices, opt_history = quantum_selector.optimize()
    
    X_optimized = X_normalized[:, selected_indices]
    
    print(f"\nFeature reduction: {X.shape[1]} → {X_optimized.shape[1]}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3: MODEL TRAINING
    # ─────────────────────────────────────────────────────────────────────────
    
    print(f"\n{'='*80}")
    print("PHASE 3: Model Training with Cross-Validation")
    print(f"{'='*80}\n")
    
    skf = StratifiedKFold(n_splits=Config.CROSS_VAL_FOLDS, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_optimized, y), 1):
        print(f"Fold {fold}/{Config.CROSS_VAL_FOLDS}")
        X_train_fold = X_optimized[train_idx]
        X_val_fold = X_optimized[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        ensemble = EnsembleClassifier()
        ensemble.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
        
        val_preds = ensemble.predict(X_val_fold)
        val_acc = accuracy_score(y_val_fold, val_preds)
        cv_scores.append(val_acc)
        print(f"  Fold {fold} Accuracy: {val_acc*100:.2f}%\n")
    
    print(f"{'='*80}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Mean Accuracy: {np.mean(cv_scores)*100:.2f}% ± {np.std(cv_scores)*100:.2f}%")
    print(f"{'='*80}\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4: FINAL MODEL TRAINING
    # ─────────────────────────────────────────────────────────────────────────
    
    print("PHASE 4: Final Model Training")
    print("-" * 80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_optimized, y, test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_SEED, stratify=y
    )
    
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_train, y_train, test_size=0.2, 
        random_state=Config.RANDOM_SEED, stratify=y_train
    )
    
    final_ensemble = EnsembleClassifier()
    final_ensemble.train(X_train_sub, y_train_sub, X_val_sub, y_val_sub)
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 5: EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    
    print(f"\n{'='*80}")
    print("PHASE 5: Final Evaluation")
    print(f"{'='*80}\n")
    
    y_pred = final_ensemble.predict(X_test)
    y_probs = final_ensemble.predict_proba(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"FINAL TEST ACCURACY: {test_acc*100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=Config.CATEGORIES, digits=4))
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    
    training_metrics = {
        'test_accuracy': test_acc,
        'cv_mean_accuracy': np.mean(cv_scores),
        'cv_std_accuracy': np.std(cv_scores),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }
    
    # ─────────────────────────────────────────────────────────────────────────
    # 🔥 PHASE 6: SAVE COMPLETE MODEL
    # ─────────────────────────────────────────────────────────────────────────
    
    model_path = model_manager.save_complete_model(
        ensemble=final_ensemble,
        selected_indices=selected_indices,
        scaler=scaler,
        training_metrics=training_metrics,
        quantum_history=opt_history
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 7: VISUALIZATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    print(f"\n{'='*80}")
    print("PHASE 6: Generating Visualizations")
    print(f"{'='*80}\n")
    
    visualizer = ResultVisualizer()
    sample_tb_path = [p for p, l in zip(image_paths, y_labels) if l == 1][0]
    
    visualizer.plot_segmentation_pipeline(sample_tb_path, segmentor, 
                                          '/mnt/user-data/outputs/segmentation_pipeline.png')
    visualizer.plot_quantum_evolution(opt_history, 
                                      '/mnt/user-data/outputs/quantum_evolution.png')
    visualizer.plot_confusion_matrix(y_test, y_pred, Config.CATEGORIES,
                                    '/mnt/user-data/outputs/confusion_matrix.png')
    visualizer.plot_roc_curve(y_test, y_probs, Config.CATEGORIES,
                             '/mnt/user-data/outputs/roc_curve.png')
    
    # Save report
    report = f"""
{'='*80}
TB DETECTION SYSTEM - TRAINING REPORT
{'='*80}

MODEL SAVED AT: {model_path}

DATASET STATISTICS:
──────────────────
Total Images: {len(X)}
Normal Cases: {np.sum(y==0)}
TB Cases: {np.sum(y==1)}

FEATURE ENGINEERING:
────────────────────
Original Features: {X.shape[1]}
After Quantum Optimization: {X_optimized.shape[1]}
Reduction: {100*(1-X_optimized.shape[1]/X.shape[1]):.1f}%

CROSS-VALIDATION RESULTS:
─────────────────────────
{Config.CROSS_VAL_FOLDS}-Fold CV Accuracy: {np.mean(cv_scores)*100:.2f}% ± {np.std(cv_scores)*100:.2f}%

FINAL TEST SET PERFORMANCE:
───────────────────────────
Accuracy: {test_acc*100:.2f}%
Sensitivity: {sensitivity*100:.2f}%
Specificity: {specificity*100:.2f}%
Precision: {precision*100:.2f}%
F1-Score: {f1*100:.2f}%

CONFUSION MATRIX:
─────────────────
                Predicted
              Normal    TB
Actual Normal   {tn:4d}  {fp:4d}
       TB       {fn:4d}  {tp:4d}

TO USE THIS MODEL:
──────────────────
from tb_detection_complete import TBPredictor

predictor = TBPredictor()
predictor.load_model()  # Loads this trained model
result = predictor.predict('new_patient_xray.png')

{'='*80}
"""
    
    report_path = '/mnt/user-data/outputs/training_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"✓ Training report saved to {report_path}")
    
    print(f"\n{'='*80}")
    print(" "*25 + "TRAINING COMPLETE")
    print(f"{'='*80}\n")
    
    return model_path


# ═══════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

def predict_single_image(image_path):
    """Predict TB status for a single image"""
    predictor = TBPredictor()
    predictor.load_model()
    result = predictor.predict(image_path)
    
    print("\n" + "="*80)
    print(" "*25 + "PREDICTION RESULT")
    print("="*80)
    print(f"Image: {image_path}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"Risk Level: {result['risk_level']}")
    print("\nDetailed Probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob*100:.1f}%")
    print("="*80 + "\n")
    
    return result


def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(
        description='TB Detection System - Training and Prediction'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict'],
        default='train',
        help='Operation mode: train a new model or predict with existing model'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to X-ray image for prediction (required when mode=predict)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to specific model to load (optional, defaults to latest)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model()
    elif args.mode == 'predict':
        if args.image is None:
            print("ERROR: --image required when mode=predict")
            return
        predict_single_image(args.image)


if __name__ == "__main__":
    # If no command line args, default to training
    import sys
    if len(sys.argv) == 1:
        print("No arguments provided. Running in TRAINING mode.")
        print("To predict: python tb_detection_complete.py --mode predict --image <path>")
        print()
        train_model()
    else:
        main()