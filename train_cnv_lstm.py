# -*- coding: utf-8 -*-
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch import cuda
import sys, os
import random
import numpy as np
from sklearn import metrics
import gc
import time
import logging
from collections import OrderedDict
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Command-line arguments
parser = argparse.ArgumentParser(description='CNV-LSTM')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate')
parser.add_argument('--model_name', type=str, default='CNV_BiLSTM', help='Model name') 
parser.add_argument('--clip', type=float, default=1, help='Gradient clipping threshold')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=192, help='Batch size')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
parser.add_argument('--save_root', type=str, default='/public/home/wangruiqi/bam/new/model_output', help='Output directory')
parser.add_argument('--data_path', type=str, default='/public/home/wangruiqi/bam/new/train_data_sample.npz', help='Feature data path')
parser.add_argument('--gpuid', type=int, default=0, help='CUDA GPU ID')
parser.add_argument('--n_features', type=int, default=10, help='Number of features')
parser.add_argument('--seq_length', type=int, default=50, help='Sequence length')
parser.add_argument('--stride', type=int, default=10, help='Sliding-window stride')
parser.add_argument('--bin_rnn_size', type=int, default=128, help='Bin RNN hidden size')
parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers')
parser.add_argument('--bidirectional', type=bool, default=True, help='Use bidirectional LSTM')
parser.add_argument('--test_size', type=float, default=0.2, help='Test set fraction')
parser.add_argument('--val_size', type=float, default=0.1, help='Validation set fraction')
parser.add_argument('--num_workers', type=int, default=0, help='Number of data-loading worker processes')
args = parser.parse_args()

# Set random seeds.
torch.manual_seed(2023)
np.random.seed(2023)
random.seed(2023)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2023)

# Configure logging.
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CNV-LSTM")

if not os.path.exists(args.save_root):
    os.makedirs(args.save_root)

# Dataset class
class CNVSequenceDataset(Dataset):
    def __init__(self, features, labels, seq_length=50, augment=False):
        self.features = features
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx].copy()
        label = int(self.labels[idx])
        # Apply on-the-fly augmentation only to training data; do not copy a window into other subsets.
        if self.augment and label in (1, 2) and np.random.random() < 0.5:
            scale = np.random.choice([0.03, 0.05, 0.07]) if label == 1 else 0.04
            x += np.random.normal(0, scale, x.shape).astype(np.float32)
        return {'features': torch.from_numpy(x),
                'label': torch.tensor([label], dtype=torch.long)}


def create_sequences(X, y, chrom_col='chrom', bin_start_col='bin_start',
                     seq_length=50, stride=10):
    columns = [c for c in X.columns if c not in ['chrom', 'bin_start', 'sample_id']]
    sequences, targets = [], []
    for chrom in X[chrom_col].unique():
        indices = np.flatnonzero(X[chrom_col].to_numpy() == chrom)
        indices = indices[np.argsort(X.iloc[indices][bin_start_col].to_numpy(), kind='stable')]
        positions = X.iloc[indices][bin_start_col].to_numpy(dtype=np.int64)
        values = X.iloc[indices][columns].to_numpy(dtype=np.float32)
        labels = y[indices]
        # Split at duplicate coordinates, gaps or discontinuities; 100 is the bin size for this dataset.
        boundaries = np.r_[0, np.flatnonzero(np.diff(positions) != 100) + 1, len(indices)]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            for offset in range(int(left), int(right) - seq_length + 1, stride):
                sequences.append(values[offset:offset + seq_length])
                targets.append(labels[offset + seq_length // 2])
    if not sequences:
        raise ValueError('No continuous interval is long enough to create sequences; check coordinates or reduce seq_length')
    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.int64)


def load_and_prepare_data(data_path, seq_length=50, stride=10):
    if seq_length < 1 or stride < 1:
        raise ValueError('seq_length and stride must be positive integers')
    if args.test_size <= 0 or args.val_size <= 0 or args.test_size + args.val_size >= 1:
        raise ValueError('Validation and test fractions must be positive and sum to less than 1')
    with np.load(data_path, allow_pickle=True) as data:
        X = pd.DataFrame(data['features'], columns=data['feature_names'])
        y = np.asarray(data['labels'], dtype=np.int64)
    if len(X) != len(y) or not np.isin(y, [0, 1, 2]).all():
        raise ValueError('Invalid label count or label classes')
    X['chrom'] = X['chrom'].astype(str)
    X['bin_start'] = pd.to_numeric(X['bin_start'], errors='raise').astype(np.int64)
    numerical_cols = [c for c in X.columns if c not in ['chrom', 'bin_start', 'sample_id']]
    X[numerical_cols] = X[numerical_cols].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(X[numerical_cols].to_numpy(dtype=float)).all():
        raise ValueError('Numeric features contain NaN or infinite values; check the data')
    chroms = np.array(sorted(X['chrom'].unique()))
    np.random.default_rng(42).shuffle(chroms)
    n_test = max(1, math.ceil(len(chroms) * args.test_size))
    n_val = max(1, math.ceil(len(chroms) * args.val_size))
    if n_test + n_val >= len(chroms):
        raise ValueError('Not enough chromosomes to create three subsets')
    split = {'test': chroms[:n_test].tolist(),
             'val': chroms[n_test:n_test+n_val].tolist(),
             'train': chroms[n_test+n_val:].tolist()}
    scaler = StandardScaler()
    scaler.fit(X.loc[X['chrom'].isin(split['train']), numerical_cols])
    datasets = {}
    for name in ['train', 'val', 'test']:
        mask = X['chrom'].isin(split[name]).to_numpy()
        subset = X.loc[mask].copy().reset_index(drop=True)
        subset[numerical_cols] = scaler.transform(subset[numerical_cols])
        seq_x, seq_y = create_sequences(subset, y[mask], seq_length=seq_length, stride=stride)
        counts = np.bincount(seq_y, minlength=3)
        logger.info('%s chromosomes=%s, sequences=%d, class counts=%s', name, split[name], len(seq_y), counts.tolist())
        if name == 'train' and (counts == 0).any():
            raise ValueError('The training set is missing a class; adjust the chromosome split')
        datasets[name] = CNVSequenceDataset(seq_x, seq_y, augment=(name == 'train'))
    with open(os.path.join(args.save_root, 'data_split.json'), 'w', encoding='utf-8') as f:
        json.dump({'chromosomes': split, 'features': numerical_cols,
                   'bin_size': 100, 'seq_length': seq_length, 'stride': stride}, f, indent=2)
    options = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                   pin_memory=torch.cuda.is_available())
    if args.num_workers > 0:
        options['prefetch_factor'] = 2
    loaders = [DataLoader(datasets[name], shuffle=(name == 'train'), **options)
               for name in ['train', 'val', 'test']]
    return *loaders, scaler, len(numerical_cols)

# Attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size, level_name="bin"):
        super(Attention, self).__init__()
        self.level_name = level_name
        self.context_vector = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.softmax = nn.Softmax(dim=1)
        # Initialize parameters.
        nn.init.uniform_(self.context_vector, -0.1, 0.1)
        
    def forward(self, x):
        # x: [batch, seq_len, hidden]
        # Calculate attention scores.
        attn_scores = torch.matmul(x, self.context_vector)  # [batch, seq_len, 1]
        attn_weights = self.softmax(attn_scores.squeeze(2))  # [batch, seq_len]
        
        # Apply attention weights.
        weighted_output = torch.bmm(attn_weights.unsqueeze(1), x)  # [batch, 1, hidden]
        return weighted_output.squeeze(1), attn_weights

# Feature-level encoder
class FeatureEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True, dropout=0.5):
        super(FeatureEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = Attention(hidden_size * self.num_directions, "bin")
        
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        outputs, _ = self.lstm(x)
        context, attention = self.attention(outputs)
        
        # Add global average pooling as a residual connection.
        global_avg = torch.mean(outputs, dim=1)
        context = context + global_avg
        
        return context, attention

# Focal loss class
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# CNV detection model
class CNVDetector(nn.Module):
    def __init__(self, args):
        super(CNVDetector, self).__init__()
        self.n_features = args.n_features
        self.seq_length = args.seq_length
        self.hidden_size = args.bin_rnn_size
        self.num_directions = 2 if args.bidirectional else 1
        
        # Feature encoders
        self.feature_encoders = nn.ModuleList()
        for i in range(self.n_features):
            self.feature_encoders.append(
                FeatureEncoder(
                    input_size=1,
                    hidden_size=self.hidden_size,
                    num_layers=args.num_layers,
                    bidirectional=args.bidirectional,
                    dropout=args.dropout
                )
            )
        
        # Feature fusion encoder
        feature_hidden_size = self.hidden_size * self.num_directions
        self.feature_fusion = nn.LSTM(
            input_size=feature_hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=args.bidirectional
        )
        
        self.feature_attention = Attention(self.hidden_size * self.num_directions, "feature")
        
        # Classifier
        classifier_input_size = self.hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(args.dropout/2),
            nn.Linear(64, 3)  # Three classes: Normal, DEL, DUP
        )
        
    def forward(self, x):
        # x: [batch, seq_len, n_features]
        batch_size = x.size(0)
        actual_features = x.size(2)  # Get the actual number of features.
        
        # Process each feature.
        feature_outputs = []
        bin_attentions = []
        
        # Process only the available features.
        for i in range(min(self.n_features, actual_features)):
            # Extract one feature.
            feature = x[:, :, i:i+1]  # [batch, seq_len, 1]
            
            # Pass through the feature encoder.
            feature_output, bin_attn = self.feature_encoders[i](feature)
            feature_outputs.append(feature_output.unsqueeze(1))  # [batch, 1, hidden]
            bin_attentions.append(bin_attn)
        
        # Combine feature representations.
        feature_repr = torch.cat(feature_outputs, dim=1)  # [batch, n_features, hidden]
        
        # Add feature interactions.
        feature_interact = torch.bmm(
            feature_repr.transpose(1, 2),  # [batch, hidden, n_features]
            feature_repr  # [batch, n_features, hidden]
        )  # [batch, hidden, hidden]
        feature_interact = F.relu(feature_interact)
        feature_interact = torch.bmm(feature_repr, feature_interact)  # [batch, n_features, hidden]
        feature_repr = feature_repr + 0.15 * feature_interact  # Residual connection
        
        # Feature fusion
        fused_output, _ = self.feature_fusion(feature_repr)
        context_vector, feature_attn = self.feature_attention(fused_output)
        
        # Classification
        logits = self.classifier(context_vector)
        
        bin_attentions_tensor = torch.stack(bin_attentions, dim=1)  # [batch, n_features, seq_len]
        
        return logits, feature_attn, bin_attentions_tensor

# Training function
def train(model, train_loader, optimizer, criterion, device, clip=1.0):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, batch in enumerate(train_loader):
        features = batch['features'].to(device)
        labels = batch['label'].squeeze(1).to(device)
        
        optimizer.zero_grad()
        outputs, feature_attn, bin_attn = model(features)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Clip gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions.
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 100 == 0:
            logger.info(f'Train Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
    
    # Calculate average loss and metrics.
    avg_loss = total_loss / len(train_loader)
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    f1 = metrics.f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, f1, all_preds, all_labels

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            labels = batch['label'].squeeze(1).to(device)
            
            outputs, feature_attn, bin_attn = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Collect predictions.
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average loss and metrics.
    avg_loss = total_loss / len(val_loader)
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    f1 = metrics.f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = metrics.confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    
    return avg_loss, accuracy, f1, conf_matrix, all_preds, all_labels

# Test function
def test(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_feature_attn = []
    all_bin_attn = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            labels = batch['label'].squeeze(1).to(device)
            
            outputs, feature_attn, bin_attn = model(features)
            
            # Collect predictions and attention weights.
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_feature_attn.append(feature_attn.cpu().numpy())
            all_bin_attn.append(bin_attn.cpu().numpy())
    
    # Calculate metrics.
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    precision = metrics.precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = metrics.recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = metrics.f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = metrics.confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    
    # Calculate per-class precision and recall.
    per_class_precision = metrics.precision_score(all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    per_class_recall = metrics.recall_score(all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    
    # Calculate per-class F1 scores.
    per_class_f1 = metrics.f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    
    # Concatenate attention weights.
    all_feature_attn = np.concatenate(all_feature_attn, axis=0)
    all_bin_attn = np.concatenate(all_bin_attn, axis=0)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'predictions': all_preds,
        'labels': all_labels,
        'feature_attention': all_feature_attn,
        'bin_attention': all_bin_attn
    }
    
    return results

# Main function
def main():
    # Check GPU availability.
    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Log model parameters.
    logger.info("======= Model parameters =======")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Sequence length: {args.seq_length}")
    logger.info(f"LSTM hidden size: {args.bin_rnn_size}")
    logger.info(f"Number of LSTM layers: {args.num_layers}")
    logger.info(f"Use bidirectional LSTM: {args.bidirectional}")
    logger.info(f"Dropout probability: {args.dropout}")
    logger.info(f"Number of data-loading workers: {args.num_workers}")
    logger.info("========================")
    
    # Load data and determine the actual number of features.
    train_loader, val_loader, test_loader, scaler, actual_features = load_and_prepare_data(
        args.data_path, args.seq_length, args.stride)
    
    # Update the number of model features.
    args.n_features = actual_features
    logger.info(f"Using the detected number of features: {args.n_features}")
    
    # Count training samples in each class.
    train_labels = []
    for batch in train_loader:
        labels = batch['label'].squeeze(1).numpy()
        train_labels.extend(labels)
    train_labels = np.array(train_labels)

    class_counts = [np.sum(train_labels==i) for i in range(3)]
    logger.info("======= Training data distribution =======")
    logger.info(f"Class 0 (Normal): {class_counts[0]} samples, fraction: {class_counts[0]/len(train_labels):.2%}")
    logger.info(f"Class 1 (DEL): {class_counts[1]} samples, fraction: {class_counts[1]/len(train_labels):.2%}")
    logger.info(f"Class 2 (DUP): {class_counts[2]} samples, fraction: {class_counts[2]/len(train_labels):.2%}")
    logger.info(f"Total samples: {len(train_labels)}")
    logger.info("===========================")
    
    # Create the model.
    model = CNVDetector(args).to(device)
    logger.info(f"Model architecture:\n{model}")
    
    # Set class weights (retain the original weights).
    weights = torch.FloatTensor([1.0, 3.0, 1.5]).to(device)

    # Use weighted focal loss.
    criterion = FocalLoss(gamma=2.0, weight=weights)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Use a learning-rate scheduler.
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Train the model.
    best_val_f1 = -float("inf")
    best_epoch = 0
    patience = 10  # Early stopping
    patience_counter = 0
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Training
        train_loss, train_acc, train_f1, _, _ = train(
            model, train_loader, optimizer, criterion, device, args.clip)
        
        # Validation
        val_loss, val_acc, val_f1, val_conf_matrix, _, _ = validate(
            model, val_loader, criterion, device)
        
        logger.info(f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        logger.info(f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        logger.info(f"Val Confusion Matrix:\n{val_conf_matrix}")
        
        # Save the best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            
            # Save the model.
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'scaler': scaler
            }, os.path.join(args.save_root, f"{args.model_name}_best.pth"))
            
            logger.info(f"Save the best model, F1: {val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping: no improvement for {patience} epochs.")
                break
        
        # Update the scheduler after validation.
        scheduler.step(val_f1)
    
    logger.info(f"Training completed! Best validation F1: {best_val_f1:.4f} (Epoch {best_epoch+1})")
    
    # Load the best model for testing.
    checkpoint = torch.load(os.path.join(args.save_root, f"{args.model_name}_best.pth"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on the test set.
    test_results = test(model, test_loader, device)
    
    logger.info("Test set results:")
    logger.info(f"Overall accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"Overall weighted precision: {test_results['precision']:.4f}")
    logger.info(f"Overall weighted recall: {test_results['recall']:.4f}")
    logger.info(f"Overall weighted F1: {test_results['f1']:.4f}")
    
    logger.info("\nPer-class performance metrics:")
    class_names = ['Normal', 'DEL', 'DUP']
    for i, class_name in enumerate(class_names):
        logger.info(f"{class_name}class - precision: {test_results['per_class_precision'][i]:.4f}, " +
                   f"recall: {test_results['per_class_recall'][i]:.4f}, " +
                   f"F1 score: {test_results['per_class_f1'][i]:.4f}")
    
    logger.info(f"\nConfusion matrix:\n{test_results['confusion_matrix']}")
    
    # Calculate and log detailed analysis from the confusion matrix.
    conf_mat = test_results['confusion_matrix']
    logger.info("\nDetailed confusion-matrix analysis:")
    for i, class_name in enumerate(class_names):
        true_positive = conf_mat[i, i]
        false_negative = np.sum(conf_mat[i, :]) - true_positive
        false_positive = np.sum(conf_mat[:, i]) - true_positive
        
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        
        logger.info(f"{class_name}class - true positives: {true_positive}, false negatives: {false_negative}, false positives: {false_positive}")
        logger.info(f"{class_name}class - precision: {precision:.4f}, recall: {recall:.4f}")
        
        # Log misclassifications into other classes.
        for j, other_class in enumerate(class_names):
            if i != j:
                logger.info(f"  {class_name} misclassified as {other_class}: {conf_mat[i, j]} ({conf_mat[i, j]/max(np.sum(conf_mat[i, :]), 1):.2%})")
    
    # Save test results.
    np.savez(os.path.join(args.save_root, f"{args.model_name}_test_results.npz"),
             predictions=test_results['predictions'],
             labels=test_results['labels'],
             feature_attention=test_results['feature_attention'],
             bin_attention=test_results['bin_attention'],
             confusion_matrix=test_results['confusion_matrix'])
    
    # Save the confusion matrix plot.
    plt.figure(figsize=(10, 8))
    conf_mat = test_results['confusion_matrix']
    classes = ['Normal', 'DEL', 'DUP']
    
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = conf_mat.max() / 2.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, format(conf_mat[i, j], fmt),
                     ha="center", va="center",
                     color="white" if conf_mat[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_root, f"{args.model_name}_confusion_matrix.png"))
    
    # Save the feature importance plot.
    feature_importance = np.mean(test_results['feature_attention'], axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Attention Weight')
    plt.savefig(os.path.join(args.save_root, f"{args.model_name}_feature_importance.png"))
    
    logger.info(f"All results saved to {args.save_root}")

if __name__ == "__main__":
    main()
