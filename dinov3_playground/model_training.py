"""
Model Training Module for DINOv3 Classification

This module contains functions for:
- Class balancing and data sampling
- Model training with validation
- Training history tracking

Author: GitHub Copilot
Date: 2025-09-11
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
import time


def balance_classes(features, labels, method='hybrid', random_state=42):
    """
    Balance classes using various methods to handle class imbalance.
    
    Parameters:
    -----------
    features : numpy.ndarray
        Feature array of shape (n_samples, n_features)
    labels : numpy.ndarray
        Label array of shape (n_samples,)
    method : str, default='hybrid'
        Balancing method: 'undersample', 'oversample', 'weighted', or 'hybrid'
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple: (balanced_features, balanced_labels, class_weights)
        class_weights is None unless method='weighted'
    """
    np.random.seed(random_state)
    
    # Get class distribution
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    print(f"\nOriginal class distribution:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    if method == 'undersample':
        # Undersample majority class
        min_count = min(class_counts)
        balanced_indices = []
        
        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            selected_indices = np.random.choice(cls_indices, min_count, replace=False)
            balanced_indices.extend(selected_indices)
        
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        return features[balanced_indices], labels[balanced_indices], None
    
    elif method == 'oversample':
        # Oversample minority classes
        max_count = max(class_counts)
        balanced_features = []
        balanced_labels = []
        
        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            cls_features = features[cls_indices]
            cls_labels = labels[cls_indices]
            
            # Sample with replacement to reach max_count
            oversample_indices = np.random.choice(len(cls_indices), max_count, replace=True)
            balanced_features.append(cls_features[oversample_indices])
            balanced_labels.append(cls_labels[oversample_indices])
        
        balanced_features = np.vstack(balanced_features)
        balanced_labels = np.hstack(balanced_labels)
        
        # Shuffle the balanced dataset
        shuffle_indices = np.random.permutation(len(balanced_labels))
        return balanced_features[shuffle_indices], balanced_labels[shuffle_indices], None
    
    elif method == 'weighted':
        # Use class weights without changing data
        class_weights = compute_class_weight(
            'balanced', 
            classes=unique_classes, 
            y=labels
        )
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        print(f"Computed class weights: {class_weight_dict}")
        return features, labels, class_weight_dict
    
    elif method == 'hybrid':
        # Combine moderate oversampling with class weights
        target_count = int(np.mean(class_counts) * 1.5)  # 1.5x average
        balanced_features = []
        balanced_labels = []
        
        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            cls_features = features[cls_indices]
            cls_labels = labels[cls_indices]
            
            current_count = len(cls_indices)
            if current_count < target_count:
                # Oversample to target count
                oversample_indices = np.random.choice(len(cls_indices), target_count, replace=True)
                balanced_features.append(cls_features[oversample_indices])
                balanced_labels.append(cls_labels[oversample_indices])
            else:
                # Keep all samples if already above target
                balanced_features.append(cls_features)
                balanced_labels.append(cls_labels)
        
        balanced_features = np.vstack(balanced_features)
        balanced_labels = np.hstack(balanced_labels)
        
        # Shuffle and compute class weights
        shuffle_indices = np.random.permutation(len(balanced_labels))
        balanced_features = balanced_features[shuffle_indices]
        balanced_labels = balanced_labels[shuffle_indices]
        
        # Compute class weights for the balanced dataset
        unique_balanced, balanced_counts = np.unique(balanced_labels, return_counts=True)
        class_weights = compute_class_weight(
            'balanced', 
            classes=unique_balanced, 
            y=balanced_labels
        )
        class_weight_dict = dict(zip(unique_balanced, class_weights))
        
        return balanced_features, balanced_labels, class_weight_dict
    
    else:
        raise ValueError(f"Unknown balancing method: {method}")


def train_classifier_with_validation(model, train_features, train_labels, 
                                   val_features, val_labels, device,
                                   epochs=100, batch_size=512, learning_rate=0.001,
                                   class_weights=None, patience=10, min_delta=0.001):
    """
    Train classifier with validation monitoring and early stopping.
    
    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model to train
    train_features : numpy.ndarray
        Training features
    train_labels : numpy.ndarray
        Training labels
    val_features : numpy.ndarray
        Validation features
    val_labels : numpy.ndarray
        Validation labels
    device : torch.device
        Device to train on
    epochs : int, default=100
        Maximum number of epochs
    batch_size : int, default=512
        Batch size for training
    learning_rate : float, default=0.001
        Learning rate
    class_weights : dict, optional
        Class weights for loss function
    patience : int, default=10
        Early stopping patience
    min_delta : float, default=0.001
        Minimum improvement for early stopping
        
    Returns:
    --------
    tuple: (trained_model, training_history)
    """
    # Move model to device
    model = model.to(device)
    
    # Setup loss function with class weights
    if class_weights is not None:
        # Convert class weights to tensor
        weight_tensor = torch.zeros(len(class_weights))
        for cls, weight in class_weights.items():
            weight_tensor[cls] = weight
        weight_tensor = weight_tensor.to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print(f"Using weighted loss with weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using unweighted loss")
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Convert data to tensors
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32).to(device)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long).to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'epochs': [],
        'learning_rates': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\nStarting training...")
    print(f"Training samples: {len(train_features)}")
    print(f"Validation samples: {len(val_features)}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Create batches
        num_batches = (len(train_features) + batch_size - 1) // batch_size
        indices = torch.randperm(len(train_features))
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(train_features))
            batch_indices = indices[start_idx:end_idx]
            
            batch_features = train_features_tensor[batch_indices].to(device)
            batch_labels = train_labels_tensor[batch_indices].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            # Process validation in batches to avoid memory issues
            val_num_batches = (len(val_features) + batch_size - 1) // batch_size
            
            for batch_idx in range(val_num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(val_features))
                
                batch_val_features = val_features_tensor[start_idx:end_idx]
                batch_val_labels = val_labels_tensor[start_idx:end_idx]
                
                outputs = model(batch_val_features)
                loss = criterion(outputs, batch_val_labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_val_labels.size(0)
                val_correct += (predicted == batch_val_labels).sum().item()
        
        # Calculate averages
        avg_train_loss = train_loss / num_batches
        avg_val_loss = val_loss / val_num_batches
        train_accuracy = 100.0 * train_correct / train_total
        val_accuracy = 100.0 * val_correct / val_total
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['epochs'].append(epoch + 1)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        if epoch % 10 == 0 or epoch < 10:
            elapsed_time = time.time() - start_time
            print(f"Epoch [{epoch+1:3d}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} "
                  f"Train Acc: {train_accuracy:.2f}% "
                  f"Val Loss: {avg_val_loss:.4f} "
                  f"Val Acc: {val_accuracy:.2f}% "
                  f"Time: {elapsed_time:.1f}s")
        
        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Restored best model from early stopping")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_outputs = model(train_features_tensor.to(device))
        val_outputs = model(val_features_tensor)
        
        _, train_pred = torch.max(train_outputs, 1)
        _, val_pred = torch.max(val_outputs, 1)
        
        final_train_acc = accuracy_score(train_labels, train_pred.cpu().numpy())
        final_val_acc = accuracy_score(val_labels, val_pred.cpu().numpy())
        
        print(f"\nFinal Results:")
        print(f"Training Accuracy: {final_train_acc:.4f}")
        print(f"Validation Accuracy: {final_val_acc:.4f}")
        
        # Detailed classification report for validation set
        print(f"\nValidation Classification Report:")
        print(classification_report(val_labels, val_pred.cpu().numpy()))
    
    return model, history


def evaluate_model(model, features, labels, device, batch_size=512):
    """
    Evaluate model on given dataset.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model
    features : numpy.ndarray
        Feature array
    labels : numpy.ndarray
        Label array
    device : torch.device
        Device to evaluate on
    batch_size : int, default=512
        Batch size for evaluation
        
    Returns:
    --------
    dict: Evaluation results including accuracy and predictions
    """
    model.eval()
    model = model.to(device)
    
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        num_batches = (len(features) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(features))
            
            batch_features = features_tensor[start_idx:end_idx].to(device)
            outputs = model(batch_features)
            
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    accuracy = accuracy_score(labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probabilities),
        'true_labels': labels,
        'classification_report': classification_report(labels, all_predictions)
    }
