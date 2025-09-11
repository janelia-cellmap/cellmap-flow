"""
Visualization Module for DINOv3 Training

This module contains functions for:
- Training history plotting
- Pipeline results visualization
- Model performance analysis

Author: GitHub Copilot
Date: 2025-09-11
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch


def plot_training_history(history, save_path=None, show_plot=True):
    """
    Plot training and validation metrics over epochs.
    
    Parameters:
    -----------
    history : dict
        Training history dictionary with keys: 
        'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'epochs'
    save_path : str, optional
        Path to save the plot
    show_plot : bool, default=True
        Whether to display the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = history['epochs']
    
    # Plot training and validation loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot training and validation accuracy
    ax2.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot learning rate if available
    if 'learning_rates' in history:
        ax3.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Learning Rate\nNot Tracked', 
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=12, fontweight='bold')
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    
    # Plot loss difference (overfitting indicator)
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
    ax4.plot(epochs, loss_diff, 'purple', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('Validation - Training Loss\n(Overfitting Indicator)', 
                  fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Difference')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, show_plot=True):
    """
    Plot confusion matrix for classification results.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        Names of the classes
    save_path : str, optional
        Path to save the plot
    show_plot : bool, default=True
        Whether to display the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add accuracy information
    accuracy = np.diag(cm).sum() / cm.sum()
    plt.text(0.02, 0.98, f'Accuracy: {accuracy:.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_roc_curves(y_true, y_proba, class_names=None, save_path=None, show_plot=True):
    """
    Plot ROC curves for multi-class classification.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities for each class
    class_names : list, optional
        Names of the classes
    save_path : str, optional
        Path to save the plot
    show_plot : bool, default=True
        Whether to display the plot
    """
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    n_classes = y_true_bin.shape[1]
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    plt.figure(figsize=(10, 8))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multi-Class Classification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_pipeline_results(model, sample_features, sample_labels, sample_images, 
                             predictions, probabilities, save_dir=None):
    """
    Visualize sample predictions from the pipeline.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model
    sample_features : numpy.ndarray
        Sample feature vectors
    sample_labels : numpy.ndarray
        True labels for samples
    sample_images : numpy.ndarray
        Original images for visualization
    predictions : numpy.ndarray
        Model predictions
    probabilities : numpy.ndarray
        Prediction probabilities
    save_dir : str, optional
        Directory to save visualizations
    """
    n_samples = min(12, len(sample_images))
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(n_samples):
        ax = axes[i]
        
        # Display image
        if sample_images[i].ndim == 3:
            # If 3D, take middle slice
            img = sample_images[i, sample_images[i].shape[0]//2, :, :]
        else:
            img = sample_images[i]
        
        ax.imshow(img, cmap='gray')
        
        # Create title with prediction info
        true_label = sample_labels[i]
        pred_label = predictions[i]
        confidence = probabilities[i, pred_label]
        
        color = 'green' if true_label == pred_label else 'red'
        title = f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.3f}'
        
        ax.set_title(title, color=color, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path = f"{save_dir}/sample_predictions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions saved to: {save_path}")
    
    plt.show()


def plot_feature_distributions(features, labels, save_path=None, show_plot=True):
    """
    Plot feature value distributions by class.
    
    Parameters:
    -----------
    features : numpy.ndarray
        Feature array of shape (n_samples, n_features)
    labels : numpy.ndarray
        Label array
    save_path : str, optional
        Path to save the plot
    show_plot : bool, default=True
        Whether to display the plot
    """
    # Calculate statistics for each class
    unique_classes = np.unique(labels)
    n_features = features.shape[1]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot 1: Feature means by class
    ax = axes[0]
    feature_means = []
    for cls in unique_classes:
        cls_mask = labels == cls
        cls_features = features[cls_mask]
        feature_means.append(np.mean(cls_features, axis=0))
    
    feature_means = np.array(feature_means)
    
    x = np.arange(min(50, n_features))  # Show first 50 features
    width = 0.35
    
    for i, cls in enumerate(unique_classes):
        ax.bar(x + i*width, feature_means[i][:len(x)], width, 
               label=f'Class {cls}', alpha=0.7)
    
    ax.set_title('Feature Means by Class', fontweight='bold')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Mean Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Feature standard deviations
    ax = axes[1]
    feature_stds = []
    for cls in unique_classes:
        cls_mask = labels == cls
        cls_features = features[cls_mask]
        feature_stds.append(np.std(cls_features, axis=0))
    
    feature_stds = np.array(feature_stds)
    
    for i, cls in enumerate(unique_classes):
        ax.bar(x + i*width, feature_stds[i][:len(x)], width, 
               label=f'Class {cls}', alpha=0.7)
    
    ax.set_title('Feature Standard Deviations by Class', fontweight='bold')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Standard Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Overall feature distribution
    ax = axes[2]
    feature_stats = {
        'mean': np.mean(features, axis=0),
        'std': np.std(features, axis=0),
        'min': np.min(features, axis=0),
        'max': np.max(features, axis=0)
    }
    
    ax.hist(feature_stats['mean'][:100], bins=30, alpha=0.7, label='Feature Means')
    ax.set_title('Distribution of Feature Means', fontweight='bold')
    ax.set_xlabel('Feature Mean Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Class distribution
    ax = axes[3]
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    colors = plt.cm.Set3(np.arange(len(unique_labels)))
    
    bars = ax.bar(unique_labels, label_counts, color=colors, alpha=0.7)
    ax.set_title('Class Distribution', fontweight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    
    # Add count labels on bars
    for bar, count in zip(bars, label_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count}', ha='center', va='bottom')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature distributions saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_comprehensive_report(model, history, evaluation_results, save_dir=None):
    """
    Create a comprehensive visualization report.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model
    history : dict
        Training history
    evaluation_results : dict
        Evaluation results from evaluate_model()
    save_dir : str, optional
        Directory to save all plots
    """
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE VISUALIZATION REPORT")
    print("="*60)
    
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    # 1. Training history
    plot_training_history(
        history, 
        save_path=f"{save_dir}/training_history.png" if save_dir else None,
        show_plot=True
    )
    
    # 2. Confusion matrix
    plot_confusion_matrix(
        evaluation_results['true_labels'],
        evaluation_results['predictions'],
        save_path=f"{save_dir}/confusion_matrix.png" if save_dir else None,
        show_plot=True
    )
    
    # 3. ROC curves (if multi-class)
    if evaluation_results['probabilities'].shape[1] > 2:
        plot_roc_curves(
            evaluation_results['true_labels'],
            evaluation_results['probabilities'],
            save_path=f"{save_dir}/roc_curves.png" if save_dir else None,
            show_plot=True
        )
    
    print(f"\nFinal Model Accuracy: {evaluation_results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(evaluation_results['classification_report'])
    
    if save_dir:
        print(f"\nAll visualizations saved to: {save_dir}")
    
    print("="*60)
