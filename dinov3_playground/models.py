"""
Model Classes for DINOv3 Classification

This module contains the neural network model classes:
- ImprovedClassifier: Enhanced classifier with batch normalization and dropout
- SimpleClassifier: Basic classifier for comparison

Author: GitHub Copilot
Date: 2025-09-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedClassifier(nn.Module):
    """
    Enhanced neural network classifier with batch normalization and dropout.
    
    Features:
    - Multiple hidden layers with configurable sizes
    - Batch normalization for training stability
    - Dropout for regularization
    - ReLU activation with optional LeakyReLU
    - Configurable architecture
    """
    
    def __init__(self, input_dim=384, hidden_dims=[256, 128, 64], num_classes=2, 
                 dropout_rate=0.3, use_batch_norm=True, activation='relu'):
        """
        Initialize the ImprovedClassifier.
        
        Parameters:
        -----------
        input_dim : int, default=384
            Input feature dimension (DINOv3 ViT-S/16 = 384)
        hidden_dims : list, default=[256, 128, 64]
            List of hidden layer dimensions
        num_classes : int, default=2
            Number of output classes
        dropout_rate : float, default=0.3
            Dropout probability
        use_batch_norm : bool, default=True
            Whether to use batch normalization
        activation : str, default='relu'
            Activation function ('relu' or 'leaky_relu')
        """
        super(ImprovedClassifier, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        
        # Build layers dynamically
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            
            # Activation
            if activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.ReLU())
            
            # Dropout
            layers.append(nn.Dropout(dropout_rate))
        
        # Final output layer
        layers.append(nn.Linear(dims[-1], num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == 'relu':
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input features of shape (batch_size, input_dim)
            
        Returns:
        --------
        torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.classifier(x)
    
    def get_feature_maps(self, x):
        """
        Get intermediate feature representations.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input features
            
        Returns:
        --------
        list: Feature maps from each layer
        """
        features = []
        current_x = x
        
        for i, layer in enumerate(self.classifier):
            current_x = layer(current_x)
            if isinstance(layer, (nn.ReLU, nn.LeakyReLU)):
                features.append(current_x.clone())
        
        return features


class SimpleClassifier(nn.Module):
    """
    Simple neural network classifier for baseline comparison.
    
    Features:
    - Single hidden layer
    - Basic ReLU activation
    - Minimal regularization
    """
    
    def __init__(self, input_dim=384, hidden_dim=128, num_classes=2):
        """
        Initialize the SimpleClassifier.
        
        Parameters:
        -----------
        input_dim : int, default=384
            Input feature dimension
        hidden_dim : int, default=128
            Hidden layer dimension
        num_classes : int, default=2
            Number of output classes
        """
        super(SimpleClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input features of shape (batch_size, input_dim)
            
        Returns:
        --------
        torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.classifier(x)


def create_model(model_type='improved', input_dim=384, num_classes=2, **kwargs):
    """
    Factory function to create model instances.
    
    Parameters:
    -----------
    model_type : str, default='improved'
        Type of model ('improved' or 'simple')
    input_dim : int, default=384
        Input feature dimension
    num_classes : int, default=2
        Number of output classes
    **kwargs : dict
        Additional model-specific parameters
        
    Returns:
    --------
    torch.nn.Module: Initialized model
    """
    if model_type == 'improved':
        return ImprovedClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == 'simple':
        return SimpleClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_info(model):
    """
    Get information about a model.
    
    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model
        
    Returns:
    --------
    dict: Model information including parameter count and layer details
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get layer information
    layers = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters())
            layers.append({
                'name': name,
                'type': type(module).__name__,
                'parameters': param_count
            })
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'layers': layers,
        'model_type': type(model).__name__
    }


def print_model_summary(model, input_shape=(1, 384)):
    """
    Print a summary of the model architecture.
    
    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model
    input_shape : tuple, default=(1, 384)
        Input tensor shape for testing
    """
    info = get_model_info(model)
    
    print(f"\n{'='*60}")
    print(f"MODEL SUMMARY: {info['model_type']}")
    print(f"{'='*60}")
    print(f"Total Parameters: {info['total_parameters']:,}")
    print(f"Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"\nLayer Details:")
    print(f"{'Name':<20} {'Type':<15} {'Parameters':<10}")
    print(f"{'-'*50}")
    
    for layer in info['layers']:
        print(f"{layer['name']:<20} {layer['type']:<15} {layer['parameters']:<10,}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        try:
            dummy_input = torch.randn(input_shape)
            output = model(dummy_input)
            print(f"\nInput Shape: {tuple(dummy_input.shape)}")
            print(f"Output Shape: {tuple(output.shape)}")
        except Exception as e:
            print(f"\nForward pass test failed: {e}")
    
    print(f"{'='*60}\n")


# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Create improved model
    improved_model = create_model('improved', 
                                hidden_dims=[256, 128, 64],
                                dropout_rate=0.3,
                                activation='relu')
    
    # Create simple model
    simple_model = create_model('simple', hidden_dim=128)
    
    # Print summaries
    print_model_summary(improved_model)
    print_model_summary(simple_model)
    
    print("Model creation test completed successfully!")
