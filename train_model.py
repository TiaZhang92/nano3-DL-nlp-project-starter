# train.py - Fixed Dog Classification Training Script with SageMaker Debugger Integration

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import copy
import argparse
import os
import logging
import sys
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# SageMaker debugger imports with error handling
try:
    import smdebug.pytorch as smd
    from smdebug import modes
    from smdebug.pytorch import get_hook
    DEBUGGER_AVAILABLE = True
    print("✓ SageMaker debugger successfully imported")
except ImportError as e:
    print(f"⚠️ SageMaker debugger not available: {e}")
    DEBUGGER_AVAILABLE = False
    # Dummy hook for compatibility
    class DummyHook:
        def register_hook(self, *args, **kwargs): pass
        def set_mode(self, *args, **kwargs): pass
        def save_scalar(self, *args, **kwargs): pass
        def register_loss(self, *args, **kwargs): pass
    def get_hook(create_if_not_exists=True):
        return DummyHook()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device, hook=None):
    """Test function with debugger integration"""
    model.eval()
    running_loss = 0
    running_corrects = 0
    
    # Set hook to evaluation mode
    if hook and DEBUGGER_AVAILABLE:
        hook.set_mode(modes.EVAL)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    # Fixed calculation (was using // instead of /)
    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects.double() / len(test_loader.dataset)
    
    # Save test metrics to debugger
    if hook and DEBUGGER_AVAILABLE:
        hook.save_scalar("test_loss", total_loss, sm_metric=True)
        hook.save_scalar("test_accuracy", total_acc, sm_metric=True)
    
    logger.info(f"Testing Loss: {total_loss:.4f}")
    logger.info(f"Testing Accuracy: {total_acc:.4f}")
    
    return total_loss, total_acc

def train(model, train_loader, validation_loader, criterion, optimizer, device, epochs=5, hook=None):
    """Training function with debugger integration and fixes"""
    best_loss = 1e6
    image_dataset = {'train': train_loader, 'valid': validation_loader}
    loss_counter = 0
    
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch+1}/{epochs}")
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                if hook and DEBUGGER_AVAILABLE:
                    hook.set_mode(modes.TRAIN)
            else:
                model.eval()
                if hook and DEBUGGER_AVAILABLE:
                    hook.set_mode(modes.EVAL)
                    
            running_loss = 0.0
            running_corrects = 0

            # Add progress bar
            phase_loader = image_dataset[phase]
            for batch_idx, (inputs, labels) in enumerate(tqdm(phase_loader, desc=f"{phase.capitalize()}")):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Save batch metrics to debugger
                if hook and DEBUGGER_AVAILABLE and batch_idx % 10 == 0:
                    hook.save_scalar(f"{phase}_batch_loss", loss.item(), sm_metric=True)

            # Fixed calculation (was using // instead of /)
            epoch_loss = running_loss / len(image_dataset[phase].dataset)
            epoch_acc = running_corrects.double() / len(image_dataset[phase].dataset)
            
            # Save epoch metrics to debugger
            if hook and DEBUGGER_AVAILABLE:
                hook.save_scalar(f"{phase}_epoch_loss", epoch_loss, sm_metric=True)
                hook.save_scalar(f"{phase}_epoch_acc", epoch_acc, sm_metric=True)
            
            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    loss_counter = 0  # Reset counter when we improve
                else:
                    loss_counter += 1

            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, best_loss))
        
        # Early stopping if loss doesn't improve
        if loss_counter >= 3:  # Stop after 3 epochs without improvement
            logger.info("Early stopping due to no improvement")
            break
            
    return model

def net():
    """Create ResNet50 model for dog classification"""
    model = models.resnet50(pretrained=True)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False   

    # Replace final layer for 133 dog breeds
    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),  # Add dropout for regularization
        nn.Linear(128, 133)
    )
    return model

def create_data_loaders(data, batch_size):
    """Create data loaders with proper transforms"""
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path = os.path.join(data, 'valid')

    # Enhanced data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Fixed: Use correct train_data_path (was using test_data_path)
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=4)
    
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(validation_data)}")
    logger.info(f"Test samples: {len(test_data)}")
    
    return train_data_loader, test_data_loader, validation_data_loader

def main(args):
    """Main training function"""
    logger.info(f'Hyperparameters are LR: {args.learning_rate}, Batch Size: {args.batch_size}, Epochs: {args.epochs}')
    logger.info(f'Data Paths: {args.data}')
    logger.info(f'Model Dir: {args.model_dir}')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize debugger hook
    hook = None
    if DEBUGGER_AVAILABLE:
        try:
            hook = get_hook(create_if_not_exists=True)
            logger.info("✓ Debugger hook initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize debugger hook: {e}")
    
    # Create data loaders
    train_loader, test_loader, validation_loader = create_data_loaders(args.data, args.batch_size)
    
    # Create model
    model = net().to(device)
    logger.info(f"Model created and moved to {device}")
    
    # Register model with debugger hook
    if hook and DEBUGGER_AVAILABLE:
        try:
            hook.register_hook(model)
            logger.info("✓ Model registered with debugger hook")
        except Exception as e:
            logger.warning(f"Could not register model with hook: {e}")
    
    # Fixed: Remove ignore_index=133 as it's larger than num_classes-1
    criterion = nn.CrossEntropyLoss()
    
    # Only train the final layer parameters
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    # Register loss with debugger hook
    if hook and DEBUGGER_AVAILABLE:
        try:
            hook.register_loss(criterion)
            logger.info("✓ Loss function registered with debugger hook")
        except Exception as e:
            logger.warning(f"Could not register loss with hook: {e}")
    
    logger.info("Starting Model Training")
    model = train(model, train_loader, validation_loader, criterion, optimizer, device, args.epochs, hook)
    
    logger.info("Testing Model")
    test_loss, test_acc = test(model, test_loader, criterion, device, hook)
    
    logger.info("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))
    
    # Save training summary
    summary = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'hyperparameters': {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'epochs': args.epochs
        }
    }
    
    import json
    with open(os.path.join(args.output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Training completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))
    
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    
    main(args)