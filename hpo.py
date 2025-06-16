import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import os
import sys
import logging

# Set up logging properly
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test(model, test_loader, criterion):
    '''
    Complete this function that can take a model and a 
    testing data loader and will get the test accuray/loss of the model
    '''
    print("Starting model testing...")
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            try:
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                if batch_idx % 10 == 0:
                    print(f"Test batch {batch_idx}/{len(test_loader)}")
            except Exception as e:
                print(f"Error in test batch {batch_idx}: {e}")
                continue
    
    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / total
    
    print(f'Testing Loss: {test_loss}')
    print(f'Testing Accuracy: {accuracy}%')
    
    return test_loss

def train(model, train_loader, criterion, optimizer, epochs=5):
    '''
    Complete this function that can take a model and
    data loaders for training and will get train the model
    '''
    print(f"Starting training for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        print(f"Starting epoch {epoch + 1}/{epochs}")
        
        try:
            for batch_idx, (data, target) in enumerate(train_loader):
                try:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                    num_batches += 1
                    
                    if batch_idx % 50 == 0:
                        print(f'Epoch {epoch+1}, Batch {batch_idx}: Loss {loss.item():.4f}')
                        
                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            break
        
        if num_batches > 0:
            epoch_loss = running_loss / num_batches
            epoch_acc = 100.0 * correct / total
            print(f'Epoch {epoch+1}/{epochs}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        else:
            print(f'Epoch {epoch+1}/{epochs}: No successful batches processed')
    
    return model

def net():
    '''
    Complete this function that initializes your model
    Remember to use a pretrained model
    '''
    print("Initializing ResNet50 model...")
    
    # Use pretrained ResNet50
    model = models.resnet50(pretrained=True)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final layer for dog breed classification (133 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 133)
    
    print("Model initialized successfully")
    return model

def create_data_loaders(data, batch_size):
    '''
    This function creates data loaders for the dog breed classification task
    '''
    print("=== CREATING DATA LOADERS ===")
    print(f"Batch size: {batch_size}")
    
    # Ensure batch_size is valid
    if batch_size is None or batch_size <= 0:
        batch_size = 32  # Conservative default
        print(f"Invalid batch_size, using default: {batch_size}")
    
    batch_size = int(batch_size)
    
    # SageMaker downloads data to these paths
    train_data_path = '/opt/ml/input/data/training'
    validation_data_path = '/opt/ml/input/data/validation'
    test_data_path = '/opt/ml/input/data/testing'
    
    print(f"Looking for data at:")
    print(f"  Training: {train_data_path}")
    print(f"  Validation: {validation_data_path}")
    print(f"  Testing: {test_data_path}")
    
    # Check if directories exist
    for path, name in [(train_data_path, "training"), (validation_data_path, "validation"), (test_data_path, "testing")]:
        if os.path.exists(path):
            try:
                contents = os.listdir(path)
                print(f"{name} directory: {len(contents)} folders found")
                if len(contents) > 0:
                    print(f"  Sample folders: {contents[:3]}")
            except Exception as e:
                print(f"Error reading {name} directory: {e}")
        else:
            print(f"ERROR: {name} directory not found!")
            return None, None, None
    
    # Simple transforms to avoid any issues
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Create datasets
        print("Creating datasets...")
        train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
        validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
        test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
        
        print(f"Dataset sizes:")
        print(f"  Training: {len(train_data)}")
        print(f"  Validation: {len(validation_data)}")
        print(f"  Testing: {len(test_data)}")
        print(f"  Classes: {len(train_data.classes)}")
        
        # Create data loaders with minimal workers to avoid issues
        print("Creating data loaders...")
        train_loader = torch.utils.data.DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Use 0 workers to avoid multiprocessing issues
            pin_memory=False
        )
        
        validation_loader = torch.utils.data.DataLoader(
            validation_data, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_data, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print("Data loaders created successfully!")
        return train_loader, test_loader, validation_loader
        
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return None, None, None

def main(args):
    print("=== STARTING MAIN FUNCTION ===")
    print(f"Arguments: {vars(args)}")
    
    # Validate arguments
    if args.batch_size is None or args.batch_size <= 0:
        args.batch_size = 32
        print(f"Fixed batch_size to: {args.batch_size}")
    
    if args.learning_rate is None or args.learning_rate <= 0:
        args.learning_rate = 0.01
        print(f"Fixed learning_rate to: {args.learning_rate}")
    
    try:
        # Initialize model
        model = net()
        
        # Create loss and optimizer
        loss_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=float(args.learning_rate))
        
        print(f"Using learning_rate: {args.learning_rate}")
        print(f"Using batch_size: {args.batch_size}")
        
        # Create data loaders
        train_loader, test_loader, validation_loader = create_data_loaders(args.data, args.batch_size)
        
        if train_loader is None:
            print("Failed to create data loaders. Exiting.")
            return
        
        # Train the model
        model = train(model, train_loader, loss_criterion, optimizer, epochs=5)
        
        # Test the model
        test_loss = test(model, test_loader, loss_criterion)
        
        # Save the model
        print(f"Saving model to {args.model_dir}")
        os.makedirs(args.model_dir, exist_ok=True)
        model_path = os.path.join(args.model_dir, 'model.pth')
        torch.save(model.state_dict(), model_path)
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        raise e

if __name__ == '__main__':
    print("=== SCRIPT STARTED ===")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))
    
    print("Environment variables:")
    for key in ['SM_MODEL_DIR', 'SM_CHANNEL_TRAINING', 'SM_CHANNEL_VALIDATION', 'SM_CHANNEL_TESTING']:
        print(f"  {key} = {os.environ.get(key, 'Not set')}")
    
    args = parser.parse_args()
    print(f"Parsed arguments: {vars(args)}")
    
    main(args)