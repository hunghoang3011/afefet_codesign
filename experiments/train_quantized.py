import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based import functional
import sys
sys.path.append('..')
from src.models.quantized_snn import QuantizedSNN
import json
import os

def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    correct = 0
    total = 0
    total_loss = 0
    
    for batch_idx, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)
        
        # Flatten image
        img = img.view(img.size(0), -1)  # [Batch, 784]
        
        # Run through time
        T = 10  # timesteps
        outputs = []
        for t in range(T):
            # Simple rate coding: use same input
            out = model(img)
            outputs.append(out)
        
        # Average spikes over time
        output = torch.stack(outputs).mean(0)  # [Batch, 10]
        
        # Loss
        loss = F.cross_entropy(output, label)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Reset neuron states
        functional.reset_net(model)
        
        # Metrics
        pred = output.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)
            img = img.view(img.size(0), -1)
            
            T = 10
            outputs = []
            for t in range(T):
                out = model(img)
                outputs.append(out)
            
            output = torch.stack(outputs).mean(0)
            pred = output.argmax(dim=1)
            
            correct += (pred == label).sum().item()
            total += label.size(0)
            
            functional.reset_net(model)
    
    return 100. * correct / total

def main():
    # Config
    batch_size = 128
    num_epochs = 10  # Start with 5 for quick test
    lr = 3e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../data/', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data/', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = QuantizedSNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training
    results = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    # Save
    os.makedirs('../results/models', exist_ok=True)
    os.makedirs('../results/metrics', exist_ok=True)
    
    torch.save(model.state_dict(), '../results/models/baseline_snn.pth')
    
    with open('../results/metrics/baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"Final test accuracy: {results['test_acc'][-1]:.2f}%")

if __name__ == '__main__':
    main()