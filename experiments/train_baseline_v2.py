import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class BaselineSNN(nn.Module):
    def __init__(self, T=20):
        super().__init__()
        self.T = T
        
        self.fc1 = layer.Linear(784, 300, bias=False)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        
        self.fc2 = layer.Linear(300, 10, bias=False)
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.lif1(x)
        x = self.fc2(x)
        x = self.lif2(x)
        return x

def train_one_epoch(model, train_loader, optimizer, device, T=20):
    model.train()
    correct = 0
    total = 0
    total_loss = 0
    
    for batch_idx, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)
        
        img = img.view(img.size(0), -1)
        
        spike_img = []
        for t in range(T):
            spike = torch.rand_like(img) < img.clamp(0, 1)
            spike_img.append(spike.float())
        spike_img = torch.stack(spike_img)
        
        output = model(spike_img)
        output_mean = output.mean(0)
        
        loss = F.cross_entropy(output_mean, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        functional.reset_net(model)
        
        pred = output_mean.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, device, T=20):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)
            
            img = img.view(img.size(0), -1)
            
            spike_img = []
            for t in range(T):
                spike = torch.rand_like(img) < img.clamp(0, 1)
                spike_img.append(spike.float())
            spike_img = torch.stack(spike_img)
            
            output = model(spike_img)
            output_mean = output.mean(0)
            pred = output_mean.argmax(dim=1)
            
            correct += (pred == label).sum().item()
            total += label.size(0)
            
            functional.reset_net(model)
    
    return 100. * correct / total

def main():
    batch_size = 128
    num_epochs = 10
    lr = 1e-3
    T = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Timesteps: {T}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../data/', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST('../data/', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = BaselineSNN(T=T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    results = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, T)
        test_acc = evaluate(model, test_loader, device, T)
        
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    os.makedirs('../results/models', exist_ok=True)
    os.makedirs('../results/metrics', exist_ok=True)
    
    torch.save(model.state_dict(), '../results/models/baseline_snn_v2.pth')
    
    with open('../results/metrics/baseline_v2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Training complete! Final test accuracy: {results['test_acc'][-1]:.2f}%")

if __name__ == '__main__':
    main()