import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader

def fit(model, dataloaders, learning_rate, momentum, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    # Dictionaries to store losses for plotting
    epoch_losses = {'train': [], 'val': []}
    
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            n_batches = 0

            for inputs, labels in dataloaders[phase]:
                labels = labels.type(torch.LongTensor)
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = F.cross_entropy(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                n_batches += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_losses[phase].append(epoch_loss)

            print(f'Epoch {epoch+1}/{num_epochs} - Phase: {phase} - '
                  f'Loss: {epoch_loss:.4f}')

    return epoch_losses
