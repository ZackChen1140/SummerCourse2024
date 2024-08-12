import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder


import numpy as np
import pandas as pd
from PIL import Image

# define a custom dataset class
class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file) # read data from csv file as a pandas dataframe
        self.transform = transform # initial transfrom

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.fromarray(self.data_frame.iloc[idx, 1:].values.reshape(28, 28).astype(np.uint8))
        
        label = int(self.data_frame.iloc[idx, 0])

        if self.transform:
            image = self.transform(image)

        return image, label
    
#Data Preprocessing
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
])

#define dataset and dataloader
train_path = 'dataset/fashion-mnist_train.csv'
test_path = 'dataset/fashion-mnist_test.csv'

train_data = MNISTDataset(csv_file=train_path, transform=train_transform)
test_data = MNISTDataset(csv_file=test_path, transform=test_transform)
testLen = int(len(test_data) * 0.5)
valLen = len(test_data) - testLen
test_data, val_data = random_split(test_data, [testLen, valLen])

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
val_loader = DataLoader(val_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#choose a model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)

#choose a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), weight_decay=0.001, lr=0.0001)

#move model to cuda/cpu
model.to(device)

#training section
num_epochs = 80
best_accuracy = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = correct_train / total_train

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

    #evaluate on validation set every epoch
    if epoch % 5 == 4:
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

            val_loss = running_val_loss / len(val_loader.dataset)
            val_accuracy = correct_val / total_val

            if val_accuracy > best_accuracy:
                # Save model weights
                torch.save(model.state_dict(), 'best_model.pth')
                best_accuracy = val_accuracy

            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

model.load_state_dict(torch.load('best_model.pth'))

#evaluate on testing set every epoch
model.eval()
running_test_loss = 0.0
correct_test = 0
total_test = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_loss = running_test_loss / len(test_loader.dataset)
test_accuracy = correct_test / total_test

print(f'Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_accuracy:.4f}')