"""
Mask Detection CNN using VGG16 transfer learning
"""

import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os


categories = ['incorrect_mask', 'with_mask', 'without_mask']
data = []
for category in categories:
    path = os.path.join('dataset', category)
    label = categories.index(category)
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        data.append([img, label])

random.shuffle(data)
X = np.array([item[0] for item in data])
y = np.array([item[1] for item in data])
X = X/255

# Split data into train, validation and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=41)  # 0.25 * 0.8 = 0.2

class MaskDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = MaskDataset(X_train, y_train, transform=transform)
val_dataset = MaskDataset(X_val, y_val, transform=transform)
test_dataset = MaskDataset(X_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize lists for tracking metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

import torch.nn as nn
import torchvision.models as models

# Load pre-trained VGG16 weights
vgg = models.vgg16(weights='IMAGENET1K_V1')

# Freeze VGG16 layers
for param in vgg.parameters():
    param.requires_grad = False

# Modify VGG16 to remove the classifier
vgg.classifier = nn.Identity()

# Create the PyTorch sequential model
pytorch_model = nn.Sequential(
    vgg,
    nn.Flatten(),
    nn.Linear(25088, 256),
    nn.ReLU(),
    nn.Dropout(0.33),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.33),
    nn.Linear(128, 3),
    nn.Softmax(dim=1)
)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)

# Print dataset information
print("Number of batches in train_loader:", len(train_loader))
print("Number of batches in val_loader:", len(val_loader))
print("Number of batches in test_loader:", len(test_loader))

epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pytorch_model.to(device)

for epoch in range(epochs):
    # Training phase
    pytorch_model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = pytorch_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_accuracy = 100 * correct / total
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)

    # Validation phase
    pytorch_model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images_val, labels_val in val_loader:
            images_val, labels_val = images_val.to(device), labels_val.to(device)
            outputs_val = pytorch_model(images_val)
            loss_val = criterion(outputs_val, labels_val)

            running_val_loss += loss_val.item() * images_val.size(0)
            _, predicted_val = torch.max(outputs_val.data, 1)
            total_val += labels_val.size(0)
            correct_val += (predicted_val == labels_val).sum().item()

    epoch_val_loss = running_val_loss / len(val_dataset)
    epoch_val_accuracy = 100 * correct_val / total_val
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_accuracy)


    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2f}%, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.2f}%")

print("Training finished.")

# Plot training and validation metrics

import matplotlib.pyplot as plt

epochs_range = range(epochs)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# Save the trained model
model_path = 'mask_detection_model.pth'
torch.save(pytorch_model.state_dict(), model_path)
print(f"Model saved to {model_path}")