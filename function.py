from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import torch
import cv2

def DataPrepare():
    base_dir = "/home/enn/workspace/deep_learning/Traffic_Sign_Classification"

    X = []
    y = []

    data_dir = Path(f"{base_dir}/CS231-TrafficSignClassification/Data")

    for class_idx, folder in enumerate(sorted(data_dir.iterdir())):
        for file in Path(folder).iterdir():
            image = cv2.resize(cv2.imread(file), (100, 100))
            X.append(image)
            y.append(class_idx)

    X = np.array(X)
    X = X.astype('float32') / 255.0
    y = np.array(y)
    
    X_train, X_val, X_test, y_train, y_val, y_test = [], [], [], [], [], []
    
    for id in range(4):
        
        indices = np.where(y == id)[0]
        X_train.extend(X[indices[:100]])
        y_train.extend([id] * 100)

        X_val.extend(X[indices[100:125]])
        y_val.extend([id] * 25)

        X_test.extend(X[indices[125:149]])
        y_test.extend([id] * 24)
    
    return np.array(X_train), np.array(X_val), np.array(X_test), np.array(y_train), np.array(y_val), np.array(y_test)

class Tensor(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
        
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label
    
def DataLoaderCreate():

    X_train, X_val, X_test, y_train, y_val, y_test = DataPrepare()
    
    train_dataset = Tensor(X_train, y_train)
    val_dataset = Tensor(X_val, y_val)
    test_dataset = Tensor(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader

def TrainModel(model, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")