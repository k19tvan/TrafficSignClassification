from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import torch
import cv2

base_dir = "/home/enn/workspace/deep_learning/Traffic_Sign_Classification"
def DataPrepare():
    
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

def ModelTraining(model, train_loader, test_loader, epochs = 20, save_after_epochs = 20, model_save_path = "model_weights.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

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
        
        if (epoch + 1) % save_after_epochs == 0:
            torch.save(model.state_dict(), model_save_path)
    
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():  
        for inputs, labels in test_loader:
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)  

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    
def ImagePrediction(model, image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))  

    image = image.astype('float32') / 255.0  
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1) 
        
    labels = ['Danger', 'Instruction', 'Neg', 'Prohibition']
    predict_label = predicted.item()
        
    print(f"This image is about {labels[predict_label]} sign" if predict_label != 3 else "This image isn't about Traffic Sign")
