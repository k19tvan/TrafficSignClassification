import torch.nn as nn
import torch.optim as optim
from basic_model import BasicModel
from function import DataLoaderCreate
import torch    

train_loader, val_loader, test_loader = DataLoaderCreate()
model = BasicModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

epochs = 20
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