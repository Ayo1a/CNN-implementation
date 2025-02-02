import os
from tkinter import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader

# טוענים את המודל המאומן מראש
model = models.resnet18(pretrained=True)  

# קופאים את כל השכבות כך שלא יתעדכנו
for param in model.parameters():
    param.requires_grad = False

# מחליפים את השכבה האחרונה בהתאם לבעיה שלנו (במקרה הזה - סיווג בינארי)
num_features = model.fc.in_features  
model.fc = nn.Sequential(
    nn.Linear(num_features, 128),  
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 1),  
    nn.Sigmoid()  # סיגמואיד כי מדובר בבעיה של סיווג בינארי
)

# מעבירים את המודל ל-GPU אם אפשר
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# פונקציית איבוד - Binary Cross Entropy כי מדובר בסיווג בינארי
criterion = nn.BCELoss()  

# רק השכבות שהוספנו יתעדכנו
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0001, weight_decay=0.001)  

num_epochs = 10  # מספר אפוכים
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:  
        images, labels = images.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

''' The ResNet and VGG pre-trained model was trained on ImageNet, where the images are normalized by:
Mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225] '''    

# טרנספורמציות מתאימות ל-ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet מצפה לגודל הזה
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_images_with_labels():
    print ("TODO")
    #TODO


def load_test_images(folder, transform):
    test_images = []
    image_names = []

    for file_name in os.listdir(folder):
        img_path = os.path.join(folder, file_name)
        img = Image.open(img_path).convert("RGB")  
        img = transform(img)  
        test_images.append(img)
        image_names.append(file_name)

    return torch.stack(test_images), image_names  

#Data
project_root = os.getcwd()  # getting current path  (project's folder)
image_folder_for_train= os.path.join(project_root, "Training_stageA")
X_train, y_train = load_images_with_labels(image_folder_for_train, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

image_folder_for_validation= os.path.join(project_root, "Validation_stageA")


