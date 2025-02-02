import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from clearml import Task, Logger

task = Task.init(project_name="Trainning_CNN_project", task_name=f'Training custom CNN {time.time()}') #in each run it ovveride the previsue run if it has the same name

# Step 1: Define the CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Define the layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 3 input channels (RGB), 32 filters
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32, 128)  # Assuming input size is 256*256, after 3 pool layers, size becomes 16x16
        # self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(128, 1)  # Output layer with 1 neuron for binary classification

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply convolutional layers with ReLU activation and pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        # Flatten the output for fully connected layers
        x = x.view(-1, 128 * 32 * 32)  # Flattening
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for binary classification
        return x

# Step 2: Define the optimizer, loss function, and the model
model = CNNModel()
# adding L2 Regularization to deal with overfitting  with weight_decay
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()  # Binary cross-entropy for binary classification

# Load existing weights if the file exists
weights_file = "cnn_model.pth"
if os.path.exists(weights_file):
    model.load_state_dict(torch.load(weights_file))
    print(f"Loaded existing weights from '{weights_file}'")
else:
    print("No existing weights found. Starting training from scratch.")
    
#Data
#data group A
project_root = os.getcwd()  # getting current path  (project's folder)
image_folder_for_train= os.path.join(project_root, "Training_stageA")
image_folder_for_validation= os.path.join(project_root, "Validation_stageA")
augmented_folder = os.path.join(project_root, "Training_stageA_Augmented")

# tarin with Data group B
image_folder_for_train_B= os.path.join(project_root, "Training_stageB")
image_folder_for_validation_B= os.path.join(project_root, "Validation_stageB")

#transformation for images 
transform = transforms.Compose([
     transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.RandomRotation(10),  # rotates images with 10 degrees Tenzor Converting
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # values normalizations[-1,1]
])

#upload images and tag 
def load_images_with_labels(folder, transform=None):
    images = []
    labels = []
    
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)

        if "notsmoking" in file_name.lower():
            label = 0
        elif "smoking" in file_name.lower():
            label = 1
        else:
            continue  

        try:
            image = Image.open(file_path).convert("RGB")  
            if transform:
                image = transform(image)  
            images.append(image)
            labels.append(label)
        except Exception as e:
            print(f"Could not load {file_path}: {e}")
            
    images_tensor = torch.stack(images)  # ממיר רשימת תמונות לטנסור עם מבנה (num_samples, 3, 256, 256)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)  # ממיר רשימת לייבלים לטנסור עם מבנה (num_samples, 1)

    return images_tensor, labels_tensor
    # return images, labels

def load_test_images(folder, transform):
    test_images = []
    image_names = []  # saving names for predictions
    for file_name in os.listdir(folder):
        img_path = os.path.join(folder, file_name)
        try:
            img = Image.open(img_path).convert("RGB")  # load as RGB
            img = transform(img)  # converting to Pytorch's tensor
            test_images.append(img)
            image_names.append(file_name)
        except Exception as e:
            print(f"Could not load {img_path}: {e}")
            
    if not test_images:
        raise ValueError("No valid images were found in the provided folder.")
    
    return torch.stack(test_images), image_names  # return all images and theis names 


# Step 3: Train the Model
def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=64):
    train_data = torch.tensor(X_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    val_data = torch.tensor(X_val, dtype=torch.float32)
    val_labels = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    
    train_loader = DataLoader(list(zip(train_data, train_labels)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(list(zip(val_data, val_labels)), batch_size=batch_size, shuffle=False)
    
    

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()  # Apply threshold for binary classification
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # Validation step
        ''' If the result on X_val is much worse than that of X_train, there is probably overfitting. '''
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = 100 * correct / total
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
            
        Logger.current_logger().report_scalar(
            title=f"Training Loss #1\n data group B", series="Loss", iteration=epoch+1, value=epoch_loss
        )
        Logger.current_logger().report_scalar(
            title=f"Training Accuracy #1\n data group B", series="Accuracy", iteration=epoch+1, value=epoch_accuracy
        )
        Logger.current_logger().report_scalar(
            title=f"Validation Loss #1\n data group B", series="Loss", iteration=epoch+1, value=val_loss
        )
        Logger.current_logger().report_scalar(
            title=f"validation Accuracy #1\n data group B", series="Accuracy", iteration=epoch+1, value=val_accuracy
        )
        

        # Save model weights after each epoch
        torch.save(model.state_dict(), weights_file)
        print(f"Model weights saved as '{weights_file}'")

'''data group A'''
# X_train, y_train = load_images_with_labels(image_folder_for_train, transform=transform)
 
#data augmentation for data group A
# X_train, y_train = load_images_with_labels(augmented_folder, transform=transform)

'''data group B '''
X_train, y_train = load_images_with_labels(image_folder_for_train_B, transform=transform)

'''validation'''
# #A
# X_val, y_val = load_images_with_labels(image_folder_for_validation, transform=transform)
#B
X_val, y_val = load_images_with_labels(image_folder_for_validation_B, transform=transform)

# Train the model
train_model(model, X_train, y_train, X_val, y_val)

# Step 4: Save the model
torch.save(model.state_dict(), "cnn_model.pth")  
print("Training complete! Model weights saved as 'cnn_model.pth'")

''' sanity test A - 2 images '''
print ("sanity check")
image_folder_for_initial_test = os.path.join(project_root, "Basic_sanity_stgA")
image_folder_for_initial_test_B = os.path.join(project_root, "Basic_sanity_stgB")

model.load_state_dict(torch.load("cnn_model.pth"))
# #A
# sanity_test_images, sanity_image_names = load_test_images(image_folder_for_initial_test, transform)
#B
sanity_test_images, sanity_image_names = load_test_images(image_folder_for_initial_test_B, transform)

model.eval()  # evaluation stats without weights update
with torch.no_grad():  # no gradients' calculation
    outputs = model(sanity_test_images)  # getting predictions
    predictions = (outputs > 0.5).float()  # convert to 1 or 0

# Results
for i, name in enumerate(sanity_image_names):
    print(f"image: {name}, prediction: {'Smoking' if predictions[i].item() == 1 else 'Not Smoking'}")

print ("end of sanity check")

''' test '''
'''data group A'''
image_folder_for_testing = os.path.join(project_root, "Testing_stageA")
'''data group B'''
image_folder_for_testing_B= os.path.join(project_root, "Testing_stageB")

model.load_state_dict(torch.load("cnn_model.pth"))
# #A
# test_images, image_names = load_test_images(image_folder_for_testing, transform)
#B
test_images, image_names = load_test_images(image_folder_for_testing_B, transform)

model.eval()  # evaluation stats without weights update
with torch.no_grad():  # no gradients' calculation
    outputs = model(test_images)  # getting predictions
    predictions = (outputs > 0.5).float()  # convert to 1 or 0
    
# Results
correct = 0
total = len(image_names)

for i, name in enumerate(image_names):
     predicted_label = 1 if predictions[i].item() >= 0.5 else 0  # threshold og 0.5 to binnary decission
     actual_label = 0 if "notsmoking" in name.lower() else 1  # 1 if it "smoking", 0 if it "notsmoking"
     
     if predicted_label == actual_label:
        correct += 1  # counting correct predictions
    
     print(f"image: {name}, prediction: {'Smoking' if predictions[i].item() == 1 else 'Not Smoking'}")
    
# accuracy calculation
accuracy = correct / total * 100
print(f"Accuracy of test : {accuracy:.2f}%")


#TODO
'''
path to sanity folder V
function to load images V
runing the model in test mode V
'''
