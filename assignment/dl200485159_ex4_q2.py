import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from clearml import Task, Logger

# Initialize ClearML Task
task = Task.init(project_name="Training_CNN_project", task_name=f'Training ResNet18 {time.time()}')

# Load pretrained ResNet18 model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)  # Binary classification

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load existing weights if available
weights_file = "resnet18_model.pth"
if os.path.exists(weights_file):
    model.load_state_dict(torch.load(weights_file))
    print(f"Loaded existing weights from '{weights_file}'")
else:
    print("No existing weights found. Starting training from scratch.")
    
# Data paths
'''Data part A'''
project_root = os.getcwd()
image_folder_for_train = os.path.join(project_root, "Training_stageA")
image_folder_for_validation = os.path.join(project_root, "Validation_stageA")
augmented_folder = os.path.join(project_root, "Training_stageA_Augmented") #for experiance with full data augmentation 

'''Data part B'''
image_folder_for_train_B = os.path.join(project_root, "Training_stageB")
image_folder_for_validation_B = os.path.join(project_root, "Validation_stageB")

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.RandomRotation(10),  #'''used in sage 2 in part 2'''
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load images and labels
def load_images_with_labels(folder, transform=None):
    images, labels = [], []
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
    return torch.stack(images), torch.tensor(labels, dtype=torch.float32).view(-1, 1)

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

   
# Train function
def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = torch.sigmoid(model(inputs))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = torch.sigmoid(model(inputs))
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        
        # Print training and validation results
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        Logger.current_logger().report_scalar(
            title=f"Training Loss round #2 Batch=54 Data group B", series="Loss", iteration=epoch+1, value=train_loss
        )
        Logger.current_logger().report_scalar(
            title=f"Training Accuracy round #2 Batch=54 Data group B", series="Accuracy", iteration=epoch+1, value=train_accuracy
        )
        Logger.current_logger().report_scalar(
            title=f"Validation Loss round #2 Batch=54 Data group B", series="Loss", iteration=epoch+1, value=val_loss
        )
        Logger.current_logger().report_scalar(
            title=f"validation Accuracy round #2 Batch=54 Data group B", series="Accuracy", iteration=epoch+1, value=val_accuracy
        )
        
        
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
        # torch.save(model.state_dict(), weights_file)
        
# Load training and validation data
'''Data group A'''
# # X_train, y_train = load_images_with_labels(image_folder_for_train, transform) #'''used in sage 2 in part 2'''
# X_train, y_train = load_images_with_labels(augmented_folder, transform) #'''used in sage 3 in part 2'''
# X_val, y_val = load_images_with_labels(image_folder_for_validation, transform)

'''Data group B'''
X_train, y_train = load_images_with_labels(image_folder_for_train_B, transform) 
X_val, y_val = load_images_with_labels(image_folder_for_validation_B, transform)


# Create DataLoaders
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=32, shuffle=True)
val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=32, shuffle=False)

# Train model
# Train model
# train_model(model, train_loader, val_loader, 10)
train_model(model, X_train, y_train, X_val, y_val)
# model.eval()
print("Training complete! Model weights saved.")

''' test '''
image_folder_for_testing = os.path.join(project_root, "Testing_stageA")
image_folder_for_testing_B= os.path.join(project_root, "Testing_stageB")

model.load_state_dict(torch.load(weights_file))

'''Data group A'''
# test_images, image_names = load_test_images(image_folder_for_testing, transform)

'''Data group B'''
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



#TODO:
#נורמליזציה V
#להוסיף גראפים למודל הראשון V
#להוציא גרפים עבור הoverfitting שנוצר V
#לייצב את המודל הראשון #להוציא גראפים V
#להתאים את המודל המאומן V
#להוסיף בדיקה קטנה V
#להוסיף בדיקת טסט V
#לנסות לחשוב על גרף שמייצג את כולם? V
