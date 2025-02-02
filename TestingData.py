import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


#Data
project_root = os.getcwd()  # getting current path  (project's folder)
image_folder_for_train = os.path.join(project_root, "Training_stageA")
image_folder_for_validation= os.path.join(project_root, "Validation_stageA")
image_folder_for_testing= os.path.join(project_root, "Testing_stageA")
augmented_folder = os.path.join(project_root, "Training_stageA_Augmented")

#transformation for images 
transform = transforms.Compose([
    transforms.ToTensor(),  # Tenzor Converting
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # values normalizations[-1,1]
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

    return images, labels

def load_test_images(folder):
    test_images = []
    image_names = []  # נשמור גם את השמות כדי לראות את התחזיות
    for file_name in os.listdir(folder):
        img_path = os.path.join(folder, file_name)
        img = Image.open(img_path).convert("RGB")  # נטען תמונה כ-RGB
        img = transform(img)  # נהפוך לטנזור של PyTorch
        test_images.append(img)
        image_names.append(file_name)
    return torch.stack(test_images), image_names  # מחזירים את כל התמונות והשמות שלהן


def augment_images_with_rotation(input_folder, output_folder, num_rotations=1, angle=10):
    """
    משכפל תמונות ומוסיף להן וריאציות מסובבות בתיקייה חדשה.
    
    :param input_folder: תיקייה עם התמונות המקוריות
    :param output_folder: תיקייה שבה יישמרו התמונות החדשות
    :param num_rotations: כמה וריאציות מסובבות ליצור לכל תמונה (ברירת מחדל: 1)
    :param angle: זווית הסיבוב המקסימלית (ברירת מחדל: 10 מעלות)
    """
    os.makedirs(output_folder, exist_ok=True)  # יצירת תיקייה אם אינה קיימת
    rotation_transform = transforms.RandomRotation(angle)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            img = Image.open(image_path)

            # שמירת התמונה המקורית
            img.save(os.path.join(output_folder, filename))

            # יצירת מספר וריאציות מסובבות ושמירתן
            for i in range(num_rotations):
                rotated_img = rotation_transform(img)
                rotated_filename = f"rotated_{i}_{filename}"
                rotated_img.save(os.path.join(output_folder, rotated_filename))

    print(f"Process complate rotated and original imaes ware saved: {output_folder}")
    
def load_augmented_images(folder, transform):
    """
    טוען תמונות מתוך תיקייה עם טרנספורמציות מוכנות.
    
    :param folder: תיקייה עם התמונות המעובדות
    :param transform: טרנספורמציות שיבוצעו על התמונות
    :return: X_train, y_train - התמונות והלייבלים שלהן
    """
    return load_images_with_labels(folder, transform=transform)  # פונקציה חיצונית שטוענת תמונות ולייבלים

# הכפלת התמונות עם סיבובים
augment_images_with_rotation(image_folder_for_train, augmented_folder, num_rotations=2, angle=10)

# טעינת התמונות החדשות
X_train, y_train = load_augmented_images(augmented_folder, transform)

# X_train, y_train = load_images_with_labels(image_folder_for_train, transform=transform)
X_val, y_val = load_images_with_labels(image_folder_for_validation, transform=transform)

#@Test
print(f"Loaded {len(X_train)} training samples.")
for i in range(5): 
    print(f"Image {i+1}: Label = {y_train[i]}") 
    
# טוענים את התמונות
X_test, test_names = load_test_images(image_folder_for_testing)

#@Test
print(f"Loaded {len(X_test)} testing samples.")
for i in range(10): 
    print(f"Image {i+1}: name = {test_names[i]}") 
