import os
import random
import shutil
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import timm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Paths
original_infected_dir = "D:/pco/infected"
original_not_infected_dir = "D:/pco/not_infected"
augmented_dir = 'D:/pco/augmented_data3'
final_split_dir = 'D:/pco/final_split3'

# Augmentation transforms
infected_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.Resize((224, 224))
])

not_infected_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224))
])

# Create augmented dataset
for label in ['Infected', 'Not_Infected']:
    src_dir = original_infected_dir if label == 'Infected' else original_not_infected_dir
    save_dir = os.path.join(augmented_dir, label)
    os.makedirs(save_dir, exist_ok=True)

    for img_name in os.listdir(src_dir):
        img_path = os.path.join(src_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        if label == 'Infected':
            for i in range(3):
                aug_img = infected_transform(img)
                aug_img.save(os.path.join(save_dir, f'{os.path.splitext(img_name)[0]}_aug{i}.jpg'))
        else:
            aug_img = not_infected_transform(img)
            aug_img.save(os.path.join(save_dir, f'{os.path.splitext(img_name)[0]}_aug.jpg'))

print('Augmentation done!')

# Prepare for train/val/test split
all_data = {'Infected': [], 'Not_Infected': []}
for label in ['Infected', 'Not_Infected']:
    label_dir = os.path.join(augmented_dir, label)
    all_data[label] = [os.path.join(label_dir, f) for f in os.listdir(label_dir)]

splits = {}
for label, paths in all_data.items():
    random.shuffle(paths)
    n_total = len(paths)
    n_test = int(0.2 * n_total)
    n_val = int(0.1 * (n_total - n_test))

    splits[label] = {
        'test': paths[:n_test],
        'val': paths[n_test:n_test+n_val],
        'train': paths[n_test+n_val:]
    }

# Save split data
for split in ['train', 'val', 'test']:
    for label in ['Infected', 'Not_Infected']:
        split_dir = os.path.join(final_split_dir, split, label)
        os.makedirs(split_dir, exist_ok=True)

        for path in splits[label][split]:
            shutil.copy(path, os.path.join(split_dir, os.path.basename(path)))

print('Data split into train/val/test!')

# Dataloaders
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

train_dataset = datasets.ImageFolder(os.path.join(final_split_dir, 'train'), transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(os.path.join(final_split_dir, 'val'), transform=data_transforms['val'])
test_dataset = datasets.ImageFolder(os.path.join(final_split_dir, 'test'), transform=data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes
print('Dataloaders ready! Classes:', class_names)

# TNT Model (Transformer in Transformer)
model_tnt = timm.create_model('tnt_s_patch16_224', pretrained=True)

# Freeze backbone
for name, param in model_tnt.named_parameters():
    if 'head' not in name:
        param.requires_grad = False

# Replace head
model_tnt.head = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model_tnt.head.in_features, 2)
)

model_tnt.to(device)

# MLP Model
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(3 * 224 * 224, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.classifier(x)

model_mlp = SimpleMLP().to(device)

# TNT training function
def train_tnt_model():
    model = model_tnt
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

    best_val_acc = 0.0
    patience = 5
    no_improve_counter = 0

    for epoch in range(50):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val
        print(f"[TNT] Epoch {epoch+1}/50, Train Loss: {running_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'D:/pco/best_tnt_model.pth')
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            if no_improve_counter >= patience:
                print("[TNT] Early stopping triggered.")
                break

        scheduler.step(val_acc)

    print(" [TNT] Training finished!")

# Shared training function
def train_model(model, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

    best_val_acc = 0.0
    patience = 5
    no_improve_counter = 0

    for epoch in range(50):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val
        print(f"[{model_name}] Epoch {epoch+1}/50, Train Loss: {running_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'D:/pco/best_{model_name}.pth')
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            if no_improve_counter >= patience:
                print(f"[{model_name}] Early stopping triggered.")
                break

        scheduler.step(val_acc)

    print(f" [{model_name}] Training finished!")

# Train models
train_tnt_model()
train_model(model_mlp, "mlp_model")

# Evaluation function
def evaluate_model(model, model_name):
    model.load_state_dict(torch.load(f'D:/pco/best_{model_name}.pth'))
    model.eval()

    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_acc = 100 * correct_test / total_test
    print(f" [{model_name}] Test Accuracy: {test_acc:.2f}%")

# Evaluate both models
evaluate_model(model_tnt, "tnt_model")
evaluate_model(model_mlp, "mlp_model")
