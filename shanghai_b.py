import os
import random
import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torchvision import models
import torch.nn.functional as F

# from google.colab import drive
from torchvision import models

# Mount Google Drive
# drive.mount('/content/drive')

## only one time execute this code


# import os
# import shutil
# from sklearn.model_selection import train_test_split

# # Define paths
# dataset_folder = "/media/iiit/Mtech/Vishal/ShanghaiTech-20240627T060555Z-001/ShanghaiTech/part_B"
# train_folder = os.path.join(dataset_folder, "train_data")
# val_folder = os.path.join(dataset_folder, "val_data")

# Create val_folder if it does not exist
# os.makedirs(val_folder, exist_ok=True)
# os.makedirs(os.path.join(val_folder, "images"), exist_ok=True)
# os.makedirs(os.path.join(val_folder, "ground-truth"), exist_ok=True)

# List all images and ground-truth files in the train_folder
# image_files = [f for f in os.listdir(os.path.join(train_folder, "images")) if f.endswith(".jpg")]
# gt_files = ["GT_" + f.split(".")[0] + ".mat" for f in image_files]

# Split the dataset
# train_images, val_images, train_gt, val_gt = train_test_split(
#   image_files, gt_files, test_size=0.2, random_state=42
# )

# Move validation files to val_folder
# for img_file, gt_file in zip(val_images, val_gt):
#    shutil.move(os.path.join(train_folder, "images", img_file), os.path.join(val_folder, "images", img_file))
#    shutil.move(os.path.join(train_folder, "ground-truth", gt_file), os.path.join(val_folder, "ground-truth", gt_file))

# print(f"Moved {len(val_images)} images and {len(val_gt)} ground-truth files to validation folder.")
# print(f"Number of images left in training folder: {len(train_images)}")


# Function to load all images and corresponding counts
def load_data(folder_path):
    image_files = [
        f for f in os.listdir(os.path.join(folder_path, "images")) if f.endswith(".jpg")
    ]

    images = []
    counts = []
    for file_name in image_files:
        # Load image
        image_path = os.path.join(folder_path, "images", file_name)
        image = Image.open(image_path).convert("RGB")
        images.append(image)

        # Load count from .mat file
        annotation_file = "GT_" + file_name.split(".")[0] + ".mat"
        annotation_path = os.path.join(folder_path, "ground-truth", annotation_file)
        mat_data = loadmat(annotation_path)
        count = mat_data["image_info"][0, 0][0, 0][1][
            0, 0
        ]  # Extracting count from the .mat structure
        counts.append(count)

    return images, counts


# Define dataset folder paths
dataset_folder = (
    "/media/iiit/Mtech/Vishal/ShanghaiTech-20240627T060555Z-001/ShanghaiTech/part_B"
)
train_folder = os.path.join(dataset_folder, "train_data")
val_folder = os.path.join(dataset_folder, "val_data")
test_folder = os.path.join(dataset_folder, "test_data")

# Load train, validation, and test datasets
train_images, train_counts = load_data(train_folder)
val_images, val_counts = load_data(val_folder)
test_images, test_counts = load_data(test_folder)


# Dataset and DataLoader
class CrowdDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [
            f
            for f in os.listdir(os.path.join(folder_path, "images"))
            if f.endswith(".jpg")
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        image_path = os.path.join(self.folder_path, "images", file_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        annotation_file = "GT_" + file_name.split(".")[0] + ".mat"
        annotation_path = os.path.join(
            self.folder_path, "ground-truth", annotation_file
        )
        mat_data = loadmat(annotation_path)
        count = mat_data["image_info"][0, 0][0, 0][1][
            0, 0
        ]  # Extracting count from the .mat structure

        # Explicitly cast count to a float
        count = float(count)

        return image, count


transform = transforms.Compose(
    [
        transforms.Resize((576, 768)),
        transforms.ToTensor(),
    ]
)

train_dataset = CrowdDataset(train_folder, transform=transform)
val_dataset = CrowdDataset(val_folder, transform=transform)
test_dataset = CrowdDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# Model Definition
class FPANet(nn.Module):
    def __init__(self):
        super(FPANet, self).__init__()
        backbone = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.conv1 = nn.Conv2d(2048, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=1)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.conv_final = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(
            256 * 144 * 192, 1
        )  # Adjust according to the feature map size

        # PSA and LCA modules
        self.psa = PyramidSpatialAttention()
        self.lca = LightWeightChannelAttention(256)

    def forward(self, x):
        c3 = self.features[:6](x)
        c4 = self.features[6:7](c3)
        c5 = self.features[7:](c4)

        p5 = self.conv1(c5)
        p4 = self.upsample(p5) + self.conv2(c4)
        p3 = self.upsample(p4) + self.conv3(c3)
        p3 = self.upsample(p3)

        # Apply PSA and LCA
        p3 = self.psa(p3)
        p3 = self.lca(p3)

        out = self.conv_final(p3)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc(out)
        return out


class PyramidSpatialAttention(nn.Module):
    def __init__(self):
        super(PyramidSpatialAttention, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool4 = nn.AdaptiveAvgPool2d(4)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(256)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        size = x.size()
        pool1 = self.pool1(x)
        pool2 = self.pool2(x)
        pool4 = self.pool4(x)

        pool1 = F.interpolate(
            self.relu(self.bn(self.conv1(pool1))),
            size[2:],
            mode="bilinear",
            align_corners=False,
        )
        pool2 = F.interpolate(
            self.relu(self.bn(self.conv2(pool2))),
            size[2:],
            mode="bilinear",
            align_corners=False,
        )
        pool4 = F.interpolate(
            self.relu(self.bn(self.conv4(pool4))),
            size[2:],
            mode="bilinear",
            align_corners=False,
        )

        out = pool1 + pool2 + pool4
        out = self.sigmoid(out)
        return x * out


# LightWeightChannelAttention remains the same as provided earlier.


class LightWeightChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(LightWeightChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = torch.mean(x, dim=(2, 3), keepdim=True)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return x * out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FPANet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the folder path
results_folder = "shanghai_b_results"

# Create the folder if it does not exist
os.makedirs(results_folder, exist_ok=True)

# File paths
training_log_path = os.path.join(results_folder, "training_log.txt")
test_log_path = os.path.join(results_folder, "test_log.txt")
model_save_path = os.path.join(results_folder, "trained_model.pth")


# Training Loop
def train_model(model, train_loader, val_loader, epochs=400, batch_size=4):
    num_epochs = epochs
    history = {"loss": [], "mae": [], "val_loss": [], "val_mae": []}
    log_file = open(training_log_path, "w")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        for images, counts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, counts = images.to(device), counts.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, counts)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_mae += mean_absolute_error(
                counts.cpu().detach().numpy(), outputs.cpu().detach().numpy()
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_mae = running_mae / len(train_loader.dataset)
        history["loss"].append(epoch_loss)
        history["mae"].append(epoch_mae)

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for images, counts in val_loader:
                images, counts = images.to(device), counts.to(device).float().view(
                    -1, 1
                )
                outputs = model(images)
                loss = criterion(outputs, counts)
                val_loss += loss.item()
                val_mae += mean_absolute_error(
                    counts.cpu().detach().numpy(), outputs.cpu().detach().numpy()
                )

        val_loss /= len(val_loader)
        val_mae /= len(val_loader.dataset)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}, MAE: {epoch_mae}, Val Loss: {val_loss}, Val MAE: {val_mae}"
        )
        log_file.write(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}, MAE: {epoch_mae}, Val Loss: {val_loss}, Val MAE: {val_mae}\n"
        )

    log_file.close()
    return history


# Plot Training History
# def plot_history(history):
#     # Create a directory to save the plots
#     output_dir = "output_plots"
#     os.makedirs(output_dir, exist_ok=True)

#     epochs = range(1, len(history["loss"]) + 1)

#     plt.figure(figsize=(12, 6))

#     # Plotting Training and Validation Loss
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, history["loss"], label="Training Loss")
#     plt.plot(epochs, history["val_loss"], label="Validation Loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.title("Training and Validation Loss")
#     plt.savefig(os.path.join(output_dir, "Training_and_Validation_Loss.png"))
#     plt.close()

#     # Plotting Training and Validation MAE
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, history["mae"], label="Training MAE")
#     plt.plot(epochs, history["val_mae"], label="Validation MAE")
#     plt.xlabel("Epochs")
#     plt.ylabel("MAE")
#     plt.legend()
#     plt.title("Training and Validation MAE")
#     plt.savefig(os.path.join(output_dir, "Training_and_Validation_MAE.png"))
#     plt.close()


# Main Script
# Display summary of loaded data
print("Train Data:")
print(f"Number of train images: {len(train_images)}")
print(f"Number of train counts: {len(train_counts)}")

print("\nValidation Data:")
print(f"Number of validation images: {len(val_images)}")
print(f"Number of validation counts: {len(val_counts)}")

print("\nTest Data:")
print(f"Number of test images: {len(test_images)}")
print(f"Number of test counts: {len(test_counts)}")

# Train the model
history = train_model(model, train_loader, val_loader, epochs=400, batch_size=4)

# Plot the training history
# plot_history(history)

# Evaluation and Prediction


# model.eval()
# all_predictions = []
# all_counts = []

# with torch.no_grad():
#     for images, counts in test_loader:
#         images, counts = images.to(device), counts.to(device).float().view(-1, 1)
#         outputs = model(images)
#         outputs = outputs.view(-1).cpu().numpy()
#         counts = counts.view(-1).cpu().numpy()
#         all_predictions.extend(outputs)
#         all_counts.extend(counts)

# r2 = r2_score(all_counts, all_predictions)
# rmse = mean_squared_error(all_counts, all_predictions, squared=False)
# print(f'R² Score: {r2}')
# print(f'RMSE: {rmse}')

# Evaluate the model on test data
model.eval()
test_loss = 0.0
test_mae = 0.0
all_predictions = []
all_counts = []

with torch.no_grad():
    for images, counts in test_loader:
        images, counts = images.to(device), counts.to(device).float().view(-1, 1)
        outputs = model(images)
        loss = criterion(outputs, counts)
        test_loss += loss.item()
        test_mae += torch.abs(outputs - counts).sum().item()

        outputs = outputs.view(-1).cpu().numpy()
        counts = counts.view(-1).cpu().numpy()
        all_predictions.extend(outputs)
        all_counts.extend(counts)

# Calculate average test loss and MAE
test_loss /= len(test_loader)
test_mae /= len(test_loader.dataset)

print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")
log_file = open(test_log_path, "w")
log_file.write(f"Test Loss: {test_loss}, Test MAE: {test_mae}\n")


# Ensure test_predictions and test_counts are 1D
all_predictions = np.array(all_predictions).reshape(-1)
all_counts = np.array(all_counts).reshape(-1)

# Calculate R² and RMSE
r2 = r2_score(all_counts, all_predictions)
rmse = mean_squared_error(all_counts, all_predictions, squared=False)
print(f"R² Score: {r2}")
print(f"RMSE: {rmse}")
log_file.write(f"R² Score: {r2}, RMSE: {rmse}\n")
log_file.close()

# Save the final trained model
torch.save(model.state_dict(), model_save_path)
print(f"Trained model saved to {model_save_path}")
