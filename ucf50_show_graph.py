import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import os

# Define the folder path
results_folder = 'ucf50_results'

# Create the folder if it does not exist
os.makedirs(results_folder, exist_ok=True)
# File paths
training_log_path = os.path.join(results_folder, 'training_log.txt')
test_log_path = os.path.join(results_folder, 'test_log.txt')

# Initialize lists to hold the data
epochs = []
loss = []
val_loss = []
mae = []
val_mae = []
# output_dir = "output_plots1"
# os.makedirs(output_dir, exist_ok=True)
# Read the log file and parse the data
with open(training_log_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        parts = line.strip().split(',')
        epoch_part = parts[0].strip().split(' ')[1]
        epoch = int(epoch_part.split('/')[0][1:])
        loss_val = float(parts[1].strip().split(' ')[1])
        mae_val = float(parts[2].strip().split(' ')[1])
        val_loss_val = float(parts[3].strip().split(' ')[2])
        val_mae_val = float(parts[4].strip().split(' ')[2])
        
        epochs.append(epoch)
        loss.append(loss_val)
        mae.append(mae_val)
        val_loss.append(val_loss_val)
        val_mae.append(val_mae_val)

# Plot Loss vs Epoch
plt1.figure(figsize=(12, 6))

plt1.subplot(1, 2, 1)
plt1.plot(epochs, loss, label='Loss')
plt1.plot(epochs, val_loss, label='Val Loss')
plt1.xlabel('Epoch')
plt1.ylabel('Loss')
plt1.legend()
plt1.title('Loss vs Epoch')
plt1.savefig(os.path.join(results_folder, "Training_and_Validation_Loss_400.png"))
plt1.close()

# Plot MAE vs Epoch
plt2.figure(figsize=(12, 6))
plt2.subplot(1, 2, 1)
plt2.plot(epochs, mae, label='MAE')
plt2.plot(epochs, val_mae, label='Val MAE')
plt2.xlabel('Epoch')
plt2.ylabel('MAE')
plt2.legend()
plt2.title('MAE vs Epoch')
plt2.savefig(os.path.join(results_folder, "Training_and_Validation_mae_400.png"))
#plt.tight_layout()
plt2.close()
