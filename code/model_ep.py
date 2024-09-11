import torch
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from scipy.stats import norm

# ==================================================================================================
# Utility Functions
# ==================================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================================================================================================
# Define the EyeFixationPredictor model
# ==================================================================================================

# Function to create a 2D Gaussian kernel
def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Generates a 2D Gaussian kernel."""
    x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()  # Normalize to ensure the sum is 1
    gauss_2d = gauss[:, None] * gauss[None, :]
    return gauss_2d

class EyeFixationPredictor(torch.nn.Module):
    def __init__(self, device='cuda', dropout=0.3, kernel_size=5):
        super(EyeFixationPredictor, self).__init__()

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.kernel_size = kernel_size

        self.fcn = fcn_resnet101(weights=FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
        self.fcn.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))

        for param in self.fcn.backbone.parameters():
            param.requires_grad = False

        self.dropout = torch.nn.Dropout(dropout)
        gaussian_kernel = gaussian(kernel_size, 4)  
        self.kernel = torch.nn.Parameter(gaussian_kernel.repeat(1, 1, 1, 1), requires_grad=False)

        center_bias_density = np.load('center_bias_density.npy')
        center_bias_density /= center_bias_density.sum()
        self.center_bias = torch.nn.Parameter(torch.log(torch.from_numpy(center_bias_density)), requires_grad=False)

        self.smoothing_conv = torch.nn.Conv2d(1, 1, kernel_size, padding='same', groups=1, bias=False)
        self.smoothing_conv.weight = self.kernel

    def forward(self, x):
        x = x.to(self.device)
        raw_predictions = self.fcn(x)['out']
        
        conv = self.smoothing_conv(raw_predictions)
        return conv

# ==================================================================================================
# Training Function
# ==================================================================================================

def train_model(model, train_dataloader, valid_dataloader, optimizer, criterion, num_epochs):
    best_model_wts = model.state_dict()
    best_rmse = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_steps = len(train_dataloader)

        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if i % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_steps}], Loss: {loss.item()}')

        epoch_loss = running_loss / total_steps
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss}, LR: {optimizer.param_groups[0]["lr"]}')

        val_rmse, val_accuracy, val_roc_auc = evaluate_model(model, valid_dataloader, criterion)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation RMSE: {val_rmse}, Accuracy: {val_accuracy}, ROC-AUC: {val_roc_auc}')

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    return model

# ==================================================================================================
# Evaluation Function
# ==================================================================================================

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    return tensor * std + mean

def compute_rmse(true_values, predicted_values):
    return np.sqrt(mean_squared_error(true_values, predicted_values))

def evaluate_model(model, data_loader, criterion, is_test=False, targets_present=True):
    model.eval()
    total_loss = 0
    total_samples = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu())

            if len(batch) > 1 and targets_present:
                targets = batch[1].to(device)
                targets = targets[:, 0:1, :, :]  # Assuming the targets have a single channel
                all_targets.append(targets.cpu())

                loss = criterion(outputs, targets)
                total_loss += loss.item()
                total_samples += targets.size(0)

    if all_predictions:
        all_predictions = torch.cat(all_predictions)
    else:
        print("No predictions found.")
        all_predictions = torch.tensor([])

    if all_targets:
        all_targets = torch.cat(all_targets)
    else:
        if not is_test:
            print("No targets found.")
        all_targets = torch.tensor([])

    if total_samples > 0 and not is_test:
        print(f"All predictions shape: {all_predictions.shape}")
        if all_targets.size(0) > 0:
            print(f"All targets shape: {all_targets.shape}")

            # Denormalize predictions and targets
            all_predictions = denormalize(all_predictions, mean=[0.5], std=[0.5])
            all_targets = denormalize(all_targets, mean=[0.5], std=[0.5])

            true_values = all_targets.flatten().numpy()
            predicted_values = all_predictions.flatten().numpy()
            rmse = compute_rmse(true_values, predicted_values)
            pred_labels = (predicted_values > 0.5).astype(int)
            true_labels = (true_values > 0.5).astype(int)

            # Check distribution of true labels
            print(f"True labels distribution: {np.bincount(true_labels)}")

            accuracy = accuracy_score(true_labels, pred_labels)
            if len(np.unique(true_labels)) > 1:
                roc_auc = roc_auc_score(true_labels, predicted_values)
            else:
                print("Only one class present in y_true. ROC AUC score is not defined in that case.")
                roc_auc = float('nan')

            print(f'Total Samples: {total_samples}, All Predictions Shape: {all_predictions.shape}, All Targets Shape: {all_targets.shape}')
            print(f'RMSE: {rmse}, Accuracy: {accuracy}, ROC-AUC: {roc_auc}')
            
            return rmse, accuracy, roc_auc
        else:
            print("No valid targets available for RMSE and accuracy calculation.")
            return float('nan'), float('nan'), float('nan')

    elif is_test:
        if all_predictions.size(0) > 0:
            print(f"Test data predictions shape: {all_predictions.shape}")

            if all_targets.size(0) > 0:
                all_targets = denormalize(all_targets, mean=[0.5], std=[0.5])

            all_predictions = denormalize(all_predictions, mean=[0.5], std=[0.5])

            true_values = all_targets.flatten().numpy()
            predicted_values = all_predictions.flatten().numpy()
            rmse = compute_rmse(true_values, predicted_values)
            pred_labels = (predicted_values > 0.5).astype(int)
            true_labels = (true_values > 0.5).astype(int)

            # Check distribution of true labels
            print(f"True labels distribution: {np.bincount(true_labels)}")

            accuracy = accuracy_score(true_labels, pred_labels)
            if len(np.unique(true_labels)) > 1:
                roc_auc = roc_auc_score(true_labels, predicted_values)
            else:
                print("Only one class present in y_true. ROC AUC score is not defined in that case.")
                roc_auc = float('nan')

            print(f'Total Samples: {total_samples}, All Predictions Shape: {all_predictions.shape}, All Targets Shape: {all_targets.shape}')
            print(f'RMSE: {rmse}, Accuracy: {accuracy}, ROC-AUC: {roc_auc}')
            
            return rmse, accuracy, roc_auc
        else:
            print("No targets available for test data. Skipping RMSE and accuracy calculation.")
            return float('nan'), float('nan'), float('nan')

    else:
        print(f"Test data predictions shape: {all_predictions.shape}")
        return float('nan'), float('nan'), float('nan')
