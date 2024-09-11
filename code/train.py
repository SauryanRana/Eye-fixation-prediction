import torch
import load_data
import model_ep
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToPILImage, Compose, ToTensor, Grayscale


# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

best_params = {
    'lr': 0.0001,
    'weight_decay': 1e-4,
    'dropout': 0.3,
    'num_epochs': 1
}

batch_size = 64
k_folds = 2

# Define transformations without normalization
simple_transform = Compose([
    ToTensor()
])

fixation_transform = Compose([
    Grayscale(),
    ToTensor()
])

def is_folder_empty(folder):
    return not any(os.scandir(folder))

# Path to save the predicted images
prediction_folder = os.path.join('prediction_folder', 'cv2_testing_data/fixations/test')
if not os.path.exists(prediction_folder):
    os.makedirs(prediction_folder)

###################################################################################################
# Generate predictions and save them
####################################################################################################

def generate_predictions(model, test_loader, prediction_folder, original_filenames):
    model.eval()
    to_pil = ToPILImage()

    predicted_fixations = []
    total_images_processed = 0

    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            if isinstance(inputs, list):
                inputs = inputs[0]  # Extract the tensor from the list if necessary
            inputs = inputs.to(device)
            outputs = model(inputs)
            batch_size = outputs.size(0)
            total_images_processed += batch_size

            # Save each predicted fixation
            for j in range(batch_size):
                if (total_images_processed - batch_size + j) < len(original_filenames):
                    # Convert the output to a PIL image after clamping values between 0 and 1
                    predicted_fixation = to_pil(outputs[j].cpu().clamp(0, 1))
                    pred_filename = original_filenames[total_images_processed - batch_size + j]
                    predicted_fixation.save(os.path.join(prediction_folder, pred_filename))
                    predicted_fixations.append(outputs[j].cpu())
                else:
                    print(f"Warning: Skipping saving prediction for index {total_images_processed - batch_size + j} due to index out of range")

    if predicted_fixations:  # Ensuring that the list is not empty
        predicted_fixations = torch.stack(predicted_fixations)
        print(f"Processed batch {i+1}/{len(test_loader)}, total images processed: {total_images_processed}")
    else:
        print("No predictions generated.")
        predicted_fixations = None

    return predicted_fixations


###################################################################################################
# Loading data
####################################################################################################

def load_images_from_dir(directory, transform):
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path).convert('RGB')
            images.append(transform(img))
    return torch.stack(images)

def load_fixations_from_dir(directory, transform):
    fixations = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            fix_path = os.path.join(directory, filename)
            fix = Image.open(fix_path).convert('L')
            fixations.append(transform(fix))
    return torch.stack(fixations)

###################################################################################################
# Main Function
####################################################################################################

def dataset_to_tensors(dataset):
    images = []
    fixations = []
    for item in dataset:
        if isinstance(item, tuple) and len(item) == 2:
            img, fix = item
        else:
            img, fix = item[0], item[1]
        images.append(img)
        fixations.append(fix)
    return torch.stack(images), torch.stack(fixations)

def main():
    # Load training and validation data
    data = load_data.data_download()

    # Combine training and validation data for cross-validation
    train_images, train_fixations = data['train']['image'], data['train']['fixations']
    val_images, val_fixations = data['validation']['image'], data['validation']['fixations']
    
    # Combine all data
    all_images = torch.cat((train_images, val_images), dim=0)
    all_fixations = torch.cat((train_fixations, val_fixations), dim=0)

    # Prepare data for K-Fold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        print(f'Fold {fold + 1}/{k_folds}')

        # Create train and validation subsets for the current fold
        train_images_fold, val_images_fold = all_images[train_idx], all_images[val_idx]
        train_fixations_fold, val_fixations_fold = all_fixations[train_idx], all_fixations[val_idx]

        # Create data loaders for train and validation subsets
        train_loader = DataLoader(TensorDataset(train_images_fold, train_fixations_fold), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_images_fold, val_fixations_fold), batch_size=batch_size, shuffle=False)

        # Initialize model with best hyperparameters
        model = model_ep.EyeFixationPredictor(dropout=best_params['dropout'])
        
        # Load the cached model if it exists
        cache_path = 'path_to_cached_model.pt'
        if os.path.exists(cache_path):
            model.load_state_dict(torch.load(cache_path))
            print(f"Loaded cached model from {cache_path}")
        
        model = model.to(device)

        # Initialize optimizer with best hyperparameters
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
        criterion = torch.nn.MSELoss()

        # Train the model
        model = model_ep.train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=best_params['num_epochs'])
        
        # Evaluate the model
        val_rmse, val_accuracy, val_roc_auc = model_ep.evaluate_model(model, val_loader, criterion)
        print(f'Fold {fold + 1}, Validation RMSE: {val_rmse}, Accuracy: {val_accuracy}, ROC-AUC: {val_roc_auc}')
        fold_results.append((val_rmse, val_accuracy, val_roc_auc))

        # Save the trained model
        torch.save(model.state_dict(), cache_path)

    # Calculate average performance across all folds
    avg_val_rmse = np.mean([result[0] for result in fold_results])
    avg_val_accuracy = np.mean([result[1] for result in fold_results])
    avg_val_roc_auc = np.mean([result[2] for result in fold_results])
    print(f'Average Validation RMSE: {avg_val_rmse}, Average Accuracy: {avg_val_accuracy}, Average ROC-AUC: {avg_val_roc_auc}')
    
    # Load test data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    test_images_dir = os.path.join(data_dir, 'cv2_testing_data/images/test')
    original_filenames = sorted(os.listdir(test_images_dir))
    
    # Load test images
    test_images = load_images_from_dir(test_images_dir, simple_transform)
    print(f"Loaded test images: {test_images.shape}")

    test_dataset = TensorDataset(test_images)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Generate predictions for the test set and save them
    predicted_fixations = generate_predictions(model, test_dataloader, prediction_folder, original_filenames)
    
    # Ensure predicted fixations are loaded correctly
    print(f"Generated predicted fixations: {predicted_fixations.shape}")

    # Load predicted fixations
    test_fixations = load_fixations_from_dir(prediction_folder, fixation_transform)
    print(f"Loaded test fixations: {test_fixations.shape}")

    # Ensure the number of predicted fixations matches the number of test images
    if test_fixations.size(0) != test_images.size(0):
        raise ValueError("Number of predicted fixations does not match number of test images")

    # Create DataLoader for test data with predicted fixations as targets
    test_loader = DataLoader(TensorDataset(test_images, test_fixations), batch_size=batch_size, shuffle=False)

    # Evaluate the model on the test set using the predicted fixations
    test_rmse, test_accuracy, test_roc_auc = model_ep.evaluate_model(model, test_loader, criterion, is_test=True, targets_present=True)

    print(f'Test RMSE: {test_rmse}, Accuracy: {test_accuracy}, ROC-AUC: {test_roc_auc}')


if __name__ == "__main__":
    main()

