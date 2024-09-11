import torch
import load_data
import model_ep
import matplotlib.pyplot as plt

# ==================================================================================================
#
#  Global variables
#
# ==================================================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

batch_size = 128
kernel_width = 5
kernel_height = 5


# ==================================================================================================
#
#  Main
#
# ==================================================================================================

data = load_data.data_download()
data_loaders = load_data.create_dataloaders(data, batch_size)
train_images, train_fixations = data['train']['image'].numpy(), data['train']['fixations'].numpy()

# Print shapes of the train_images and train_fixations
print(f'Shape of train_images: {train_images.shape}')
print(f'Shape of train_fixations: {train_fixations.shape}')

# Define the parameter grid
param_grid = {
    'lr': [0.01, 0.001, 0.0001],
    'weight_decay': [0, 1e-4, 1e-5],
    'dropout': [0.3, 0.4, 0.5],
    'num_epochs': [5, 10, 15]
}

# Perform manual grid search
best_params, best_score, results = model_ep.manual_grid_search(train_images, train_fixations, param_grid)

# Print the best parameters and score
print("Best parameters found: ", best_params)
print("Best cross-validation score: ", best_score)

# Function to abbreviate parameter sets for shorter labels
def abbreviate_params(params):
    abbreviations = {
        'lr': 'lr',
        'weight_decay': 'wd',
        'dropout': 'do',
        'num_epochs': 'ep'
    }
    return ', '.join([f"{abbreviations[key]}={value}" for key, value in params.items()])

# Generate shorter labels for the x-axis
short_labels = [abbreviate_params(result[0]) for result in results]

# Plot the hyperparameter tuning results
params, scores = zip(*results)
scores = [-score for score in scores]  
plt.figure(figsize=(12, 6))
plt.plot(range(len(scores)), scores, marker='o')
plt.xticks(range(len(scores)), short_labels, rotation=45, ha='right')
plt.xlabel('Parameter Set')
plt.ylabel('Mean Squared Error')
plt.title('Hyperparameter Tuning Results')
plt.grid(True)
plt.tight_layout()
plt.savefig('hyperparameter_tuning_results.png')
plt.show()