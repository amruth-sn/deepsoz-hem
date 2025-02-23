import json
import numpy as np
import os
import matplotlib.pyplot as plt

final_path = '/projectnb/seizuredet/Sz-challenge/repeated_crossval/baselines/'

for m in ['txlstm']:
    # Initialize lists to store losses for all iterations and folds
    all_inner_train_losses = []
    all_inner_val_losses = []
    all_outer_train_losses = []
    all_outer_test_losses = []

    
    path = f'/projectnb/seizuredet/Sz-challenge/repeated_crossval/baselines/{m}/outer_{0}/'
    print(path)
    # Check if files exist and are non-empty
    try:
        file_size = os.path.getsize(path + 'inner_0/inner_0.json')
        if file_size == 0:
            continue

        file_size = os.path.getsize(path + f'outer_{0}.json')
        if file_size == 0:
            continue
    except FileNotFoundError:
        continue
    
    # Load inner losses
    with open(path + 'inner_0/inner_0.json', 'r') as f:
        inner_data = json.load(f)

    # Collect inner fold losses
    all_inner_train_losses.append(inner_data['train_losses'])
    all_inner_val_losses.append(inner_data['val_losses'])

    # Load outer losses
    with open(path + f'outer_{0}.json', 'r') as f:
        outer_data = json.load(f)

    # Collect outer fold losses
    all_outer_train_losses.append(outer_data['train_losses'])
    all_outer_test_losses.append(outer_data['test_losses'])

    # Convert to numpy arrays and calculate the mean across all iterations and folds
    all_inner_train_losses = np.array(all_inner_train_losses)
    all_inner_val_losses = np.array(all_inner_val_losses)
    all_outer_train_losses = np.array(all_outer_train_losses)
    all_outer_test_losses = np.array(all_outer_test_losses)

    mean_inner_train_losses = np.mean(all_inner_train_losses, axis=0)
    mean_inner_val_losses = np.mean(all_inner_val_losses, axis=0)
    mean_outer_train_losses = np.mean(all_outer_train_losses, axis=0)
    mean_outer_test_losses = np.mean(all_outer_test_losses, axis=0)

# Assuming mean_outer_train_losses, mean_outer_test_losses, std_outer_train_losses, std_outer_test_losses are defined

# Calculate the standard deviation (example data, replace with actual calculations)
    std_outer_train_losses = np.std(mean_outer_train_losses, axis=0)
    std_outer_test_losses = np.std(mean_outer_test_losses, axis=0)

    # Plot the mean inner losses
    m = 'DeepSOZ'

    # Plot the mean outer losses with standard deviation
    plt.figure(figsize=(10, 5))
    epochs = range(len(mean_outer_train_losses))

    # Plot mean training loss with standard deviation
    plt.plot(epochs, mean_outer_train_losses, label='Mean Training Loss')
    plt.fill_between(epochs, mean_outer_train_losses - std_outer_train_losses, mean_outer_train_losses + std_outer_train_losses, color='blue', alpha=0.2)
    plt.fill_between(epochs, mean_outer_train_losses - 2*std_outer_train_losses, mean_outer_train_losses + 2*std_outer_train_losses, color='blue', alpha=0.025)

    # Plot mean testing loss with standard deviation
    plt.plot(epochs, mean_outer_test_losses, label='Mean Testing Loss')
    plt.fill_between(epochs, mean_outer_test_losses - std_outer_test_losses, mean_outer_test_losses + std_outer_test_losses, color='orange', alpha=0.2)
    plt.fill_between(epochs, mean_outer_test_losses - 2*std_outer_test_losses, mean_outer_test_losses + 2*std_outer_test_losses, color='orange', alpha=0.025)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.title(f'Mean Outer Fold Training and Testing Loss for {m} with lr={5e-05}')
    plt.legend()
    plt.savefig(f'/projectnb/seizuredet/Sz-challenge/repeated_crossval/baselines/{m}_mean_outer_losses_with_std.png')
    plt.close()
