import neural_network as nn
import matplotlib.pyplot as plt
import numpy as np


X_train, y_train, X_val, y_val = nn.load_data_large()


def avg_cross_entropy_vs_hidden_units():

    # Define hyperparameters
    num_epochs = 50
    hidden_units_list = [5, 20, 50, 100, 200]
    init_rand = True  # Initialize weights randomly
    learning_rate = 0.01

    # Initialize lists to store average training and validation cross-entropy
    train_losses = []
    valid_losses = []

    # Iterate over different numbers of hidden units
    for num_hidden in hidden_units_list:
        # Train the neural network
        loss_per_epoch_train, loss_per_epoch_val, _, _, _, _ = nn.train_and_valid(X_train, y_train, X_val, y_val, num_epochs, num_hidden, init_rand, learning_rate)
        
        # Calculate average training and validation cross-entropy
        avg_train_loss = np.mean(loss_per_epoch_train)
        avg_valid_loss = np.mean(loss_per_epoch_val)
        
        # Append to lists
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(hidden_units_list, train_losses, label='Average Training Cross-Entropy')
    plt.plot(hidden_units_list, valid_losses, label='Average Validation Cross-Entropy')
    plt.xlabel('Number of Hidden Units')
    plt.ylabel('Average Cross-Entropy')
    plt.title('Average Training and Validation Cross-Entropy vs Number of Hidden Units')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'entropy_vs_hidden_units.png')
    plt.show()

def plot_entropy_for_learning_rates():
    # Define hyperparameters
    num_epochs = 50
    learning_rates = [0.1, 0.01, 0.001]
    hidden_units = 50
    init_rand = True  # Initialize weights randomly

    # Iterate over different learning rates
    for lr in learning_rates:
        # Train the neural network
        print(lr)
        loss_per_epoch_train, loss_per_epoch_val, _, _, _, _ = nn.train_and_valid(X_train, y_train, X_val, y_val, num_epochs, hidden_units, init_rand, lr)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), loss_per_epoch_train, label='Average Training Cross-Entropy')
        plt.plot(range(1, num_epochs + 1), loss_per_epoch_val, label='Average Validation Cross-Entropy')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Cross-Entropy')
        plt.title(f'Learning Rate = {lr}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'learning_rate_{lr}.png')
        plt.show()

avg_cross_entropy_vs_hidden_units()
plot_entropy_for_learning_rates()
