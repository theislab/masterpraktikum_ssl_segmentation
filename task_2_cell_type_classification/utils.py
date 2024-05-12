import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def df_to_tensor_dataset(features_df, labels_df):
    """
    Converts features and labels DataFrames into a PyTorch TensorDataset.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing the features.
        labels_df (pd.DataFrame): DataFrame containing the labels.
        
    Returns:
        TensorDataset: A PyTorch TensorDataset containing features and labels.
    """
    # Convert the DataFrames to NumPy arrays
    features_np = features_df.to_numpy(dtype=float)
    labels_np = labels_df.to_numpy(dtype=float).squeeze()  # Squeeze is used to ensure labels are in the correct shape
    
    # Convert the NumPy arrays to PyTorch tensors
    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_np, dtype=torch.float32)
    
    # Create a TensorDataset
    dataset = TensorDataset(features_tensor, labels_tensor)
    
    return dataset

def create_data_loaders(dataset, batch_size=64):
    # Calculate sizes for splits
    total_size = len(dataset)
    test_val_size = int(0.1 * total_size)  # 10% for test, 10% for validation
    train_size = total_size - 2 * test_val_size

    # Shuffle the dataset indices
    indices = torch.randperm(total_size).tolist()

    # Create data subsets for train, validation, and test
    train_indices, val_indices, test_indices = indices[:train_size], indices[train_size:train_size+test_val_size], indices[train_size+test_val_size:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(train_subset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_subset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    }

    return dataloaders

def evaluate_model(model, test_loader, device):
    """
    Evaluates the trained model on the test set and reports various performance scores.
    
    Args:
        model (torch.nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Device to perform computation on ('cuda' or 'cpu').
    """
    model.eval()
    model.to(device)
    true_labels = []
    predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.view(-1).cpu().numpy())
            true_labels.extend(labels.view(-1).cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')

    print("Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    metrics = {}
    metrics['accuracy'] = accuracy
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    return metrics


def plot_label_distribution(labels_df):
    """
    Plots the distribution of labels.
    
    Args:
        labels_df (pd.DataFrame): DataFrame containing the labels.
    """
    label_counts = labels_df.iloc[:, 0].value_counts()  # Assuming the labels are in the first column
    plt.figure(figsize=(10, 6))
    label_counts.plot(kind='bar')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title('Distribution of Labels')
    plt.grid(True)
    plt.show()



def plot_losses(epoch_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses['train'], label='Train Loss')
    plt.plot(epoch_losses['val'], label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def random_baseline(test_loader, num_classes):
    """
    Evaluates a random baseline using labels from a DataLoader.
    
    Args:
    - test_loader (DataLoader): DataLoader for the test set containing features and labels.
    - num_classes (int): The number of distinct classes.
    
    Returns:
    - None: Prints performance metrics.
    """
    true_labels = []
    random_labels = []

    # Gather all true labels and generate random labels
    for _, labels in test_loader:
        true_labels.extend(labels.numpy())
        random_labels.extend(np.random.randint(0, num_classes, size=len(labels)))

    # Calculate metrics
    accuracy = accuracy_score(true_labels, random_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, random_labels, average='weighted')
    
    print("Random Baseline Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    metrics = {}
    metrics['accuracy'] = accuracy
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    return metrics

def categorical_baseline(test_loader, train_loader, num_classes):
    """
    Assigns labels based on the relative frequency of classes in the training dataset and evaluates performance.
    
    Args:
    - test_loader (DataLoader): DataLoader for the test set containing features and labels.
    - train_loader (DataLoader): DataLoader for the training set containing features and labels.
    - num_classes (int): The number of distinct classes.
    
    Returns:
    - None: Prints performance metrics.
    """
    train_labels = []
    true_labels = []
    frequency_labels = []

    # Collect all labels from the training set
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())
    
    # Calculate class frequencies
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_probabilities = class_counts / class_counts.sum()

    # Assign labels based on calculated probabilities and collect true labels
    for _, labels in test_loader:
        true_labels.extend(labels.numpy())
        frequency_labels.extend(np.random.choice(np.arange(num_classes), size=len(labels), p=class_probabilities))

    # Calculate metrics
    accuracy = accuracy_score(true_labels, frequency_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, frequency_labels, average='weighted')
    
    print("Frequency-based Baseline Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    metrics = {}
    metrics['accuracy'] = accuracy
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    return metrics

def plot_metrics(metrics_dict):
    """
    Plots the metrics from a dictionary of dictionaries in a single plot.

    Args:
        metrics_dict_of_dicts (dict): Dictionary of dictionaries containing metrics, where each key corresponds to a model name.
    """
    num_models = len(metrics_dict)
    metrics = list(next(iter(metrics_dict.values())).keys())

    fig, axes = plt.subplots(1, num_models, figsize=(16, 6), sharey=True)

    for i, (model_name, metrics_dict) in enumerate(metrics_dict.items()):
        ax = axes[i]
        values = [metrics_dict[metric] for metric in metrics]
        ax.bar(metrics, values)
        ax.set_title(model_name)
        ax.set_ylabel('Value')

    plt.tight_layout()
    plt.show()
