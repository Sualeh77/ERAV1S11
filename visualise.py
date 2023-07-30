import matplotlib.pyplot as plt
import math
from main import train_losses, train_acc
from main import test_losses, test_acc
import torch
import itertools
import numpy as np
import random
from utils import image_denormalizer, get_GradCAM_heatmap

#########################################################################################################################################################

def print_samples(loader, class_map, count=16):
    """
    Print samples input images

    Args:
        loader (DataLoader): dataloader for training data
        count (int, optional): Number of samples to print. Defaults to 16.
    """
    # Print Random Samples
    if not count % 8 == 0:
        return
    label_mapping = {idx:cls for cls, idx in class_map.items()}
    fig = plt.figure(figsize=(15, 5))
    for imgs, labels in loader:
        for i in range(count):
            ax = fig.add_subplot(int(count/8), 8, i + 1, xticks=[], yticks=[])
            ax.set_title(label_mapping[labels[i].item()])
            plt.imshow(imgs[i].numpy().transpose(1, 2, 0))
        break

#########################################################################################################################################################

def print_class_scale(loader, class_map):
    """
    Print Dataset Class scale

    Args:
        loader (DataLoader): Loader instance for dataset
        class_map (dict): mapping for class names
    """
    labels_count = {k: v for k, v in zip(
        range(0, len(class_map)), [0]*len(class_map))}
    for _, labels in loader:
        for label in labels:
            labels_count[label.item()] += 1

    labels = list(class_map.keys())
    values = list(labels_count.values())

    plt.figure(figsize=(15, 5))

    # creating the bar plot
    plt.bar(labels, values, width=0.5)
    plt.legend(labels=['Samples Per Class'])
    for l in range(len(labels)):
        plt.annotate(values[l], (-0.15 + l, values[l] + 50))
    plt.xticks(rotation=45)
    plt.xlabel("Classes")
    plt.ylabel("Class Count")
    plt.title("Classes Count")
    plt.show()

#########################################################################################################################################################

def draw_graphs():
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    
#########################################################################################################################################################

def plot_network_performance(epochs, schedule, train_loss, valid_loss, train_correct, valid_correct):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(epochs), schedule, 'r', label='One Cycle LR')
    plt.title('Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(epochs), train_loss, 'g', label='Training loss')
    plt.plot(range(epochs), valid_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(epochs), train_correct, 'g', label='Training Accuracy')
    plt.plot(range(epochs), valid_correct, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

#########################################################################################################################################################

def plot_incorrect_predictions(predictions, class_map, count=10):
    """
    Plot Incorrect predictions

    Args:
        predictions (list): List of all incorrect predictions
        class_map (dict): Lable mapping
        count (int, optional): Number of samples to print, multiple of 5. Defaults to 10.
    """
    print(f'Total Incorrect Predictions {len(predictions)}')

    if not count % 5 == 0:
        print("Count should be multiple of 10")
        return

    classes = list(class_map.values())

    fig = plt.figure(figsize=(10, 5))
    for i, (d, t, p, o) in enumerate(predictions):
        ax = fig.add_subplot(int(count/5), 5, i + 1, xticks=[], yticks=[])
        ax.set_title(f'{classes[t.item()]}/{classes[p.item()]}')
        plt.imshow(d.cpu().numpy().transpose(1, 2, 0))
        if i+1 == 5*(count/5):
            break

#########################################################################################################################################################

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap = plt.cm.Blues):
    """
    Plot Confusion Matrix

    Args:
        cm (tensor): Confusion Matrix
        classes (list): Class lables
        normalize (bool, optional): Enable/Disable Normalization. Defaults to False.
        title (str, optional): Title for plot. Defaults to 'Confusion matrix'.
        cmap (str, optional): Colour Map. Defaults to plt.cm.Blues.
    """
    if normalize:
        cm = cm.type(torch.float32) / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#########################################################################################################################################################

def plot_model_comparison(trainers, epochs):
    """
    Plot comparison charts for models

    Args:
        trainers (list): List or all trainers for different experiments
        epochs (int): Number or training loops
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(epochs),
            trainers[0].list_valid_loss, 'b', label='BN + L1 loss')
    plt.plot(range(epochs), trainers[1].list_valid_loss, 'r', label='GN loss')
    plt.plot(range(epochs), trainers[2].list_valid_loss, 'm', label='LN loss')
    plt.title('Validation losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs),
            trainers[0].list_valid_correct, 'b', label='BN + L1 Accuracy')
    plt.plot(range(epochs),
            trainers[1].list_valid_correct, 'r', label='GN Accuracy')
    plt.plot(range(epochs),
            trainers[2].list_valid_correct, 'm', label='LN Accuracy')
    plt.title('Validation Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

#########################################################################################################################################################

def plot_incorrect_pred_GradCAM(model, target_layers:list, dataset_std:tuple, dataset_mean:tuple, incorrect_predictions:list, labels_mapping:dict, use_cuda:bool=False,
                                 count:int=10, transparency:float=0.7):
    """
        Function to plot Grad-CAM heatmap over images incorrectly predicted by model

        args:
        model : CNN Model to be used
        target_layers:list : Layer to be consider for extracting activation maps(Channels), Gradients(weight) of predicted class w.r.t activation maps
        datset_std:tuple : standard daviation used for normalizing dataset images, Required for denarmalizing image
        dataset_mean:tuple : Mean used for normalizing dataset images, Required for denarmalizing image
        incorrect_predictions:list : Incorrect predictions by model.
        labels_mapping:dict : Dictionary of Class mapping with labels.
        use_cuda:bool : Default = False
        count:int : default=10, number of images to plot
        transparency:float : To set transparency ratio of heatmap and image
    """
    
    random.seed(10)

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Target/Predicted")
    
    for i, data in enumerate(random.choices(incorrect_predictions, k=count)):
        img, target, predicted, score = data
        denorm_img = image_denormalizer(img, dataset_std, dataset_mean)
        visualization = get_GradCAM_heatmap(model, target_layers, use_cuda, target, denorm_img, transparency)

        # Plot Grad-CAM
        ax = fig.add_subplot(int(count/5), 5, i + 1, xticks=[], yticks=[])
        ax.set_title(f'{labels_mapping[target.item()]}/{labels_mapping[predicted.item()]}')
        plt.imshow(visualization)

#########################################################################################################################################################