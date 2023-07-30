# CIFAR10 Image Classification using Resnet18

## Summary
This project is an image classification project. Dataset used for training is CIFAR10. Model used for traning is Resnet18. Albumentation is used for image transformation. One Cycle LR policy is used for best LR range.

Model is trained for 20 epochs, with batch size of 512.
For Identifying reasons of incorrect predictions Grad-CAM is implemented.

## File Structure
### S11.ipynb : 
is the file where all of the defined implementations are fetched and used. EG: Setting up train/test data loader, Visualising sample images and effect of transformations, Use of LR finder to find max best LR, Running Training pipeline, Visualising Model performance graphs/ Incorrect Predicted images, Grad-CAM heatmap on Incorrect predictions etc...

### models :
This is the directory where all models are/will be stored. It currently includes resnet18.py file which hase resnet18 architecture implemented in it.

### dataset.py :
Dataset class is definrd in this.

### main.py : ***
Some of the important functions are defined in this file.

    1- set_optim() : sets up the optimizer

    2- get_one_cycle_lr_scheduler() : sets up scheduler

    3- model_train() : Training Loop

    4- model_test() : Test Loop

    5- get_loader() : Sets up Train/Test Dataloader

### utils.py :
Utitlity functions are kept in this file
    
    1- GetCorrectPredCount() : Returns Correct prediction count

    2- load_weights_from_path() : Loads trained model's weights

    3- get_incorrrect_predictions() : Returns Incorrect predictions by model over a dataset inferred.

    4- get_all_predictions() : Returns all model's predictions.

    5- get_device() : Check and returns device (Cuda, MPS, CPU) available in machine.

    6- image_denormalizer() : Denormalizes normalised image.

    7- get_GradCAM_heatmap() : Returns Heatmap created using activation maps and gradient of the layer passed.

### visualise.py :
In this file all visualisation functions of the project are kept. Eg: plotting Grad-CAM, Showing Image samples etc..



