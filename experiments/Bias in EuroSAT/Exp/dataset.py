import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision import transforms
from skimage.color import rgb2hsv


def load_dataset_filtered_eurosat(path='data/EuroSAT_2_classes/'):
    # Copyright 2020 by Laurent Risser, Quentin Vincenot, Nicolas Couellan, Jean-Michel Loubes
    # All rights reserved.
    # This code is based on the following paper published by the authors :
    # (https://arxiv.org/pdf/1908.05783.pdf)
    """
    Load the EuroSAT dataset, filtered to showcase a specific unfairness case.
	The bias is showed with a few satellite images possessing a "blue shaded" aspect, that could influence a model predictions.

	Preprocessing of the dataset is completely done inside this method.

	:return X_train: The training set tensor containing the satellites images
	:return S_train: The training set sensibility vector, computed whether images possess a blue shade or not (0/1)
	:return y_train: The training set true labels for the data, whether they are Highways or Rivers (0/1)
	:return X_test: The test set tensor containing the satellites images
	:return S_test: The test set sensibility vector, computed whether images possess a blue shade or not (0/1)
	:return y_test: The test set set true labels for the data, whether they are Highways or Rivers (0/1)

	:rtype: NumPy array of (A, 3, 64, 64) elements
	:rtype: NumPy array of (A, 1) elements
	:rtype: NumPy array of (A, 1) elements
	:rtype: NumPy array of (B, 3, 64, 64) elements
	:rtype: NumPy array of (B, 1) elements
    :rtype: NumPy array of (B, 1) elements
    """
    # Load the dataset from the main directory (each subdirectory is a class)
    transformations = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=path, transform=transformations)

    # Extract indices for train/test, for each class in the dataset
    wanted_classes = ['Highway', 'River']
    indices_current_class = np.where(np.array(dataset.targets) == dataset.class_to_idx[wanted_classes[0]])[0]
    amount_of_current_class = len(indices_current_class)
    split_index = int(0.75 * amount_of_current_class)
    train_indices = np.array(indices_current_class[:split_index])
    test_indices = np.array(indices_current_class[split_index:])
    for current_class in range(1, len(wanted_classes)):
        indices_current_class = np.where(np.array(dataset.targets)
                                         == dataset.class_to_idx[wanted_classes[current_class]])[0]
        amount_of_current_class = len(indices_current_class)
        split_index = int(0.75 * amount_of_current_class)
        train_indices = np.concatenate((train_indices, indices_current_class[:split_index]), axis=0)
        test_indices = np.concatenate((test_indices, indices_current_class[split_index:]), axis=0)

    # Create samplers and loaders to generate batches among sets of data
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = DataLoader(dataset, batch_size=512, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=170, sampler=test_sampler)

    def extract_images_and_labels_from_loader(loader):
        # Extract in NumPy arrays images/labels from a PyTorch loader
        images = np.array([])
        targets = np.array([])
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                images = data.cpu().numpy() if images.shape[0] == 0 else np.concatenate((images, data.cpu().numpy()))
                targets = target.cpu().numpy() if targets.shape[0] == 0 else np.concatenate(
                    (targets, target.cpu().numpy()))

        return images, targets

    def extract_all(loaders):
        # For every PyTorch loaders, extract images/labels into NumPy arrays
        images, labels = extract_images_and_labels_from_loader(loaders[0])
        for loader in loaders[1:]:
            temp_images, temp_labels = extract_images_and_labels_from_loader(loader)
            images = np.concatenate((images, temp_images))
            labels = np.concatenate((labels, temp_labels))

        return images, labels

    def __compute_means_on_channels(test_images, space='rgb'):
        mean_channels = np.zeros((test_images.shape[0], 3))
        for i in range(test_images.shape[0]):
            if space == 'rgb':
                image = np.moveaxis(test_images[i], 0, -1)
                channels_range = [255., 255., 255.]
            elif space == 'hsv':
                # Convert the image into HSV color space
                image = rgb2hsv(np.moveaxis(test_images[i], 0, -1))
                channels_range = [360., 1., 1.]

                # Get pixel values for each channel
                image_rh, image_gs, image_bv = image[:, :, 0].ravel(), image[:, :, 1].ravel(), image[:, :, 2].ravel()
                intensities_rh, counts_rh = np.unique(image_rh, return_counts=True)
                intensities_gs, counts_gs = np.unique(image_gs, return_counts=True)
                intensities_bv, counts_bv = np.unique(image_bv, return_counts=True)

                # Compute mean of RGB~HSV channels for each image
                mean_channels[i] = [round(np.mean(image_rh) * channels_range[0], 3),
                                    round(np.mean(image_gs) * channels_range[1], 3),
                                    round(np.mean(image_bv) * channels_range[2], 3)]

        return mean_channels

    def __flag_blue_images(images, space='rgb'):
        # Compute the mean on each channel
        mean_channels = __compute_means_on_channels(images, space)

        # Filter all images at once to get indices of images which are "blue-shaded" because of atmospheric phenomenon
        blue_images = (mean_channels[:, 0] > 210) & (mean_channels[:, 0] < 270) & (mean_channels[:, 1] > 0.35) & (
                mean_channels[:, 2] > 0.4)  # HSV

        # Build the sensibility vector thanks to indices of images which passed the filter
        selected_indices = np.where(blue_images)[0]
        sensibility_vector = np.ones((images.shape[0],))
        sensibility_vector[selected_indices] = 0

        return sensibility_vector, mean_channels

    # Extract images and their associated labels
    x_train, y_train = extract_all([train_loader])
    x_test, y_test = extract_all([test_loader])

    # Build the corresponding sensibilty vectors
    s_train, _ = __flag_blue_images(x_train, space='hsv')
    s_test, _ = __flag_blue_images(x_test, space='hsv')

    return x_train, s_train, y_train, x_test, s_test, y_test
