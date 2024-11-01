import os
import matplotlib.pyplot as plt
import argparse 
import random as r
import cv2
import torchvision
import splitfolders

transformations = torchvision.transforms.Compose([
    # torchvision.transforms.ToPILImage(), # as I upload raw images

    torchvision.transforms.Resize(size=(224,224)), # resize images to the needed size of ResNet50

    torchvision.transforms.ToTensor(), # convert images to tensors

    torchvision.transforms.Normalize(
        
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )    

])


class_dist = {}

def calculateClassDistribution(dir):
    """Calculate Class Distribution of the Dataset.

    Args:
        dir (Str): The directory containing Dataset.

    Returns:
        class_dist : A dictionary containing each class with the number of images corresponding to its label.
    """

    for subdir in os.listdir(dir):

        class_dist[subdir] = len(os.listdir(os.path.join(dir, subdir)))

    return class_dist


def classDistribution():

    """Plot Bar Chart Showing Frequency of Each Class
    """

    plt.figure(figsize=(10,10))

    plt.bar(
        class_dist.keys(),
        class_dist.values(),
        color = ["grey", "red", "orange", "yellow", "green", "blue", "purple", "pink"]
    )

    plt.xticks(rotation=90)
    plt.title("Class Distribution of Cell Types in Dataset.")
    plt.xlabel("Class Label")
    plt.ylabel("Frequency of Class")
    plt.savefig("plots/dataset_class_distribution.png")
    plt.show()
    print("Bar Chart Distribution Plotted.")

def getSamples(dir):

    rows = 2
    columns = 4
    c = 0

    fig, axs = plt.subplots(rows, columns, figsize=(10,10))

    for subdir in os.listdir(dir):

        img_path = r.choice(os.listdir(os.path.join(dir,subdir)))

        # read the image then compute the corresponding row and column according to the count
        image = cv2.imread(os.path.join(dir, subdir, img_path))

        axs[c//columns, c%columns].imshow(image)
        axs[c//columns, c%columns].set_title(subdir)
        c+=1
        if c>= rows*columns:
            break

    fig.suptitle("Samples of the Dataset")
    plt.subplots_adjust(bottom=0.35, top=0.98, hspace=0.15)
    plt.savefig("plots/samples_from_dataset.png")
    plt.show()
    print("Samples Plotted.")


def dataSplit(dir, split_dir, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):


    splitfolders.ratio(input=dir,
                       output=split_dir,
                       seed=42,
                       ratio=(train_ratio, validation_ratio, test_ratio),
                       group_prefix=None,
                       move=False)








