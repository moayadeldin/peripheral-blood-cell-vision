import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision

transformations = torchvision.transforms.Compose([
    # torchvision.transforms.ToPILImage(), # as I upload raw images

    torchvision.transforms.Resize(size=(224,224)), # resize images to the needed size of ResNet50

    torchvision.transforms.ToTensor(), # convert images to tensors

    torchvision.transforms.Normalize(
        
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )    

])

class medicalImageDataset(Dataset):

    def __init__(self,root_dir, transform=None):

        """Args:
            root_dir (string) : Directory with all images, organized in class folders.
            transform (callable, optional): Transformations to be applied.
        """

        self.root_dir = root_dir
        if transform is None:
            self.transform = transformations
        else:
            self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir)) # returning the names of the folders as each name represents a class

        # We load all images and their respective labels and append their paths to their corresponding lists.

        for label_idx, class_folder in enumerate(self.classes):

            class_folder_path = os.path.join(root_dir,class_folder)

            for img_name in os.listdir(class_folder_path):

                img_path = os.path.join(class_folder_path,img_name)

                self.image_paths.append(img_path)

                self.labels.append(label_idx)

    
    def __len__(self):

        return len(self.image_paths)
    
    def __getitem__(self,idx):

        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(img)

    
        return image,label
    


