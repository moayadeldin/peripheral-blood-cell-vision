"""This extract_embeddings.py is integrated in order to use fine-tuned ResNet50 model as an embeddings extractor for the training images, to evaluate the performance of different classic ML algorithms who excel in tabular data.
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from resnet import adjustedResNet
from dataset import medicalImageDataset
from dataset import transformations

TRAIN_DIR = "DataSet_Splitted/train"
BATCH_SIZE = 16

model = adjustedResNet()
model.load_state_dict(torch.load('ResNet50_results_weights.pth', weights_only=True))

def adjustModel(model):

    """The model is modified to be prepared for extracting feature embeddings rather than making predictions.
    """

    
    # take all layers except the last one (one used for classification) to make it output feature embeddings.
    modules = list(model.children())[:-1]

    # unpacking the layers in modules and now it contains the entire model minus the last one.
    model = nn.Sequential(*modules)

    return model

adjusted_model = adjustModel(model=model)

for p in adjusted_model.parameters(): # stop gradient calculations
    p.requires_grad = False

adjusted_model.cuda() # move the model to cuda

train_dataset = medicalImageDataset(root_dir=TRAIN_DIR)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

def extractEmbeddings():

    extracted_labels = []
    extracted_embeddings = []

    torch.cuda.empty_cache()
    model.eval()

    for data, labels in tqdm(train_loader):

        new_labels = labels.numpy().tolist()

        extracted_labels += new_labels

        data = data.cuda()

        embeddings = adjusted_model(data.cuda())

        extracted_embeddings.append(np.reshape(embeddings.detach().cpu().numpy(),(len(new_labels),-1)))

    extracted_embeddings = np.vstack(extracted_embeddings)

    return extracted_embeddings, extracted_labels

embeddings, labels = extractEmbeddings()

print(f"The shape of the embeddings matrix on training set is{embeddings.shape}")
print(f"The shape of the labels vector on training set is{labels}")

# save the output as CSV file
np.savetxt("training_embeddings.csv", embeddings,delimiter=",")
np.savetxt("labels_embeddings.csv", np.array(labels), delimiter=",")







    
