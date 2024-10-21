import torchvision
import torch
from dataset import medicalImageDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.models import ResNet50_Weights
import argparse
import sys
import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
LEARNING_RATE=0.001

class adjustedResNet50(nn.Module):

    def __init__(self, num_classes=8):

        super(adjustedResNet50, self).__init__()

        self.model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # remove the final connected layer by putting a placeholder
        self.model.fc = nn.Identity()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048,1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):

        x = self.model(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x
    
model = adjustedResNet50()

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

criterion = nn.CrossEntropyLoss()

model = model.to(DEVICE)

parser = argparse.ArgumentParser(description="The root directory of each of the three splits.")

parser.add_argument('--train_dir', type=str, required=True, help='Path to training data')
parser.add_argument('--val_dir', type=str, required=True, help='Path to validation data')
parser.add_argument('--test_dir', type=str, required=True, help='Path to test data')

args = parser.parse_args()


train_dataset = medicalImageDataset(root_dir=args.train_dir)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

val_dataset = medicalImageDataset(root_dir=args.val_dir)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

test_dataset = medicalImageDataset(root_dir=args.test_dir)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)


# model metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# logging info
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger()

def train(model, train_loader, criterion, check_val_every_n_epoch, epochs):

    for epoch in range(epochs):

        model.train()

        # loss tracking metrics

        running_loss=0.0
        running_vloss=0.0
        batch_loss=0.0
        running_acc=0.0
        running_val_acc = 0

        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (inputs, labels) in train_pbar:

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            running_acc += computeAccuracy(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_loss += loss.item()

            if i % 10 == 0: # calculate average loss across every 10 batches

                batch_loss = batch_loss / 10
                train_pbar.set_postfix({"loss": round(batch_loss,5)})
                batch_loss = 0.0

        
        # now we compute the average loss and accuracy for each epoch
        train_accuracy_per_epoch = running_acc / len(train_loader)
        train_accuracies.append((epoch, train_accuracy_per_epoch.cpu()))

        avg_loss = running_loss / len(train_loader)
        train_losses.append((epoch, avg_loss))

        # evaluating the model after certain number of epochs
        if epoch % check_val_every_n_epoch == 0:

            model.eval()

            val_pbar = tqdm(enumerate(test_loader), total=len(test_loader))

            with torch.no_grad():

                for i, (input,labels) in val_pbar:

                    inputs, labels = input.to(DEVICE), labels.to(DEVICE)

                    outputs = model(inputs)

                    loss = criterion(outputs, labels)

                    running_vloss+= loss.item()

                    # compute validation accuracy for this epoch

                    running_val_acc += computeAccuracy(outputs,labels)

            val_accuracy_per_epoch = running_val_acc / len(val_loader)
            val_accuracies.append((epoch, val_accuracy_per_epoch.cpu()))

            avg_vloss = running_vloss / len(val_loader)
            val_losses.append((epoch, avg_vloss))

            logger.info(
                    f"[EPOCH {epoch + 1}] Training Loss= {avg_loss} Validation Loss={avg_vloss} | Training Accuracy={train_accuracy_per_epoch} val={val_accuracy_per_epoch}"
                )     

def test(model, test_loader):

    correct = 0 # we want to know how many images in the test set was predicted correctly (matched the label) so we keep adding the results of this with each batch running with specific input.

    model.eval()

    test_pbar = tqdm(enumerate(test_loader), total=len(test_loader))

    with torch.no_grad():

        for i, (input, labels) in test_pbar:

            inputs, labels = input.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)

            correct+=computeAccuracy(outputs,labels)

    logger.info(f"Test accuracy: {(correct / len(test_loader)) * 100}%")
            

def computeAccuracy(outputs, labels):

    """Compute accuracy given outputs as logits.
    """

    preds = torch.argmax(outputs, dim=1)
    return torch.sum(preds == labels) / len(preds)


train(model=model, train_loader=train_loader, criterion=criterion,check_val_every_n_epoch=1, epochs=1)

test(model=model, test_loader=test_loader)
