import torchvision
import torch
from dataset import medicalImageDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import ResNet18_Weights
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LEARNING_RATE=0.001

model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# drop last linear layer and fit new linear layer for our dataset
model.fc = nn.Linear(in_features=512, out_features=8)

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

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

def train(model, train_loader, criterion,epochs=1):

    for epoch in range(epochs):

        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader))

        running_loss=0.0

        for i, (inputs, labels) in train_pbar:

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss+= loss.item()

            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

def test(model,test_loader):

    model.eval()

    test_pbar = tqdm(enumerate(test_loader), total=len(test_loader))

    with torch.no_grad():

        for i, (input, labels) in test_pbar:

            inputs, labels = input.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)

            predictions = torch.argmax(outputs, dim=1)

            print(f"Prediction: {predictions.cpu().numpy()} vs Ground Truth: {labels.cpu().numpy()}")


train(model=model, train_loader=train_loader, criterion=criterion,epochs=1)

test(model=model, test_loader=test_loader)





