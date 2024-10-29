import os
import torch
from dataset import medicalImageDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import logging
from utilities import transformations

torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=16
LEARNING_RATE=0.0001
NUM_EPOCHS = 10
CHECK_VAL_EVERY_N_EPOCH = 1

TRAIN_DIR = "DataSet_Splitted/train"
VAL_DIR = "DataSet_Splitted/val"
TEST_DIR = "DataSet_Splitted/test"

###############################################################
# ADJUST THE TRANSFORMATIONS ACCORDING TO THE MODEL YOU WILL USE
###############################################################

train_dataset = medicalImageDataset(root_dir=TRAIN_DIR, transform=transformations)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

val_dataset = medicalImageDataset(root_dir=VAL_DIR, transform=transformations)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

test_dataset = medicalImageDataset(root_dir=TEST_DIR, transform=transformations)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

class Trainer:

    def __init__(self, model):
        
        self.model = model
        self.model.to(DEVICE)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []


        # logging info
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()

    def train(self,train_loader=train_loader,val_loader=val_loader):
    
        best_val_acc=0.0 # to save the best fitting model on the validation set

        for epoch in range(NUM_EPOCHS):

            self.model.train()

            # loss tracking metrics

            running_loss=0.0
            running_vloss=0.0
            batch_loss=0.0
            running_acc=0.0
            running_val_acc = 0

            train_pbar = tqdm(enumerate(train_loader), total=len(train_loader))

            for i, (inputs, labels) in train_pbar:

                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                running_acc += self.computeAccuracy(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                batch_loss += loss.item()

                if i % 10 == 0: # calculate average loss across every 10 batches

                    batch_loss = batch_loss / 10
                    train_pbar.set_postfix({"loss": round(batch_loss,5)})
                    batch_loss = 0.0

        
            # now we compute the average loss and accuracy for each epoch
            train_accuracy_per_epoch = running_acc / len(train_loader)
            self.train_accuracies.append((epoch, train_accuracy_per_epoch.cpu()))

            avg_loss = running_loss / len(train_loader)
            self.train_losses.append((epoch, avg_loss))

            # evaluating the model after certain number of epochs
            if epoch % CHECK_VAL_EVERY_N_EPOCH == 0:

                self.model.eval()

                val_pbar = tqdm(enumerate(test_loader), total=len(test_loader))

                with torch.no_grad():

                    for i, (input,labels) in val_pbar:

                        inputs, labels = input.to(DEVICE), labels.to(DEVICE)

                        outputs = self.model(inputs)

                        loss = self.criterion(outputs, labels)

                        running_vloss+= loss.item()

                        # compute validation accuracy for this epoch

                        running_val_acc += self.computeAccuracy(outputs,labels)

                val_accuracy_per_epoch = running_val_acc / len(val_loader)
                self.val_accuracies.append((epoch, val_accuracy_per_epoch.cpu()))

                avg_vloss = running_vloss / len(val_loader)
                self.val_losses.append((epoch, avg_vloss))

		# save the best model achieving on val dataset
                if val_accuracy_per_epoch > best_val_acc:

                    best_val_acc = val_accuracy_per_epoch
                    self.saveModel(f"{epoch}_best_model_weights.pth")

                self.logger.info(
                        f"[EPOCH {epoch + 1}] Training Loss= {avg_loss} Validation Loss={avg_vloss} | Training Accuracy={train_accuracy_per_epoch} val={val_accuracy_per_epoch}"
                    )

    def test(self, test_loader=test_loader, best_model_weights_path=None):

        if best_model_weights_path is None:
            pass
        else:
            self.model.load_state_dict(torch.load(best_model_weights_path))
            print('Model into the path is loaded.')

        correct = 0 # we want to know how many images in the test set was predicted correctly (matched the label) so we keep adding the results of this with each batch running with specific input.

        self.model.eval()

        test_pbar = tqdm(enumerate(test_loader), total=len(test_loader))

        with torch.no_grad():

            for i, (input, labels) in test_pbar:

                inputs, labels = input.to(DEVICE), labels.to(DEVICE)

                outputs = self.model(inputs)

                correct+=self.computeAccuracy(outputs,labels)

        self.logger.info(f"Test accuracy: {(correct / len(test_loader)) * 100}%")

    def saveModel(self, path):

        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model Saved to {path}")
            

    def computeAccuracy(self,outputs, labels):

        """Compute accuracy given outputs as logits.
        """

        preds = torch.argmax(outputs, dim=1)
        return torch.sum(preds == labels) / len(preds)

    def plotMetrics(self):

        os.makedirs("plots", exist_ok=True)

        # extracting iterations and losses
        
        t_iters = [item[0] for item in self.train_losses]
        t_loss = [item[1] for item in self.train_losses]

        # extracting validation losses without using zip
        v_loss = [item[1] for item in self.val_losses]

        # extracting train & val accuracies
        acc = [item[1] for item in self.train_accuracies]
        v_acc = [item[1] for item in self.val_accuracies]

        fig, ax = plt.subplots(1,2, figsize=(12,5))
        fig.suptitle(f"ResNet 18 Trained on Peripheral Blood Cells Dataset")

        ax[0].set_title(f"Loss with Batch Size={BATCH_SIZE} & Learning Rate = {LEARNING_RATE}")
        ax[0].plot(t_iters, t_loss)
        ax[0].plot(t_iters, v_loss)
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend(["Train", "Validation"])
        ax[0].set_xticks(t_iters[::5])
        ax[0].set_xticklabels(t_iters[::5], rotation=45)


        ax[1].set_title(f"Accuracy with Batch Size={BATCH_SIZE} & Learning Rate = {LEARNING_RATE}")
        ax[1].plot(t_iters,acc)
        ax[1].plot(t_iters, v_acc)
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend(["Train", "Validation"])
        ax[1].set_xticks(t_iters)

        fig.savefig(f"plots/evaluation_metrics.png")
        plt.show()

# if __name__ == "__main__":

#     model = adjustedResNet()

#     model_trainer = Trainer(model=model)

#     model_trainer.train()

#     model_trainer.test()

#     model_trainer.saveModel('best_weights.pth')


