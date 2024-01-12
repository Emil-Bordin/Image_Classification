import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as torchvision
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import torchmetrics

#Exercise 2: Convolutional Neural Network Architecture Definition
class CropsCNN(pl.LightningModule):
    def __init__(self):
        super(CropsCNN, self).__init__()

        self.model_ft = models.vgg16(pretrained=True)
        for param in self.model_ft.parameters():
            param.requires_grad = False
        num_ftrs = self.model_ft.classifier[6].in_features
        self.model_ft.classifier[6] = nn.Linear(num_ftrs, 5)

        print(self.model_ft)

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)

    def forward(self, x):
        x = self.model_ft(x)
        return x
        
#Exercise 3: Optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
        
#Exercise 4: Training, Validation and Test Step
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True)
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_epoch=True)
        return loss
        

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss, on_step=True)
        self.val_acc(y_hat, y)
        self.log("val_acc", self.val_acc, on_epoch=True)
        return val_loss
        

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        test_loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", test_loss, on_step=True)
        self.test_acc(y_hat, y)
        self.log("test_acc", self.test_acc, on_epoch=True)
        return test_loss
        
