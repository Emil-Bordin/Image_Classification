import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from EmilBordin969367_data_loading_assignment_4 import DataLoading
from EmilBordin969367_neural_network_assignment_4 import CropsCNN

#Exercise 6: Training and Evaluation
if __name__ == "__main__":
    data_loading = DataLoading()
    train_loader, val_loader, test_loader = data_loading.load_data()

    model = CropsCNN()

#Exercise 7: Tensorboard
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")

    trainer = pl.Trainer(devices=1, accelerator="auto", max_epochs=100, logger=tb_logger, log_every_n_steps=10,
                         callbacks=EarlyStopping(monitor="val_loss", min_delta=0.01, patience=5, verbose=True))

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(ckpt_path="best", dataloaders=test_loader)

#Exercise 8: Results
#Training Accuracy: 0.8049
#Test Accuracy: 0.6222222447395325
