import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data.cifar_datamodule import CIFAR10DataModule
from models.resnet_module import ResNetLightningModule


def main():
    # Set seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Hyperparameters
    MAX_EPOCHS = 200
    BATCH_SIZE = 128
    LEARNING_RATE = 0.1
    WEIGHT_DECAY = 5e-4
    
    # Initialize DataModule
    data_module = CIFAR10DataModule(
        data_dir="./data",
        batch_size=BATCH_SIZE,
        num_workers=4,
        val_split=0.1,
    )
    
    # Initialize Model
    model = ResNetLightningModule(
        num_classes=10,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_epochs=MAX_EPOCHS,
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="resnet18-cifar10-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Logger
    logger = TensorBoardLogger("logs/", name="resnet18_cifar10")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        deterministic=True,
        precision="16-mixed" if torch.cuda.is_available() else "32",
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Test
    trainer.test(model, data_module, ckpt_path="best")
    
    print(f"\nBest model path: {checkpoint_callback.best_model_path}")
    print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()