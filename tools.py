from lightning.pytorch.callbacks import ModelCheckpoint

# Model Checkpoint to save models for all epochs(not tracking any metrics)
checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch:02d}",
        save_top_k=-1,  
        mode="min",
        monitor="Val_Loss",
        save_weights_only=True,
        every_n_epochs=1
    )


from lightning.pytorch.loggers import CSVLogger
log_name = ""
# logging the required metrics
logger = CSVLogger("logs", name=log_name)


# distributed data parallel training
trainer = Trainer(max_epochs=1,
                  accelerator="cuda",
                  strategy="ddp", # use ddp_notebook if used in notebooks directly
                  devices=2,
                  logger=logger)
