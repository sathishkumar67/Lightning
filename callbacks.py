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
