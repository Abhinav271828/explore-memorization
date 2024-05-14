from defs.model import TransformerLMLightning
from defs.data import LMData
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

for size in [1e3, 1e4, 1e5]:

    config = {'ntoken': 19, 'd_model': 64, 'nhead': 2, 'd_hid': 256, 'nlayers': 3, 'dataset_size' : size}
    model = TransformerLMLightning(**config, from_pretrained=False)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=-1, dirpath='models/', filename=f'size={size}.ckpt')

    logger = WandbLogger(project="Memorization (Prelim)", name=f'size={size}-local', config=config)

    trainer = Trainer(max_epochs=100, callbacks=[early_stopping, model_checkpoint], logger=logger)
    trainer.fit(model)
    trainer.test(model)

    logger.experiment.finish()