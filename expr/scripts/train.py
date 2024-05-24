from defs.model import TransformerLMLightning
from defs.data import LMData
from utils import get_name_from_config
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

for size in [1e3, 1e4]:
    for config in \
        [{'ntoken': 19, 'd_model': 64,  'nhead': 2, 'd_hid': 256,  'nlayers': 3, 'dataset_size' : size},
         {'ntoken': 19, 'd_model': 128, 'nhead': 4, 'd_hid': 512,  'nlayers': 3, 'dataset_size' : size},
         {'ntoken': 19, 'd_model': 256, 'nhead': 8, 'd_hid': 1024, 'nlayers': 6, 'dataset_size' : size}]:

        name = get_name_from_config(config)
        model = TransformerLMLightning(**config, from_pretrained=False)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        model_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=-1, dirpath='models/', filename=name)

        logger = WandbLogger(project="Memorization (Prelim)", name=name, config=config)

        trainer = Trainer(max_epochs=100, callbacks=[early_stopping, model_checkpoint], logger=logger)
        trainer.fit(model)
        trainer.test(model)
        torch.save(model.train_dataloader().dataset, f'datasets/{name}.pt')

        logger.experiment.finish()