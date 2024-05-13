from defs.model import TransformerLMLightning
from defs.data import LMData
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

for size in [1e3, 1e4, 1e5, 1e6]:
    train_data = LMData('data/base-data.txt', size, 'train')
    torch.save(train_data, f'data/train_data_size={size}.pt')
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
    dev_dl = DataLoader(LMData('data/base-data.txt', size, 'dev'), batch_size=32)
    test_dl = DataLoader(LMData('data/base-data.txt', size, 'test'), batch_size=32)

    config = {'ntoken': 20, 'd_model': 64, 'nhead': 2, 'd_hid': 256, 'nlayers': 3}
    model = TransformerLMLightning(**config)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=-1, dirpath='models/', filename=f'size={size}.ckpt')

    logger = WandbLogger(project="Memorization (Prelim)", name=f'size={size}', config=config)

    trainer = Trainer(max_epochs=100, callbacks=[early_stopping, model_checkpoint], logger=logger)
    trainer.fit(model, train_dl, dev_dl)
    trainer.test(model, test_dl)