import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from lightning import LightningModule
import math

from utils import PositionalEncoding
from defs.data import LMData

class TransformerLM(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Embedding(ntoken, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[batch_size, seq_len, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[1])
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output
    
    def complete_sequence(self, src: Tensor, length: int, argmax=False):
        """
        Complete the sequence given by `src` with `length` elements.
        If `src` is batched, return a whole batch of predictions.
        """
        src_ = src.unsqueeze(0) if len(src.shape) == 1 else src
        # src_ : [bz, seq]

        # We only work with src_ from now on. `src` is the original input.
        while src_.size(1) < src.size(-1) + length:
            outputs = self(src_)
            # [bz, seq, ntoken]
            if not argmax:
                probabilities = F.softmax(outputs[:, -1, :], dim=-1)
                preds = torch.multinomial(probabilities, 1)
            else:
                preds = torch.argmax(outputs[:, -1:, :], dim=-1)
            # [bz, 1]
            src_ = torch.concat([src_, preds], dim=1)
        
        return src_

class TransformerLMLightning(LightningModule):
    def __init__(self, ntoken: int = 19, d_model: int = 64, nhead: int = 2, d_hid: int = 256,
                 nlayers: int = 3, dropout: float = 0.5, dataset_size : int = 10, from_pretrained : bool = True):
        super().__init__()
        self.model = TransformerLM(ntoken, d_model, nhead, d_hid, nlayers, dropout)
        self.dataset_size = dataset_size
        self.criterion = nn.CrossEntropyLoss(ignore_index=18)
        self.save_hyperparameters()

    def train_dataloader(self):
        return DataLoader(LMData('data/base-data.txt', self.dataset_size, 'train'), batch_size=32, shuffle=True)
    def val_dataloader(self):
        return DataLoader(LMData('data/base-data.txt', self.dataset_size, 'dev'), batch_size=32)
    def test_dataloader(self):
        return DataLoader(LMData('data/base-data.txt', self.dataset_size, 'test'), batch_size=32)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        return self.model(src, src_mask)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        src, tgt = batch
        output = self.model(src)
        loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        src, tgt = batch
        output = self.model(src)
        loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        src, tgt = batch
        output = self.model(src)
        loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
