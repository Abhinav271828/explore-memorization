import torch
from torch import Tensor
from torch import nn
from lightning import LightningModule
import math

from utils import PositionalEncoding

class TransformerLM(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, device: str = 'cpu'):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Embedding(ntoken, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        self.device = device

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

class TransformerLMLightning(LightningModule):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, device: str = 'cpu'):
        super().__init__()
        self.model = TransformerLM(ntoken, d_model, nhead, d_hid, nlayers, dropout, device)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        return self.model(src, src_mask)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        src, tgt = batch
        src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.model.device)
        output = self.model(src, src_mask)
        loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        src, tgt = batch
        src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.model.device)
        output = self.model(src, src_mask)
        loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss