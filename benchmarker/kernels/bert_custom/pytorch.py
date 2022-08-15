"""
This is transformer-based language model for benchmarker
it is based on the torch sample code and is not identical
to the original BERT model from Vaswani et al., 2017 paper.
It should, however, expose similar performace behaviour.

Multuple parameters can be specified for this model:
number of layers, attention heads, hidden size etc.

One thing to keep in mind is that this should not be used
for comparison between different framworks.

Hopefully this should be fixed when we start importing models from ONNX
"""

import math

import torch
import torch.nn as nn
# import torch.nn.functional as F
from benchmarker.kernels.bert_custom import estimate_gflop_per_sample


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntokens, ninp, nhead, dim_mlp, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(d_model=ninp,
                                                 nhead=nhead,
                                                 dim_feedforward=dim_mlp,
                                                 dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntokens, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntokens)
        self.ntokens = ntokens
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_ids, labels, has_mask=True):
        if has_mask:
            device = input_ids.device
            if self.src_mask is None or self.src_mask.size(0) != len(input_ids):
                mask = self._generate_square_subsequent_mask(len(input_ids)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        input_ids = self.encoder(input_ids) * math.sqrt(self.ninp)
        input_ids = self.pos_encoder(input_ids)
        output = self.transformer_encoder(input_ids, self.src_mask)
        output = output[-1]
        output = self.decoder(output)
        # TODO: return softmax or cross_entropy depending on the mode
        return output
        # return F.log_softmax(output, dim=-1)


def get_kernel(params):
    # TODO: use cnt_tokens in data generation as max rand!
    cnt_samples = params["problem"]["size"][0]
    len_seq = params["problem"]["size"][1]
    dim_mlp = params["problem"]["cnt_units"] * 4
    gflop_per_sample = estimate_gflop_per_sample(
        len_seq=len_seq,
        embed_dim=params["problem"]["cnt_units"],
        lin_dim=dim_mlp,
        nb_layers=params["problem"]["cnt_layers"],
    )
    gflop_estimated = gflop_per_sample * cnt_samples * params["nb_epoch"]
    params["problem"]["gflop_estimated"] = gflop_estimated
    Net = TransformerModel(ntokens=params["problem"]["cnt_tokens"],
                           ninp=params["problem"]["cnt_units"],
                           nhead=params["problem"]["cnt_heads"],
                           dim_mlp=dim_mlp,
                           nlayers=params["problem"]["cnt_layers"],
                           dropout=0.5)
    return Net
