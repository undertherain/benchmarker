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

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, ntokens, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
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

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = output[-1]
        output = self.decoder(output)
        # TODO: return softmax or cross_entropy depending on the mode
        return output
        # return F.log_softmax(output, dim=-1)


def get_kernel(params, unparsed_args=None):
    # assert params["mode"] == "inference"
    parser = argparse.ArgumentParser(description='Benchmark lstm kernel')
    parser.add_argument('--cnt_units', type=int, default=512)
    parser.add_argument('--cnt_heads', type=int, default=8)
    parser.add_argument('--cnt_layers', type=int, default=1)
    parser.add_argument('--cnt_tokens', type=int, default=1000)
    parser.add_argument('--bidirectional', type=bool, default=False)
    args = parser.parse_args(unparsed_args)
    params["problem"].update(vars(args))
    # print(params["problem"])
    # TODO: use cnt_tokens in data generation as max rand!
    Net = TransformerModel(ntokens=params["problem"]["cnt_tokens"],
                           ninp=params["problem"]["cnt_units"],
                           nhead=params["problem"]["cnt_heads"],
                           nhid=params["problem"]["cnt_units"],
                           nlayers=params["problem"]["cnt_layers"],
                           dropout=0.5)
    return Net
