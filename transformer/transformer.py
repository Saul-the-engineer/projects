import torch
import torch.nn as nn
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.n_heads = heads
        self.head_size = embed_size // self.n_heads
        assert (
            self.head_size * self.n_heads == embed_size
        ), "Embed size needs to be divisible by heads"
        # query: What do we want to pay attention to?
        # key: What do we want to compare our query to?
        # value: What do we want to output?
        # These are the same, so we use nn.Linear to create them
        self.values = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)
        self.fc_out = nn.Linear(self.n_heads * self.head_size, self.embed_size)

    def forward(self, values, keys, queries, mask):
        sample_size = queries.size(0)
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        # print(values.shape)

        # Split the embedding into self.heads different pieces
        values = values.reshape(sample_size, value_len, self.n_heads, self.head_size)
        keys = keys.reshape(sample_size, key_len, self.n_heads, self.head_size)
        queries = queries.reshape(sample_size, query_len, self.n_heads, self.head_size)
        # print("values reshape: ", values.shape)

        # passing keys into linear layer
        values = self.values(values)  # (N, value_len, heads, head_size)
        keys = self.keys(keys)  # (N, key_len, heads, head_size)
        queries = self.queries(queries)  # (N, query_len, heads, head_size)
        # print("values: ", values.shape)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm
        energy = torch.einsum("bqhd, bkhd -> bhqk", [queries, keys])
        # print("energy: ", energy.shape)
        # Normalize energy values so that they sum to 1
        energy = energy / (self.embed_size ** (1 / 2))

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # calculate attention weights
        attention = torch.softmax(energy, dim=3)
        # print("attention: ", attention.shape)

        out = torch.einsum("bhql, blhd -> bqhd", [attention, values]).reshape(
            sample_size, query_len, self.n_heads * self.head_size
        )
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        # forward expansion is the size of the feed forward network
        # forward expansion allows us to learn more complex functions
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        # layer norm is similar to batch norm
        # takes average and std across the embedding dimension
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # value, key, and query shape: (N, seq_len, embed_size)
        # mask shape: (N, seq_len)
        attention = self.attention(value, key, query, mask)
        # add skip connection, residual connection
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        # target mask is for the decoder
        # source mask is for the encoder
        # target mask pads the sequence so that the decoder can't see future tokens
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        # how do we know vocab size?
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        # we will have num_layers of transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x shape: (N, seq_len)
        N, seq_len = x.shape
        # creates a tensor of positions from 0 to seq_len
        # expands the tensor to N rows and seq_len columns
        # essentially a 2d tensor size N x seq_len
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        # positions shape: (N, seq_len)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        # out shape: (N, seq_len, embed_size)
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src shape: (N, src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask shape: (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        # trg_mask shape: (N, 1, trg_len, trg_len)
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        # enc_src shape: (N, src_len, embed_size)
        enc_src = self.encoder(src, src_mask)
        # out shape: (N, trg_len, trg_vocab_size)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
