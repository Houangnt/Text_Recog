import copy
import torch
import torch.nn as nn

from network.sequence_modeling.transformer_layers import MultiHeadedAttention, PositionalEncoding, \
    PositionwiseFeedForward, Embeddings, LayerNorm, SublayerConnection, clones

__all__ = ['transformer']


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers"
    """

    def __init__(self, layer, N):
        """
        :param layer: Layer class
        :param N: Number of encoder layers in the stack
        """

        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """

        :param x: data
        :param mask: mask, not used
        :return: forward pass
        """

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        :param size: size of the output
        :param self_attn: self attention function
        :param feed_forward: feed forward layer
        :param dropout: dropout rate
        """

        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        :param x: input tensor
        :param mask: mask, not used
        :return:
        """

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """

    def __init__(self, layer, N):
        """
        :param layer: layer class
        :param N: Number of Decoder layers as stack
        """

        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Forward pass for the decoder
        :param x: data
        :param memory: embeddings from the encoder
        :param src_mask: mask for the source
        :param tgt_mask: target mask, hide the targets for the next time step predicting during parallel training of the decoder
        :return: decoder output
        """

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        :param size:
        :param self_attn:
        :param src_attn:
        :param feed_forward:
        :param dropout:
        """

        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Forward pass of the decoder layer
        :param x:
        :param memory:
        :param src_mask:
        :param tgt_mask:
        :return:
        """
        m = memory
        x = self.sublayer[0](x, lambda k: self.self_attn(k, k, k, tgt_mask))
        # x = self.sublayer[0](x, sub_layer1)
        x = self.sublayer[1](x, lambda k: self.src_attn(k, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, tgt_embed, device, direction_embed=None, bidirectional_decoding=False):
        """
        :param encoder: Encoder class
        :param decoder: Decoder class
        :param tgt_embed: lookup with the target embeddings
        :param device: CPU or GPU
        :param direction_embed: Lookup for the direction embeddings
        :param bidirectional_decoding: If true, decode bidirectional
        """

        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.direction_embed = direction_embed
        self.bidirectional_decoding = bidirectional_decoding
        self.device = device

        self.ltr_attn_dist = None
        self.rtl_attn_dist = None

    def forward(self, image, src_mask, tgt_embedding_mask=None, ltr_targets=None, rtl_targets=None, decode=True,
                encode=True, **kwargs):
        """
        Take in and process masked src and target sequences.
        :param image: input image
        :param src_mask: mask for encoder input, not used
        :param tgt_embedding_mask: mask for the target embeddings
        :param ltr_targets: targets left-to-right decoding
        :param rtl_targets: targets right-to-left decoding
        :param decode: include decode if True
        :param decode: include encode if True
        :return:
        """
        if decode and encode:
            return self.decode(self.encode(image, src_mask), src_mask, tgt_embedding_mask, ltr_targets, rtl_targets)
        elif encode:
            return self.encode(image, src_mask)
        else:
            return self.decode(image, src_mask, ltr_targets=ltr_targets, rtl_targets=ltr_targets, **kwargs)

    def encode(self, x, src_mask):
        """
        :param x: input feature
        :param src_mask: mask for the input image, not used
        :return: encoded image representation
        """
        return self.encoder(x, src_mask)

    def decode(self, memory, src_mask, ltr_tgt_mask, ltr_targets, rtl_targets=None, rtl_tgt_mask=None):
        """
        :param memory: the encoded image embeddings
        :param src_mask:
        :param ltr_tgt_mask: masking out the future targets
        :param ltr_targets: output targets
        :param rtl_targets: targets of the right-to-left sequence
        :param rtl_tgt_mask: targets of the left-to-right sequence
        :return:
        """
        nbatches = memory.size(0)

        if not self.bidirectional_decoding or rtl_targets is None:
            ltr = self.decoder(self.tgt_embed(ltr_targets) +
                               self.direction_embed(torch.zeros((nbatches, 1)).to(self.device)),
                               memory, src_mask, ltr_tgt_mask)
            return ltr, None
        else:
            if rtl_tgt_mask is None:
                rtl_tgt_mask = ltr_tgt_mask

            ltr = self.decoder(self.tgt_embed(ltr_targets) +
                               self.direction_embed(torch.zeros((nbatches, 1)).to(self.device)),
                               memory, src_mask, ltr_tgt_mask)
            rtl = self.decoder(self.tgt_embed(rtl_targets) +
                               self.direction_embed(torch.ones((nbatches, 1)).to(self.device)),
                               memory, src_mask, rtl_tgt_mask)

            return ltr, rtl


def transformer(opt):
    c = copy.deepcopy
    dropout = opt.sequence_modeling.dropout
    d_model = opt.sequence_modeling.d_model
    attn = MultiHeadedAttention(opt.sequence_modeling.n_heads, d_model)
    ff = PositionwiseFeedForward(d_model, opt.sequence_modeling.d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    direction_embed = Embeddings(d_model, 2)

    encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), opt.sequence_modeling.n_layers)
    decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), opt.sequence_modeling.n_layers)
    tgt_embed = nn.Sequential(Embeddings(d_model, opt.num_classes), c(position))
    return EncoderDecoder(encoder, decoder, tgt_embed, opt.device, direction_embed, bidirectional_decoding=True)
