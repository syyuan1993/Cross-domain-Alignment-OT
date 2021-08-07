import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import math, copy, time
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_ as clip_grad_norm
import numpy as np
from collections import OrderedDict
import sys
from .transformer import clones

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        # if self.use_bi_gru:
        #     embed_size = int(embed_size/2)
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            # print(cap_emb.size(2))
            cap_emb = (cap_emb[:,:,:int(cap_emb.size(2)/2)] + cap_emb[:,:,int(cap_emb.size(2)/2):])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len, x


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

# Word Embedding Based Language Model
class WordEmbedding(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(WordEmbedding, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # # caption embedding
        # self.use_bi_gru = use_bi_gru
        # if self.use_bi_gru:
        #     embed_size = int(embed_size/2)
        # self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        # packed = pack_padded_sequence(x, lengths, batch_first=True)
        #
        # # Forward propagate RNN
        # out, _ = self.rnn(packed)
        #
        # # Reshape *final* output to (batch_size, hidden_size)
        # padded = pad_packed_sequence(out, batch_first=True)
        # cap_emb, cap_len = padded
        #
        # # if self.use_bi_gru:
        # #     print(cap_emb.size(2))
        # #     cap_emb = (cap_emb[:,:,:cap_emb.size(2)/2] + cap_emb[:,:,cap_emb.size(2)/2:])/2
        #
        # normalization in the joint embedding space
        if not self.no_txtnorm:
            x = l2norm(x, dim=-1)

        cap_emb=x
        cap_len=lengths

        return cap_emb, cap_len, x



# Word Embedding and MLP Based Language Model
class LabelEmbedding(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(LabelEmbedding, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        self.linear = nn.Linear(word_dim, embed_size)

        # # caption embedding
        # self.use_bi_gru = use_bi_gru
        # if self.use_bi_gru:
        #     embed_size = int(embed_size/2)
        # self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        x = self.linear(x)
        # packed = pack_padded_sequence(x, lengths, batch_first=True)
        #
        # # Forward propagate RNN
        # out, _ = self.rnn(packed)
        #
        # # Reshape *final* output to (batch_size, hidden_size)
        # padded = pad_packed_sequence(out, batch_first=True)
        # cap_emb, cap_len = padded
        #
        # # if self.use_bi_gru:
        # #     print(cap_emb.size(2))
        # #     cap_emb = (cap_emb[:,:,:cap_emb.size(2)/2] + cap_emb[:,:,cap_emb.size(2)/2:])/2
        #
        # normalization in the joint embedding space
        if not self.no_txtnorm:
            x = l2norm(x, dim=-1)

        cap_emb=x
        cap_len=lengths

        return cap_emb, cap_len, x

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# Transformer Based Language Model
class Transformer(nn.Module):
	def __init__(self, vocab_size, word_dim, embed_size, num_layers,
	             use_bi_gru=False, no_txtnorm=False):
		super(Transformer, self).__init__()
		self.embed_size = embed_size
		self.no_txtnorm = no_txtnorm

		# word embedding
		self.embed = nn.Embedding(vocab_size, word_dim)

		self.init_weights()
		self.fc1 = nn.Linear()
		self.fc2 = nn.Linear()

	def init_weights(self):
		self.embed.weight.data.uniform_(-0.1, 0.1)

	def forward(self, x, lengths):
		"""Handles variable size captions
		"""
		# Embed word ids to vectors
		x = self.embed(x)
		# packed = pack_padded_sequence(x, lengths, batch_first=True)
		#
		# # Forward propagate RNN
		# out, _ = self.rnn(packed)
		#
		# # Reshape *final* output to (batch_size, hidden_size)
		# padded = pad_packed_sequence(out, batch_first=True)
		# cap_emb, cap_len = padded
		#
		# if self.use_bi_gru:
		# 	# print(cap_emb.size(2))
		# 	cap_emb = (cap_emb[:, :, :int(cap_emb.size(2) / 2)] + cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2

		# normalization in the joint embedding space

		cap_emb, p = attention(x, x, x)
		if not self.no_txtnorm:
			cap_emb = l2norm(cap_emb, dim=-1)
			x = l2norm(x, dim=-1)
		cap_len = lengths

		return cap_emb, cap_len, x


def l1norm(X, dim, eps=1e-8):
	"""L1-normalize columns of X
	"""
	norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
	X = torch.div(X, norm)
	return X


def l2norm(X, dim, eps=1e-8):
	"""L2-normalize columns of X
	"""
	norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
	X = torch.div(X, norm)
	return X



class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1, no_normalize = True):
		"Take in model size and number of heads."
		super(MultiHeadedAttention, self).__init__()
		assert d_model % h == 0
		# We assume d_v always equals d_k
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)
		self.no_norm = no_normalize

	def forward(self, query, key, value, mask=None):
		"Implements Figure 2"
		if mask is not None:
			# Same mask applied to all h heads.
			mask = mask.unsqueeze(1)
		nbatches = query.size(0)

		# 1) Do all the linear projections in batch from d_model => h x d_k
		query, key, value = \
			[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
			 for l, x in zip(self.linears, (query, key, value))]

		# 2) Apply attention on all the projected vectors in batch.
		x, self.attn = attention(query, key, value, mask=mask,
		                         dropout=self.dropout)

		# 3) "Concat" using a view and apply a final linear.
		x = x.transpose(1, 2).contiguous() \
			.view(nbatches, -1, self.h * self.d_k)
		results = self.linears[-1](x)
		if not self.no_norm:
			results = l2norm(results, dim=-1)

		return results


class PositionalEncoding(nn.Module):
	"Implement the PE function."

	def __init__(self, d_model, dropout, max_len=200):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len).unsqueeze(1).float()
		div_term = torch.exp(torch.arange(0, d_model, 2).float() *
		                     -(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + Variable(self.pe[:, :x.size(1)],
		                 requires_grad=False).cuda()
		return self.dropout(x)


class Batch:
	"Object for holding a batch of data with mask during training."
	def __init__(self, src, trg=None, pad=0):
		self.src = src
		self.src_mask = (src != pad).unsqueeze(-2)
		if trg is not None:
			self.trg = trg[:, :-1]
			self.trg_y = trg[:, 1:]
			self.trg_mask = \
				self.make_std_mask(self.trg, pad)
			self.ntokens = (self.trg_y != pad).data.sum()

	@staticmethod
	def make_std_mask(tgt, pad):
		"Create a mask to hide padding and future words."
		tgt_mask = (tgt != pad).unsqueeze(-2)
		tgt_mask = tgt_mask & Variable(
			subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
		return tgt_mask

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def padding_mask_single(bs, max_len, lengths):
    x_mask = np.zeros((bs, max_len)).astype('float32')
    for idx in range(bs):
        x_mask[idx, :lengths[idx]] = 1. # change to remove the real END token
    return torch.from_numpy(x_mask).unsqueeze(-2).cuda()

def padding_mask_both(seq_k, seq_q):
    # seq_k 和 seq_q 的形状都是 [B,L]
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(1)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1).cuda()  # shape [B, L_q, L_k]
    return pad_mask

def padding_mask(bs, max_len, lengths):
	padding = padding_mask_single(bs, max_len, lengths)
	return padding_mask_both(padding, padding)



class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
