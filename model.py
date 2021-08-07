# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""SCAN model"""

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_ as clip_grad_norm
import numpy as np
from collections import OrderedDict
from TextModels import BagOfWords
from OT_torch_ import cost_matrix_batch_torch, IPOT_distance_torch_batch_uniform, GW_distance_uniform


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


def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic', 
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc

def Image_Embedding(img_dim, embed_size, num_layers, precomp_enc_type='basic',
                 no_imgnorm=False):
    """A wrapper to image encoders. Encode image features into probability vectors
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp_emb(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp_emb(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


def InitImgEmbMap(embed_size, num_img_features):
    return torch.Variable(torch.randn(embed_size, num_img_features))

class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImagePrecomp_emb(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp_emb, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()
        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embeddi
        # ng space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)
        features = self.softmax(features)
        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp_emb(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp_emb, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)
        features = self.softmax(features)
        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)

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
        # # normalization in the joint embedding space
        # if not self.no_txtnorm:
        #     cap_emb = l2norm(cap_emb, dim=-1)

        cap_emb=x
        cap_len=lengths

        return cap_emb, cap_len, x


def func_attention(query, context, opt, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)


    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if opt.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = nn.Softmax(dim=1)(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "l1norm":
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score_t2i(images, captions, cap_lens, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    
    return similarities


def xattn_score_i2t(images, captions, cap_lens, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities


def xattn_score_t2i_OT(images, captions, cap_lens, x, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    similarities_OT = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        x_i = x[i, :n_word, :].unsqueeze(0).contiguous()
        x_i_expand = x_i.repeat(n_image, 1, 1)
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        x_i_expand = x_i_expand.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))

        cos_distance = cost_matrix_batch_torch(x_i_expand.transpose(2,1), images.transpose(2,1))
        OT_sim = IPOT_distance_torch_batch_uniform(cos_distance, n_image, images.size(1), n_word, opt.OT_iteration)

        similarities.append(row_sim)
        similarities_OT.append(OT_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    similarities_OT = torch.cat(similarities_OT, 1)

    return similarities, similarities_OT


def xattn_score_i2t_OT(images, captions, cap_lens, x, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    similarities_OT = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        x_i = x[i, :n_word, :].unsqueeze(0).contiguous()
        x_i_expand = x_i.repeat(n_image, 1, 1)
        # --> (n_image, n_word, d)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))

        cos_distance = cost_matrix_batch_torch(x_i_expand.transpose(2, 1), images.transpose(2, 1))
        OT_sim = IPOT_distance_torch_batch_uniform(cos_distance, n_image, images.size(1), n_word, opt.OT_iteration)


        similarities.append(row_sim)
        similarities_OT.append(OT_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    similarities_OT = torch.cat(similarities_OT, 1)
    return similarities, similarities_OT



def xattn_score_t2i_GW(images, captions, cap_lens, x, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    similarities_GW = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        x_i = x[i, :n_word, :].unsqueeze(0).contiguous()
        x_i_expand = x_i.repeat(n_image, 1, 1)
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        x_i_expand = x_i_expand.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))

        # cos_distance = cost_matrix_batch_torch(x_i_expand.transpose(2,1), images.transpose(2,1))
        # OT_sim = IPOT_distance_torch_batch_uniform(cos_distance, n_image, images.size(1), n_word, opt.OT_iteration)

        GW_sim = GW_distance_uniform(cap_i_expand.transpose(2,1), images.transpose(2,1), iteration=opt.GW_iteration, OT_iteration=opt.OT_iteration)

        similarities.append(row_sim)
        similarities_GW.append(GW_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    similarities_GW = torch.cat(similarities_GW, 1)

    return similarities, similarities_GW


def xattn_score_i2t_GW(images, captions, cap_lens, x, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    similarities_GW = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        x_i = x[i, :n_word, :].unsqueeze(0).contiguous()
        x_i_expand = x_i.repeat(n_image, 1, 1)
        # --> (n_image, n_word, d)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))

        # cos_distance = cost_matrix_batch_torch(x_i_expand.transpose(2, 1), images.transpose(2, 1))
        # OT_sim = IPOT_distance_torch_batch_uniform(cos_distance, n_image, images.size(1), n_word, opt.OT_iteration)

        GW_sim = GW_distance_uniform(cap_i_expand.transpose(2,1), images.transpose(2,1), iteration=opt.GW_iteration, OT_iteration=opt.OT_iteration)

        similarities.append(row_sim)
        similarities_GW.append(GW_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    similarities_GW = torch.cat(similarities_GW, 1)
    return similarities, similarities_GW


def computeMatchmap_batch(I, A):
    assert(I.dim() == 3)
    assert(A.dim() == 3)
    assert(I.size(2) == A.size(2))
    bs = I.size(0)
    D = I.size(2)
    # Ir = I.view(bs, -1, D)
    # Ar= A.view(bs, D, -1)
    matchmap = torch.bmm(I, A.transpose(2,1))
    return matchmap

def matchmapSim_batch(M, simtype):
    assert(M.dim() == 3)
    if simtype == 'SISA':
        return M.mean(dim=2).mean(dim=1, keepdim=True)
    elif simtype == 'MISA':
        M_maxH, _ = M.max(1)
        # M_maxHW, _ = M_maxH.max(1)
        return M_maxH.mean(dim=1, keepdim=True)
    elif simtype == 'SIMA':
        M_maxT, _ = M.max(2)
        return M_maxT.mean(dim=1, keepdim=True)#.mean(dim=1)
    else:
        raise ValueError

def dot_product_score(images, captions, cap_lens, x, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    similarities_OT = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        x_i = x[i, :n_word, :].unsqueeze(0).contiguous()
        x_i_expand = x_i.repeat(n_image, 1, 1)
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """

        row_sim = matchmapSim_batch(computeMatchmap_batch(images, cap_i_expand),opt.sim_type_dot)

        #
        # weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        # cap_i_expand = cap_i_expand.contiguous()
        # weiContext = weiContext.contiguous()
        # x_i_expand = x_i_expand.contiguous()
        # # (n_image, n_word)
        #
        #
        # row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        # if opt.agg_func == 'LogSumExp':
        #     row_sim.mul_(opt.lambda_lse).exp_()
        #     row_sim = row_sim.sum(dim=1, keepdim=True)
        #     row_sim = torch.log(row_sim) / opt.lambda_lse
        # elif opt.agg_func == 'Max':
        #     row_sim = row_sim.max(dim=1, keepdim=True)[0]
        # elif opt.agg_func == 'Sum':
        #     row_sim = row_sim.sum(dim=1, keepdim=True)
        # elif opt.agg_func == 'Mean':
        #     row_sim = row_sim.mean(dim=1, keepdim=True)
        # else:
        #     raise ValueError("unknown aggfunc: {}".format(opt.agg_func))

        # cos_distance = cost_matrix_batch_torch(x_i_expand.transpose(2,1), images.transpose(2,1))
        # OT_sim = IPOT_distance_torch_batch_uniform(cos_distance, n_image, images.size(1), n_word)

        similarities.append(row_sim)
        # similarities_OT.append(OT_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    # similarities_OT = torch.cat(similarities_OT, 1)

    return similarities, similarities_OT


def dot_product_score_WE(images, captions, cap_lens, x, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    similarities_OT = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        x_i = x[i, :n_word, :].unsqueeze(0).contiguous()
        x_i_expand = x_i.repeat(n_image, 1, 1)
        # --> (n_image, n_word, d)
        # cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """

        row_sim = matchmapSim_batch(computeMatchmap_batch(images, x_i_expand),opt.sim_type_dot)

        #
        # weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        # cap_i_expand = cap_i_expand.contiguous()
        # weiContext = weiContext.contiguous()
        # x_i_expand = x_i_expand.contiguous()
        # # (n_image, n_word)
        #
        #
        # row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        # if opt.agg_func == 'LogSumExp':
        #     row_sim.mul_(opt.lambda_lse).exp_()
        #     row_sim = row_sim.sum(dim=1, keepdim=True)
        #     row_sim = torch.log(row_sim) / opt.lambda_lse
        # elif opt.agg_func == 'Max':
        #     row_sim = row_sim.max(dim=1, keepdim=True)[0]
        # elif opt.agg_func == 'Sum':
        #     row_sim = row_sim.sum(dim=1, keepdim=True)
        # elif opt.agg_func == 'Mean':
        #     row_sim = row_sim.mean(dim=1, keepdim=True)
        # else:
        #     raise ValueError("unknown aggfunc: {}".format(opt.agg_func))

        # cos_distance = cost_matrix_batch_torch(x_i_expand.transpose(2,1), images.transpose(2,1))
        # OT_sim = IPOT_distance_torch_batch_uniform(cos_distance, n_image, images.size(1), n_word)

        similarities.append(row_sim)
        # similarities_OT.append(OT_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    # similarities_OT = torch.cat(similarities_OT, 1)

    return similarities, similarities_OT

def dot_product_score_OT(images, captions, cap_lens, x, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    similarities_OT = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # x_i = x[i, :n_word, :].unsqueeze(0).contiguous()
        # x_i_expand = x_i.repeat(n_image, 1, 1)
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """

        row_sim = matchmapSim_batch(computeMatchmap_batch(images, cap_i_expand), opt.sim_type_dot)

        #
        # weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        # cap_i_expand = cap_i_expand.contiguous()
        # weiContext = weiContext.contiguous()
        # x_i_expand = x_i_expand.contiguous()
        # # (n_image, n_word)
        #
        #
        # row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        # if opt.agg_func == 'LogSumExp':
        #     row_sim.mul_(opt.lambda_lse).exp_()
        #     row_sim = row_sim.sum(dim=1, keepdim=True)
        #     row_sim = torch.log(row_sim) / opt.lambda_lse
        # elif opt.agg_func == 'Max':
        #     row_sim = row_sim.max(dim=1, keepdim=True)[0]
        # elif opt.agg_func == 'Sum':
        #     row_sim = row_sim.sum(dim=1, keepdim=True)
        # elif opt.agg_func == 'Mean':
        #     row_sim = row_sim.mean(dim=1, keepdim=True)
        # else:
        #     raise ValueError("unknown aggfunc: {}".format(opt.agg_func))

        cos_distance = cost_matrix_batch_torch(cap_i_expand.transpose(2,1), images.transpose(2,1))
        OT_sim = IPOT_distance_torch_batch_uniform(cos_distance, n_image, images.size(1), n_word, opt.OT_iteration)

        similarities.append(row_sim)
        similarities_OT.append(OT_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    similarities_OT = torch.cat(similarities_OT, 1)

    return similarities, similarities_OT



def dot_product_score_OT_WE(images, captions, cap_lens, x, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    similarities_OT = []
    n_image = images.size(0)
    n_caption = x.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        # cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        x_i = x[i, :n_word, :].unsqueeze(0).contiguous()
        x_i_expand = x_i.repeat(n_image, 1, 1)
        # --> (n_image, n_word, d)
        # cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """

        row_sim = matchmapSim_batch(computeMatchmap_batch(images, x_i_expand), opt.sim_type_dot)

        #
        # weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        # cap_i_expand = cap_i_expand.contiguous()
        # weiContext = weiContext.contiguous()
        # x_i_expand = x_i_expand.contiguous()
        # # (n_image, n_word)
        #
        #
        # row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        # if opt.agg_func == 'LogSumExp':
        #     row_sim.mul_(opt.lambda_lse).exp_()
        #     row_sim = row_sim.sum(dim=1, keepdim=True)
        #     row_sim = torch.log(row_sim) / opt.lambda_lse
        # elif opt.agg_func == 'Max':
        #     row_sim = row_sim.max(dim=1, keepdim=True)[0]
        # elif opt.agg_func == 'Sum':
        #     row_sim = row_sim.sum(dim=1, keepdim=True)
        # elif opt.agg_func == 'Mean':
        #     row_sim = row_sim.mean(dim=1, keepdim=True)
        # else:
        #     raise ValueError("unknown aggfunc: {}".format(opt.agg_func))

        cos_distance = cost_matrix_batch_torch(x_i_expand.transpose(2,1), images.transpose(2,1))
        OT_sim = IPOT_distance_torch_batch_uniform(cos_distance, n_image, images.size(1), n_word, opt.OT_iteration)

        similarities.append(row_sim)
        similarities_OT.append(OT_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    similarities_OT = torch.cat(similarities_OT, 1)

    return similarities, similarities_OT


def dot_product_score_GW(images, captions, cap_lens, x, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    similarities_OT = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        x_i = x[i, :n_word, :].unsqueeze(0).contiguous()
        x_i_expand = x_i.repeat(n_image, 1, 1)
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """

        row_sim = matchmapSim_batch(computeMatchmap_batch(images, cap_i_expand), opt.sim_type_dot)

        #
        # weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        # cap_i_expand = cap_i_expand.contiguous()
        # weiContext = weiContext.contiguous()
        # x_i_expand = x_i_expand.contiguous()
        # # (n_image, n_word)
        #
        #
        # row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        # if opt.agg_func == 'LogSumExp':
        #     row_sim.mul_(opt.lambda_lse).exp_()
        #     row_sim = row_sim.sum(dim=1, keepdim=True)
        #     row_sim = torch.log(row_sim) / opt.lambda_lse
        # elif opt.agg_func == 'Max':
        #     row_sim = row_sim.max(dim=1, keepdim=True)[0]
        # elif opt.agg_func == 'Sum':
        #     row_sim = row_sim.sum(dim=1, keepdim=True)
        # elif opt.agg_func == 'Mean':
        #     row_sim = row_sim.mean(dim=1, keepdim=True)
        # else:
        #     raise ValueError("unknown aggfunc: {}".format(opt.agg_func))

        # cos_distance = cost_matrix_batch_torch(x_i_expand.transpose(2,1), images.transpose(2,1))
        cap_i_expand_ = cap_i_expand.transpose(2,1)
        images_ = images.transpose(2,1)
        OT_sim = GW_distance_uniform(cap_i_expand_, images_, iteration=opt.GW_iteration, OT_iteration=opt.OT_iteration)

        similarities.append(row_sim)
        similarities_OT.append(OT_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    similarities_OT = torch.cat(similarities_OT, 1)

    return similarities, similarities_OT

def dot_product_score_OT_emb(images, captions, cap_lens, x, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    similarities_OT = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        x_i = x[i, :n_word, :].unsqueeze(0).contiguous()
        x_i_expand = x_i.repeat(n_image, 1, 1)
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """

        row_sim = matchmapSim_batch(computeMatchmap_batch(images, cap_i_expand), opt.sim_type_dot)

        #
        # weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        # cap_i_expand = cap_i_expand.contiguous()
        # weiContext = weiContext.contiguous()
        # x_i_expand = x_i_expand.contiguous()
        # # (n_image, n_word)
        #
        #
        # row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        # if opt.agg_func == 'LogSumExp':
        #     row_sim.mul_(opt.lambda_lse).exp_()
        #     row_sim = row_sim.sum(dim=1, keepdim=True)
        #     row_sim = torch.log(row_sim) / opt.lambda_lse
        # elif opt.agg_func == 'Max':
        #     row_sim = row_sim.max(dim=1, keepdim=True)[0]
        # elif opt.agg_func == 'Sum':
        #     row_sim = row_sim.sum(dim=1, keepdim=True)
        # elif opt.agg_func == 'Mean':
        #     row_sim = row_sim.mean(dim=1, keepdim=True)
        # else:
        #     raise ValueError("unknown aggfunc: {}".format(opt.agg_func))

        cos_distance = cost_matrix_batch_torch(x_i_expand.transpose(2,1), images.transpose(2,1))
        OT_sim = IPOT_distance_torch_batch_uniform(cos_distance, n_image, images.size(1), n_word, opt.OT_iteration)

        similarities.append(row_sim)
        similarities_OT.append(OT_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    similarities_OT = torch.cat(similarities_OT, 1)

    return similarities, similarities_OT


class ImgEncoder(nn.Module):

	def __init__(self, img_dim, embed_size, no_imgnorm=False):
		super(ImgEncoder, self).__init__()
		self.embed_size = embed_size
		self.no_imgnorm = no_imgnorm
		self.fc = nn.Linear(img_dim, embed_size)
		self.init_weights()

	def init_weights(self):
		"""Xavier initialization for the fully connected layer
		"""
		r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
		                          self.fc.out_features)
		self.fc.weight.data.uniform_(-r, r)
		self.fc.bias.data.fill_(0)

	def forward(self, images):
		"""Extract image feature vectors."""
		# assuming that the precomputed features are already l2-normalized

		features = self.fc(images)

		# normalize in the joint embedding space
		if not self.no_imgnorm:
			features = l2norm(features, dim=-1)

		return features

	def load_state_dict(self, state_dict):
		"""Copies parameters. overwritting the default one to
		accept state_dict from Full model
		"""
		own_state = self.state_dict()
		new_state = OrderedDict()
		for name, param in state_dict.items():
			if name in own_state:
				new_state[name] = param

		super(ImgEncoder, self).load_state_dict(new_state)

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l):
        # compute image-sentence score matrix
        if self.opt.cross_attn == 't2i':
            scores = xattn_score_t2i(im, s, s_l, self.opt)
        elif self.opt.cross_attn == 'i2t':
            scores = xattn_score_i2t(im, s, s_l, self.opt)
        else:
            raise ValueError("unknown first norm type:", opt.raw_feature_norm)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()



class ContrastiveLoss_dot(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss_dot, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l, x):
        # compute image-sentence score matrix

        scores, _ = dot_product_score(im, s, s_l, x, self.opt)
        #
        # if self.opt.cross_attn == 't2i':
        #     scores = xattn_score_t2i(im, s, s_l, self.opt)
        # elif self.opt.cross_attn == 'i2t':
        #     scores = xattn_score_i2t(im, s, s_l, self.opt)
        # else:
        #     raise ValueError("unknown first norm type:", opt.raw_feature_norm)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()



class ContrastiveLoss_dot_WE(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss_dot_WE, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l, x):
        # compute image-sentence score matrix

        scores, _ = dot_product_score_WE(im, s, s_l, x, self.opt)
        #
        # if self.opt.cross_attn == 't2i':
        #     scores = xattn_score_t2i(im, s, s_l, self.opt)
        # elif self.opt.cross_attn == 'i2t':
        #     scores = xattn_score_i2t(im, s, s_l, self.opt)
        # else:
        #     raise ValueError("unknown first norm type:", opt.raw_feature_norm)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()



class ContrastiveLoss_OT(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss_OT, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l, x):
        # x is the word embedding of sentence
        # compute image-sentence score matrix
        if self.opt.cross_attn == 't2i':
            scores, scores_OT = xattn_score_t2i_OT(im, s, s_l, x, self.opt)
        elif self.opt.cross_attn == 'i2t':
            scores, scores_OT = xattn_score_i2t_OT(im, s, s_l, x, self.opt)
        else:
            raise ValueError("unknown first norm type:", self.opt.raw_feature_norm)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        diagonal_OT = scores_OT.diag().view(im.size(0), 1)
        d1_OT = diagonal_OT.expand_as(scores_OT)
        d2_OT = diagonal_OT.t().expand_as(scores_OT)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_s_OT = (self.margin + scores_OT - d1_OT).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)
        cost_im_OT = (self.margin + scores_OT - d2_OT).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        cost_s_OT = cost_s_OT.masked_fill_(I, 0)
        cost_im_OT = cost_im_OT.masked_fill_(I, 0)

        alpha = self.opt.alpha#.cuda()
        cost_s = cost_s + alpha*cost_s_OT
        cost_im = cost_im + alpha*cost_im_OT

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
            cost_s_OT = cost_s_OT.max(1)[0]
            cost_im_OT = cost_im_OT.max(0)[0]

        return cost_s.sum() + cost_im.sum(), cost_s_OT.sum()+ cost_im_OT.sum()


class ContrastiveLoss_dot_OT(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss_dot_OT, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l, x):
        # x is the word embedding of sentence
        # compute image-sentence score matrix
        # if self.opt.cross_attn == 't2i':
        #     scores, scores_OT = xattn_score_t2i_OT(im, s, s_l, x, self.opt)
        # elif self.opt.cross_attn == 'i2t':
        #     scores, scores_OT = xattn_score_i2t_OT(im, s, s_l, x, self.opt)
        # else:
        #     raise ValueError("unknown first norm type:", self.opt.raw_feature_norm)

        scores, scores_OT = dot_product_score_OT(im, s, s_l, x, self.opt)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        diagonal_OT = scores_OT.diag().view(im.size(0), 1)
        d1_OT = diagonal_OT.expand_as(scores_OT)
        d2_OT = diagonal_OT.t().expand_as(scores_OT)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_s_OT = (self.margin + scores_OT - d1_OT).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)
        cost_im_OT = (self.margin + scores_OT - d2_OT).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        cost_s_OT = cost_s_OT.masked_fill_(I, 0)
        cost_im_OT = cost_im_OT.masked_fill_(I, 0)

        alpha = self.opt.alpha#.cuda()
        cost_s = cost_s + alpha*cost_s_OT
        cost_im = cost_im + alpha*cost_im_OT

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
            cost_s_OT = cost_s_OT.max(1)[0]
            cost_im_OT = cost_im_OT.max(0)[0]

        return cost_s.sum() + cost_im.sum(), cost_s_OT.sum()+ cost_im_OT.sum()



class ContrastiveLoss_dot_OT_WE(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss_dot_OT_WE, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l, x):
        # x is the word embedding of sentence
        # compute image-sentence score matrix
        # if self.opt.cross_attn == 't2i':
        #     scores, scores_OT = xattn_score_t2i_OT(im, s, s_l, x, self.opt)
        # elif self.opt.cross_attn == 'i2t':
        #     scores, scores_OT = xattn_score_i2t_OT(im, s, s_l, x, self.opt)
        # else:
        #     raise ValueError("unknown first norm type:", self.opt.raw_feature_norm)

        scores, scores_OT = dot_product_score_OT_WE(im, s, s_l, x, self.opt)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        diagonal_OT = scores_OT.diag().view(im.size(0), 1)
        d1_OT = diagonal_OT.expand_as(scores_OT)
        d2_OT = diagonal_OT.t().expand_as(scores_OT)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_s_OT = (self.margin + scores_OT - d1_OT).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)
        cost_im_OT = (self.margin + scores_OT - d2_OT).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        cost_s_OT = cost_s_OT.masked_fill_(I, 0)
        cost_im_OT = cost_im_OT.masked_fill_(I, 0)

        alpha = self.opt.alpha#.cuda()
        cost_s = cost_s + alpha*cost_s_OT
        cost_im = cost_im + alpha*cost_im_OT

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
            cost_s_OT = cost_s_OT.max(1)[0]
            cost_im_OT = cost_im_OT.max(0)[0]

        return cost_s.sum() + cost_im.sum(), cost_s_OT.sum()+ cost_im_OT.sum()


class ContrastiveLoss_dot_GW(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss_dot_GW, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l, x):
        # x is the word embedding of sentence
        # compute image-sentence score matrix
        # if self.opt.cross_attn == 't2i':
        #     scores, scores_OT = xattn_score_t2i_OT(im, s, s_l, x, self.opt)
        # elif self.opt.cross_attn == 'i2t':
        #     scores, scores_OT = xattn_score_i2t_OT(im, s, s_l, x, self.opt)
        # else:
        #     raise ValueError("unknown first norm type:", self.opt.raw_feature_norm)

        scores, scores_OT = dot_product_score_GW(im, s, s_l, x, self.opt)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        diagonal_OT = scores_OT.diag().view(im.size(0), 1)
        d1_OT = diagonal_OT.expand_as(scores_OT)
        d2_OT = diagonal_OT.t().expand_as(scores_OT)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_s_OT = (self.margin + scores_OT - d1_OT).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)
        cost_im_OT = (self.margin + scores_OT - d2_OT).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        cost_s_OT = cost_s_OT.masked_fill_(I, 0)
        cost_im_OT = cost_im_OT.masked_fill_(I, 0)

        # alpha = self.opt.alpha#.cuda()
        # cost_s = cost_s + alpha*cost_s_OT
        # cost_im = cost_im + alpha*cost_im_OT

        cost_s = cost_s_OT
        cost_im = cost_im_OT

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum(), cost_s_OT.sum()+ cost_im_OT.sum()



class ContrastiveLoss_dot_OT_emb(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss_dot_OT_emb, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l, x):
        # x is the word embedding of sentence
        # compute image-sentence score matrix
        # if self.opt.cross_attn == 't2i':
        #     scores, scores_OT = xattn_score_t2i_OT(im, s, s_l, x, self.opt)
        # elif self.opt.cross_attn == 'i2t':
        #     scores, scores_OT = xattn_score_i2t_OT(im, s, s_l, x, self.opt)
        # else:
        #     raise ValueError("unknown first norm type:", self.opt.raw_feature_norm)

        scores, scores_OT = dot_product_score_OT_emb(im, s, s_l, x, self.opt)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        diagonal_OT = scores_OT.diag().view(im.size(0), 1)
        d1_OT = diagonal_OT.expand_as(scores_OT)
        d2_OT = diagonal_OT.t().expand_as(scores_OT)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_s_OT = (self.margin + scores_OT - d1_OT).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)
        cost_im_OT = (self.margin + scores_OT - d2_OT).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        cost_s_OT = cost_s_OT.masked_fill_(I, 0)
        cost_im_OT = cost_im_OT.masked_fill_(I, 0)

        alpha = self.opt.alpha#.cuda()
        cost_s = cost_s + alpha*cost_s_OT
        cost_im = cost_im + alpha*cost_im_OT

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum(), cost_s_OT.sum()+ cost_im_OT.sum()



class SCAN_OT(object):
    """
    Stacked Cross Attention Network (SCAN) model plus Optimal Transport
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss_OT(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list), embedded_sentence
        cap_emb, cap_lens, x = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens, x

    def forward_loss(self, img_emb, cap_emb, cap_len, x, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss, loss_OT = self.criterion(img_emb, cap_emb, cap_len, x)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        self.logger.update('Le_OT', loss_OT.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        image: bs * 36 * 2048
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens, x = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, cap_lens, x)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()





class SCAN(object):
    """
    Stacked Cross Attention Network (SCAN) model
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers, 
                                   use_bi_gru=opt.bi_gru,  
                                   no_txtnorm=opt.no_txtnorm)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list)
        cap_emb, cap_lens, x = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens, x

    def forward_loss(self, img_emb, cap_emb, cap_len, x, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb, cap_len)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens, x = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, cap_lens, x)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()



class Model_dot(object):
    """
    Stacked Cross Attention Network (SCAN) model plus Optimal Transport
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss_dot(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if volatile:
            with torch.no_grad():
                images = Variable(images)
                captions = Variable(captions)
        else:
            images = Variable(images)
            captions = Variable(captions)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()



        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list), embedded_sentence
        cap_emb, cap_lens, x = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens, x

    def forward_loss(self, img_emb, cap_emb, cap_len, x, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb, cap_len, x)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        # self.logger.update('Le_OT', loss_OT.data[0], img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        image: bs * 36 * 2048
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens, x = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, cap_lens, x)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()


class Model_(object):
    """
    Stacked Cross Attention Network (SCAN) model plus Optimal Transport
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss_dot(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if volatile:
            with torch.no_grad():
                images = Variable(images)
                captions = Variable(captions)
        else:
            images = Variable(images)
            captions = Variable(captions)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()



        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list), embedded_sentence
        cap_emb, cap_lens, x = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens, x

    def forward_loss(self, img_emb, cap_emb, cap_len, x, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb, cap_len, x)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        # self.logger.update('Le_OT', loss_OT.data[0], img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        image: bs * 36 * 2048
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens, x = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, cap_lens, x)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()



class Model_dot_WE(object):
    """
    Stacked Cross Attention Network (SCAN) model plus Optimal Transport
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = WordEmbedding(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss_dot_WE(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if volatile:
            with torch.no_grad():
                images = Variable(images)
                captions = Variable(captions)
        else:
            images = Variable(images)
            captions = Variable(captions)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()



        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list), embedded_sentence
        cap_emb, cap_lens, x = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens, x

    def forward_loss(self, img_emb, cap_emb, cap_len, x, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb, cap_len, x)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        # self.logger.update('Le_OT', loss_OT.data[0], img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        image: bs * 36 * 2048
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens, x = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, cap_lens, x)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()


class Model_dot_OT(object):
    """
    Stacked Cross Attention Network (SCAN) model plus Optimal Transport
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss_dot_OT(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list), embedded_sentence
        cap_emb, cap_lens, x = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens, x

    def forward_loss(self, img_emb, cap_emb, cap_len, x, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss, loss_OT = self.criterion(img_emb, cap_emb, cap_len, x)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        self.logger.update('Le_OT', loss_OT.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        image: bs * 36 * 2048
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens, x = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, cap_lens, x)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()



class Model_dot_OT_WE(object):
    """
    Stacked Cross Attention Network (SCAN) model plus Optimal Transport
    Word embedding is used directly, no GRU is used for text encoding
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = WordEmbedding(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss_dot_OT_WE(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list), embedded_sentence
        cap_emb, cap_lens, x = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens, x

    def forward_loss(self, img_emb, cap_emb, cap_len, x, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss, loss_OT = self.criterion(img_emb, cap_emb, cap_len, x)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        self.logger.update('Le_OT', loss_OT.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        image: bs * 36 * 2048
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens, x = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, cap_lens, x)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()


class Model_dot_GW(object):
    """
    Stacked Cross Attention Network (SCAN) model plus Optimal Transport
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss_dot_GW(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list), embedded_sentence
        cap_emb, cap_lens, x = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens, x

    def forward_loss(self, img_emb, cap_emb, cap_len, x, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss, loss_GW = self.criterion(img_emb, cap_emb, cap_len, x)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        self.logger.update('Le_GW', loss_GW.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        image: bs * 36 * 2048
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens, x = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, cap_lens, x)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()


class Model_dot_OT_emb(object):
    """
    Stacked Cross Attention Network (SCAN) model plus Optimal Transport
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.img_embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)
        self.img_emb = Image_Embedding(opt.img_embed_size*opt.num_regions, opt.num_img_features,
                                       opt.num_layers, precomp_enc_type=opt.precomp_enc_type,
                                       no_imgnorm=opt.no_imgnorm)
        self.img_emb_map = InitImgEmbMap(opt.embed_size, opt.num_img_features)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss_dot_OT_emb(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        params += list(self.img_emb_map)

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.img_emb_map]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.img_emb_map = state_dict[2]

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list), embedded_sentence
        cap_emb, cap_lens, x = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens, x

    def forward_loss(self, img_emb, cap_emb, cap_len, x, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss, loss_OT = self.criterion(img_emb, cap_emb, cap_len, x)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        self.logger.update('Le_OT', loss_OT.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        image: bs * 36 * 2048
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens, x = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, cap_lens, x)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()


class OT_score(object):
	def __init__(self, opt):
		super(OT_score, self).__init__()
		self.vocab_size = opt.vocab_size
		self.softmax_temp = opt.softmax_temp
		self.img_emb_size = opt.img_emb_size

		# self.image_model = VGG16(pretrained=opt.pretrained_image_model)
		self.text_model = BagOfWords(self.vocab_size)
		self.img_encoder = ImgEncoder(50176, self.img_emb_size)
		if torch.cuda.is_available():
			# self.image_model.cuda()
			cudnn.benchmark = True
			self.img_encoder.cuda()

		self.image_emb_map = torch.autograd.Variable(torch.tensor(torch.randn(opt.emb_dim, opt.img_emb_size)),requires_grad = True)
		self.word_emb_map = torch.autograd.Variable(torch.tensor(opt.word_embedding.transpose()), requires_grad=True)
		self.softmax_ = torch.nn.Softmax(dim=1)

	def state_dict(self):
		state_dict_ = [self.image_emb_map, self.word_emb_map]
		return state_dict_

	def load_state_dict(self, state_dict):
		# self.image_model.load_state_dict(state_dict[0])
		self.image_embed = state_dict[0]
		self.word_embed = state_dict[1]

	def train_start(self):
		"""switch to train mode"""
		# self.image_model.train()
		self.img_encoder.train()
		# self.txt_enc.train()

	def val_start(self):
		"""switch to evaluate mode"""
		# self.image_model.eval()
		self.img_encoder.eval()
		# self.txt_enc.eval()

	# def get_fimg(self, img):
	# 	img_features = self.get_img_emb(img)
	# 	img_features = self.img_encoder(img_features.reshape(img.size(0), -1))
	# 	fimg = self.softmax_(self.softmax_temp*img_features)
	# 	return fimg

	def get_fimg(self, img_features):
		# img_features = self.get_img_emb(img)
		img_features = self.img_encoder(img_features.reshape(img_features.size(0), -1))
		fimg = self.softmax_(self.softmax_temp*img_features)
		return fimg

	def get_fsent(self, sent, lsent):
		sent_features = self.text_model.calculate(sent, lsent)
		fsent = self.softmax_(self.softmax_temp*sent_features)
		return fsent

	def get_cost_matrix(self):
		return cost_matrix_torch(self.image_emb_map, self.word_emb_map)

	# def get_img_emb(self, img):
	# 	return self.image_model(img)

	def parameters(self):
		list_param = []
		list_param.append(self.word_emb_map)
		list_param.append(self.image_emb_map)
		for p in self.img_encoder.parameters():
			if p.requires_grad:
				list_param.append(p)
		# for p in self.image_model.parameters():
		# 	if p.requires_grad:
		# 		list_param.append(p)
		return list_param
