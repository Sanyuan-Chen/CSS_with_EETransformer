#!/usr/bin/env python

import math
import torch.nn as nn
import numpy as np
import torch


class RelativePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, maxlen=1000, embed_v=False):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.maxlen = maxlen
        self.pe_k = torch.nn.Embedding(2*maxlen, d_model)
        if embed_v:
            self.pe_v = torch.nn.Embedding(2*maxlen, d_model)
        self.embed_v = embed_v

    def forward(self, pos_seq):
        pos_seq.clamp_(-self.maxlen, self.maxlen - 1)
        pos_seq = pos_seq + self.maxlen
        if self.embed_v:
            return self.pe_k(pos_seq), self.pe_v(pos_seq)
        else:
            return self.pe_k(pos_seq), None


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward

    :param int idim: input dimension
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate
    """

    def __init__(self, idim, hidden_units, dropout_rate):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)

        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)


    def forward(self, query, key, value, pos_k, pos_v, mask):
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :return torch.Tensor: attentioned and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)  #(b, t, d)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)   #(b, t, d)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        A = torch.matmul(q, k.transpose(-2, -1))
        if pos_k is not None:
            reshape_q = q.contiguous().view(n_batch * self.h, -1, self.d_k).transpose(0,1)
            B = torch.matmul(reshape_q, pos_k.transpose(-2, -1))
            B = B.transpose(0, 1).view(n_batch, self.h, pos_k.size(0), pos_k.size(1))
            scores = (A + B) / math.sqrt(self.d_k)
        else:
            scores = A / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        if pos_v is not None:
            reshape_attn = p_attn.contiguous().view(n_batch * self.h, pos_v.size(0), pos_v.size(1)).transpose(0,1)      #(t1, bh, t2)

            attn_v = torch.matmul(reshape_attn, pos_v).transpose(0,1).contiguous().view(n_batch, self.h, pos_v.size(0), self.d_k)
            x = x + attn_v
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)


class EncoderLayer(nn.Module):
    """Encoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
        feed forward module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(self, size, self_attn, feed_forward, dropout_rate,
                 normalize_before=True, concat_after=False, attention_heads=8):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = torch.nn.LayerNorm(size, eps=1e-12)
        self.norm2 = torch.nn.LayerNorm(size, eps=1e-12)
        self.norm_k = torch.nn.LayerNorm(size//attention_heads, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(self, x, pos_k, pos_v, mask):
        """Compute encoded features.

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
            if pos_k is not None:
                pos_k = self.norm_k(pos_k)
            if pos_v is not None:
                pos_v = self.norm_v(pos_v)
        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x, x, x, pos_k, pos_v, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x, x, x, pos_k, pos_v, mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask


class EETransformerEncoder(torch.nn.Module):
    """Early Exit Transformer encoder module.

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(self,
                 idim=1799,
                 attention_dim=256,
                 attention_heads=4,
                 linear_units=2048,
                 num_blocks=16,
                 dropout_rate=0.1,
                 positional_dropout_rate=0.1,
                 attention_dropout_rate=0.0,
                 relative_pos_emb=True,
                 normalize_before=True,
                 concat_after=False,
                 exit_classifiers=None):
        super(EETransformerEncoder, self).__init__()

        self.embed = torch.nn.Sequential(
            torch.nn.Linear(idim, attention_dim),
            torch.nn.LayerNorm(attention_dim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
        )

        if relative_pos_emb:
            self.pos_emb = RelativePositionalEncoding(attention_dim // attention_heads, 1000)
        else:
            self.pos_emb = None

        self.encoders = torch.nn.Sequential(*[EncoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim, attention_dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
                attention_heads
            ) for _ in range(num_blocks)])

        self.dropout_layer = torch.nn.Dropout(p=positional_dropout_rate)
        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-12) if normalize_before else None

        self.exit_classifiers = exit_classifiers
        self.inference_exit_layers = []

    def forward(self, xs, masks, early_exit_threshold=0):
        xs = self.embed(xs)

        if self.pos_emb is not None:
            x_len = xs.shape[1]
            pos_seq = torch.arange(0, x_len).long().to(xs.device)
            pos_seq = pos_seq[:, None] - pos_seq[None, :]
            pos_k, pos_v = self.pos_emb(pos_seq)
        else:
            pos_k, pos_v = None, None

        xs = self.dropout_layer(xs)

        if self.training:
            # during training, return the estimated masks of all the layers
            results = []
            for i, layer in enumerate(self.encoders):
                xs, _ = layer(xs, pos_k, pos_v, masks)
                output = self.after_norm(xs) if self.after_norm else xs
                output = self.exit_classifiers[i](output)
                output = torch.sigmoid(output)
                results.append(output)
        else:
            # We dynamically stop the inference if the predictions from two consecutive layers are sufficiently similar
            last_predicts = None
            calculated_layer_num = 0
            for i, layer in enumerate(self.encoders):
                calculated_layer_num += 1
                xs, _ = layer(xs, pos_k, pos_v, masks)
                output = self.after_norm(xs) if self.after_norm else xs
                logits = self.exit_classifiers[i](output)
                predicts = torch.sigmoid(logits)
                predicts = predicts.detach()
                if (last_predicts is not None) and torch.dist(last_predicts, predicts, p=2) / last_predicts.shape[1] / last_predicts.shape[2] < early_exit_threshold:
                    last_predicts = predicts
                    break
                else:
                    last_predicts = predicts
            results = [last_predicts]
            self.inference_exit_layers.append(calculated_layer_num)
        return results, masks


default_encoder_conf = {
    "attention_dim": 256,
    "attention_heads": 4,
    "linear_units": 2048,
    "num_blocks": 16,
    "dropout_rate": 0.1,
    "positional_dropout_rate": 0.1,
    "attention_dropout_rate": 0.0,
    "relative_pos_emb": True,
    "normalize_before": True,
    "concat_after": False,
}


class EETransformerCSS(nn.Module):
    """
    Early Exit Transformer speech separation model
    """
    def __init__(self,
                 stats_file=None,
                 in_features=1799,
                 num_bins=257,
                 num_spks=2,
                 num_nois=1,
                 transformer_conf=default_encoder_conf):
        super(EETransformerCSS, self).__init__()

        # input normalization layer
        if stats_file is not None:
            stats = np.load(stats_file)
            self.input_bias = torch.from_numpy(
                np.tile(np.expand_dims(-stats['mean'].astype(np.float32), axis=0), (1, 1, 1)))
            self.input_scale = torch.from_numpy(
                np.tile(np.expand_dims(1 / np.sqrt(stats['variance'].astype(np.float32)), axis=0), (1, 1, 1)))
            self.input_bias = nn.Parameter(self.input_bias, requires_grad=False)
            self.input_scale = nn.Parameter(self.input_scale, requires_grad=False)
        else:
            self.input_bias = torch.zeros(1, 1, in_features)
            self.input_scale = torch.ones(1, 1, in_features)
            self.input_bias = nn.Parameter(self.input_bias, requires_grad=False)
            self.input_scale = nn.Parameter(self.input_scale, requires_grad=False)

        self.num_bins = num_bins
        self.num_spks = num_spks
        self.num_nois = num_nois
        self.linear = nn.ModuleList([nn.Linear(transformer_conf["attention_dim"], num_bins * (num_spks + num_nois))
                                     for _ in range(transformer_conf['num_blocks'])])

        # Transformers
        self.transformer = EETransformerEncoder(in_features, **transformer_conf, exit_classifiers=self.linear)

    def forward(self, f, early_exit_threshold=0):
        """
        args
            f: N x * x T
        return
            m: [N x F x T, ...]
        """
        # N x * x T => N x T x *
        f = f.transpose(1, 2)

        # global feature normalization
        f = f + self.input_bias
        f = f * self.input_scale

        m_list, _ = self.transformer(f, masks=None, early_exit_threshold=early_exit_threshold)
        res_m = []
        for m in m_list:
            # N x T x F => N x F x T
            m = m.transpose(1, 2)
            m = torch.chunk(m, self.num_spks + self.num_nois, 1)
            res_m.append(m)

        if not self.training:
            assert len(res_m) == 1
            res_m = res_m[0]

        return res_m

