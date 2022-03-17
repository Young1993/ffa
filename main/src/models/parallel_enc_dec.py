#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'fkb'
__email__ = 'fkb@zjuici.com'

import torch
import torch.nn as nn
from transformers import BartModel, FunnelModel


decode_model_name = "facebook/bart-large"
encode_model_name = "funnel-transformer/xlarge"


class ParallelEndecoderGraph(nn.Module):
    """docstring for LanModelGraph"""

    def __init__(self, config):
        super(ParallelEndecoderGraph, self).__init__()
        self.config = config
        self.encode_layer = FunnelModel.from_pretrained(encode_model_name, hidden_dropout=config.xfmr_hidden_dropout_prob)
        self.decode_layer = BartModel.from_pretrained(decode_model_name, use_cache=False, gradient_checkpointing=True, output_hidden_states=True, dropout=config.xfmr_hidden_dropout_prob)

        fusion_in_features = 2*config.lan_hidden_size
        self.fusion_layer = nn.TransformerEncoderLayer(
             d_model=fusion_in_features
           , nhead=config.xfmr_num_attention_heads
           , dim_feedforward=config.xfmr_intermediate_size
           , dropout=config.xfmr_hidden_dropout_prob
           , activation='gelu'
              )

        self.dropout_layer = nn.Dropout(config.xfmr_hidden_dropout_prob)
        self.out_layer = nn.Linear(
                in_features=fusion_in_features
                , out_features=len(config.label2idx_dict))

        # freeze the Pre-trained Language Model
        if config.freeze_lan_model:
            for param in self.lan_layer.base_model.parameters():
                param.requires_grad = False

    def forward(self, xs, x_masks, y_tags=None, y_mask=None):
        xs_decode = self.decode_layer(xs, attention_mask=x_masks)[0]
        xs_encode = self.encode_layer(xs, attention_mask=x_masks)[0]

        xs = torch.cat((xs_decode, xs_encode), dim=-1)
        xs = self.fusion_layer(xs)
        x = self.dropout_layer(xs)
        ys = self.out_layer(x)

        return ys

