#!/user/bin/env python
# -*- coding:utf-8 -*-

import os
# ckpt
column_index = 2
num_classes = 4
num_epochs = 35
batch_size = 32
vocab_size = 210000
tokenize_style = "word"
embed_size = 100
lstm_dim = 128
filter_sizes = [4]
num_filters = 64
max_len = 501   # 必须是top_k的倍数
top_k = 20
learning_rate = 0.001
clip_gradients = 3.0
validate_every = 1
dropout_keep_prob = 0.5
