#!/user/bin/env python
# -*- coding:utf-8 -*-

column_index = 2
num_classes = 4
num_epochs = 30
batch_size = 256
vocab_size = 30000
tokenize_style = "word"
embed_size = 300
filter_sizes = 3
num_filters = 128
max_len = 501   # 必须是top_k的倍数
top_k = 3
learning_rate = 0.001
clip_gradients = 3.0
validate_every = 1
dropout_keep_prob = 0.8
