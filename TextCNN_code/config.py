#!/user/bin/env python
# -*- coding:utf-8 -*-

import os
model_save_path = os.path.abspath('..') + "/data"
num_epochs = 10
batch_size = 64
tokenize_style = "word"
embed_size = 100
filter_sizes = [3, 4, 5]
num_filters = 64
sentence_len = 39   # 必须是top_k的倍数
top_k = 1
learning_rate = 0.001
clip_gradients = 3.0
validate_every = 1
dropout_keep_prob = 0.5
