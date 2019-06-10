import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from Attention_RCNN import config


class TextCNN:
    def __init__(self):
        # 初始化参数
        self.num_classes = config.num_classes
        self.sequence_length = config.max_len
        self.vocab_size = config.vocab_size + 2
        self.embed_size = config.embed_size
        self.hidden_size = config.embed_size
        self.lr = config.learning_rate
        self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name="learning_rate")
        decay_rate_big = 0.50
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes = config.filter_sizes  # it is a list of int. e.g. [3,4,5]
        self.num_filters = config.num_filters
        self.rnn_dim = config.rnn_dim
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.clip_gradients = config.clip_gradients
        self.top_k = config.top_k
        # 设置占位符和变量
        self.Embedding_word2vec = tf.get_variable("Embedding_word2vec", shape=[self.vocab_size, self.embed_size], initializer=self.initializer)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x1")  # sentences
        self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")  # labels:[None,num_classes]
        self.weights = tf.placeholder(tf.float32, [None, ], name="weights_label")  # 标签权重
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.iter = tf.placeholder(tf.int32)  # 记录training iteration
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        used = tf.sign(tf.abs(self.input_x))  # 计算序列中索引非0字符的数量
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)  # 记录序列除去padding（索引为0）的真实长度
        self.sequence_mask = tf.cast(tf.greater(self.input_x, 0), 'float32')
        self.batch_size = tf.shape(self.input_x)[0]
        self.num_steps = tf.shape(self.input_x)[-1]  # 序列总长度

        # 构造图
        feature = self.embed(self.input_x)
        feature = self.bi_rnn_1(feature)
        # feature = self.bi_rnn_2(feature)
        # feature = self.cnn(feature)
        feature = self.cnn_resnet(feature)
        feature_att = self.attention(feature)
        feature = self.pool_concat(feature, feature_att)
        print("feature:", feature)
        print("num_classes:", self.num_classes)
        self.logits = tf.layers.dense(feature, self.num_classes, activation=tf.nn.softmax)
        self.predictions = tf.argmax(self.logits, axis=1)
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        self.loss_val = self.loss()
        self.train_op = self.train()

    def embed(self, inputs):
        with tf.variable_scope("birnn_1"):
            embedded_words = tf.nn.embedding_lookup(self.Embedding_word2vec, inputs)
            embedded_words = tf.nn.dropout(embedded_words, keep_prob=self.dropout_keep_prob)
        return embedded_words

    def bi_rnn_1(self, inputs):
        with tf.variable_scope("birnn_1"):
            cell_for = tf.contrib.rnn.BasicLSTMCell(self.rnn_dim, state_is_tuple=True)
            cell_back = tf.contrib.rnn.BasicLSTMCell(self.rnn_dim, state_is_tuple=True)
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_for,
                                                                     cell_back,
                                                                     inputs,
                                                                     dtype=tf.float32,
                                                                     sequence_length=self.lengths,   # 之前在该项目中使用RNN时，没有考虑padding字符
                                                                     time_major=False)
            outputs = tf.concat(outputs, axis=2)
            mask = tf.expand_dims(self.sequence_mask, -1)
            outputs -= (1 - mask) * 1e10
            outputs = tf.nn.relu(outputs)
            # outputs = tf.nn.dropout(tf.nn.relu(outputs), keep_prob=self.dropout_keep_prob)
        return outputs

    def bi_rnn_2(self, inputs):
        with tf.variable_scope("birnn_2"):
            cell_for = tf.contrib.rnn.BasicLSTMCell(self.rnn_dim, state_is_tuple=True)
            cell_back = tf.contrib.rnn.BasicLSTMCell(self.rnn_dim, state_is_tuple=True)
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_for,
                                                                     cell_back,
                                                                     inputs,
                                                                     dtype=tf.float32,
                                                                     sequence_length=self.lengths,
                                                                     time_major=False)
            outputs = tf.concat(outputs, axis=2)
            outputs = tf.nn.dropout(tf.nn.relu(outputs), keep_prob=self.dropout_keep_prob)
        return outputs

    def cnn_resnet(self, inputs_origin):
        with tf.variable_scope("cnn_1"):
            inputs = tf.expand_dims(inputs_origin, -1)
            filters = tf.get_variable("filters", [self.filter_sizes, self.rnn_dim*2, 1, self.rnn_dim*2], initializer=self.initializer)
            outputs = tf.nn.conv2d(inputs, filters, strides=[1, 1, self.rnn_dim*2, 1], padding="SAME", name="conv")
            outputs = tf.reshape(outputs, [-1, self.sequence_length, self.rnn_dim*2])
            outputs = tf.nn.relu(outputs)
            outputs = tf.layers.batch_normalization(outputs)
        with tf.variable_scope("cnn_2"):
            inputs = tf.expand_dims(outputs, -1)
            filters = tf.get_variable("filters", [self.filter_sizes, self.rnn_dim*2, 1, self.rnn_dim*2], initializer=self.initializer)
            outputs = tf.nn.conv2d(inputs, filters, strides=[1, 1, self.rnn_dim*2, 1], padding="SAME", name="conv")
            outputs = tf.reshape(outputs, [-1, self.sequence_length, self.rnn_dim*2])
            # outputs = tf.nn.relu(outputs)
            outputs = tf.layers.batch_normalization(outputs)
        outputs = tf.nn.relu(tf.add(inputs_origin, outputs))
        outputs = tf.layers.dense(outputs, self.num_filters, activation=tf.nn.relu)
        # outputs = tf.nn.dropout(outputs, keep_prob=self.dropout_keep_prob)
        return outputs


    # def cnn(self, inputs):
    #     with tf.variable_scope("cnn"):
    #         inputs = tf.expand_dims(inputs, -1)
    #         filters = tf.get_variable("filters", [self.filter_sizes, self.rnn_dim*2, 1, self.num_filters], initializer=self.initializer)
    #         outputs = tf.nn.conv2d(inputs, filters, strides=[1, 1, self.rnn_dim*2, 1], padding="SAME", name="conv")
    #         outputs = tf.reshape(outputs, [-1, self.sequence_length, self.num_filters])
    #         outputs = tf.nn.relu(outputs)
    #     outputs = tf.nn.dropout(outputs, keep_prob=self.dropout_keep_prob)
    #     return outputs

    def attention(self, inputs):
        e = tf.layers.dense(tf.reshape(inputs, shape=[-1, self.num_filters]), 1, use_bias=False)
        e = tf.nn.tanh(tf.reshape(e, shape=[-1, self.num_steps]))
        a = tf.exp(e) / (tf.reduce_sum(e, axis=1, keep_dims=True) + 0.00001)
        print("a:", a)
        a = tf.expand_dims(a, 2)
        outputs = a * inputs
        outputs = tf.reduce_sum(outputs, axis=1)
        return outputs

    def pool_concat(self, inputs, inputs_att):
        inputs = tf.expand_dims(inputs, axis=-1)
        max_pool = tf.nn.max_pool(inputs, [1, self.sequence_length, 1, 1], [1, 1, 1, 1], padding='VALID')
        max_pool = tf.reshape(max_pool, shape=[-1, self.num_filters])
        print("max_pool:", max_pool)
        avg_pool = tf.nn.avg_pool(inputs, [1, self.sequence_length, 1, 1], [1, 1, 1, 1], padding='VALID')
        avg_pool = tf.reshape(avg_pool, shape=[-1, self.num_filters])
        outputs = tf.concat([max_pool, avg_pool, inputs_att], axis=1)
        return outputs

    def loss(self, l2_lambda=0.0003):
        with tf.name_scope("loss"):
            losses = tf.losses.sparse_softmax_cross_entropy(self.input_y, self.logits, weights=self.weights)
            loss_main = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss_main+l2_losses
        return loss

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, 1, 1, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate,
                                                   optimizer="Adam", clip_gradients=self.clip_gradients)
        return train_op
