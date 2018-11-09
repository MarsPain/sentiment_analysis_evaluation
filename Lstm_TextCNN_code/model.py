import tensorflow as tf
import numpy as np
from Lstm_TextCNN_code import config
import Lstm_TextCNN_code.rnncell as rnn


class Bilstm:
    def __init__(self):
        # 初始化参数
        self.num_classes = config.num_classes
        self.sequence_length = config.max_len
        self.vocab_size = config.vocab_size + 2
        self.embed_size = config.embed_size
        self.lstm_dim = config.lstm_dim
        self.hidden_size = config.embed_size
        self.lr = config.learning_rate
        self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name="learning_rate")
        decay_rate_big = 0.50
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes = config.filter_sizes  # it is a list of int. e.g. [3,4,5]
        self.num_filters = config.num_filters
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.num_filters_total = self.num_filters * len(config.filter_sizes)  # 卷积核filter的数量
        self.clip_gradients = config.clip_gradients
        self.top_k = config.top_k
        # 设置占位符和变量
        self.Embedding_word2vec = tf.get_variable("Embedding_word2vec", shape=[self.vocab_size, self.embed_size], initializer=self.initializer)
        self.Embedding_fasttext = tf.get_variable("Embedding_fasttext", shape=[self.vocab_size, self.embed_size], initializer=self.initializer)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x1")  # sentences
        # print("input_x:", self.input_x)
        self.features_vector = tf.placeholder(tf.float32, [None, self.sequence_length], name="features_vector")  # features_vector
        # print("features_vector:", self.features_vector)
        self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")  # labels:[None,num_classes]
        # print("input_y:", self.input_y)
        self.weights = tf.placeholder(tf.float32, [None, ], name="weights_label")  # 标签权重
        # print("weights:", self.weights)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.iter = tf.placeholder(tf.int32)  # 记录training iteration
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        # 构造图
        # embed = self.embedding_layer()
        # self.lstm_outputs = self.bilstm_layer(embed)
        self.logits = self.inference_cnn()
        self.loss_val = self.loss_layer()
        self.train_op = self.train()    # 更新参数
        self.predictions = tf.argmax(self.logits, 1, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

    def embedding_layer(self):
        """
        词嵌入层，将语句的词序列转换为词向量与分割特征序列转换为词向量
        :return:[batch_size, num_steps, embedding size]
        """
        embedding_list = []
        embedded_words = tf.nn.embedding_lookup(self.Embedding_fasttext, self.input_x)
        embedding_list.append(embedded_words)
        embed = tf.concat(embedding_list, axis=-1)
        return embed

    def bilstm_layer(self, lstm_inputs):
        with tf.variable_scope("char_BiLSTM"):
            cell = tf.contrib.rnn.BasicLSTMCell(config.lstm_dim)
            outputs, final_states = tf.nn.dynamic_rnn(
                cell,
                lstm_inputs,
                dtype=tf.float32,
                time_major=False)
        # tf.transpose用于交换矩阵的维度，tf.unstack用于对矩阵进行分解，从而可以选取最后一个时间步的输出
        # hidden = tf.reshape(outputs, [-1, self.sequence_length, config.lstm_dim])
        outputs_all = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
        output_final = outputs_all[-1]
        return output_final

    def inference_cnn(self):
        # cnn features from sentences_1 and sentences_2
        x = self.conv_layers(self.input_x, 1, self.Embedding_fasttext)  # [None,num_filters_total]
        # h_cnn = self.additive_attention(x, self.hidden_size / 2, "cnn_attention")
        h_lstm = self.bilstm_layer(x)
        h = h_lstm
        h = tf.layers.dense(h, self.hidden_size, activation=tf.nn.relu, use_bias=True)  # fully connected layer
        h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)
        with tf.name_scope("output"):
            logits = tf.layers.dense(h, self.num_classes, use_bias=False)
        return logits

    def conv_layers(self, input_x, name_scope, embedding, reuse_flag=False):
        embedded_words = tf.nn.embedding_lookup(embedding, input_x)    # [None,sentence_length,embed_size]
        # idf_attention_matrix = tf.tile(self.idf_attention, [1, 1, 100])
        # embedded_words = embedded_words * idf_attention_matrix
        # [None,sentence_length,embed_size,1] expand dimension so meet input requirement of 2d-conv
        sentence_embeddings_expanded = tf.expand_dims(embedded_words, -1)   # 词向量可以是多通道的
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope(str(name_scope)+"convolution-pooling-%s" % filter_size, reuse=reuse_flag):
                # 1.create filter
                filters = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters], initializer=self.initializer)
                # 2.conv operation: conv2d===>computes a 2-D convolution given 4-D `input` and `filter` tensors.
                conv = tf.nn.conv2d(sentence_embeddings_expanded, filters, strides=[1, 1, 1, 1],
                                    padding="VALID", name="conv")   # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                # print("conv:", conv)
                # 3. apply nolinearity
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")  # [batch_size,sequence_length - filter_size + 1,1,num_filters]
                h = tf.reshape(h, [-1, self.sequence_length - filter_size + 1, self.num_filters])  # [batch_size,sequence_length - filter_size + 1,num_filters]
                # h = tf.transpose(h, [0, 2, 1])  # [batch_size,num_filters,sequence_length - filter_size + 1]
                # 4. k-max pooling
                # h = tf.nn.top_k(h, k=self.top_k, name='top_k')[0]  # [batch_size,num_filters,self.k]
                # h = tf.reshape(h, [-1, self.num_filters*self.top_k])  # [batch_size,num_filters*self.k]
                pooled_outputs.append(h)
        # 5. combine all pooled features, and flatten the feature.output' shape is a [1,None]
        h_pool = tf.concat(pooled_outputs, 2)  # shape:[batch_size, num_filters_total*self.k]
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total*self.top_k])  # shape should be:[None,num_filters_total]
        # print("h_pool_flat:", h_pool_flat)
        # 6. add dropout
        with tf.name_scope("dropout"):
            h = tf.nn.dropout(h_pool, keep_prob=self.dropout_keep_prob)    # [None,num_filters_total]
        return h

    def additive_attention(self, x, dimension_size, vairable_scope):
        with tf.variable_scope(vairable_scope):
            g = tf.get_variable("attention_g", initializer=tf.sqrt(1.0 / self.hidden_size))
            b = tf.get_variable("bias", shape=[dimension_size], initializer=tf.zeros_initializer)
            x = tf.layers.dense(x, dimension_size)  # [batch_size,hidden_size]
            h = g*tf.nn.relu(x + b)  # [batch_size,hidden_size]
        return h

    def loss_layer(self, l2_lambda=0.0003):
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
