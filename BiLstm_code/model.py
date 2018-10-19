import tensorflow as tf
import numpy as np
from BiLstm_code import config
import BiLstm_code.rnncell as rnn


class Bilstm:
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
        self.lstm_dim = config.lstm_dim
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.clip_gradients = config.clip_gradients
        # 设置占位符和变量
        self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size], initializer=self.initializer)
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
        embed = self.embedding_layer()
        lstm_outputs = self.bilstm_layer(embed)
        self.logits = self.project_layer(lstm_outputs)
        self.loss_val = self.loss_layer()
        print("loss:", self.loss_val)
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
        embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        embedding_list.append(embedded_words)
        embed = tf.concat(embedding_list, axis=-1)
        return embed

    def bilstm_layer(self, lstm_inputs):
        with tf.variable_scope("char_BiLSTM"):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        self.lstm_dim,
                        use_peepholes=True, initializer=self.initializer, state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                time_major=False)
        outputs_all = tf.concat(outputs, axis=2)
        print("outputs:", outputs_all)
        # tf.transpose用于交换矩阵的维度，tf.unstack用于对矩阵进行分解，从而可以选取最后一个时间步的输出
        outputs_all = tf.unstack(tf.transpose(outputs_all, [1, 0, 2]))
        print("outputs:", outputs_all)
        output_final = outputs_all[-1]
        print("output_final:", output_final)
        return output_final

    def project_layer(self, lstm_outputs):
        """
        根据lstm的输出对序列中每个字符进行预测，得到每个字符是每个标签的概率
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"):
            # 隐层的计算
            with tf.variable_scope("hidden"):
                w = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, w, b))
            # 得到标签概率
            with tf.variable_scope("logits"):
                w = tf.get_variable("W", shape=[self.lstm_dim, self.num_classes],
                                    dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable("b", shape=[self.num_classes], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                pred = tf.nn.xw_plus_b(hidden, w, b)
            # print("pred:", pred)
            return tf.reshape(pred, [-1, self.num_classes])

    def loss_layer(self):
        with tf.name_scope("loss"):
            losses = tf.losses.sparse_softmax_cross_entropy(self.input_y, self.logits, weights=self.weights)
            loss_main = tf.reduce_mean(losses)
        return loss_main

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, 1, 1, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate,
                                                   optimizer="Adam", clip_gradients=self.clip_gradients)
        return train_op
