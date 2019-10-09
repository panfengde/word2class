import tensorflow as tf
import numpy as np
import pandas as pd


class word2class():
    def __init__(self, vec_dim, n_class):
        self.global_steps = tf.Variable(0, trainable=False)
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=[None, vec_dim], name='input_dim')
        self.target_placeholder = tf.placeholder(tf.int32, shape=[None], name='input_target')
        self.n_class = n_class
        self.learning_rate = 0.002
        self.epoch = 50
        self.batch_size = 256
        self.sess = tf.Session()
        self.build()

    def build_graphy(self):
        with tf.variable_scope('graphy'):
            middle = tf.layers.dense(self.inputs_placeholder, 56)
            self.result = tf.layers.dense(middle, self.n_class)
            self.pre_result = tf.argmax(tf.nn.softmax(self.result), 1)
            new_step = tf.add(self.global_steps, tf.constant(1))
            self.update_step = tf.assign(self.global_steps, new_step)

    def add_loss(self):

        self.loss_list = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.result,
                                                                        labels=self.target_placeholder)
        self.batch_loss = tf.reduce_mean(self.loss_list)
        tf.summary.scalar('batch_loss', self.batch_loss)

    def add_training_op(self):
        self.training_op = tf.train.RMSPropOptimizer(self.learning_rate, name='training_op').minimize(
            self.batch_loss)
        self.summaries = tf.summary.merge_all()

    def build(self):
        self.build_graphy()
        self.add_loss()
        self.add_training_op()

    def readData(self, data_path):
        """
        数据为csv文件，

        格式
        ------------------------
        |  vec(字符串) | label |
        | '1,2,3,4'   |   4   |
        ---------------------
        :return:
        """
        chunker = pd.read_csv(data_path, chunksize=self.batch_size)
        return chunker

    def train(self, data_path, model_save_path):
        writer = tf.summary.FileWriter("./tensorboard", self.sess.graph)
        saver = tf.train.Saver(max_to_keep=3)
        self.sess.run(tf.global_variables_initializer())
        model_file = tf.train.latest_checkpoint(model_save_path)
        if model_file:
            saver.restore(self.sess, model_file)
        print('--开始训练')
        # for epoch in range(self.epoch):
        #     input_batch_tokens = np.random.randn(10, 158)
        #     target_batch = np.random.randint(0, 4, (10), dtype=np.int32)
        #     # target_batch = np.zeros([10, 5], dtype=np.int32)
        #     # for row in target_batch:
        #     #     row[np.random.randint(0, 4, dtype=np.int32)] = 1
        step = 0
        for k in range(self.epoch):
            chunker = self.readData(data_path)
            for one_chunk in chunker:
                input_batch_tokens = [one.split(',') for one in one_chunk['vec'].tolist()]
                target_batch = one_chunk['label']
                _, summ, step = self.sess.run([self.training_op, self.summaries, self.update_step], feed_dict={
                    self.inputs_placeholder: input_batch_tokens,
                    self.target_placeholder: target_batch
                })
                writer.add_summary(summ, global_step=(step))
            saver.save(self.sess, model_save_path + 'model' + str(k) + '.ckpt')
            print(k, "------------------", step)
        self.sess.close()


oneModel = word2class(158, 5)
oneModel.train('./data.csv', './model_save/')
