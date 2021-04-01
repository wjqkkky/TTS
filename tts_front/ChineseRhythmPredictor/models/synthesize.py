# -*- coding: utf-8 -*-
"""
@Time: 2020/7/23
@author: JiangPeipei
"""
import tensorflow as tf
# from .bilstm_cbow_pred_jiang_test_haitian import BiLSTM
from tts_front.ChineseRhythmPredictor.jiang_data_processing import data_main
from tts_front.ChineseRhythmPredictor import util
import gc
# from memory_profiler import profile


class Synthesizer:
    def __init__(self, use_gpu=False, allow_soft_placement=False):
        self.use_gpu = use_gpu
        self.allow_soft_placement = allow_soft_placement

    def load(self, meta_graph, checkepoint):
        """
        加载模型
        :param meta_graph:
        :param checkepoint:
        :return: None
        """
        # ###
        """
        self.graph = tf.Graph()
        self.saver = tf.train.import_meta_graph(meta_graph, graph=self.graph)
        self.sess = tf.Session(graph=self.graph)
        # ckpt = tf.train.latest_checkpoint(checkepoint)
        self.saver.restore(self.sess, checkepoint)
        """
        self.saver = tf.train.import_meta_graph(meta_graph)
        self.graph = tf.get_default_graph()
        self.x_p = self.graph.get_operation_by_name("input_p").outputs[0]
        self.seq_len_p = self.graph.get_operation_by_name("seq_len").outputs[0]
        self.pos_p = self.graph.get_operation_by_name("pos_p").outputs[0]
        self.length_p = self.graph.get_operation_by_name("length_p").outputs[0]
        self.position_p = self.graph.get_operation_by_name("position_p").outputs[0]
        self.keep_prob_p = self.graph.get_operation_by_name("keep_prob_p").outputs[0]
        self.input_keep_prob_p = self.graph.get_operation_by_name("input_keep_prob_p").outputs[0]
        self.output_keep_prob_p = self.graph.get_operation_by_name("output_keep_prob_p").outputs[0]
        self.pred_pw = self.graph.get_operation_by_name("pw/pred").outputs[0]
        self.pred_pph = self.graph.get_operation_by_name("pph/pred").outputs[0]
        # self.sess = tf.Session()
        config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        config.gpu_options.allow_growth = self.use_gpu
        config.allow_soft_placement = self.allow_soft_placement
        self.sess = tf.Session(config=config)
        # ckpt = tf.train.latest_checkpoint(checkepoint)
        self.saver.restore(self.sess, checkepoint)

    # @profile
    def synth(self, x_test, pos_test, len_test, length_test, position_test):
        """
        模型预测
        :param x_test:
        :param pos_test:
        :param len_test:
        :param length_test:
        :param position_test:
        :return:
        """
        feed_dict = {self.x_p: x_test,
                     self.seq_len_p: len_test,
                     self.pos_p: pos_test,
                     self.length_p: length_test,
                     self.position_p: position_test,
                     self.keep_prob_p: 1.0,
                     self.input_keep_prob_p: 1.0,
                     self.output_keep_prob_p: 1.0}

        pred_pw, pred_pph = self.sess.run(fetches=[self.pred_pw, self.pred_pph], feed_dict=feed_dict)
        doc = util.recover2(
            X=x_test,
            preds_pw=pred_pw,
            preds_pph=pred_pph
        )
        del feed_dict, pred_pw, pred_pph
        gc.collect()

        return doc

    # @profile
    def pred(self, chinese):
        """
        模型预测韵律函数，
        :param chinese: 汉字字符
        :return: 带韵律的字符
        """
        # 数据转换，将汉语字符数据转化为数组格式
        x_test, pos_test, len_test, length_test, position_test = data_main(chinese)
        pos = self.synth(x_test, pos_test, len_test, length_test, position_test)
        gc.collect()
        return pos


# @profile
def get_pos(ch):
    print('ch', ch)
    x_test, pos_test, len_test, length_test, position_test = data_main(ch)
    pos = syn.pred(x_test, pos_test, len_test, length_test, position_test)
    return pos


if __name__ == '__main__':
    syn = Synthesizer()
    syn.load("./tts_front/ChineseRhythmPredictor/models/models/test/bilstm/my-model-5.meta",
             "./tts_front/ChineseRhythmPredictor/models/models/test/bilstm/my-model-5")
    # ch = '你好，欢迎来到汽车之家。'
    while True:
        ch = input('enter message : ')
        ch = ch.encode('utf-8', errors='surrogateescape').decode('utf-8')
        if ch == 'end':
            break
        post = get_pos(ch)
