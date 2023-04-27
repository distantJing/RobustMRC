import tensorflow as tf
import tensorflow_hub as tfh
# from tensorflow_hub import *
# from bert_serving import client, server
from func_add_knowledge import dropout, gru_n, cudnn_gru, dense, softmax
from func_add_knowledge import highway, highway_rnn
from func_add_knowledge import DCN_plus_single_step, interactive_aligning, easy_dot_attention
from func_add_knowledge import my_pointer_network, memory_based_answer_pointer, question_focused_attentional_pointer

from func_add_knowledge import semantic_fusion_unit, compute_similarity_matrix
from func_add_knowledge import interactive_aligning_add_knowledge

class Model(object):
    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True, bert_client=None):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        # self.c:     context_idx                [batch_size, para_limit]
        # self.q:     question_idx               [batch_size, ques_limit]
        # self.ch:    context_char_idx           [batch_size, para_limit, char_limit]
        # self.qh:    question_idx               [batch_size, ques_limit, char_limit]
        # self.y1:    [para_limit]  概率分布      [batch_size, para_limit]
        # self.y2:    [para_limit]  概率分布      [batch_size, para_limit]
        # self.qa_id: question_id                 [batch_size]
        # self.context_tokens：  context string to elmo input [batch_size, a string] 
        # self.question_tokens： question string to elmo input [batch_size, a string]
            # there is no blank substring in context tokens, so the lenth of context_string is the 
                # smallest length in this batch data
        # self.passage_connections： passage to passage connections  [batch_size, para_limit, para_limit]
        # self.question_connections: question to passage connections  [batch_size, ques_limit, para_limit]
        # self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id = batch.get_next()
        self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id, self.context_tokens, self.question_tokens, self.passage_connections, self.question_connections, self.na = batch.get_next()
        self.na = tf.cast(self.na, tf.float32)        
        self.is_train = tf.get_variable("is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32),
                                        trainable=False)
        self.char_mat = tf.get_variable("char_mat", char_mat.shape, dtype=tf.float32)

        self.c_mask = tf.cast(self.c, tf.bool)    # [batch, para_limit]
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1) # [batch]
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
        self.answer_maxlen = config.answer_limit

        # self.elmo_model = tfh.Module(spec=config.elmo_url, trainable=config.elmo_trainable)
        self.elmo_model = tfh.Module(spec=config.elmo_url, trainable=config.elmo_trainable)

        if opt:
            N, CL = config.batch_size, config.char_limit
            self.c_maxlen = tf.reduce_max(self.c_len)
            self.q_maxlen = tf.reduce_max(self.q_len)
            self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
            self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
            self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
            self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
            self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
            self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
            # self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
            # self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
            self.passage_connections = tf.slice(self.passage_connections, [0,0,0], [N, self.c_maxlen, self.c_maxlen])
            self.question_connections = tf.slice(self.question_connections, [0,0,0], [N, self.q_maxlen, self.c_maxlen])
        else:
            self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

        self.ch_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
        self.qh_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

        # 使用rnn将passage, question 分别编码
        self.embedding()  # word + char + elmo embeddings
        self.embedding_keys() # add passage and question connections
        # self.embedding_char_glove_elmo_bert()
        # self.highway_conv_1()
        # self.highway_rnn_1()
        # self.highway_encoding()
        # self.highway_rnn_encoding()
        # 交互部分
        # self.DCN_plus()
        # self.dot_attention()
        # self.dot_self_attention()
        self.multi_interactive_aligning_add_knowledge()
        self.match_keys()
        # self.multi_interactive_aligning()
        # self.highway_conv_2()
        # self.highway_rnn_2()
        self.multi_interactive_self_aligning_add_knowledge()
        # self.multi_interactive_self_aligning()
        # self.highway_conv_3()
        # self.highway_rnn_3()
        # 答案预测部分
        self.pointer_network()
        # self.memory_based_pointer_network()
        # self.pointer_network_multi_loss()
        # self.question_focused_pointing_network()

        if trainable:
            self.lr = tf.get_variable("lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)
            tf.train.AdamOptimizer()
            reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4),
                                                         tf.trainable_variables())
            self.loss = self.loss + reg
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)

        # if trainable:
        #     self.lr1 = tf.minimum(config.learning_rate,
        #                          0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
        #     self.opt = tf.train.AdamOptimizer(learning_rate = self.lr1, beta1 = 0.8, beta2 = 0.999, epsilon = 1e-7)
        #     grads = self.opt.compute_gradients(self.loss)
        #     gradients, variables = zip(*grads)
        #     capped_grads, _ = tf.clip_by_global_norm(
        #         gradients, config.grad_clip)
        #     self.train_op = self.opt.apply_gradients(
        #         zip(capped_grads, variables), global_step=self.global_step)



    def embedding(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        keep_prob = self.config.keep_prob
        with tf.variable_scope("embedding"):
            with tf.variable_scope("char"):
                c_maxlen = self.c_maxlen
                q_maxlen = self.q_maxlen
                char_limit = self.config.char_limit
                char_dim = self.config.char_dim
                char_hidden_size = self.config.char_hidden_size
                self.ch_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.ch),
                                         [batch_size*c_maxlen, char_limit, char_dim])
                self.qh_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.qh),
                                         [batch_size*q_maxlen, char_limit, char_dim])
                ch_emb = dropout(self.ch_emb, keep_prob=keep_prob, is_train=self.is_train)   # mode = embedding ???
                qh_emb = dropout(self.qh_emb, keep_prob=keep_prob, is_train=self.is_train)
                print("self.c", self.c)
                print("self.ch", self.ch)
                print("self.ch_emb", self.ch_emb)
                print("ch_emb", ch_emb)
                cell_fw = gru_n(hidden_size=char_hidden_size, num_layer=1, is_train=self.is_train)
                cell_bw = gru_n(hidden_size=char_hidden_size, num_layer=1, is_train=self.is_train)
                _, c_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                           cell_bw=cell_bw,
                                                           inputs=ch_emb,
                                                           sequence_length=self.ch_len,
                                                           dtype=tf.float32)
                self.ch_enc = tf.reshape(tf.concat(c_state, axis=1), [batch_size, c_maxlen, 2*char_hidden_size])
                _, q_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                             cell_bw=cell_bw,
                                                             inputs=qh_emb,
                                                             sequence_length=self.qh_len,
                                                             dtype=tf.float32)
                self.qh_enc = tf.reshape(tf.concat(q_state, axis=1), [batch_size, q_maxlen, 2*char_hidden_size])
            with tf.variable_scope("word"):
                self.c_word_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                self.q_word_emb = tf.nn.embedding_lookup(self.word_mat, self.q)
            with tf.variable_scope("elmo"):
                self.c_elmo = self.elmo_model(self.context_tokens, signature="default", as_dict=True)["elmo"]
                self.q_elmo = self.elmo_model(self.question_tokens, signature="default", as_dict=True)["elmo"]

            # self.c_emb = tf.concat([self.c_word_emb, self.ch_enc], axis=2)
            # self.q_emb = tf.concat([self.q_word_emb, self.qh_enc], axis=2)
            self.c_emb = tf.concat([self.c_word_emb, self.ch_enc, self.c_elmo], axis=2)
            self.q_emb = tf.concat([self.q_word_emb, self.qh_enc, self.q_elmo], axis=2)

        with tf.variable_scope("embedding_encoding"):
            num_layer = 1
            inputs_size = self.c_emb.get_shape().as_list()[-1]
            rnn = cudnn_gru(num_layers=num_layer, num_units=hidden_size, batch_size=batch_size,
                            input_size=inputs_size, keep_prob=self.config.keep_prob, is_train=self.is_train)
            self.c_enc = rnn(self.c_emb, self.c_len)
            self.q_enc = rnn(self.q_emb, self.q_len) 


    def embedding_char_glove_elmo_bert(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        keep_prob = self.config.keep_prob
        with tf.variable_scope("embedding"):
            with tf.variable_scope("char"):
                c_maxlen = self.c_maxlen
                q_maxlen = self.q_maxlen
                char_limit = self.config.char_limit
                char_dim = self.config.char_dim
                char_hidden_size = self.config.char_hidden_size
                self.ch_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.ch),
                                         [batch_size*c_maxlen, char_limit, char_dim])
                self.qh_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.qh),
                                         [batch_size*q_maxlen, char_limit, char_dim])
                ch_emb = dropout(self.ch_emb, keep_prob=keep_prob, is_train=self.is_train)   # mode = embedding ???
                qh_emb = dropout(self.qh_emb, keep_prob=keep_prob, is_train=self.is_train)
                print("self.c", self.c)
                print("self.ch", self.ch)
                print("self.ch_emb", self.ch_emb)
                print("ch_emb", ch_emb)
                cell_fw = gru_n(hidden_size=char_hidden_size, num_layer=1, is_train=self.is_train)
                cell_bw = gru_n(hidden_size=char_hidden_size, num_layer=1, is_train=self.is_train)
                _, c_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                           cell_bw=cell_bw,
                                                           inputs=ch_emb,
                                                           sequence_length=self.ch_len,
                                                           dtype=tf.float32)
                self.ch_enc = tf.reshape(tf.concat(c_state, axis=1), [batch_size, c_maxlen, 2*char_hidden_size])
                _, q_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                             cell_bw=cell_bw,
                                                             inputs=qh_emb,
                                                             sequence_length=self.qh_len,
                                                             dtype=tf.float32)
                self.qh_enc = tf.reshape(tf.concat(q_state, axis=1), [batch_size, q_maxlen, 2*char_hidden_size])
            with tf.variable_scope("word"):
                self.c_word_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                self.q_word_emb = tf.nn.embedding_lookup(self.word_mat, self.q)
            with tf.variable_scope("elmo"):
                self.c_elmo = self.elmo_model(self.context_tokens, signature="default", as_dict=True)["elmo"]
                self.q_elmo = self.elmo_model(self.question_tokens, signature="default", as_dict=True)["elmo"]
            # with tf.variable_scope("bert"):
            #     bert_input = [item] 
            #     bert_client.encode()
            self.c_emb = tf.concat([self.c_word_emb, self.ch_enc], axis=2)
            self.q_emb = tf.concat([self.q_word_emb, self.qh_enc], axis=2)
            # self.c_emb = tf.concat([self.c_word_emb, self.ch_enc, self.c_elmo], axis=2)
            # self.q_emb = tf.concat([self.q_word_emb, self.qh_enc, self.q_elmo], axis=2)

        with tf.variable_scope("embedding_encoding"):
            num_layer = 1
            inputs_size = self.c_emb.get_shape().as_list()[-1]
            rnn = cudnn_gru(num_layers=num_layer, num_units=hidden_size, batch_size=batch_size,
                            input_size=inputs_size, keep_prob=self.config.keep_prob, is_train=self.is_train)
            self.c_enc = rnn(self.c_emb, self.c_len)
            self.q_enc = rnn(self.q_emb, self.q_len) 

    
    def embedding_keys(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        keep_prob = self.config.keep_prob
        with tf.variable_scope("passage_keys"):
            passage_additional_info = tf.matmul(
                softmax(tf.math.multiply(self.passage_connections, # [b, p, p]
                            compute_similarity_matrix(self.c_enc, self.c_mask, self.c_enc, self.c_mask,
                                    keep_prob=keep_prob, hidden=hidden_size, is_train=self.is_train)),# [b, p, p]
                        axis=2, 
                        mask=self.passage_connections), # [b, p, p]
                self.c_enc) # [b, p, h]  -> [b, p, h]
            self.passage_keys = semantic_fusion_unit(self.c_enc, passage_additional_info, keep_prob,is_train=self.is_train)
        with tf.variable_scope("question_keys"):
            question_additional_info = tf.matmul(
                softmax(tf.math.multiply(self.question_connections, # [b, q, p]
                            compute_similarity_matrix(self.q_enc, self.q_mask, self.c_enc, self.c_mask,
                                    keep_prob=keep_prob, hidden=hidden_size, is_train=self.is_train)),# [b, q, p]
                        axis=2, 
                        mask=self.question_connections), # [b, q, p]
                self.c_enc) # [b, p, h]  -> [b, q, h]
            self.question_keys = semantic_fusion_unit(self.q_enc, question_additional_info, keep_prob,is_train=self.is_train)
       

    def multi_interactive_aligning_add_knowledge(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        keep_prob = self.config.keep_prob
        with tf.variable_scope("multi_interactive_aligning"):
            c_t, self.M = interactive_aligning_add_knowledge(self.c_enc, self.passage_keys, self.c_mask, self.q_enc, self.question_keys, 
                                self.q_mask, hidden_size, layer=1, keep_prob=keep_prob, # similarity_mode="dot_attention",
                                       is_train=self.is_train)
            rnn = cudnn_gru(num_layers=1, num_units=hidden_size, batch_size=batch_size,
                            input_size=c_t.get_shape().as_list()[-1], keep_prob=keep_prob,
                            is_train=self.is_train, scope="align_rnn")
            self.match = rnn(c_t, self.c_len)

    def match_keys(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        keep_prob = self.config.keep_prob
        with tf.variable_scope("match_keys"):
            match_additional_info = tf.matmul(
                softmax(tf.math.multiply(self.passage_connections, # [b, p, p]
                            compute_similarity_matrix(self.match, self.c_mask, self.match, self.c_mask,
                                    keep_prob=keep_prob, hidden=hidden_size, is_train=self.is_train)),# [b, p, p]
                        axis=2, 
                        mask=self.passage_connections), # [b, p, p]
                self.match) # [b, p, h]  -> [b, p, h]
            self.match_keys = semantic_fusion_unit(self.match, match_additional_info, keep_prob,is_train=self.is_train)
  

    def multi_interactive_self_aligning_add_knowledge(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        keep_prob = self.config.batch_size
        with tf.variable_scope("multi_interactive_self_aligning"):
            c_t, self.self_M = interactive_aligning_add_knowledge(self.match, self.match_keys, self.c_mask, self.match,self.match_keys,self.c_mask, hidden_size,
                                       layer=1, keep_prob=keep_prob, # similarity_mode="dot_attention",
                                       is_train=self.is_train)
            rnn_self = cudnn_gru(num_layers=1, num_units=hidden_size, batch_size=batch_size,
                                 input_size=c_t.get_shape().as_list()[-1], keep_prob=keep_prob,
                                 is_train=self.is_train, scope="self_align_rnn")
            self.self_match = rnn_self(c_t, self.c_len)



















    def highway_conv_1(self):
        self.c_enc = highway(self.c_enc, scope="highway_1", reuse=None)
        self.q_enc = highway(self.q_enc, scope="highway_1", reuse=True)
    def highway_rnn_1(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        keep_prob = self.config.keep_prob
        self.c_enc = highway_rnn(self.c_enc, self.c_len, num_layer=1, hidden_size=hidden_size, batch_size=batch_size,
                                 keep_prob=keep_prob, is_train=self.is_train, scope="highway", reuse=None)
        self.q_enc = highway_rnn(self.q_enc, self.q_len, num_layer=1, hidden_size=hidden_size, batch_size=batch_size,
                                 keep_prob=keep_prob, is_train=self.is_train, scope="highway", reuse=True)
    def highway_encoding(self):
        with tf.variable_scope("highway_embedding_encoding"):
            # todo: 如何加入mask信息
            self.c_enc = highway(self.c_emb, scope="highway", reuse=None)
            self.q_enc = highway(self.q_emb, scope="highway", reuse=True)
    def highway_rnn_encoding(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        keep_prob = self.config.keep_prob
        with tf.variable_scope("highway_embedding_encoding"):
            self.c_enc = highway_rnn(self.c_emb, self.c_len, num_layer=1, hidden_size=hidden_size, batch_size=batch_size,
                                     keep_prob=keep_prob, is_train=self.is_train, scope="highway", reuse=None)
            self.q_enc = highway_rnn(self.q_emb, self.q_len, num_layer=1, hidden_size=hidden_size, batch_size=batch_size,
                                     keep_prob=keep_prob, is_train=self.is_train, scope="highway", reuse=True)


    def dot_attention(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        keep_prob = self.config.keep_prob
        with tf.variable_scope("match_attention"):
            qc_att = easy_dot_attention(self.c_enc, self.c_mask, self.q_enc, self.q_mask,
                                        hidden_size, keep_prob, is_train=self.is_train, scope="match_attention")
            rnn = cudnn_gru(num_layers=1, num_units=hidden_size, batch_size=batch_size,
                             input_size=qc_att.get_shape().as_list()[-1], keep_prob=keep_prob,
                             is_train=self.is_train, scope="rnn")
            self.match = rnn(qc_att, seq_len=self.c_len)
            print("self_match", self.match)


    def dot_self_attention(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        keep_prob = self.config.keep_prob
        with tf.variable_scope("self_match_attention"):
            cc_att = easy_dot_attention(self.match, self.c_mask, self.match, self.c_mask,
                                        hidden_size, keep_prob, self_attention=True,
                                        is_train=self.is_train, scope="self_match_attention")
            rnn = cudnn_gru(num_layers=1, num_units=hidden_size, batch_size=batch_size,
                            input_size=cc_att.get_shape().as_list()[-1], keep_prob=keep_prob,
                            is_train=self.is_train, scope="rnn_2")
            self.self_match = rnn(cc_att, seq_len=self.c_len)


    def DCN_plus(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        keep_prob = self.config.keep_prob
        with tf.variable_scope("match_attention"):
            with tf.variable_scope("pro_question"):
                q_last_units = self.q_enc.get_shape().as_list()[-1]
                Q = tf.nn.tanh(dense(self.q_enc, q_last_units, use_bias=True, scope="q"))
                self.q_enc = Q
            with tf.variable_scope("co_attention_layer_one"):
                S_1_D, S_1_Q, C_1_D = DCN_plus_single_step(self.c_enc, self.c_mask, self.q_enc, self.q_mask,
                                                           hidden_size, keep_prob, self.is_train)
                rnn1 = cudnn_gru(num_layers=1, num_units=hidden_size, batch_size=batch_size,
                                 input_size=S_1_D.get_shape().as_list()[-1], keep_prob=keep_prob,
                                 is_train=self.is_train, scope="rnn1")
                E_2_D = rnn1(S_1_D, self.c_len)
                E_2_Q = rnn1(S_1_Q, self.q_len)
            with tf.variable_scope("co_attention_layer_two"):
                S_2_D, S_2_Q, C_2_D = DCN_plus_single_step(E_2_D, self.c_mask, E_2_Q, self.q_mask,
                                                           hidden_size, keep_prob, self.is_train)
            with tf.variable_scope("co_attention_combine"):
                new_inputs = tf.concat([self.c_enc, E_2_D, S_1_D, S_2_D, C_1_D, C_2_D], axis=2)
                rnn2 = cudnn_gru(num_layers=1, num_units=hidden_size, batch_size=batch_size,
                                 input_size=new_inputs.get_shape().as_list()[-1], keep_prob=keep_prob,
                                 is_train=self.is_train, scope="rnn2")
                self.match = rnn2(new_inputs, self.c_len)


    def multi_interactive_aligning(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        keep_prob = self.config.keep_prob
        with tf.variable_scope("multi_interactive_aligning"):
            c_t, self.M = interactive_aligning(self.c_enc, self.c_mask, self.q_enc, self.q_mask, hidden_size,
                                       layer=2, keep_prob=keep_prob, # similarity_mode="dot_attention",
                                       is_train=self.is_train)
            rnn = cudnn_gru(num_layers=1, num_units=hidden_size, batch_size=batch_size,
                            input_size=c_t.get_shape().as_list()[-1], keep_prob=keep_prob,
                            is_train=self.is_train, scope="align_rnn")
            self.match = rnn(c_t, self.c_len)

    def highway_conv_2(self):
        self.match = highway(self.match, scope="highway_2", reuse=None)
    def highway_rnn_2(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        keep_prob = self.config.keep_prob
        self.match = highway_rnn(self.match, self.c_len, num_layer=1, hidden_size=hidden_size, batch_size=batch_size,
                                 keep_prob=keep_prob, is_train=self.is_train, scope="highway", reuse=None)

    def multi_interactive_self_aligning(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        keep_prob = self.config.batch_size
        with tf.variable_scope("multi_interactive_self_aligning"):
            c_t, M = interactive_aligning(self.match, self.c_mask, self.match, self.c_mask, hidden_size,
                                       layer=1, keep_prob=keep_prob, # similarity_mode="dot_attention",
                                       is_train=self.is_train)
            rnn_self = cudnn_gru(num_layers=1, num_units=hidden_size, batch_size=batch_size,
                                 input_size=c_t.get_shape().as_list()[-1], keep_prob=keep_prob,
                                 is_train=self.is_train, scope="self_align_rnn")
            self.self_match = rnn_self(c_t, self.c_len)

    def highway_conv_3(self):
        self.self_match = highway(self.self_match, scope="highway_3", reuse=None)
    def highway_rnn_3(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        keep_prob = self.config.keep_prob
        self.self_match = highway_rnn(self.self_match, self.c_len, num_layer=1, hidden_size=hidden_size,
                                      batch_size=batch_size, keep_prob=keep_prob, is_train=self.is_train,
                                      scope="highway", reuse=None)

    # def pointer_network(self):
    #     hidden_size = self.config.hidden_size
    #     batch_size = self.config.batch_size
    #     # final_output = self.match
    #     final_output = self.self_match
    #     # final_output = tf.concat([self.match, self.self_match], axis=2)
    #     with tf.variable_scope("pointer_network"):
    #         logits1, logits2 = my_pointer_network(final_output, self.c_mask, self.q_enc, self.q_mask,
    #                                               batch_size, hidden_size, self.config.ptr_keep_prob,
    #                                               self.is_train, scope="my_pointer_network")

    #     with tf.variable_scope("predict"):
    #         # outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
    #         #                   tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            

            
    #         y1 = tf.one_hot(self.y1, depth=self.c_maxlen, dtype=tf.float32)
    #         y2 = tf.one_hot(self.y2, depth=self.c_maxlen, dtype=tf.float32)

            
    #         na_bias = tf.get_variable("na_bias", shape=[1], dtype=tf.float32)
    #         na_bias_tiled = tf.tile(tf.reshape(na_bias, [1, 1]), [batch_size, 1]) #  [N, 1]
    #         flat_logits = logits1  # [N, P]
    #         print("na_bias:", na_bias_tiled)
    #         print("logits: ", flat_logits)
    #         concat_flat_logits = tf.concat([na_bias_tiled, flat_logits], axis=-1)
    #         concat_flat_yp = tf.nn.softmax(concat_flat_logits)  # [-1, P+1]
    #         na_prob = tf.squeeze(tf.slice(concat_flat_yp, [0, 0], [-1, 1]), [1])  # [N]
    #         flat_yp = tf.slice(concat_flat_yp, [0, 1], [-1, -1]) # [N,P]
    #         yp = flat_yp

    #         flat_logits2 = logits2 # [N, P]
    #         concat_flat_logits2 = tf.concat([na_bias_tiled, flat_logits2], axis=1)
    #         concat_flat_yp2 = tf.nn.softmax(concat_flat_logits2)
    #         na_prob2 = tf.squeeze(tf.slice(concat_flat_yp2, [0, 0], [-1, 1]), [1])  # [N]
    #         flat_yp2 = tf.slice(concat_flat_yp2, [0, 1], [-1, -1])
    #         yp2 = flat_yp2

    #          ##################################################
    #         na_bias_tiled_2 = tf.layers.dense(tf.reduce_mean(final_output, axis=1), 2)
    #         na_flag = tf.cast(self.na, tf.int32)
    #         y_na = tf.one_hot(na_flag, depth=2, dtype=tf.float32)
    #         loss_na = tf.nn.softmax_cross_entropy_with_logits(
    #             logits=na_bias_tiled_2, labels=y_na)
    #         na_hot = tf.nn.softmax(na_bias_tiled_2)
    #         self.na_hot = na_hot[:,1]
    #         ##################################################

    #         self.logits = flat_logits
    #         self.logits2 = flat_logits2
    #         self.concat_logits = concat_flat_logits
    #         self.concat_logits2 = concat_flat_logits2
    #         self.yyp = yp  # start prob dist
    #         self.yyp2 = yp2  # end prob dist
    #         self.na_prob = na_prob * na_prob2

    #         ## compute loss
    #         na = tf.reshape(self.na, [-1, 1])
    #         concat_y = tf.concat([na, y1], axis=1)
    #         loss1 = tf.nn.softmax_cross_entropy_with_logits(
    #             logits=self.concat_logits, labels=tf.cast(concat_y, 'float'))
    #         # tf.add_to_collection('losses', ce_loss)
    #         concat_y2 = tf.concat([na, y2], axis=1)
    #         loss2 = tf.nn.softmax_cross_entropy_with_logits(
    #             logits=self.concat_logits2, labels=tf.cast(concat_y2, 'float'))
    #         self.loss = tf.reduce_mean(loss1 + loss2)
    #         ########################## loss function 2 ######################

    #         ################# new softmax logits ####################
    #         outer = tf.matmul(tf.expand_dims(softmax(self.logits, axis=1, mask=self.c_mask), axis=2),
    #                           tf.expand_dims(softmax(self.logits2, axis=1, mask=self.c_mask), axis=1))
    #         ################# new softmax logits ####################
    #         # outer = tf.matrix_band_part(outer, 0, 15) # max_answer_len = 15
    #         outer = tf.matrix_band_part(outer, 0, self.answer_maxlen) # max_answer_len = 15
    #         self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
    #         self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

    def pointer_network(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        # final_output = self.match
        final_output = self.self_match
        with tf.variable_scope("pointer_network"):
            logits1, logits2 = my_pointer_network(final_output, self.c_mask, self.q_enc, self.q_mask,
                                                  batch_size, hidden_size, self.config.ptr_keep_prob,
                                                  self.is_train, scope="my_pointer_network")

        with tf.variable_scope("predict"):
            y1 = tf.one_hot(self.y1, depth=self.c_maxlen, dtype=tf.float32)
            y2 = tf.one_hot(self.y2, depth=self.c_maxlen, dtype=tf.float32)

            ##################################################
            na_bias = tf.get_variable("na_bias", shape=[1], dtype=tf.float32)
            na_bias_tiled = tf.tile(tf.reshape(na_bias, [1, 1]), [batch_size, 1]) #  [N, 1]
            ##################################################
            na_bias_tiled_2 = tf.layers.dense(tf.reduce_mean(final_output, axis=1), 2)
            na_flag = tf.cast(self.na, tf.int32)
            y_na = tf.one_hot(na_flag, depth=2, dtype=tf.float32)
            loss_na = tf.nn.softmax_cross_entropy_with_logits(
                logits=na_bias_tiled_2, labels=y_na)
            na_hot = tf.nn.softmax(na_bias_tiled_2)
            self.na_hot = na_hot[:,1]
            ##################################################

            flat_logits = logits1  # [N, P]
            print("na_bias:", na_bias_tiled)
            print("logits: ", flat_logits)
            concat_flat_logits = tf.concat([na_bias_tiled, flat_logits], axis=-1)
            concat_flat_yp = tf.nn.softmax(concat_flat_logits)  # [-1, P+1]
            na_prob = tf.squeeze(tf.slice(concat_flat_yp, [0, 0], [-1, 1]), [1])  # [N]
            flat_yp = tf.slice(concat_flat_yp, [0, 1], [-1, -1]) # [N,P]
            yp = flat_yp

            flat_logits2 = logits2 # [N, P]
            concat_flat_logits2 = tf.concat([na_bias_tiled, flat_logits2], axis=1)
            concat_flat_yp2 = tf.nn.softmax(concat_flat_logits2)
            na_prob2 = tf.squeeze(tf.slice(concat_flat_yp2, [0, 0], [-1, 1]), [1])  # [N]
            flat_yp2 = tf.slice(concat_flat_yp2, [0, 1], [-1, -1])
            yp2 = flat_yp2

            self.logits = flat_logits
            self.logits2 = flat_logits2
            self.concat_logits = concat_flat_logits
            self.concat_logits2 = concat_flat_logits2
            self.yyp = yp  # start prob dist
            self.yyp2 = yp2  # end prob dist
            self.na_prob = na_prob * na_prob2

            ## compute loss
            na = tf.reshape(self.na, [-1, 1])
            concat_y = tf.concat([na, y1], axis=1)
            loss1 = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.concat_logits, labels=tf.cast(concat_y, 'float'))
            # tf.add_to_collection('losses', ce_loss)
            concat_y2 = tf.concat([na, y2], axis=1)
            loss2 = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.concat_logits2, labels=tf.cast(concat_y2, 'float'))
            self.loss = tf.reduce_mean(loss1) + tf.reduce_mean(loss2) + tf.reduce_mean(loss_na)
            ########################## loss function 2 ######################

            ################# new softmax logits ####################
            outer = tf.matmul(tf.expand_dims(softmax(self.logits, axis=1, mask=self.c_mask), axis=2),
                              tf.expand_dims(softmax(self.logits2, axis=1, mask=self.c_mask), axis=1))
            ################# new softmax logits ####################
            # outer = tf.matrix_band_part(outer, 0, 15) # max_answer_len = 15
            outer = tf.matrix_band_part(outer, 0, self.answer_maxlen) # max_answer_len = 15
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            self.s = softmax(self.logits, axis=1, mask=self.c_mask)
            self.e = softmax(self.logits2, axis=1, mask=self.c_mask)

    def pointer_network_multi_loss(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        def compute_loss(logits1, logits2, y1, y2):
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, 15)
            yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits1, labels=y1)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits2, labels=y2)
            loss = tf.reduce_mean(losses + losses2)
            return loss, yp1, yp2
        with tf.variable_scope("pointer_network_match"):
            logits1, logits2 = my_pointer_network(self.match, self.c_mask, self.q_enc, self.q_mask,
                                                  batch_size, hidden_size, self.config.ptr_keep_prob,
                                                  self.is_train, scope="pointer_network_1")
            logits2_1, logits2_2 = my_pointer_network(self.self_match, self.c_mask, self.q_enc, self.q_mask,
                                                  batch_size, hidden_size, self.config.ptr_keep_prob,
                                                  self.is_train, scope="pointer_network_2")
            loss1, yp1_1, yp1_2 = compute_loss(logits1, logits2, self.y1, self.y2)
            loss2, self.yp1, self.yp2 = compute_loss(logits2_1, logits2_2, self.y1, self.y2)
            m = 1
            n = 4
            self.loss = (m * loss1 + n * loss2) / 5
            # self.yp1 = (m * yp1_1 + n * yp2_1) / 10
            # self.yp2 = (m * yp1_2 + n * yp2_2) / 10


    def memory_based_pointer_network(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        filnal_output = self.match
        with tf.variable_scope("pointer_network"):
            logits1, logits2 = memory_based_answer_pointer(filnal_output, self.c_mask, self.q_enc, self.q_mask,
                                                           batch_size, hidden_size, self.config.ptr_keep_prob,
                                                           1, self.is_train, scope="memory_based_pointer_network")
        with tf.variable_scope("predict"):
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, 15)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=self.y1)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=self.y2)
            self.loss = tf.reduce_mean(losses + losses2)


    def question_focused_pointing_network(self):
        hidden_size = self.config.hidden_size
        batch_size = self.config.batch_size
        final_output = self.self_match
        with tf.variable_scope("pointer_network"):
            logits1, logits2 = question_focused_attentional_pointer(final_output, self.c_mask, self.c_len,
                                                                    self.q_enc, self.q_mask, self.q_len,
                                                                    self.M, batch_size, hidden_size,
                                                                    self.config.ptr_keep_prob, self.is_train,
                                                                    scope="question_focused_attentional_pointer")
        with tf.variable_scope("predict"):
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, 15)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=self.y1)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=self.y2)
            self.loss = tf.reduce_mean(losses + losses2)

    def get_loss(self):
        return self.loss


    def get_global_step(self):
        return self.global_step



