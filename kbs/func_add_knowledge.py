import tensorflow as tf

initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                             mode='FAN_IN',
                                                             uniform=False,
                                                             dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)

def gru(units, is_train=True):
    # units: GRU's hidden size
    gru_cell = tf.nn.rnn_cell.GRUCell(num_units=units)
    return gru_cell


def gru_n(hidden_size, num_layer, is_train=True):
    '''
    create a stacked gated recurrent units, with num_layers-layer
    '''
    if num_layer == 1:
        return gru(hidden_size, is_train)
    else:
        stacked_gru_cells = tf.nn.rnn_cell.MultiRNNCell(
            [gru(hidden_size, is_train) for _ in range(num_layer)]
        )
        return stacked_gru_cells


class cudnn_gru_tensorflow_1_4:
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.params = []
        self.inits = []
        self.dropout_mask = []
        print("type input_size ", type(input_size))
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(
                num_layers=1, num_units=num_units, input_size=input_size_)
            gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(
                num_layers=1, num_units=num_units, input_size=input_size_)
            print("cudnn input_size", gru_bw.input_size)
            param_fw = tf.Variable(tf.random_uniform(
                [gru_fw.params_size()], -0.1, 0.1), validate_shape=False)
            param_bw = tf.Variable(tf.random_uniform(
                [gru_bw.params_size()], -0.1, 0.1), validate_shape=False)
            init_fw = tf.Variable(tf.zeros([1, batch_size, num_units]))
            init_bw = tf.Variable(tf.zeros([1, batch_size, num_units]))
            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.params.append((param_fw, param_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        # inputs:[batch_size, c_max_len, glove_dim]
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            param_fw, param_bw = self.params[layer]
            init_fw, init_bw = self.inits[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]
            with tf.variable_scope("fw"):
                out_fw, _ = gru_fw(outputs[-1] * mask_fw, init_fw, param_fw)
            with tf.variable_scope("bw"):
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                out_bw, _ = gru_bw(inputs_bw, init_bw, param_bw)
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res

class cudnn_gru:
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.cudnn_rnn.CudnnLSTM(
                num_layers=1, num_units=num_units) 
            gru_bw = tf.contrib.cudnn_rnn.CudnnLSTM(
                num_layers=1, num_units=num_units)
            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        # inputs:[batch_size, c_max_len, glove_dim]
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]
            with tf.variable_scope("fw_{}".format(layer)): #, reuse=tf.AUTO_REUSE):
                out_fw, _ = gru_fw(outputs[-1] * mask_fw)
            with tf.variable_scope("bw_{}".format(layer)): #, reuse=tf.AUTO_REUSE):
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                    # outputs[-1], seq_lengths = seq_len, seq_dim = 0, batch_dim = 1)
                print("in ", inputs_bw)
                out_bw, _ = gru_bw(inputs_bw)
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res


class cudnn_gru_dynamic:
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.dropout_mask = []
        self.num_units = num_units
        self.batch_size = batch_size
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.nn.rnn_cell.GRUCell(num_units=num_units)
            gru_bw = tf.nn.rnn_cell.GRUCell(num_units=num_units)
            mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))


    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        cell_fw, cell_bw = self.grus[0]
        mask_fw, mask_bw = self.dropout_mask[0]
        inputs_ = inputs * mask_fw
        state_f, state_b = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                               inputs=inputs_, sequence_length=seq_len, dtype=tf.float32)
        shape = inputs.get_shape().as_list()
        batch_size, max_len = shape[0], shape[1]
        # print("inputs: ", inputs)
        # print("shape: ", shape)
        # print("state f: ", state_f)
        # print("state b: ", state_b)
        state = tf.concat(state_f, axis=2)
        print("state: ", state)
        outputs = tf.reshape(state, [batch_size, -1, 2*self.num_units])
        return outputs


class cudnn_gru_cpu:
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="native_gru",
                 cell=None):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2*num_units
            if cell is None:
                gru_fw = tf.nn.rnn_cell.GRUCell(num_units)
                gru_bw = tf.nn.rnn_cell.GRUCell(num_units)
            else:
                gru_fw, gru_bw = cell, cell
            init_fw = tf.Variable(tf.zeros([batch_size, num_units]))
            init_bw = tf.Variable(tf.zeros([batch_size, num_units]))
            mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train)
            self.grus.append((gru_fw, gru_bw))
            self.inits.append((init_fw, init_bw))
            self.dropout_mask.append((mask_fw, mask_bw))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [inputs]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]
                with tf.variable_scope("fw_{}".format(layer)):
                    out_fw, _ = tf.nn.dynamic_rnn(cell=gru_fw, inputs=outputs[-1]*mask_fw, sequence_length=seq_len,
                                                  initial_state=init_fw, dtype=tf.float32)
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(input=outputs[-1]*mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    out_bw, _ = tf.nn.dynamic_rnn(cell=gru_bw, inputs=inputs_bw, sequence_length=seq_len,
                                                  initial_state=init_bw, dtype=tf.float32)
                    out_bw = tf.reverse_sequence(out_bw, seq_len, seq_dim=1, batch_dim=0)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
            if concat_layers:
                res = tf.concat(outputs[1:], axis=2)
            else:
                res = outputs[-1]
            return res


def dropout(args, keep_prob, is_train, mode="recurrent"):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0    # pre- trained
        shape = tf.shape(args)
        if mode == "embedding":   # 丢弃部分embedding,但最后要将未丢弃的embedding还原到正确embedding状态
            noise_shape = [shape[0], 1]
            scale = keep_prob
        # if mode=="recurrent" and len(args.get_shape().as_list())==3:
        #     noise_shape = [shape[0], 1, shape[-1]] # 为什么要对所有的列进行相同的处理？？？
        args = tf.cond(is_train, lambda: tf.nn.dropout(args, keep_prob, noise_shape) * scale, lambda: args)
    return args


def softmax(target, axis, mask, epsilon=1e-12, name=None):
    # 对target进行axis上的softmax
    # mask为与target相同大小的mask, dtype=tf.float32
    # 先减去max, 求exp时避免float溢出
    mask = tf.to_float(mask)
    with tf.op_scope([target], name, 'softmax'):
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target - max_axis) * mask
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = target_exp / (normalize + epsilon)
        return softmax


# def softmax(target, axis, mask, INF=1e30, name=None):
#     target_mask = sotfmask_mask(target, mask, INF)
#     target_softmax = tf.nn.softmax(target_mask, axis)
#     return target_softmax
# def sotfmask_mask(val, mask, INF=1e30):
#     # mask 为val的mask
#     x = -INF * (1 - tf.cast(mask, tf.float32))   # true: 0,   false: -inf
#     y = val * tf.cast(mask, tf.float32)          # true: val, false: 0
#     return x + y


def dense(inputs, hidden, use_bias=False, scope="dense"):
    '''
    将inputs矩阵最后一维大小转换为hidden大小
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[i] for i in range(len(inputs.get_shape().as_list())-1)] + [hidden]
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(tf.reshape(inputs, [-1, dim]), W)  # [-1, hidden]
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res


def conv(inputs, output_size, bias = None, activation = None, kernel_size = 1, name = "conv", reuse = None):
    with tf.variable_scope(name, reuse = reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1,kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,1,output_size]
            strides = [1,1,1,1]
        else:
            filter_shape = [kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                        filter_shape,
                        dtype = tf.float32,
                        regularizer=regularizer,
                        initializer = initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, "VALID")
        if bias:
            outputs += tf.get_variable("bias_",
                        bias_shape,
                        regularizer=regularizer,
                        initializer = tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs


def highway(x, size = None, activation = tf.nn.relu,
            num_layers = 2, scope = "highway", reuse = None):
    with tf.variable_scope(scope, reuse):
        if size is None:
            size = x.shape.as_list()[-1]
        else:
            x = conv(x, size, name = "input_projection", reuse = reuse)
        for i in range(num_layers):
            T = conv(x, size, bias = True, activation = tf.sigmoid,
                     name = "gate_%d"%i, reuse = reuse)
            H = conv(x, size, bias = True, activation = activation,
                     name = "activation_%d"%i, reuse = reuse)
            x = H * T + x * (1.0 - T)
        return x

def highway_rnn(inputs, inputs_len, num_layer, hidden_size, batch_size, keep_prob, is_train,
                size=None, activation=tf.nn.sigmoid, scope="highway", reuse=None):
    with tf.variable_scope(scope, reuse):
        inputs_size = inputs.get_shape().as_list()[-1]
        rnn1 = cudnn_gru(num_layers=num_layer, num_units=hidden_size, batch_size=batch_size,
                        input_size=2*hidden_size, keep_prob=keep_prob, is_train=is_train, scope="rnn1")
        rnn2 = cudnn_gru(num_layers=num_layer, num_units=hidden_size, batch_size=batch_size,
                         input_size=2*hidden_size, keep_prob=keep_prob, is_train=is_train, scope="rnn2")
        rnn3 = cudnn_gru(num_layers=num_layer, num_units=hidden_size, batch_size=batch_size,
                         input_size=inputs_size, keep_prob=keep_prob, is_train=is_train, scope="rnn3")
        if size is None:
            x = rnn3(inputs, inputs_len)
        T = rnn1(x, inputs_len)
        H = rnn2(x, inputs_len)
        if activation is not None:
            H = activation(H)
            T = activation(T)
        res = H * T + x * (1.0 - T)
        return res


# sim 1
def compute_similarity_matrix(inputs, inputs_mask, memory, memory_mask, keep_prob=1.0, hidden=None,
                              self_attention=False, is_train=None, mode=None,
                              scope="compute_similarity_matrix_action"):
    '''
    计算相似度矩阵，不进行softmax
    :param inputs:       [batch, P, dim_p]
    :param inputs_mask:  [batch, P]
    :param memory:       [batch, Q, dim_q]
    :param memory_mask:  [batch, Q]
    '''
    d_inputs = dropout(inputs, keep_prob, is_train)  # [b, p, p_dim]
    d_memory = dropout(memory, keep_prob, is_train)  # [b, q, q_dim]
    if mode is None:
        mode = "dot_attention"
        mode = "multiplication_attention"
    if mode == "multiplication_attention":
        M = tf.matmul(inputs, memory, adjoint_b=True)
    if mode == "":
        pass
    elif mode == "dot_attention":
        with tf.variable_scope(scope):
            inputs_ = tf.nn.relu(dense(d_inputs, hidden, use_bias=False, scope="inputs"))
            memory_ = tf.nn.relu(dense(d_memory, hidden, use_bias=False, scope="memory"))
            M = tf.matmul(inputs_, memory_, adjoint_b=True)
            M = M / (hidden ** 0.5)
    print("this is M0", M)
    # todo: self attention, diag matrix
    # if self_attention is True:
    #     shape = tf.shape(M)
    #     c = tf.ones(shape)
    #     c = tf.matrix_band_part(c, 0, 0)
    #     M = M * (1 - c)
    #     print("this is M", M)
    return M


##  sim 2
# def compute_similarity_matrix(inputs, inputs_mask, memory, memory_mask, keep_prob=1.0, hidden=None,
#                               self_attention=False, is_train=None, mode=None,
#                               scope="compute_similarity_matrix_action"):
#     '''
#     计算相似度矩阵，不进行softmax
#     :param inputs:       [batch, P, dim_p]
#     :param inputs_mask:  [batch, P]
#     :param memory:       [batch, Q, dim_q]
#     :param memory_mask:  [batch, Q]
#     '''
#     inputs_WEIGHT = tf.get_variable(name="inputs_WEIGHT", shape=[1, inputs.shape.as_list()[-1]] )
#     memory_WEIGHT = tf.get_variable(name="memory_WEIGHT", shape=[1, memory.shape.as_list()[-1]])
#     PRODUCT_WEIGHT = tf.get_variable(name="PRODUCT_WEIGHT", shape=[1, inputs.shape.as_list()[-1]])
#     DIFFERENCE_WEIGHT = tf.get_variable(name="DIFFERENCE_WEIGHT", shape=[1, inputs.shape.as_list()[-1]])
#     return tf.math.add_n(
#             [
#                 tf.broadcast_to(
#                     input=tf.reshape(tf.linalg.matmul(a=tf.reshape(inputs, [-1, inputs.shape.as_list()[-1]]),
#                                                     b=inputs_WEIGHT, 
#                                                     transpose_b=True),
#                                     [tf.shape(inputs)[0], tf.shape(inputs)[1], 1]),
#                     shape=[tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(memory)[1]]
#                 ),
#                 tf.broadcast_to(
#                     input=tf.reshape(tf.linalg.matmul(a=tf.reshape(memory, [-1, memory.shape.as_list()[-1]]),
#                                                     b=memory_WEIGHT,
#                                                     transpose_b=True),
#                                     [tf.shape(memory)[0], 1, tf.shape(memory)[1]]
#                                     ),
#                     shape=[tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(memory)[1]]
#                 ),
#                 tf.linalg.matmul(
#                     a=tf.math.multiply(x=inputs, y=PRODUCT_WEIGHT),
#                     b=memory,
#                     transpose_b=True
#                 ),
#                 tf.math.subtract(
#                     x=tf.reshape(tf.linalg.matmul(a=tf.reshape(inputs, [-1, inputs.shape.as_list()[-1]]), 
#                                     b=DIFFERENCE_WEIGHT, 
#                                     transpose_b=True),
#                                 [tf.shape(inputs)[0], tf.shape(inputs)[1], 1]),
#                     y=tf.reshape(tf.linalg.matmul(a=tf.reshape(memory, [-1, memory.shape.as_list()[-1]]),
#                                     b=DIFFERENCE_WEIGHT, 
#                                     transpose_b=True),
#                                 [tf.shape(memory)[0], 1, tf.shape(memory)[1]])
#                 )
#             ]
#         )


def easy_dot_attention(inputs, inputs_mask, memory, memory_mask, hidden, keep_prob=1.0, self_attention=False,
                       is_train=None, scope="easy_dot_attention"):
    with tf.variable_scope(scope):
        M = compute_similarity_matrix(inputs, inputs_mask, memory, memory_mask, keep_prob, hidden, self_attention,
                                      is_train=is_train, mode="dot_attention") # [b, p, q]
        M_mask = tf.to_float(tf.matmul(tf.expand_dims(tf.cast(inputs_mask, tf.int32), -1),
                                       tf.expand_dims(tf.cast(memory_mask, tf.int32), -1), adjoint_b=True)) # [b, p, q]
        a_t = softmax(M, 2, M_mask) # [b, p, q]
        outputs = tf.matmul(a_t, memory) # [b, p, q_dim]
        res = tf.concat([inputs, outputs], axis=2)
    with tf.variable_scope("gate"):
        dim = res.get_shape().as_list()[-1]
        d_res = dropout(res, keep_prob, is_train)
        gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False, scope="g"))
        return res * gate


def DCN_attention(inputs, inputs_mask, memory, memory_mask, hidden, keep_prob=1.0, is_train=None,
                  scope="DCN_attention"):
    with tf.variable_scope(scope):
        M = compute_similarity_matrix(inputs, inputs_mask, memory, memory_mask, keep_prob, hidden,
                                      is_train=is_train, mode="multiplication_attention")
        M_mask = tf.to_float(tf.matmul(tf.expand_dims(tf.cast(inputs_mask, tf.int32), -1),
                                       tf.expand_dims(tf.cast(memory_mask, tf.int32), -1), adjoint_b=True))  # [b, p, q]
        A_Q = softmax(M, 1, M_mask)
        A_D = softmax(M, 2, M_mask)
        c_q = tf.matmul(A_Q, inputs, adjoint_a=True) # [b,p,q]*[b,p,2h] => [b, q, 2h]
        c_d = tf.matmul(A_D, tf.concat([memory, c_q], axis=2)) # [b,p,q]*[b,q,4h] => [b, p, 4h]
        res = tf.concat([inputs, c_d], axis=2)
        return res


def DCN_plus_single_step(inputs, inputs_mask, memory, memory_mask, hidden, keep_prob=1.0, is_train=None,
                         scope="DCN_plus_single_plus"):
    with tf.variable_scope(scope):
        M = compute_similarity_matrix(inputs, inputs_mask, memory, memory_mask, keep_prob, hidden,
                                      is_train=is_train, mode="multiplication_attention")
        M_mask = tf.to_float(tf.matmul(tf.expand_dims(tf.cast(inputs_mask, tf.int32), -1),
                                       tf.expand_dims(tf.cast(memory_mask, tf.int32), -1), adjoint_b=True))  # [b, p, q]
        E_1_D = inputs
        E_1_Q = memory
        S_1_D = tf.matmul(softmax(M, 2, M_mask), E_1_Q)  # [b,p,q]*[b,q,2h] => [b, P, 2h]
        S_1_Q = tf.matmul(softmax(M, 1, M_mask), E_1_D, adjoint_a=True)  # [b,p,q]*[b,p,2h] => [b, Q, 2h]
        C_1_D = tf.matmul(softmax(M, 1, M_mask), S_1_Q)  # [b,p,q]*[b,q,2h] => [b, P, 2h]
        return S_1_D, S_1_Q, C_1_D


def interactive_aligning(inputs, inputs_mask, memory, memory_mask, hidden, layer=1, keep_prob=1.0,
                         is_train=None, similarity_mode=None, scope="interactive_aligning"):
    with tf.variable_scope(scope):
        if similarity_mode is None:
            similarity_mode = "multiplication_attention"
        c_t_1 = inputs
        for _ in range(layer):
            M = compute_similarity_matrix(c_t_1, inputs_mask, memory, memory_mask, keep_prob, hidden,
                                          is_train=is_train, mode=similarity_mode)
            M_mask = tf.to_float(tf.matmul(tf.expand_dims(tf.cast(inputs_mask, tf.int32), -1),
                                           tf.expand_dims(tf.cast(memory_mask, tf.int32), -1), adjoint_b=True))
            q_t = tf.matmul(softmax(M, 2, M_mask), memory) # [b,p,q]*[b,q,2h] => [b, P, 2h]
            c_t = semantic_fusion_unit(c_t_1, [q_t, c_t_1*q_t, c_t_1-q_t], keep_prob, is_train, scope="SFU")
            c_t_1 = c_t
        return c_t, M


def interactive_aligning_add_knowledge(inputs, input_keys, inputs_mask, memory, memory_keys, memory_mask, hidden, layer=1, keep_prob=1.0,
                         is_train=None, similarity_mode=None, scope="interactive_aligning"):
    with tf.variable_scope(scope):
        if similarity_mode is None:
            similarity_mode = "multiplication_attention"
        c_t_1 = inputs
        for _ in range(layer):
            M = compute_similarity_matrix(input_keys, inputs_mask, memory_keys, memory_mask, keep_prob, hidden,
                                            is_train=is_train, mode=similarity_mode)
            M_mask = tf.to_float(tf.matmul(tf.expand_dims(tf.cast(inputs_mask, tf.int32), -1),
                                           tf.expand_dims(tf.cast(memory_mask, tf.int32), -1), adjoint_b=True))
            M_ = softmax(M, 2, M_mask)
            q_t = tf.matmul(M_, memory) # [b,p,q]*[b,q,2h] => [b, P, 2h]
            c_t = semantic_fusion_unit(c_t_1, [q_t, c_t_1*q_t, c_t_1-q_t], keep_prob, is_train, scope="SFU")
        return c_t, M_
# def interactive_aligning_add_knowledge(inputs, input_keys, inputs_mask, memory, memory_keys, memory_mask, hidden, layer=1, keep_prob=1.0,
#                          is_train=None, similarity_mode=None, scope="interactive_aligning"):
#     with tf.variable_scope(scope):
#         if similarity_mode is None:
#             similarity_mode = "multiplication_attention"
#         c_t_1 = inputs
#         for _ in range(layer):
#             M = compute_similarity_matrix(input_keys, inputs_mask, memory_keys, memory_mask, keep_prob, hidden,
#                                             is_train=is_train, mode=similarity_mode)
#             M_mask = tf.to_float(tf.matmul(tf.expand_dims(tf.cast(inputs_mask, tf.int32), -1),
#                                            tf.expand_dims(tf.cast(memory_mask, tf.int32), -1), adjoint_b=True))
#             q_t = tf.matmul(softmax(M, 2, M_mask), memory) # [b,p,q]*[b,q,2h] => [b, P, 2h]
#             c_t = semantic_fusion_unit(c_t_1, [q_t, c_t_1*q_t, c_t_1-q_t], keep_prob, is_train, scope="SFU")
#         return c_t, M


def semantic_fusion_unit(r, f_sets, keep_prob=1.0, is_train=None, scope="semantic_fusion_unit"):
    # r: [b, p, 2h]
    with tf.variable_scope(scope):
        dim = r.get_shape().as_list()[-1]
        f_unit = tf.concat(f_sets, axis=-1)
        res = tf.concat([r, f_unit], axis=-1)
        d_res = dropout(res, keep_prob, is_train)
        r_hat = tf.nn.tanh(dense(d_res, dim, use_bias=True, scope="SFU_r")) # [b, p, 2h]
        g = tf.nn.sigmoid(dense(d_res, dim, use_bias=True, scope="SFU_g")) # [b, p, 2h]
        output = g * r_hat + (1 - g) * r
        return output





def my_pointer_network(context, context_mask, question, question_mask, batch_size, units,
                       keep_prob, is_train=None, cell=tf.nn.rnn_cell.GRUCell, scope="my_pointer_network"):
    '''
    :param context:           [batch_size, P, h/2h/?]    => h
    :param context_mask:      [batch_size, P]
    :param question:          [batch_size, Q, 2h/6h/?]   => 2h
                            => w_u_q * q.enc + w_v_q * v_r_q
    :param question_mask:     [batch_size, Q]
    :param batch_size:
    :param units:
    :param cell:              无括号
    :param scope:
    :return:   logits1, logits2 概率分布 [b, P]
    '''
    with tf.variable_scope(scope):
        # initializer = VSI(mode='FAN_AVG')
        context_units = context.get_shape().as_list()[-1]
        question_units = question.get_shape().as_list()[-1]
        W_u_Q = tf.get_variable("W_u_Q", shape=[question_units, units], dtype=tf.float32)
        W_h_P = tf.get_variable("W_h_P", shape=[context_units, units], dtype=tf.float32)
        W_h_a = tf.get_variable("W_h_a", shape=[question_units, units], dtype=tf.float32)
        V1 = tf.get_variable("V1", shape=[units, 1], dtype=tf.float32)
        V2 = tf.get_variable("V2", shape=[units, 1], dtype=tf.float32)

        # 计算initial state
        d_question = dropout(question, keep_prob, is_train)
        sum1 = tf.matmul(tf.reshape(d_question, [-1, question_units]), W_u_Q)
        s1 = tf.matmul(tf.tanh(sum1), V1)
        s1 = tf.reshape(s1, [batch_size, -1])  # [batch_size, Q]
        a_t1 = softmax(s1, 1, question_mask)   # [batch_size, Q]
        init_state = tf.reduce_sum(question * tf.expand_dims(a_t1, -1), 1)  # [batch_size, h_q]
        state_dropout_mask = dropout(tf.ones([batch_size, question_units], dtype=tf.float32), keep_prob, is_train)
        d_init_state = init_state * state_dropout_mask
        # print('init_state', init_state)

        # 计算p1
        d_context = dropout(context, keep_prob, is_train)
        sum2_1 = tf.matmul(tf.reshape(d_context, [-1, context_units]), W_h_P)
        sum2_1 = tf.reshape(sum2_1, [batch_size, -1, units])
        sum2_2 = tf.matmul(d_init_state, W_h_a)
        sum2 = sum2_1 + tf.expand_dims(sum2_2, 1)     # [b, p, h]
        s2 = tf.matmul(tf.reshape(tf.tanh(sum2), [-1, units]), V2)
        s2 = tf.reshape(s2, [batch_size, -1])  # [batch_size, P]
        logits1 = s2     # 注意，logits = s2 为没有softmax的数据
        at_2 = softmax(s2, 1, context_mask)   # [batch_size, P]
        c_t = tf.reduce_sum(context * tf.expand_dims(at_2, -1), 1) # [batch_size, h_c]
        print('ct', c_t)

        rnn = cell(question_units)
        output, next_state = rnn(c_t, init_state)  # output: [batch, 2h] [batch, 2h]
        d_next_state = next_state * state_dropout_mask

        # 计算logits2
        sum3_1 = tf.matmul(tf.reshape(context, [-1, context_units]), W_h_P)
        sum3_1 = tf.reshape(sum3_1, [batch_size, -1, units])
        sum3_2 = tf.matmul(d_next_state, W_h_a)
        sum3 = sum3_1 + tf.expand_dims(sum3_2, 1)
        # sum3 = tf.matmul(tf.reshape(context, [-1, context_units]), W_h_P) + tf.matmul(next_state, W_h_a)
        s3 = tf.matmul(tf.reshape(tf.tanh(sum3), [-1, units]), V2)
        s3 = tf.reshape(s3, [batch_size, -1])
        logits2 = s3
        at_3 = softmax(s3, 1, context_mask)
        print('s3', s3)
        return logits1, logits2
        ################ logits softmax processing
        # return at_2, at_3

def memory_based_answer_pointer(context, context_mask, question, question_mask, batch_size, units,
                                keep_prob=1.0, layer=1, is_train=None, scope="memory_based_answer_pointer"):
    with tf.variable_scope(scope):
        question_summary = question[:,-1,:]  # 注意是否添加mask_len信息？？？？？ [b, 2h]
        z_s = question_summary
        context_maxlen = context.get_shape().as_list()[1]
        context_maxlen = tf.shape(context)[1]
        sum1_0 = context  # [b, p, 2h]
        sum1_1 = tf.tile(tf.expand_dims(z_s, 1), [1, context_maxlen, 1]) # [b, p, 2h]
        sum1_2 = context * tf.expand_dims(z_s, 1) # [b, p, 2h] * [b, 1, 2h] => [b, p, 2h]
        sum1 = tf.concat([sum1_0, sum1_1, sum1_2], axis=2)  # [b, p, 6h]
        # todo： feedforward neural network
        s_1 = tf.reshape(dense(sum1, 1, use_bias=False, scope="s_l"), [batch_size, context_maxlen])
        logits_1 = softmax(s_1, 1, context_mask)  # [b, p]

        u_s = tf.reshape(tf.matmul(context, tf.expand_dims(logits_1, -1), adjoint_a=True), (batch_size, -1))
        # [b, p, 2h] * [b, p, 1]  => [b, 2h, 1]
        z_e = semantic_fusion_unit(z_s, u_s)
        sum2_0 = context
        sum2_1 = tf.tile(tf.expand_dims(z_e, 1), [1, context_maxlen, 1]) # [b, p, 2h]
        sum2_2 = context * tf.expand_dims(z_e, 1)
        sum2 = tf.concat([sum2_0, sum2_1, sum2_2], axis=2)
        # todo: FFN
        e_1 = tf.reshape(dense(sum2, 1, use_bias=False, scope="e_l"), [batch_size, context_maxlen])
        logits_2 = softmax(e_1, 1, context_mask)
        return s_1, e_1



def question_focused_attentional_pointer(context, context_mask, context_len, question, question_mask, question_len,
                                         similarity, batch_size, units, keep_prob=1, is_train=None,
                                         scope="question_focused_attentional_pointer"):
    M_mask = tf.to_float(tf.matmul(tf.expand_dims(tf.cast(context_mask, tf.int32), -1),
                                   tf.expand_dims(tf.cast(question_mask, tf.int32), -1), adjoint_b=True))
    A_ = tf.reduce_max(M_mask * similarity, axis=1) # [b, q]
    print("this is A_", A_) # [64, ?]
    k = softmax(A_, axis=1, mask=question_mask) # [b, q]
    print("this is k: ", k) # [?, 64, ?]
    encoding_dim = question.get_shape().as_list()[-1]
    # [b, 1, q ] * [b, q, 2h] => [b, 2h]
    q_ma = tf.reshape(tf.matmul(tf.expand_dims(k, axis=1), question), (batch_size, encoding_dim)) # [b, 2h]
    q_ma_dropout_mask = dropout(tf.ones([batch_size, encoding_dim], dtype=tf.float32), keep_prob, is_train)
    d_q_ma = q_ma * q_ma_dropout_mask
    print("q_ma", q_ma)
    print("d_q_ma")
    q_f = tf.slice(question, [0, 0, 0], [batch_size, 2, encoding_dim]) # [b, 2, 2h]
    q_f = tf.reshape(q_f, (batch_size, -1)) # [b, 4h]
    print("q_f", q_f)
    
    q_hat = tf.nn.tanh(dense(tf.concat([d_q_ma, q_f], axis=1), hidden=encoding_dim, use_bias=True, scope="q_hat"))
    print("q_hat", q_hat) # [b, 2h]

    rnn_begin = cudnn_gru(num_layers=1, num_units=units, batch_size=batch_size,
                         input_size=context.get_shape().as_list()[-1], keep_prob=keep_prob,
                         is_train=is_train, scope="pointer_network_begin")
    rnn_end = cudnn_gru(num_layers=1, num_units=units, batch_size=batch_size,
                         input_size=context.get_shape().as_list()[-1], keep_prob=keep_prob,
                         is_train=is_train, scope="pointer_network_end")
    B = rnn_begin(context, context_len, is_train) # [b, p, 2h]
    E = rnn_end(context, context_len, is_train) # [b, p, 2h]

    logits1 = tf.matmul(B, tf.expand_dims(q_hat, axis=2))  # [b, p, 2h] * [b, 2h, 1] => [b, p, 1]
    logits2 = tf.matmul(E, tf.expand_dims(q_hat, axis=2))  # [b, p, 2h] * [b, 2h, 1] => [b, p, 1]

    logits1 = tf.reshape(logits1, [batch_size, -1])
    logits2 = tf.reshape(logits2, [batch_size, -1])

    return logits1, logits2

