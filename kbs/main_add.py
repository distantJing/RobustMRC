import numpy as np
import json
import tensorflow as tf
import os
# from bert_serving import client, server
# import matplotlib as plt
from tqdm import tqdm
from util_add import get_record_parser, get_batch_dataset, get_dataset, convert_tokens, evaluate
# from model_origin import Model
from model_add import Model
import csv

def write_csv(csv_file, data_row):
    with open(csv_file, "a+") as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)

def train(config):
    with open(config.word_emb_file, "r", encoding="utf-8") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r", encoding="utf-8") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r", encoding="utf-8") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r", encoding="utf-8") as fh:
        dev_eval_file = json.load(fh)
    with open(config.meta_file, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    
    result_file_path = os.path.join(config.log_dirs, "result.csv")
    print("???", os.path.exists(result_file_path))
    if not os.path.exists(result_file_path):
        print("create result csv")
        head = ["num_steps", "train/EM", "train/F1", "train/loss", "dev12/EM", "dev12/F1", "dev12/loss", "lr"]
        write_csv(result_file_path, head)
    else:
        print(result_file_path)

    # if config.use_bert == True:
    #     bert_server = server.BertServer(
    #             server.get_args_parser().parse_args(
    #                 [
    #                     # "-max_seq_len", "NONE",
    #                     ############### add by zhijing #############
    #                     "-max_seq_len", str(position_limit),
    #                     "-max_batch_size", "8",
    #                     "-gpu_memory_fraction", "0.2",
    #                     "-pooling_strategy", "NONE",
    #                     "-pooling_layer", "-1", "-2", "-3", "-4",
    #                     "-model_dir", bert_archive_path,
    #                     "-num_worker", "{}".format(gpu_count)
    #                 ]
    #             )
    #         )
    #     bert_server.start()
    #     bert_client = client.BertClient(output_fmt="list")
    # else:
    #     bert_server = None
    #     bert_client = None
    # client.BertClient(output_fmt="list").encode()


    dev_total = meta["dev_total"]
    train_total = meta["train_total"]


    print("Building model...")
    parser = get_record_parser(config)  # 设置单个example的parse
    train_dataset = get_batch_dataset(config.train_record_file, parser, config)
    dev_dataset = get_dataset(config.dev_record_file, parser, config)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

    # model = Model(config, iterator, word_mat, char_mat, tag_mat)
    model = Model(config, iterator, word_mat, char_mat)

    # todo: what's this mean
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    loss_save = 100.0
    patience = 0
    lr = config.init_lr

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())

        print("************* trainable variables *******************")
        for v in tf.trainable_variables():
            print(v)
        print("************* trainable variables *******************")

        writer = tf.summary.FileWriter(config.log_dirs, graph=tf.get_default_graph())
        # writer = tf.summary.FileWriter(config.log_dir)
        saver = tf.train.Saver(max_to_keep=15)
        if config.load == True:
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            print("**********************************loading checkpoint in {}".format(config.save_dir))
        else:
            print("**********************************training from 0")

        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())

        sess.run(tf.assign(model.is_train, tf.constant(True, tf.bool)))
        # sess.run(tf.assign(model.lr, tf.constant(lr, tf.float32)))
        if config.load == False:
            sess.run(tf.assign(model.lr, tf.constant(lr, tf.float32)))
        else:
            lr = sess.run(model.lr)
            print("this is lr: ", lr)

        for _ in tqdm(range(config.num_steps)):
            global_step = sess.run(model.global_step) + 1
            loss, train_op = sess.run([model.loss, model.train_op], feed_dict={handle: train_handle})
            # print("loss: ", loss)
            # test_num, a, a_ = sess.run([model.test, model.alpha, model.alpha_test], feed_dict={handle: train_handle})
            # # print("test_num: ", test_num)
            # print('a: ', a)
            # print('a_: ', a_)

            if global_step % config.period == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss), ])
                writer.add_summary(loss_sum, global_step)
            if global_step % config.checkpoint == 0:
                result_row = []
                sess.run(tf.assign(model.is_train, tf.constant(False, tf.bool)))
                metrics, summ = evaluate_batch(model, config.val_num_batches, train_eval_file, sess, "train",
                                               handle, train_handle)
                for sum in summ:
                    writer.add_summary(sum, global_step)
                result_row += [global_step, metrics["exact_match"], metrics["f1"], metrics["loss"]]                

                metrics, summ = evaluate_batch(
                    model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)
                result_row += [metrics["exact_match"], metrics["f1"], metrics["loss"]]
                result_row += [lr]
                write_csv(result_file_path, result_row)
                print(metrics)
                sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
                dev_loss = metrics["loss"]
                if dev_loss < loss_save:
                    loss_save = dev_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= config.patience:
                    lr /= 2.0
                    loss_save = dev_loss
                    patience = 0
                print("learning rate is {} at global_step {} ".format(lr, global_step))
                sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
                for s in summ:
                    writer.add_summary(s, global_step)
                writer.flush()
                filename = os.path.join(
                    config.save_dir, "model_{}.ckpt".format(global_step))
                saver.save(sess, filename)
            # if global_step % 3400 == 0:
            # if global_step % 18152 == 0:    
            if global_step % 10920 == 0:           
                filename = os.path.join(
                    config.save_dir, "epoch", "model_{}.ckpt".format(global_step))
                saver.save(sess, filename)
            
def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    answer_dict = {}
    losses = []
    for _ in tqdm(range(num_batches)):
        qa_id, loss, yp1, yp2 = sess.run([model.qa_id, model.loss, model.yp1, model.yp2], feed_dict={handle: str_handle})
        answer_dict_, _ = convert_tokens(eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
    return metrics, [loss_sum, f1_sum, em_sum]

    # parser = get_record_parser(config)
    # num_threads = tf.constant(4, dtype=tf.int32)
    # datasets = tf.data.TFRecordDataset(config.dev_record_file)
    # print(datasets)
    # datasets = datasets.map(parser)
    # print(datasets)
    # datasets = datasets.batch(50)
    # print(datasets)


def test(config):
    with open(config.word_emb_file, "r", encoding="utf-8") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r", encoding="utf-8") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    # with open(config.tag_emb_file, "r", encoding="utf-8") as fh:
    #     tag_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.meta_file, "r", encoding="utf-8") as fh:
        meta = json.load(fh)


    # # with open(config.test1_eval_file, "r", encoding="utf-8") as fh:
    # # with open(config.addonesent_eval_file, "r", encoding="utf-8") as fh:
    # with open(config.addsent_eval_file, "r", encoding="utf-8") as fh:
    #     eval_file = json.load(fh)
    # # total = meta["dev_total"]
    # # total = meta["test1_total"]
    # # total = meta["addonesent"]
    # total = meta["addsent"]
    # print("Loading model...")
    # # test_batch = get_dataset(config.test1_record_file, get_record_parser(
    # # test_batch = get_dataset(config.addonesent_record_file, get_record_parser(    
    # test_batch = get_dataset(config.addsent_record_file, get_record_parser(    
    #     config, is_test=True), config).make_one_shot_iterator()

    if config.comment == "addonesent":
        with open(config.addonesent_eval_file, "r", encoding="utf-8") as fh:
            eval_file = json.load(fh)
        total = meta["addonesent"]
        test_batch = get_dataset(config.addonesent_record_file, get_record_parser(    
            config, is_test=True), config).make_one_shot_iterator()
    elif config.comment == "addsent":
        with open(config.addsent_eval_file, "r", encoding="utf-8") as fh:
            eval_file = json.load(fh)
        total = meta["addsent"]
        print("Loading model...")
        test_batch = get_dataset(config.addsent_record_file, get_record_parser(    
            config, is_test=True), config).make_one_shot_iterator()
    elif  config.comment == "test1":
        with open(config.test1_eval_file, "r", encoding="utf-8") as fh:
            eval_file = json.load(fh)
        total = meta["test1_total"]
        test_batch = get_dataset(config.test1_record_file, get_record_parser(
            config, is_test=True), config).make_one_shot_iterator()

    # model = Model(config, test_batch, word_mat, char_macdt, tag_mat, trainable=False)
    model = Model(config, test_batch, word_mat, char_mat, trainable=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    print(config.save_dir)

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print("load model from {}".format(config.save_dir))
        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
        losses = []
        answer_dict = {}
        remapped_dict = {}
        for step in tqdm(range(total // config.batch_size + 1)):
            qa_id, loss, yp1, yp2 = sess.run(
                [model.qa_id, model.loss, model.yp1, model.yp2])
            answer_dict_, remapped_dict_ = convert_tokens(
                eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
            answer_dict.update(answer_dict_)
            remapped_dict.update(remapped_dict_)
            losses.append(loss)
        loss = np.mean(losses)
        metrics = evaluate(eval_file, answer_dict)
        with open(config.answer_file, "w") as fh:
            json.dump(remapped_dict, fh)
        print("Exact Match: {}, F1: {}".format(
            metrics['exact_match'], metrics['f1']))



def test_no_answer(config):
    with open(config.word_emb_file, "r", encoding="utf-8") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r", encoding="utf-8") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    # with open(config.tag_emb_file, "r", encoding="utf-8") as fh:
    #     tag_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test2_eval_file, "r", encoding="utf-8") as fh:
    # with open(config.addonesent_eval_file, "r", encoding="utf-8") as fh:
    # with open(config.addsent_eval_file, "r", encoding="utf-8") as fh:
        eval_file = json.load(fh)
    with open(config.meta_file, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    # total = meta["dev_total"]
    total = meta["test2_total"]
    # total = meta["addonesent"]
    # total = meta["addsent"]

    print("Loading model...")
    test_batch = get_dataset(config.test2_record_file, get_record_parser(
    # test_batch = get_dataset(config.addonesent_record_file, get_record_parser(    
    # test_batch = get_dataset(config.addsent_record_file, get_record_parser(    
        config, is_test=True), config).make_one_shot_iterator()

    # model = Model(config, test_batch, word_mat, char_macdt, tag_mat, trainable=False)
    model = Model(config, test_batch, word_mat, char_mat, trainable=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print("load model from {}".format(config.save_dir))
        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
        losses = []
        answer_dict = {}
        remapped_dict = {}
        diff_dict = {}
        diff_2_dict = {}
        
        for step in tqdm(range(total // config.batch_size + 1)):
            qa_id, loss, yp1, yp2, na_prob, na_hot = sess.run(
                [model.qa_id, model.loss, model.yp1, model.yp2, model.na_prob, model.na_hot])
            answer_dict_, remapped_dict_, diff_dict_, diff_2_dict_ = convert_tokens_with_no_answer(
                eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist(), na_prob.tolist(), na_hot.tolist())
            # answer_dict_, remapped_dict_ = convert_tokens(
            #     eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
            answer_dict.update(answer_dict_)
            remapped_dict.update(remapped_dict_)
            diff_dict.update(diff_dict_)
            diff_2_dict.update(diff_2_dict_)
            losses.append(loss)
        loss = np.mean(losses)
        metrics = evaluate(eval_file, answer_dict)
        # with open(config.answer_file_2, "w") as fh:
        #     json.dump(remapped_dict, fh)
        # with open("diff.json", "w") as fh:
        #     json.dump(diff_dict, fh)
        # with open("diff2.json", "w") as fh:
        #     json.dump(diff_2_dict, fh)
        if config.comment == "1":
            with open("dev1.json", "w") as fh:
                json.dump(remapped_dict, fh)
            with open("diff1.1.json", "w") as fh:
                json.dump(diff_dict, fh)
            with open("diff1.2.json", "w") as fh:
                json.dump(diff_2_dict, fh)
        if config.comment == "2":
            with open("dev2.json", "w") as fh:
                json.dump(remapped_dict, fh)
            with open("diff2.1.json", "w") as fh:
                json.dump(diff_dict, fh)
            with open("diff2.2.json", "w") as fh:
                json.dump(diff_2_dict, fh)

        print("Exact Match: {}, F1: {}".format(
            metrics['exact_match'], metrics['f1']))

######################################################################################

def convert_tokens_with_no_answer(eval_file, qa_id, pp1, pp2, na_prob, na_hot_prob):
    answer_dict = {}
    remapped_dict = {}
    diff_dict = {}
    diff_dict_2 = {}
    for qid, p1, p2, na, na2 in zip(qa_id, pp1, pp2, na_prob, na_hot_prob):
        # print(qid, p1, p2, max_score, null_score)
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        # if null_score - max_score > thresh_hold:
        #     answer_dict[str(qid)] = context[start_idx: end_idx]
        #     remapped_dict[uuid] = context[start_idx: end_idx]
        # else:
        #     answer_dict[str(qid)] = ""
        #     remapped_dict[uuid] = ""
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
        diff_dict[uuid] = na
        diff_dict_2[uuid] = na2
    return answer_dict, remapped_dict, diff_dict, diff_dict_2

    ############# test
    # with tf.Session(config=sess_config) as sess:
    #     sess.run(tf.global_variables_initializer())
    #     saver = tf.train.Saver()
    #     print("load model from {}".format(config.save_dir))
    #     saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
    #     sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
    #     losses = []
    #     answer_dict = {}
    #     remapped_dict = {}
    #     for step in tqdm(range(total // config.batch_size + 1)):
    #         qa_id, loss, yp1, yp2 = sess.run(
    #             [model.qa_id, model.loss, model.yp1, model.yp2])
    #         for a, b in zip(yp1, yp2):
    #             if a==0 and b==0:
    #                 print("--------------------------")
    #                 print(yp1)
    #                 print(yp2)


################################################################
def test_2(config):
    # with open(config.word_emb_file, "r", encoding="utf-8") as fh:
    #     word_mat = np.array(json.load(fh), dtype=np.float32)
    # with open(config.char_emb_file, "r", encoding="utf-8") as fh:
    #     char_mat = np.array(json.load(fh), dtype=np.float32)
    
    word_mat = np.zeros([69295,300], dtype=np.float32)
    char_mat = np.zeros([1315,8], dtype=np.float32)

    with open(config.meta_file, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    # if config.comment == "addonesent":
    #     with open(config.addonesent_eval_file, "r", encoding="utf-8") as fh:
    #         eval_file = json.load(fh)
    #     total = meta["addonesent"]
    #     test_batch = get_dataset(config.addonesent_record_file, get_record_parser(    
    #         config, is_test=True), config).make_one_shot_iterator()
    # elif config.comment == "addsent":
    #     with open(config.addsent_eval_file, "r", encoding="utf-8") as fh:
    #         eval_file = json.load(fh)
    #     total = meta["addsent"]
    #     print("Loading model...")
    #     test_batch = get_dataset(config.addsent_record_file, get_record_parser(    
    #         config, is_test=True), config).make_one_shot_iterator()
    # elif  config.comment == "test1":
    #     with open(config.test1_eval_file, "r", encoding="utf-8") as fh:
    #         eval_file = json.load(fh)
    #     total = meta["test1_total"]
    #     test_batch = get_dataset(config.test1_record_file, get_record_parser(
    #         config, is_test=True), config).make_one_shot_iterator()
        
    with open(config.dev_eval_file, "r", encoding="utf-8") as fh:
        eval_file = json.load(fh)
        total = meta["dev_total"]
        # test_batch = get_dataset(config.dev_record_file, get_record_parser(    
        #     config, is_test=False), config).make_one_shot_iterator()
        test_batch = get_dataset(config.dev_record_file, get_record_parser(    
            config, is_test=True), config).make_one_shot_iterator()

    # model = Model(config, test_batch, word_mat, char_macdt, tag_mat, trainable=False)
    model = Model(config, test_batch, word_mat, char_mat, trainable=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    print(config.save_dir)

    # total = 10
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print("load model from {}".format(config.save_dir))
        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
        def count_flops(graph):
            flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
            print('FLOPs: {}'.format(flops.total_float_ops))
        def count():
            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                # shape = variable.get_shape()
                # # shape = np.shape(reader.get_tensor(key))  #get the shape of the tensor in the model
                # print(shape)
                shape = list(np.shape(variable))
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            print("paras: ", total_parameters)
            # return total_parameters
        # count()
        count_flops(tf.get_default_graph())

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print("load model from {}".format(config.save_dir))
        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
        losses = []
        answer_dict = {}
        remapped_dict = {}
        s_dict = {}
        e_dict = {}
        M_dict = {}
        self_M_dict = {}

        for step in tqdm(range(total // config.batch_size + 1)):
            qa_id, loss, yp1, yp2 = sess.run(
                [model.qa_id, model.loss, model.yp1, model.yp2])
            answer_dict_, remapped_dict_ = convert_tokens(
                eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
            answer_dict.update(answer_dict_)
            remapped_dict.update(remapped_dict_)
            losses.append(loss)
        loss = np.mean(losses)
        metrics = evaluate(eval_file, answer_dict)
        with open("predictions.json", "w") as fh:
            json.dump(remapped_dict, fh)

        print("Exact Match: {}, F1: {}".format(
            metrics['exact_match'], metrics['f1']))

def convert_tokens_2(eval_file, qa_id, pp1, pp2, ss, ee, MM, self_MM):
    answer_dict = {}
    remapped_dict = {}
    s, e, M, self_M = {}, {}, {}, {}
    for qid, p1, p2, l, r, m, self_m in zip(qa_id, pp1, pp2, ss, ee, MM, self_MM):
        # print(p1)
        # print(l)
        # print(m.shape)
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
        s[str(qid)] = l.tolist()
        e[str(qid)] = r.tolist()
        M[str(qid)] = m.tolist()
        self_M[str(qid)] = self_m.tolist()
    return answer_dict, remapped_dict, s, e, M, self_M



def save_mat(config):
    with open(config.word_emb_file, "r", encoding="utf-8") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r", encoding="utf-8") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    # with open(config.tag_emb_file, "r", encoding="utf-8") as fh:
    #     tag_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.meta_file, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    with open(config.char2idx_dict, "r", encoding="utf-8") as fh:
        char2idx_dict = json.load(fh)

    if config.comment == "addonesent":
        with open(config.addonesent_eval_file, "r", encoding="utf-8") as fh:
            eval_file = json.load(fh)
        total = meta["addonesent"]
        test_batch = get_dataset(config.addonesent_record_file, get_record_parser(    
            config, is_test=True), config).make_one_shot_iterator()
    elif config.comment == "addsent":
        with open(config.addsent_eval_file, "r", encoding="utf-8") as fh:
            eval_file = json.load(fh)
        total = meta["addsent"]
        print("Loading model...")
        test_batch = get_dataset(config.addsent_record_file, get_record_parser(    
            config, is_test=True), config).make_one_shot_iterator()
    elif  config.comment == "test1":
        with open(config.test1_eval_file, "r", encoding="utf-8") as fh:
            eval_file = json.load(fh)
        total = meta["test1_total"]
        test_batch = get_dataset(config.test1_record_file, get_record_parser(
            config, is_test=True), config).make_one_shot_iterator()

    # model = Model(config, test_batch, word_mat, char_macdt, tag_mat, trainable=False)
    model = Model(config, test_batch, word_mat, char_mat, trainable=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    print(config.save_dir)

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print("load model from {}".format(config.save_dir))
        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))

        char_mat = sess.run(model.char_mat)
        print("char_mat: ", char_mat.shape)

        char_embedding_dict = {}
        for token, idx in char2idx_dict.items():
            char_embedding_dict[token] = char_mat[idx]
        char_lines = ""
        for token, vec in char_embedding_dict.items():
            line = " ".join([str(token), " ".join([str(i) for i in vec])])
            char_lines = "\n".join([char_lines, line])

        with open("char_152000", "w", encoding="utf-8") as fh:
            fh.write(char_lines)