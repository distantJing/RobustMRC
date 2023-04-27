import tensorflow as tf
import os
import csv


# from main import train, test
from main_add import train
from main_add import test, test_no_answer, test_2, save_mat
from prepro_add_multy_process import prepro

flags = tf.app.flags

# ----------------------- squad dataset -----------------------------#
train_file = os.path.join("squad2.0", "train-v2.0.json")
# dev_file = os.path.join("squad2.0", "dev-v2.0.json")      ############
test_file = os.path.join("squad2.0", "dev-v2.0.json")
addonesent_file = os.path.join("addonesent_dataset")
addsent_file = os.path.join("addsent_dataset")

# dev_file = "/home/wzj/resp/bert/squad1.0/addonesent_dataset"
dev_file = "addonesent_dataset"



# scp -P 1302 *.py zhoujing@101.6.96.160:/data/zhoujing/resp/multi_answer_QA/
# scp answer_compare.py config.py config_iar.py util_add.py func_add.py main_add.py model_add.py prepro_add.py  wzj@166.111.138.128:/home/wzj/resp/qa_aaai
# python config.py --mode --batch_size --load --elmo_trainable 

# glove 数据
# glove_file_path_home = "/data/disk1/sharing/pretrained_embedding/glove"  # iar
# glove_file_path_home = "/data/disk1/sharing/pretrained_embedding/glove"  # iar-msa
# glove_file = os.path.join(glove_file_path_home, "glove.840B.300d.txt")
glove_file = "glove.840B.300d.txt"
# sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
flags.DEFINE_integer("glove_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")

# ----------------------- squad dataset -----------------------------#
# 原始数据处理 输出 文件路径 + 文件名

flags.DEFINE_bool("elmo_trainable", True, "")

# target_dir = "/data/disk1/wzj/KBS_source_code_format/multi_answer_QA_all/data/prepro/add_con_all_new" # iar
target_dir = ""

num_model = "all_new_know"   # (+joint + know)

# save_dir = "log/model_" + num_model
# save_dir = "/data/disk1/wzj/KBS_source_code_format/multi_answer_QA_all/log/model_all_new_know"
save_dir = "log"
answer_dir = save_dir
log_dir = save_dir
# ----------------------- squad dataset -----------------------------#

char_file_path = save_dir
char_file = os.path.join(char_file_path, "char.3384.8d.txt")
flags.DEFINE_integer("char_size", 3384, "Corpus size for char")
flags.DEFINE_integer("char_dim", 8, "Embedding dimension for char")

train_record_file = os.path.join(target_dir, "train.tfrecords")
dev_record_file = os.path.join(target_dir, "dev.tfrecords")
test_record_file = os.path.join(target_dir, "test.tfrecords")
test12_record_file = os.path.join(target_dir, "test12.tfrecords")
test1_record_file = os.path.join(target_dir, "test1.tfrecords")
test2_record_file = os.path.join(target_dir, "test2.tfrecords")

addonesent_record_file = os.path.join(target_dir, "addonesent.tfrecords")
addsent_record_file = os.path.join(target_dir, "addsent.tfrecords")

word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
tag_emb_file = os.path.join(target_dir, "tag_emb.json")
word2idx_dict = os.path.join(target_dir, "word2idx_dict.json")
char2idx_dict = os.path.join(target_dir, "char2idx_dict.json")
tag2idx_dict = os.path.join(target_dir, "tag2idx_dict.json")
train_eval_file = os.path.join(target_dir, "train_eval.json")
dev_eval_file = os.path.join(target_dir, "dev_eval.json")
test_eval_file = os.path.join(target_dir, "test_eval.json")
test12_eval_file = os.path.join(target_dir, "test12_eval.json")
test1_eval_file = os.path.join(target_dir, "test1_eval.json")
test2_eval_file = os.path.join(target_dir, "test2_eval.json")

addonesent_eval_file = os.path.join(target_dir, "addonesent_eval.json")
addsent_eval_file = os.path.join(target_dir, "addsent_eval.json")

meta_file = os.path.join(target_dir, "meta.json")
answer_file = os.path.join(answer_dir, "answer_"+num_model+".json")
answer_con_file = os.path.join(answer_dir, "answer_con_"+num_model+".json")
answer_file_2 = os.path.join(answer_dir, "answer"+num_model+"_no_answer.json")
answer_con_file_2 = os.path.join(answer_dir, "answer_con_"+num_model+"_2.json")


# if not os.path.exists(target_dir):
#     os.makedirs(target_dir)
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
# if not os.path.exists(answer_dir):
#     os.makedirs(answer_dir)
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# if not os.path.exists(os.path.join(save_dir, "epoch")):
#     os.makedirs(os.path.join(save_dir, "epoch"))

flags.DEFINE_string("train_file", train_file, "train source file")
flags.DEFINE_string("dev_file", dev_file, "dev source file")
flags.DEFINE_string("test_file", test_file, "test source file")
# flags.DEFINE_string("test12_file", test12_file, "test source file")
# flags.DEFINE_string("test1_file", test1_file, "test source file")
# flags.DEFINE_string("test2_file", test2_file, "test source file")

flags.DEFINE_string("addonesent_file", addonesent_file, "addonesent_file source file")
flags.DEFINE_string("addsent_file", addsent_file, "addsent_file source file")

flags.DEFINE_string("glove_file", glove_file, "glove source file")
flags.DEFINE_string("char_file", char_file, "char source file")
flags.DEFINE_string("target_dir", target_dir, "")
flags.DEFINE_string("answer_dir", answer_dir, "")
flags.DEFINE_string("log_dirs", log_dir, "")
flags.DEFINE_string("save_dir", save_dir, "")
flags.DEFINE_string("train_record_file", train_record_file, "")
flags.DEFINE_string("dev_record_file", dev_record_file, "")
flags.DEFINE_string("test_record_file", test_record_file, "")
flags.DEFINE_string("test12_record_file", test12_record_file, "")
flags.DEFINE_string("test1_record_file", test1_record_file, "")
flags.DEFINE_string("test2_record_file", test2_record_file, "")

flags.DEFINE_string("addonesent_record_file", addonesent_record_file, "")
flags.DEFINE_string("addsent_record_file", addsent_record_file, "")

flags.DEFINE_string("word_emb_file", word_emb_file, "")
flags.DEFINE_string("char_emb_file", char_emb_file, "")
flags.DEFINE_string("tag_emb_file", tag_emb_file, "")
flags.DEFINE_string("word2idx_dict", word2idx_dict, "")
flags.DEFINE_string("char2idx_dict", char2idx_dict, "")
flags.DEFINE_string("tag2idx_dict", tag2idx_dict, "")
flags.DEFINE_string("train_eval_file", train_eval_file, "")
flags.DEFINE_string("dev_eval_file", dev_eval_file, "")
flags.DEFINE_string("test_eval_file", test_eval_file, "")
flags.DEFINE_string("test12_eval_file", test12_eval_file, "")
flags.DEFINE_string("test1_eval_file", test1_eval_file, "")
flags.DEFINE_string("test2_eval_file", test2_eval_file, "")

flags.DEFINE_string("addonesent_eval_file", addonesent_eval_file, "")
flags.DEFINE_string("addsent_eval_file", addsent_eval_file, "")

flags.DEFINE_string("meta_file", meta_file, "")
flags.DEFINE_string("answer_file", answer_file, "")
flags.DEFINE_string("answer_con_file", answer_con_file, "")
flags.DEFINE_string("answer_file_2", answer_file_2, "")
flags.DEFINE_string("answer_con_file_2", answer_con_file_2, "")


# essential training 程序运行方式
flags.DEFINE_string("mode", "test", "prepro | train | test | debug")

# 为context question 设置最大长度
flags.DEFINE_integer("para_limit", 550, "max length of train context")
flags.DEFINE_integer("ques_limit", 50, "max length of train question")
flags.DEFINE_integer("test_para_limit", 1000, "max length of test context")
flags.DEFINE_integer("test_ques_limit", 100, "max length of test question")
flags.DEFINE_integer("char_limit", 16, "limit length for character")  # 多篇文章该指标为16
flags.DEFINE_integer("answer_limit", 15, "limit length for answer") # 限定答案长度

# tfrecord 文件读取参数
flags.DEFINE_integer("capacity", 15000, "")
flags.DEFINE_integer("num_threads", 4, "number of threads in input pipeline")

# model training 模型训练参数
flags.DEFINE_integer("batch_size", 64, "")
flags.DEFINE_float("init_lr", 0.5, "")
flags.DEFINE_float("learning_rate", 0.001, "adam")
flags.DEFINE_integer("hidden_size", 100, "")
flags.DEFINE_integer("char_hidden_size", 100, "")
flags.DEFINE_float("keep_prob", 0.7, "Dropout keep prob in rnn")
flags.DEFINE_float("ptr_keep_prob", 0.7, "Dropout keep prob for pointer network")
flags.DEFINE_integer("num_steps", 50000, "number of training steps")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate") #todo###############################3
flags.DEFINE_integer("val_num_batches", 150, "Number of batches to evaluate the model")
# model 保存等checkpoint
flags.DEFINE_integer("patience", 2, "patience for learning rate decay")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("checkpoint", 3000, "checkpoint to save and evaluate the model")
# comment
flags.DEFINE_string("comment", "", "")

# bert
flags.DEFINE_integer("doc_stride", 128, "doc stride")
flags.DEFINE_integer("multi_answer", 3, "number of answers")

flags.DEFINE_bool("load", False, "")

flags.DEFINE_string("elmo_url", "2", "")  # iar small; iar
# flags.DEFINE_string("elmo_url", "/data/zhoujing/elmo/2", "")  # zhoujing

# flags.DEFINE_bool("load", False, "")
flags.DEFINE_float("no_answer_thresh_hold", 0.0, "no answer thresh hold")

# python config.py --mode prepro
# python config.py --mode test --elmo_url

def main(_):
    config = flags.FLAGS
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 修改此处‘0’，即可使用指定的显卡计算
    if config.mode == "prepro":
        prepro(config)
    elif config.mode == "train":
        train(config)
    elif config.mode == "test":
        test_2(config)
    # elif config.mode == "test_no_answer":
    #     test_no_answer(config)
    else:
        print("wrong")
    


if __name__ == "__main__":
    tf.app.run()
