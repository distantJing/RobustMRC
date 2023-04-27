import tensorflow as tf
import re
import string
from collections import Counter

# get batch data
# evaluate
def get_record_parser(config, is_test=False):
    def parse(example):
        para_limit = config.test_para_limit if is_test else config.para_limit
        ques_limit = config.test_ques_limit if is_test else config.ques_limit
        char_limit = config.char_limit
        features = tf.parse_single_example(example,
                                           features={
                                               "context_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs": tf.FixedLenFeature([], tf.string),
                                               "context_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "y1": tf.FixedLenFeature([], tf.int64),
                                               "y2": tf.FixedLenFeature([], tf.int64),
                                               "id": tf.FixedLenFeature([], tf.int64),
                                               "context_tokens":tf.FixedLenFeature([], tf.string),
                                               "question_tokens":tf.FixedLenFeature([], tf.string),
                                               "passage_connections":tf.FixedLenFeature([], tf.string),
                                               "question_connections":tf.FixedLenFeature([], tf.string),
                                               "p_connections_num":tf.FixedLenFeature([], tf.int64),
                                               "q_connections_num":tf.FixedLenFeature([], tf.int64),
                                           })
        context_idxs = tf.reshape(tf.decode_raw(features["context_idxs"], tf.int32), [para_limit])
        ques_idxs = tf.reshape(tf.decode_raw(features["ques_idxs"], tf.int32), [ques_limit])
        context_char_idxs = tf.reshape(tf.decode_raw(features["context_char_idxs"], tf.int32), [para_limit, char_limit])
        ques_char_idxs = tf.reshape(tf.decode_raw(features["ques_char_idxs"], tf.int32), [ques_limit, char_limit])
        y1 = features["y1"]   # an int value
        y2 = features["y2"]
        qa_id = features["id"]

        na = tf.cast(tf.cast(tf.cast(y1+1, tf.bool),tf.int32)-1, tf.bool)

        # context_tokens = tf.string_split(features["context_tokens"], " ")
        # question_tokens = tf.string_split(features["question_tokens"], " ")
        context_tokens = features["context_tokens"]
        question_tokens = features["question_tokens"]

        p_connections_num, q_connections_num = features["p_connections_num"], features["q_connections_num"]
    
        passage_connections = tf.reshape(tf.decode_raw(features["passage_connections"], tf.int64), [-1, 2])
        question_connections = tf.reshape(tf.decode_raw(features["question_connections"], tf.int64), [-1, 2])
        
        # p_connections = tf.sparse.SparseTensor(indices=passage_connections, values=tf.ones([q_connections_num], dtype=tf.float32), 
            # dense_shape=[para_limit, para_limit])
        # q_connections = tf.sparse.SparseTensor(indices=question_connections, values=tf.ones([p_connections_num], dtype=tf.float32),
            # dense_shape=[ques_limit, para_limit])
        p_connections = tf.sparse_to_dense(sparse_indices=passage_connections, output_shape=[para_limit, para_limit],
            sparse_values=tf.ones(p_connections_num, dtype=tf.float32), default_value=0.0)
        q_connections = tf.sparse_to_dense(sparse_indices=question_connections, output_shape=[ques_limit, para_limit],
            sparse_values=tf.ones(q_connections_num, dtype=tf.float32), default_value=0.0)
        
        # return context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id
#         return context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id, context_tokens, question_tokens, passage_connections[0], question_connections[0],p_connections_num, q_connections_num, p_sum, q_sum
        return context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id, context_tokens, question_tokens, p_connections, q_connections, na
    return parse


# origin net na 
# def get_record_parser(config, is_test=False):
#     def parse(example):
#         para_limit = config.test_para_limit if is_test else config.para_limit
#         ques_limit = config.test_ques_limit if is_test else config.ques_limit
#         char_limit = config.char_limit
#         features = tf.parse_single_example(example,
#                                            features={
#                                                "context_idxs": tf.FixedLenFeature([], tf.string),
#                                                "ques_idxs": tf.FixedLenFeature([], tf.string),
#                                                "context_char_idxs": tf.FixedLenFeature([], tf.string),
#                                                "ques_char_idxs": tf.FixedLenFeature([], tf.string),
#                                                "y1": tf.FixedLenFeature([], tf.int64),
#                                                "y2": tf.FixedLenFeature([], tf.int64),
#                                                "id": tf.FixedLenFeature([], tf.int64),
#                                                "context_tokens":tf.FixedLenFeature([], tf.string),
#                                                "question_tokens":tf.FixedLenFeature([], tf.string),
#                                                "passage_connections":tf.FixedLenFeature([], tf.string),
#                                                "question_connections":tf.FixedLenFeature([], tf.string),
#                                                "p_connections_num":tf.FixedLenFeature([], tf.int64),
#                                                "q_connections_num":tf.FixedLenFeature([], tf.int64),
#                                            })
#         context_idxs = tf.reshape(tf.decode_raw(features["context_idxs"], tf.int32), [para_limit])
#         ques_idxs = tf.reshape(tf.decode_raw(features["ques_idxs"], tf.int32), [ques_limit])
#         context_char_idxs = tf.reshape(tf.decode_raw(features["context_char_idxs"], tf.int32), [para_limit, char_limit])
#         ques_char_idxs = tf.reshape(tf.decode_raw(features["ques_char_idxs"], tf.int32), [ques_limit, char_limit])
#         y1 = features["y1"]   # an int value
#         y2 = features["y2"]
#         qa_id = features["id"]

#         na = tf.cast(tf.cast(tf.cast(y1+1, tf.bool),tf.int32)-1, tf.bool)

#         return context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id, na
#         # return context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id, context_tokens, question_tokens, p_connections, q_connections
#     return parse

    
def get_record_parser_old(config, is_test=False):
    def parse(example):
        para_limit = config.test_para_limit if is_test else config.para_limit
        ques_limit = config.test_ques_limit if is_test else config.ques_limit
        char_limit = config.char_limit
        features = tf.parse_single_example(example,
                                           features={
                                               "context_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs": tf.FixedLenFeature([], tf.string),
                                               "context_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "y1": tf.FixedLenFeature([], tf.int64),
                                               "y2": tf.FixedLenFeature([], tf.int64),
                                               "id": tf.FixedLenFeature([], tf.int64),
                                               "context_tokens":tf.FixedLenFeature([], tf.string),
                                               "question_tokens":tf.FixedLenFeature([], tf.string),
                                               "passage_connections":tf.FixedLenFeature([], tf.string),
                                               "question_connections":tf.FixedLenFeature([], tf.string)
                                           })
        context_idxs = tf.reshape(tf.decode_raw(features["context_idxs"], tf.int32), [para_limit])
        ques_idxs = tf.reshape(tf.decode_raw(features["ques_idxs"], tf.int32), [ques_limit])
        context_char_idxs = tf.reshape(tf.decode_raw(features["context_char_idxs"], tf.int32), [para_limit, char_limit])
        ques_char_idxs = tf.reshape(tf.decode_raw(features["ques_char_idxs"], tf.int32), [ques_limit, char_limit])
        # y1 = tf.reshape(tf.decode_raw(features["y1"], tf.float32), [para_limit])
        # y2 = tf.reshape(tf.decode_raw(features["y2"], tf.float32), [para_limit])
        y1 = features["y1"]
        y2 = features["y2"]
        qa_id = features["id"]

        # context_tokens = tf.string_split(features["context_tokens"], " ")
        # question_tokens = tf.string_split(features["question_tokens"], " ")
        # context_tokens += ["" for _ in range(para_limit-len(context_tokens))]
        # question_tokens += ["" for _ in range(ques_limit-len(question_tokens))]
        context_tokens = features["context_tokens"]
        question_tokens = features["question_tokens"]
    
        passage_connections = tf.reshape(tf.decode_raw(features["passage_connections"], tf.int32), [para_limit, para_limit])
        question_connections = tf.reshape(tf.decode_raw(features["question_connections"], tf.int32), [ques_limit, para_limit])
        
        # return context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id
        return context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id, context_tokens, question_tokens, passage_connections, question_connections

    return parse



def get_record_parser_tag(config, is_test=False):
    def parse(example):
        para_limit = config.test_para_limit if is_test else config.para_limit
        ques_limit = config.test_ques_limit if is_test else config.ques_limit
        char_limit = config.char_limit
        features = tf.parse_single_example(example,
                                           features={
                                               "context_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs": tf.FixedLenFeature([], tf.string),
                                               "context_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "context_tag_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_tag_idxs": tf.FixedLenFeature([], tf.string),
                                               "y1": tf.FixedLenFeature([], tf.string),
                                               "y2": tf.FixedLenFeature([], tf.string),
                                               "id": tf.FixedLenFeature([], tf.int64)
                                           })
        context_idxs = tf.reshape(tf.decode_raw(features["context_idxs"], tf.int32), [para_limit])
        ques_idxs = tf.reshape(tf.decode_raw(features["ques_idxs"], tf.int32), [ques_limit])
        context_char_idxs = tf.reshape(tf.decode_raw(features["context_char_idxs"], tf.int32), [para_limit, char_limit])
        ques_char_idxs = tf.reshape(tf.decode_raw(features["ques_char_idxs"], tf.int32), [ques_limit, char_limit])
        context_tag_idxs = tf.reshape(tf.decode_raw(features["context_tag_idxs"], tf.int32), [para_limit])
        ques_tag_idxs = tf.reshape(tf.decode_raw(features["ques_tag_idxs"], tf.int32), [ques_limit])
        y1 = tf.reshape(tf.decode_raw(features["y1"], tf.float32), [para_limit])
        y2 = tf.reshape(tf.decode_raw(features["y2"], tf.float32), [para_limit])
        qa_id = features["id"]
        return context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_tag_idxs, ques_tag_idxs, y1, y2, qa_id
    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file)\
        .map(parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    dataset = dataset.batch(config.batch_size)
    return dataset

def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file)\
        .map(parser, num_parallel_calls=num_threads).repeat().batch(config.batch_size)
    return dataset


######################################################################################

def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score,
                                            prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1, "total": total}

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    # ################ has no answer (squad 2.0) #####################
    # if len(scores_for_ground_truths)>0:
    #     return max(scores_for_ground_truths)
    # else:
    #     return 0
    # ################ has no answer (squad 2.0) #####################
    return max(scores_for_ground_truths)