import json
import spacy
import random
import tensorflow as tf
import numpy as np
from collections import Counter
from tqdm import tqdm
import time

from nltk.corpus import stopwords, wordnet

wordnet_posmap = {
    "NN": wordnet.NOUN, "NNP": wordnet.NOUN, "NNPS": wordnet.NOUN, "NNS": wordnet.NOUN,
    "VB": wordnet.VERB, "VBD": wordnet.VERB, "VBG": wordnet.VERB,
    "VBN": wordnet.VERB, "VBP": wordnet.VERB, "VBZ": wordnet.VERB,
    "JJ": wordnet.ADJ, "JJR": wordnet.ADJ, "JJS": wordnet.ADJ,
    "RB": wordnet.ADV, "RBR": wordnet.ADV, "RBS": wordnet.ADV, "RP": wordnet.ADV
}

wordnet_relations = [
    "hypernyms", "instance_hypernyms",
    "hyponyms", "instance_hyponyms",
    "member_holonyms", "substance_holonyms", "part_holonyms",
    "member_meronyms", "substance_meronyms", "part_meronyms",
    "attributes", "entailments", "causes", "also_sees", "verb_groups", "similar_tos"
]

# spacy_nlp = spacy.load(name="en_core_web_lg", disable=["parser", "ner"])
# spacy_nlp = spacy.blank("en")
spacy_nlp = spacy.load("en", disable=["parser", "ner"])
stopwords_words = stopwords.words("english")
wordnet_relation_hop_count = 3

# nlp = spacy.blank("en")
# def word_tokenize(sent):
#     doc = nlp(sent)
#     return [token.text for token in doc]


# def word_tokenize(text):
#     import nltk
#     doc = nltk.word_tokenize(text)
#     return [token.replace("''", '"').replace("``", '"') for token in doc]
#     return [token.text for token in doc]

global word_counter, char_counter
word_counter, char_counter = Counter(), Counter()

def convert_idx(text, tokens):
    current = 0
    spans = []
    # print(text)
    for token in tokens:
        # print(token)
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current+len(token)))
        current += len(token)
    return spans

def get_text_normals(text_tokens):
    return [token.norm_ for token in text_tokens if len(token.text.strip())!=0]

def get_text_postags(text_tokens):
    return [token.tag_ for token in text_tokens if len(token.text.strip())!=0]

def get_text_symbols(text_tokens):
    return [token.text for token in text_tokens if len(token.text.strip())!=0]

def get_connections(passage_normals, passage_postags, question_normals, question_postags):
    def get_text_nodes(text_normals, text_postags):
        text_nodes = []

        for normal, postag in zip(text_normals, text_postags):
            direct_synsets = set()

            # zhijing: 去除停用词，只是用wordnet中指定词性的词
            if normal not in stopwords_words and postag in wordnet_posmap:
                direct_synsets.update(wordnet.synsets(lemma=normal, pos=wordnet_posmap[postag]))

            spread_synsets = direct_synsets.copy()

            if len(spread_synsets) > 0:
                current_synsets = spread_synsets

                for _ in range(wordnet_relation_hop_count):
                    current_synsets = set(
                        target
                        for synset in current_synsets
                        for relation in wordnet_relations
                        for target in getattr(synset, relation)()
                    )

                    spread_synsets.update(current_synsets)

            text_nodes.append({"direct_synsets": direct_synsets, "spread_synsets": spread_synsets})

        return text_nodes

    def get_text_connections(subject_nodes, object_nodes):
        return np.asarray(
            a=[
                  [subject_index, object_index]
                  for subject_index, subject_node in enumerate(subject_nodes)
                  for object_index, object_node in enumerate(object_nodes)
                  if subject_node is not object_node and
                     len(subject_node["spread_synsets"].intersection(object_node["direct_synsets"])) > 0
              ] or np.empty(shape=[0, 2], dtype=np.int),
            dtype=np.int
        )
    
    passage_nodes = get_text_nodes(passage_normals, passage_postags)
    question_nodes = get_text_nodes(question_normals, question_postags)
    passage_connections = get_text_connections(passage_nodes, passage_nodes)
    question_connections = get_text_connections(question_nodes, passage_nodes)
    return passage_connections, question_connections


def process_single_triple(article):
    examples = []
    current_id = 0
    this_word_counter, this_char_counter = Counter(), Counter()
    for para in article["paragraphs"]:
        context = para["context"].strip().replace("''", '" ').replace("``", '" ').lower()
        # context_token: 将原始context段落分割为 token 序列
        context_tokens_spacy = spacy_nlp(context)
        # context_tokens = word_tokenize(context)
        context_tokens = get_text_symbols(context_tokens_spacy)
        context_normals = get_text_normals(context_tokens_spacy)
        context_postags = get_text_postags(context_tokens_spacy)
        # print("tokens      ", context_tokens)
        # print("normals     ", context_normals)
        # print("pos          ", context_postags)
        # context_chars: 将context_tokens里每一个token分割为 char 序列
        context_chars = [list(token) for token in context_tokens]
        # 计算context_token里的每一个token，在原始context段落中的char位置
        # 如：context = "how are you"
        #     spans = [(0,3), (4,7), (8,11)]
        spans = convert_idx(context, context_tokens)
        for token in context_tokens:
            this_word_counter[token] += len(para["qas"])
            for char in token:
                this_char_counter[char] += len(para["qas"])

        for qa in para["qas"]:
            current_id += 1
            ques = qa["question"].replace("''", '" ').replace("``", '" ').lower()
            # ques_tokens = word_tokenize(ques)
            ques_tokens_spacy = spacy_nlp(ques)
            ques_tokens = get_text_symbols(ques_tokens_spacy)
            ques_normals = get_text_normals(ques_tokens_spacy)
            ques_postags = get_text_postags(ques_tokens_spacy)
            ques_chars = [list(token) for token in ques_tokens]
            for token in ques_tokens:
                this_word_counter[token] += 1
                for char in token:
                    this_char_counter[char] += 1
            y1s, y2s = [], []
            answer_texts = []

            # for answer in qa["answers"]:
            #     answer_text = answer["text"]
            #     answer_start = answer["answer_start"]
            #     answer_end = answer_start + len(answer_text)
            #     answer_texts.append(answer_text)
            #     answer_span = []
            #     for idx, span in enumerate(spans):
            #         if not (answer_end<=span[0] or answer_start>=span[1]):
            #             answer_span.append(idx)
            #     y1, y2 = answer_span[0], answer_span[-1]
            #     y1s.append(y1)
            #     y2s.append(y2)
            if qa.get("is_impossible", False) == False:
                for answer in qa["answers"]:
                    answer_text = answer["text"]
                    answer_start = answer["answer_start"]
                    answer_end = answer_start + len(answer_text)
                    answer_texts.append(answer_text)
                    answer_span = []
                    for idx, span in enumerate(spans):
                        if not (answer_end<=span[0] or answer_start>=span[1]):
                            answer_span.append(idx)
                    y1, y2 = answer_span[0], answer_span[-1]
                    y1s.append(y1)
                    y2s.append(y2)
            else:
                y1s.append(-1)
                y2s.append(-1)
                answer_texts.append("")
            
            passage_connections, question_connections = get_connections(context_normals, context_postags, ques_normals, ques_postags)
            # print(passage_connections)
            # print(question_connections)
            # print(np.reshape(passage_connections, [-1]))
            example = {"context_tokens":context_tokens, "context_chars":context_chars,
                        "ques_tokens":ques_tokens, "ques_chars":ques_chars,
                        "y1s":y1s, "y2s":y2s, "id":current_id,
                    #    "context_normals":context_normals, "context_postags":context_postags,
                    #    "ques_normals":ques_normals, "ques_postags":ques_postags
                        "passage_connections":passage_connections,
                        "question_connections":question_connections,
                        "eval_examples": {
                "context":context, "spans":spans, "answers":answer_texts, "uuid":qa["id"],
                "passage_connections":passage_connections.tolist(), "question_connections":question_connections.tolist(),
                "context_tokens":context_tokens, "ques_tokens":ques_tokens
                }   
            }
            examples.append(example)
    # print("part this_word_counter: ", this_word_counter)
    # print("part this_char_counter: ", this_char_counter)
    # print("part add: ", this_word_counter + this_char_counter)
    return examples, this_word_counter, this_char_counter
            

def get_examples(file_name, data_type, n_processes):
    '''
    :param word_counter: 所有(passage, question)对 中word char出现次数 统计
    :param char_counter:
    :return: examples, eval_examples
             每一个训练样本，(passage, question, answer_s, id) 对
             example[] = {"context_tokens": context_tokens, "context_chars": context_chars,
                        "ques_tokens": ques_tokens, "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
             记录各example的 context, context的spans, 该问题的所有answer, 该问题的原始question_id
             eval_examples[str(total)] = { "context": context, "spans": spans,
                                            "answers": answer_texts, "uuid": qa["id"]}
    '''
    print("Generating {} exampels...".format(data_type))
    examples_raws = []
    source = json.load(open(file_name, "r", encoding='utf-8'))
    st = time.time()
    from multiprocessing import Pool
    with Pool(processes=n_processes) as pool:
        multi_res = [pool.apply_async(process_single_triple, 
                                        (article, )) for article in tqdm(source["data"])]
        # print("len(multi_res): ", len(multi_res))
        result = [res.get() for res in tqdm(multi_res) if len(res.get())>0]
        # print("result", len(result[0]))
        for i in range(len((result))): #  result[i] = examples, word_counter, char_counter
            examples_raws = examples_raws + result[i][0]
            word_counter.update(result[i][1])
            char_counter.update(result[i][2])
        print("len(examples_raws): ", len(examples_raws))
    print("multicore time: ", time.time() - st)
    
    examples = []
    eval_examples = {}
    current_id = 0
    for t in tqdm(examples_raws):
        current_id += 1
        example = {"context_tokens":t["context_tokens"], "context_chars":t["context_chars"],
                    "ques_tokens":t["ques_tokens"], "ques_chars":t["ques_chars"],
                    "y1s":t["y1s"], "y2s":t["y2s"], "id":current_id,
                                "passage_connections":t["passage_connections"],
                                "question_connections":t["question_connections"],
                                
                    }
        examples.append(example)
        eval_examples[str(current_id)] = t["eval_examples"]
    random.shuffle(examples)
    print("{} quesitons in total".format(len(examples)))
    # print("all word_counter: ", word_counter)
    # print("all char_counter: ", char_counter)
    return examples, eval_examples


def get_embedding(counter, data_type, emb_file=None, emb_size=None, vec_size=None, limit=-1):
    '''
    简化glove形式，简化为： 对给定 token，得到其token编号，按编号直接得到其 word_vector
    :param counter:    word_counter or char_counter
    :param data_type:  'word' or 'char'
    :param limit:      只保留counter次数达到limit的 word 或 char
    :param emb_file:   glove file: file_path + file_name
    :param size:       glove file 的字典数
                       sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    :param vec_size:   使用的glove embedding size = {50, 100, 200, 300}
    :return:    emb_mat:         emb_mat[idx] = token_idx.vector
                token2_idx_dict: token2idx_dict[token] = idx
    '''
    print("Generating {} embedding...".format(data_type))
    # 保存有效词的vector:  embedding_dict[word] = vector
    embedding_dict = {}
    filter_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        # word counter 计算 glove
        assert emb_size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding='utf-8') as fh:
            for line in tqdm(fh, total=emb_size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word]>limit:
                    embedding_dict[word] = vector
                # embedding_dict[word] = vector
        print("{} / {} tokens have corresponding embedding vector".format(
            len(embedding_dict), len(filter_elements)
        ))
    else:
        # char counter 计算 char embedding
        assert vec_size is not None
        for token in filter_elements:
            embedding_dict[token] = [0. for _ in range(vec_size)]
        print("{} chars have corresponding embedding vector".format(len(filter_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token:idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_festures(config, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
    '''
    example = {"context_tokens": context_tokens, "context_chars": context_chars,
               "ques_tokens": ques_tokens, "ques_chars": ques_chars,
               "y1s": y1s, "y2s": y2s, "id": current_id,
               "passage_connections":passage_connections,
                "question_connections":question_connections}
    '''
    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    char_limit = config.char_limit

    # example过滤函数，将context_tokens question_tokens 数量过大的example过滤掉
    def filter_func(example):
        return len(example["context_tokens"])>para_limit or len(example["ques_tokens"])>ques_limit

    print("Processing {} examples".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0    # 有效example数量
    total_ = 0   # 全部example数量
    for example in tqdm(examples):
        total_ += 1
        if filter_func(example):
            continue

        total += 1
        # 初始化能够feed到 model placeholder的信息
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        # y1 = np.zeros([para_limit], dtype=np.float32)
        # y2 = np.zeros([para_limit], dtype=np.float32)

        # 返回word的index
        def _get_word(word):
            # lowercase; 首字母大写; uppercase
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1
        # 1 为OOV, out-of-vocab

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i][j] = _get_char(char)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i][j] = _get_char(char)

        # 使用最后一个答案
        # squad 2.0中有不含 answer 的 sample
        if len(example["y1s"]) > 0:
            # start, end = example["y1s"][-1], example["y2s"][-1]
            start, end = example["y1s"][0], example["y2s"][0]
            # y1[start], y2[end] = 1.0, 1.0
            y1, y2 = start, end

        # p_connections = np.zeros(shape=[para_limit, para_limit], dtype=np.int32)
        # q_connections = np.zeros(shape=[ques_limit, para_limit], dtype=np.int32)
        # for (src, tar) in example["passage_connections"]:
        #     p_connections[src][tar] = 1.0
        # for (src, tar) in example["question_connections"]:
        #     q_connections[src][tar] = 1.0

        p_connections, q_connections = example["passage_connections"], example["question_connections"]

        record = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
                    "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
                    "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
                    "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
                    # "y1":tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
                    # "y2":tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
                    "y1":tf.train.Feature(int64_list=tf.train.Int64List(value=[y1])),
                    "y2":tf.train.Feature(int64_list=tf.train.Int64List(value=[y2])),
                    "id":tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]])),
                    "context_tokens": tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(" ".join(example["context_tokens"]), encoding='utf-8')])),
                    "question_tokens": tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(" ".join(example["ques_tokens"]), encoding="utf-8")])),
                    # "passage_connections": tf.train.Feature(bytes_list=tf.train.BytesList(value=[p_connections.tostring()])),
                    # "question_connections": tf.train.Feature(bytes_list=tf.train.BytesList(value=[q_connections.tostring()])),
                    # "passage_connections":tf.train.Feature(int64_list=tf.train.Int64List(value=np.reshape(p_connections, [-1]))),
                    # "question_connections":tf.train.Feature(int64_list=tf.train.Int64List(value=np.reshape(q_connections,[-1]))),
                    "passage_connections": tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.reshape(p_connections, [-1]).tostring()])),
                    "question_connections": tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.reshape(q_connections,[-1]).tostring()])),
                    "p_connections_num":tf.train.Feature(int64_list=tf.train.Int64List(value=[p_connections.shape[0]])),
                    "q_connections_num":tf.train.Feature(int64_list=tf.train.Int64List(value=[q_connections.shape[0]])),
                }
            )
        )
        writer.write(record.SerializeToString())
    print("Build {} / {} instances of features in total".format(total, total_))
    writer.close()
    return total


def save(file_name, obj, message):
    if message is not None:
        print("Saving {}...".format(message))
        with open(file_name, "w") as fh:
            json.dump(obj, fh)


def prepro(config):
    n_processes = 6
    # print(word_counter)
    # print(char_counter)
    dev12_examples, dev_eval = get_examples(config.dev_file, "dev12", n_processes)
    # addsent_examples, addsent_eval = get_examples(config.addsent_file, "addsent", n_processes)
    # print(word_counter)
    # print(char_counter)
    # addoneset_examples, addonesent_eval = get_examples(config.addonesent_file, "addonesent", n_processes)
    # test1_examples, test1_eval = get_examples(config.test1_file, "test1",n_processes)
    # test2_examples, test2_eval = get_examples(config.test2_file, "test2",n_processes)
    # dev12_examples, dev_eval = get_examples(config.dev_file, "dev12", n_processes)
    # test12_examples, test12_eval = get_examples(config.test12_file, "test12",n_processes)
    # train12_examples, train_eval = get_examples(config.train_file, "train12", n_processes)

    # word_emb_mat, word2idx_dict = get_embedding(word_counter, "word", config.glove_file, config.glove_size, config.glove_dim)
    # char_emb_mat, char2idx_dict = get_embedding(char_counter, "char", vec_size=config.char_dim)
    with open("word2idx_dict.json") as fh:
        word2idx_dict = json.load(fh)
    with open("char2idx_dict.json") as fh:
        char2idx_dict = json.load(fh)

    # train_total = build_festures(config, train12_examples, "train", config.train_record_file, word2idx_dict, char2idx_dict)
    dev_total = build_festures(config, dev12_examples, "dev", config.dev_record_file, word2idx_dict, char2idx_dict, is_test=True)
    # test12_total = build_festures(config, test12_examples, "test", config.test12_record_file, word2idx_dict, char2idx_dict, is_test=True)
    # test1_total = build_festures(config, test1_examples, "test", config.test1_record_file, word2idx_dict, char2idx_dict, is_test=True)
    # test2_total = build_festures(config, test2_examples, "test", config.test2_record_file, word2idx_dict, char2idx_dict, is_test=True)
    # addonesent_total = build_festures(config, addoneset_examples, "addonesent", config.addonesent_record_file, word2idx_dict, char2idx_dict, is_test=True)
    # addsent_total = build_festures(config, addsent_examples, "addsent", config.addsent_record_file, word2idx_dict, char2idx_dict, is_test=True)

    # meta = {"train_total":train_total, "dev_total":dev_total, "test12_total":test12_total,
    #         "test1_total":test1_total, "test1_total":test2_total,
    #         "addonesent":addonesent_total, "addsent":addsent_total}

    meta = {"dev_total":dev_total}

    # save(config.word_emb_file, word_emb_mat, "word embedding")
    # save(config.char_emb_file, char_emb_mat, "char embedding")
    # save(config.word2idx_dict, word2idx_dict, "word2idx dictionary")
    # save(config.char2idx_dict, char2idx_dict, "char2idx dictionary")
    # save(config.train_eval_file, train_eval, "train eval")
    save(config.dev_eval_file, dev_eval, "dev eval")
    # save(config.test12_eval_file, test12_eval, "test eval")
    # save(config.test1_eval_file, test1_eval, "test eval")
    # save(config.test2_eval_file, test2_eval, "test eval")
    save(config.meta_file, meta, "meta")

    # save(config.addonesent_eval_file, addonesent_eval, "addonesent eval")
    # save(config.addsent_eval_file, addsent_eval, "addsent eval")



# word embedding: 91585 / 111127 tokens
# char embedding: 1425 chars
# train examples: 87391 / 87599 instances of features in total
# dev examples:   10483 / 10570 instances of features in total
# test examples:  10570 / 10570 instances of features in total
