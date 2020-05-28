#! -*- coding: utf-8 -*-
# 百度LIC2020的事件抽取赛道，非官方baseline
# 直接用RoBERTa+CRF
# 在第一期测试集上能达到0.78的F1，优于官方baseline

import json
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import pylcs
import linecache
from keras.callbacks import LearningRateScheduler
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 5 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.2)
        print("lr changed to {}".format(lr * 0.2))
    return K.get_value(model.optimizer.lr)

# 基本信息
maxlen = 256
epochs = 10
batch_size = 2
learning_rate = 2e-4
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率

# bert配置
config_path = 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json'
checkpoint_path = 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt'
dict_path = 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt'


def load_data(filename):
    D = []
    with open(filename) as f:
        for l in f:
            l = json.loads(l)
            arguments = {}
            for event in l['event_list']:
                for argument in event['arguments']:
                    key = argument['argument']
                    value = (event['event_type'], argument['role'])  #事件类型+论元角色
                    arguments[key] = value
            D.append((l['text'], arguments))
    return D


# 读取数据
train_data = load_data('train_data/train.json')
valid_data = load_data('dev_data/dev.json')

# 读取schema  事件模式
with open('event_schema/event_schema.json') as f:
    id2label, label2id, n = {}, {}, 0
    for l in f:
        l = json.loads(l)
        for role in l['role_list']:
            key = (l['event_type'], role['role'])
            id2label[n] = key
            label2id[key] = n
            n += 1
    num_labels = len(id2label) * 2 + 1

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):   # 迭代器 ： batch id segment_id label
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, arguments) in self.sample(random):
            """
            try:
                text=list(arguments.items())[0][1][0]+"|"+text
            except IndexError:
                pass
            else :
                text = list(arguments.items())[0][1][0] + "|" + text
            """
            token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
            labels = [0] * len(token_ids)
            for argument in arguments.items():
                a_token_ids = tokenizer.encode(argument[0])[0][1:-1]
                start_index = search(a_token_ids, token_ids)
                if start_index != -1:
                    labels[start_index] = label2id[argument[1]] * 2 + 1
                    for i in range(1, len(a_token_ids)):
                        labels[start_index + i] = label2id[argument[1]] * 2 + 2
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

def data_process(train_data, batch_size,random=False):
    """数据生成
    """
       # 迭代器 ： batch id segment_id label
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for is_end, (text, arguments) in train_data(random):
        token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
        labels = [0] * len(token_ids)
        for argument in arguments.items():
            a_token_ids = tokenizer.encode(argument[0])[0][1:-1]
            start_index = search(a_token_ids, token_ids)
            if start_index != -1:
                labels[start_index] = label2id[argument[1]] * 2 + 1
                for i in range(1, len(a_token_ids)):
                    labels[start_index + i] = label2id[argument[1]] * 2 + 2
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append(labels)
        if len(batch_token_ids) == batch_size or is_end:
            batch_token_ids = sequence_padding(batch_token_ids)
            batch_segment_ids = sequence_padding(batch_segment_ids)
            batch_labels = sequence_padding(batch_labels)
            yield [batch_token_ids, batch_segment_ids], batch_labels
            batch_token_ids, batch_segment_ids, batch_labels = [], [], []


model = build_transformer_model(
    config_path,
    checkpoint_path,
)
unfreezon=['Transformer-11-MultiHeadSelfAtt','Transformer-11-FeedForward',
           'Transformer-11-FeedForward-Drop','Transformer-11-FeedForward-Add',
           'Transformer-11-FeedForward-Norm']
for layer in model.layers:
    if not layer.name in unfreezon:
        pass #layer.trainable = False
output = Dense(num_labels)(model.output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
model.summary()
# 编译模型 损失函数
model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learning_rate),  # lr scheduler
    metrics=[CRF.sparse_accuracy]
)


def viterbi_decode(nodes, trans):
    """Viterbi算法求最优路径
    其中nodes.shape=[seq_len, num_labels],
        trans.shape=[num_labels, num_labels].
    """
    labels = np.arange(num_labels).reshape((1, -1))
    scores = nodes[0].reshape((-1, 1))
    scores[1:] -= np.inf  # 第一个标签必然是0
    paths = labels
    for l in range(1, len(nodes)):
        M = scores + trans + nodes[l].reshape((1, -1))
        idxs = M.argmax(0)
        scores = M.max(0).reshape((-1, 1))
        paths = np.concatenate([paths[:, idxs], labels], 0)
    return paths[:, scores[:, 0].argmax()]


def extract_arguments(text):
    """arguments抽取函数 　冻结部分Bert 层
    """
    tokens = tokenizer.tokenize(text)  #转化为tokens
    while len(tokens) > 510:           #大于510，pop
        tokens.pop(-2)
    mapping = tokenizer.rematch(text, tokens)    # 进行文本和token的匹配
    token_ids = tokenizer.tokens_to_ids(tokens)  # 找到tokens的ID
    segment_ids = [0] * len(token_ids)           #找到segment的ID
    nodes = model.predict([[token_ids], [segment_ids]])[0]   #模型预测
    trans = K.eval(CRF.trans)      #
    labels = viterbi_decode(nodes, trans)
    arguments, starting = [], False
    for i, label in enumerate(labels):
        if label > 0:
            if label % 2 == 1:
                starting = True
                arguments.append([
                    [i], id2label[(label - 1) // 2]
                ])
            elif starting:
                arguments[-1][0].append(i)
            else:
                starting = False
        else:
            starting = False
    for w, l in arguments:
        if w[-1] == len(tokens) - 1: w[-1] = len(tokens) - 2
    return {
        text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1]: l
        for w, l in arguments
    }


def evaluate(data):
    """评测函数（跟官方评测结果不一定相同，但很接近）
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for text, arguments in tqdm(data):
        inv_arguments = {v: k for k, v in arguments.items()}
        pred_arguments = extract_arguments(text)
        pred_inv_arguments = {v: k for k, v in pred_arguments.items()}
        Y += len(pred_inv_arguments)
        Z += len(inv_arguments)
        for k, v in pred_inv_arguments.items():
            if k in inv_arguments:
                # 用最长公共子串作为匹配程度度量
                l = pylcs.lcs(v, inv_arguments[k])
                X += 2. * l / (len(v) + len(inv_arguments[k]))
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


def predict_to_file(in_file, out_file):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    i=0
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            arguments = extract_arguments(l["text"])
            event_list = []
            for k, v in arguments.items():
                flag=0
                for i in range(len(event_list)):
                    if event_list[i]['event_type']==v[0]:
                        event_list[i]['arguments'].append({
                            'role': v[1],
                            'argument': k})
                        flag=1
                if not flag:
                    event_list.append({
                        'event_type': v[0],
                        'arguments': [{
                            'role': v[1],
                            'argument': k
                        }]
                    })
            l['event_list'] = event_list
            l = json.dumps(l, ensure_ascii=False)
            fw.write(l + '\n')
            i = i + 1
    fw.close()


class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """
    def __init__(self):
        self.best_val_f1 = 0.
    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('best_model_256.weights')
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
if __name__ == '__main__':
    train_generator = data_generator(train_data, batch_size)
    #data_process(train_data, batch_size)
    evaluator = Evaluator()
    """
    model.fit(x=train_data,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[evaluator]
              )
            """

    model.load_weights('best_model_256.weights')
    reduce_lr = LearningRateScheduler(scheduler)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator,reduce_lr]
    )
    predict_to_file('dev_data/dev.json', 'ee_pred_dev_s1.json')
    predict_to_file('test1_data/test1.json', 'ee_pred_test1_s1.json')
    predict_to_file('test1_data/test2.json', 'ee_pred_test2_s1.json')


