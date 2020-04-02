import torch

from torchtext.data import Field, TabularDataset
import numpy as np

import random
import jieba


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def tokenize_post(s):
    sentences = s.split(' ')
    r = []
    for sentence in sentences:
        r.extend(jieba.lcut(sentence))
        r.append('<eos>')
    r.pop(-1)
    return r


def tokenize_resp(s):
    return jieba.lcut(s)


def main():
    setup_seed(2020)
    POST = Field(tokenize=tokenize_post, init_token='<sos>', eos_token='<eos>')
    RESP = Field(tokenize=tokenize_resp, init_token='<sos>', eos_token='<eos>')
    fields = {'post': ('post', POST), 'resp': ('resp', RESP)}
    train_data, valid_data, test_data = TabularDataset.splits(path='./data', train='train.json'
                                                              , validation='valid.json', test='test.json',
                                                              format='json', fields=fields)
    POST.build_vocab(train_data, min_freq=1)
    RESP.build_vocab(train_data, min_freq=1)


if __name__ == '__main__':
    main()
