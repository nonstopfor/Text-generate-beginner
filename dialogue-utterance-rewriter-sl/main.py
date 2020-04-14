import torch

from torchtext.data import Field, TabularDataset
from torchtext.data import BucketIterator
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
    # two sentences
    sentences = s.split(' ')
    r = []
    for sentence in sentences:
        r.extend(jieba.lcut(sentence))
        r.append('<eos>')
    r.pop(-1)
    return r


def tokenize_query(s):
    # one sentence
    return jieba.lcut(s)


def tokenize_resp(s):
    # one sentence
    return jieba.lcut(s)


def main():
    setup_seed(2020)
    POST = Field(tokenize=tokenize_post, init_token='<sos>', eos_token='<eos>')
    QUERY = Field(tokenize=tokenize_query, init_token='<sos>', eos_token='<eos>')
    RESP = Field(tokenize=tokenize_resp, init_token='<sos>', eos_token='<eos>')
    # first 'post' is the key in loaded json, second 'post' is the key in batch
    fields = {'post': ('post', POST), 'query': ('query', QUERY), 'resp': ('resp', RESP)}
    train_data, valid_data, test_data = TabularDataset.splits(path='./data', train='train.json'
                                                              , validation='valid.json', test='test.json',
                                                              format='json', fields=fields)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    POST.build_vocab(train_data, min_freq=1)
    QUERY.build_vocab(train_data, min_freq=1)
    RESP.build_vocab(train_data, min_freq=1)
    # print(POST.vocab.__dict__)

    batch_size = 10
    train_iter, val_iter, test_iter = BucketIterator.splits((train_data, valid_data, test_data),
                                                            batch_sizes=(batch_size, batch_size, batch_size),
                                                            device=device,
                                                            sort_key=lambda x: len(x.post),
                                                            sort_within_batch=True,
                                                            # sort according to the len, for padding in LSTM
                                                            repeat=False)
    cnt = 0
    for i, batch in enumerate(train_iter):
        if cnt == 0:
            post = batch.post
            print(post.size())
            print(batch.post)
        cnt += 1


if __name__ == '__main__':
    main()
