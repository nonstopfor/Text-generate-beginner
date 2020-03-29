import torch

from torchtext.data import Field
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
        r.append(jieba.lcut(sentence))
        r.append('<eos>')
    r.pop(-1)
    return r

def tokenize_response(s):
    return jieba.lcut(s)

def main():
    setup_seed(2020)
    SRC = Field()
