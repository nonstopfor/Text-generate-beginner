import jieba

def tokenize_post(s):
    sentences=s.split(' ')
    r=[]
    for sentence in sentences:
        r.append(jieba.lcut(sentence))
        r.append('<eos>')
    r.pop(-1)
    return r

s='能给我签名吗 出专辑再议 我现在就要'
print(tokenize_post(s))