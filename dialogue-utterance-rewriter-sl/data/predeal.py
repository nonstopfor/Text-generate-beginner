import json

in_file = open('corpus.txt', 'r', encoding='utf-8')

out_file_train = open('train.json', 'w', encoding='utf-8')
out_file_valid = open('valid.json', 'w', encoding='utf-8')
out_file_test = open('test.json', 'w', encoding='utf-8')
lines = in_file.readlines()

cnt = 0
for line in lines:
    line = line.split()
    post = ' '.join(line[:-1])
    resp = line[-1]
    d = {'post': post, 'resp': resp}
    if cnt < 18000:
        json.dump(d, out_file_train)
        out_file_train.write('\n')
    elif cnt < 19000:
        json.dump(d, out_file_valid)
        out_file_valid.write('\n')
    else:
        json.dump(d, out_file_test)
        out_file_test.write('\n')

    cnt += 1
