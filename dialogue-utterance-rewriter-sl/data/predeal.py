import json

in_file=open('corpus.txt','r',encoding='utf-8')

out_file=open('train.json','w',encoding='utf-8')

lines=in_file.readlines()

cnt=0
for line in lines:
    line=line.split()
    post=' '.join(line[:-1])
    resp=line[-1]
    d={'post':post,'resp':resp}
    json.dump(d,out_file)
    out_file.write('\n')
    if cnt==0:
        print(line)
        print(post)
    cnt+=1