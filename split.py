import os
import random
random.seed(2)
def split(full_list,shuffle=False,ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
        sublist_1 = full_list[:offset]
        sublist_2 = full_list[offset:]
    return sublist_1,sublist_2
triple=[]
with open('triple.txt') as f:
    li=f.readlines()
    for line in li:
        triple.append(line)
index = list(range(len(triple)))
random.shuffle(triple)
train_len = int(len(triple) * 0.7)
test_len = int(len(triple) * 0.2)
train_data = triple[:train_len]
with open('train2id.txt','w') as f:
    for i in train_data:
        f.write(i)
# train_labels = self.gt[:train_len]       
test_data = triple[train_len:train_len+test_len]  
dev_data = triple[train_len+test_len:]     
with open('test2id.txt','w') as f:
    for i in test_data:
        f.write(i)
with open('valid2id.txt','w') as f:
    for i in dev_data:
        f.write(i)