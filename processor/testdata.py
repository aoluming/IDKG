# from msilib import sequence
import random
import os
import torch
import json
import ast
from PIL import Image
import torchvision.models as models
# from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer
from transformers import ViltProcessor
from torchvision import transforms
import logging
logger = logging.getLogger(__name__)
import re
import numpy as np
import pdb
import random
random.seed(2)
from collections import OrderedDict, Counter
from sklearn.preprocessing import MultiLabelBinarizer
class mmimdbDataset(Dataset):
    def __init__(self, data_path='/data/ljq/mmim/mmimdb', img_path=None,max_seq=40, sample_ratio=1.0, mode="train") -> None:
        self.data_path=data_path
        with open(os.path.join(self.data_path,'list.txt'), 'r') as f:
            files = f.read().splitlines()
        self.vocab_counts=[]
        self.movies=[]
        n_classes=23
        for i, file in enumerate(files):
            with open(os.path.join(self.data_path,file)) as f:
                data = json.load(f)
                data['imdb_id'] = file.split('/')[-1].split('.')[0]
                im_file = os.path.join(self.data_path,file.replace('json', 'jpeg'))
                if all([k in data for k in ('genres', 'plot')] + [os.path.isfile(im_file)]):
                    plot_id = np.array([len(p) for p in data['plot']]).argmax()
                    data['plot'] = self.normalizeText(data['plot'][plot_id])
                    if len(data['plot']) > 0:
                        self.vocab_counts.extend(data['plot'])
                        self.movies.append(data)
        counts = OrderedDict(
            Counter([g for m in self.movies for g in m['genres']]).most_common())
        target_names = list(counts.keys())[:n_classes]
        # print(counts)
        # print(target_names)
        # 
        le = MultiLabelBinarizer()
        self.Y = le.fit_transform([m['genres'] for m in self.movies])
        self.le=le
        self.labels = np.nonzero(le.transform([[t] for t in target_names]))[1]
        self.gt=self.Y[:,self.labels]
        self.target_names=target_names
        index = list(range(len(self.movies)))
        random.shuffle(index)
        self.m=[]
        self.g=[]
        for i in range(1000):
            self.m.append(self.movies[i])
            self.g.append(self.gt[i])
            i+=1
        self.movies=self.m
        self.gt=self.g
        # self.gt=self.gt[0]*1000
        train_len = int(len(self.movies) * 0.6)
        test_len = int(len(self.movies) * 0.3)
        train_data = self.movies[:train_len]
        train_labels = self.gt[:train_len]       
        test_data = self.movies[train_len:train_len+test_len]
        test_labels = self.gt[train_len:train_len+test_len]       
        dev_data = self.movies[train_len+test_len:]
        dev_labels = self.gt[train_len+test_len:]
               
        if mode=='train':
            self.samples=train_data
            self.gengt=train_labels
        elif mode=='test':
            self.samples=test_data
            self.gengt=test_labels
        else:
            self.samples=dev_data
            self.gengt=dev_labels

        # pdb.set_trace()

        # self.tokenizer = self.processor.tokenizer
    def normalizeText(self,text):
        text = text.lower()
        text = re.sub(r'<br />', r' ', text).strip()
        text = re.sub(r'^https?:\/\/.*[\r\n]*', ' L ', text, flags=re.MULTILINE)
        text = re.sub(r'[\~\*\+\^`_#\[\]|]', r' ', text).strip()
        text = re.sub(r'[0-9]+', r' N ', text).strip()
        text = re.sub(r'([/\'\-\.?!\(\)",:;])', r' \1 ', text).strip()
        return text.split()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        return 0
class gmudataset(mmimdbDataset):
    def __init__(self,data_path='/data/ljq/mmim/mmimdb', word2vec_path='/data/ljq/mmim/mmimdb/GoogleNews-vectors-negative300.bin',mode='dev') -> None:
        
        super().__init__(data_path=data_path,mode=mode)
        vocab_counts = OrderedDict(Counter(self.vocab_counts).most_common())
        vocab = ['_UNK_'] + [v for v in vocab_counts.keys()]
        self.googleword2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        self.ix_to_word = dict(zip(range(len(vocab)), vocab))
        self.word_to_ix = dict(zip(vocab, range(len(vocab))))
        # lookup = np.array([self.googleword2vec[v] for v in vocab if v in googleword2vec])   
        self.vocab=vocab
        self.data_path=data_path
        self.vgg = models.vgg19(pretrained=True).cuda()
        self.transform = transforms.Compose([
            transforms.Resize((160,256)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # wordinput = np.zeros((300), dtype='float32')
        words=self.samples[idx]['plot']
        sequences= [self.word_to_ix[w] if w in self.vocab else unk_idx for w in words]
        wordinput = np.array([self.googleword2vec[w]
                            for w in words if w in self.googleword2vec]).mean(axis=0)
        img_path=os.path.join(self.data_path,'dataset',self.samples[idx]['imdb_id']+'.jpeg')
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        # pdb.set_trace()
        # a=self.vgg.features(image)#使用feature获取特征层（卷积层）的特征；输出特征维度为【1，512，4，4】
        # b=self.vgg.avgpool(a)#使用vgg定义的池化操作；输出特征维度为【1，512，7，7】
        # b=torch.flatten(b)#将特征变成一维度；输出特征维度为【1，25088】
        # c=self.vgg.classifier[:4](b)#使用分类层的的第一层，当然可以选择数；输出特征维度为【1，4096】

        return image,wordinput,self.gengt[idx]   


    
if __name__ == "__main__":
    # data=mmimdbDataset(
    data=gmudataset()
    # print("123",data[0][1].shape)
    train_dataloader = DataLoader(data, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    for i,k in enumerate(train_dataloader):
        print(k[2])
        pdb.set_trace()
    pdb.set_trace()