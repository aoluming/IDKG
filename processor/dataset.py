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
import torch
from PIL import Image
import open_clip
import random
random.seed(10)
from collections import OrderedDict, Counter
from sklearn.preprocessing import MultiLabelBinarizer

Image.MAX_IMAGE_PIXELS = 2300000000


class imdbDataset(Dataset):
    def __init__(self, data_path='/data/ljq/imdb', img_path=None,max_seq=40, sample_ratio=1.0, mode="train") -> None:
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
                im_file = os.path.join(self.data_path,file.replace('json', 'jpg'))
                if all([k in data for k in ('genres', 'plot')] + [os.path.isfile(im_file)]):
                    # plot_id = np.array([len(p) for p in data['plot']]).argmax()
                    # pdb.set_trace()
                    data['plot'] = self.normalizeText(data['plot'])
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
        train_len = int(len(self.movies) * 0.6)
        test_len = int(len(self.movies) * 0.3)
        
 
      
        train_data = [self.movies[index[i]] for i in range(train_len)]
        train_labels = [self.gt[index[i]] for i in range(train_len)]
        test_data = [self.movies[index[i]] for i in range(train_len,train_len+test_len)]
        test_labels = [self.gt[index[i]] for i in range(train_len,train_len+test_len)]
        dev_data = [self.movies[index[i]] for i in range(train_len+test_len,len(index))]
        dev_labels = [self.gt[index[i]] for i in range(train_len+test_len,len(index))]
        with open('imdbtrain.txt','w') as f:
                for i in train_data:
                    id=i['imdb_id']
                    f.write(str(id)+'\n')
        with open('imdbtest.txt','w') as f:
            for i in test_data:
                id=i['imdb_id']
                f.write(str(id)+'\n')
        with open('imdbdev.txt','w') as f:
            for i in dev_data:
                id=i['imdb_id']
                f.write(str(id)+'\n')
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

    
class imdbclipdataset(imdbDataset):
    def __init__(self,data_path='/data/ljq/imdb', entity_path='/home/lijiaqi/idkg/entity2id.txt',kgparam_path='/home/lijiaqi/idkg/OpenKE/transh_paramsimdb.json',model=None,processor=None,mode='dev') -> None:
        
        super().__init__(data_path=data_path,mode=mode)
        self.entity_path=entity_path
        self.kg_path=kgparam_path
        self.entityid={}
        with open(self.entity_path) as f:
            lines=f.readlines()[1:]
            for line in lines:
                # pdb.set_trace()
                self.entityid[line.strip().split('\t')[0]]=int(line.strip().split('\t')[-1])
        with open(self.kg_path) as f:
            kgparams=json.load(f)
            self.kgemb=kgparams['ent_embeddings.weight']
        vocab_counts = OrderedDict(Counter(self.vocab_counts).most_common())
        vocab = ['_UNK_'] + [v for v in vocab_counts.keys()]

        self.ix_to_word = dict(zip(range(len(vocab)), vocab))
        self.word_to_ix = dict(zip(vocab, range(len(vocab))))
 
        self.vocab=vocab
        self.data_path=data_path
        self.preprocessor=processor
        # self.vgg = models.vgg19(pretrained=True).cuda()
        self.transform = transforms.Compose([
            transforms.Resize((160,256)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.samples)
    def normalizeText(self,text):
        text = text.lower()
        text = re.sub(r'<br />', r' ', text).strip()
        text = re.sub(r'^https?:\/\/.*[\r\n]*', ' L ', text, flags=re.MULTILINE)
        text = re.sub(r'[\~\*\+\^`_#\[\]|]', r' ', text).strip()
        text = re.sub(r'[0-9]+', r' N ', text).strip()
        text = re.sub(r'([/\'\-\.?!\(\)",:;])', r' \1 ', text).strip()
        return text
    def __getitem__(self, idx):
        # wordinput = np.zeros((300), dtype='float32')
        # pdb.set_trace()
        words=self.samples[idx]['plot']
        words = open_clip.tokenize(words).squeeze(0)
        # kgembedding=np.array()
        #kgembedding
        kgem=[]
        kgweight=0
        labelem=[self.kgemb[i] for i in range(23)]
        labelem=np.array(labelem)
        i=0
        if 'cast' in self.samples[idx].keys():
            for actor in self.samples[idx]['cast']:
                if actor in self.entityid.keys():
                    kgem.append(self.kgemb[self.entityid[actor]])
                    i+=1

        if 'director' in self.samples[idx].keys():
            
            dir=self.samples[idx]['director']
            if dir in self.entityid.keys():
                i+=1
                

                kgem.append(self.kgemb[self.entityid[dir]])
        if len(kgem)==0:
            kgem=np.zeros(200)
        else:
            kgem=np.array(kgem).sum(axis=0)

        img_path=os.path.join(self.data_path,'dataset',self.samples[idx]['imdb_id']+'.jpg')
        # image = Image.open(img_path).convert('RGB')
        kgweight=i/(i+6)#10
        image = Image.open(img_path)
        image = self.preprocessor(image)


        return image,words,self.gengt[idx],kgem,labelem,kgweight

if __name__ == "__main__":
    # data=mmimdbDataset(
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
    # data=clipdataset(processor=preprocess,mode='dev')
    # # print(data[0])
    # x=0
    # # print("123",data[0][1].shape)
    # train_dataloader = DataLoader(data, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    # for i,k in enumerate(train_dataloader):
    #     for s in k[2]:
    #         if s[-1]==1:
    #             x+=1
    data=imdbDataset()
    pdb.set_trace()
    # pdb.set_trace()


    # data=mmimdbDataset()