import os
import argparse
import logging
import sys
sys.path.append("..")
import logging
import pdb
import os

# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader

from processor.dataset import imdbclipdataset
# from processor.testdata import gmudataset
from configs.exp_1 import Exp1
from modules.train import clipweightTrainer
import warnings
from module.modeling_vilt import ViltModel
import torch
from PIL import Image
import open_clip
from model.clipclass import *
warnings.filterwarnings("ignore", category=UserWarning)
from tensorboardX import SummaryWriter

# logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt = '%m/%d/%Y %H:%M:%S',
#                     level = logging.INFO)
# logger = logging.getLogger(__name__)

# MODEL_CLASSES = {
#     'MRE': ConditionedVilt,

# }
DATASET = {

    'clipweight':imdbclipdataset,
}


        
# }
TRAINER_CLASSES = {
    'clipweight':clipweightTrainer,

}


def set_seed(seed=2021):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='MRE', type=str, help="The name of dataset.")
    parser.add_argument('--model_name', default='dandelin/vilt-b32-mlm', type=str, help="Pretrained language model path")
    parser.add_argument('--num_epochs', default=30, type=int, help="num training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=1, type=int, help="epoch to start evluate")
    parser.add_argument('--seed', default=1, type=int, help="random seed, default is 1")
    parser.add_argument('--prompt_len', default=10, type=int, help="prompt length")
    parser.add_argument('--prompt_dim', default=800, type=int, help="mid dimension of prompt project layer")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default=None, type=str, help="save model at save_path")
    parser.add_argument('--write_path', default=None, type=str, help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--use_prompt', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--vis_dim', default=4096, type=int)
    parser.add_argument('--text_dim', default=300, type=int)
    parser.add_argument('--kg_dim', default=200, type=int)
    parser.add_argument('--n_classes', default=23, type=int)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--ignore_idx', default=-100, type=int)
    parser.add_argument('--modelclass', default='kg', type=str)
    parser.add_argument('--sample_ratio', default=1.0, type=float, help="only for low resource.")

    args = parser.parse_args()

    # data_path, img_path = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name]
    Trainer = TRAINER_CLASSES[args.dataset_name]
    # data_process, dataset_class = DATA_PROCESS[args.dataset_name]
    dataset=DATASET[args.dataset_name]
    clipmodel, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')

    set_seed(args.seed) # set seed, default is 1
    if args.save_path is not None:  # make save_path dir
        # args.save_path = os.path.join(args.save_path, args.dataset_name+"_"+str(args.batch_size)+"_"+str(args.lr)+"_"+args.notes)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    print(args)
    logdir = "logs/" + args.dataset_name+ "_"+str(args.batch_size) + "_" + str(args.lr) + args.notes
    writer = SummaryWriter(logdir=logdir)
    # writer=None

    
    # pdb.set_trace()
    train_dataset = dataset(mode='train',processor=preprocess)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # for i,sample in enumerate(train_dataloader):
    #     print(sample)
    #     pdb.set_trace()
    dev_dataset = dataset(mode='dev',processor=preprocess)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_dataset = dataset(mode='test',processor=preprocess)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # pdb.set_trace()

    if args.modelclass=='clipweight':
        model = clipweightClassifier(args.vis_dim, args.text_dim,args.kg_dim,args.n_classes,args.hidden_size)
        # for name, param in model.named_parameters():
        #     print(name)
    # pdb.set_trace()
    trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model, clipmodel=clipmodel,args=args, logger=logger, writer=writer,target_names=train_dataset.target_names)
    if args.do_train:
        # train
        trainer.train()
        # test best model
        args.load_path = os.path.join(args.save_path, 'best_model.pth')
        trainer.test()

    if args.only_test:
        # only do test
        trainer.test()

    torch.cuda.empty_cache()
    # writer.close()
    

if __name__ == "__main__":
    main()