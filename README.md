# 
This is the repo of 'Incorporating Domain Knowledge Graph into Multimodal Movie Genre Classification with Self-Supervised Attention and Contrastive Learning' in MM 2023.

please download the MM-IMDB 2.0 dataset in the following link:https://drive.google.com/file/d/1fmU3ZKM3ieTDeTAeyK1uT9sFVvaGplIp/view?usp=sharing
and change the path in IDKG/processor/dataset.py

We have train the Open KG in advance and download it in :https://drive.google.com/file/d/1-yszovzxKTXi1284HUuJz-sNjs1dFfZQ/view?usp=sharing. Also you need to change the kgparam_path in the dataset.py.

For training the code:
sh run_imdbclip.sh

