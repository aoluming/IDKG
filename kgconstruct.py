





import json
import os
import pdb






with open('/home/lijiaqi/mmim/processor/imdbtrain.txt') as f:
    trainli=[]
    lines=f.readlines()
    for line in lines:
        trainli.append(line.strip())
        # print(trainli)
        # pdb.set_trace()


# with open('entity2id.txt','w') as f:
# gens={'Drama':0,'Comedy':1,'Romance':2,'Thriller':3,'Crime':4,'Action':5,'Adventure':6,'Horror':7,'Documentary':8,'Mystery':9,'Sci-Fi':10,'Fantasy':11,'Family':12,'Biography':13,'War':14,
# 'History':15,'Music':16,'Animation':17,'Musical':18,'Western':19,'Sport':20,'Short':21,'Film-Noir':22}
# k=23
# enti={'Drama':0,'Comedy':1,'Romance':2,'Thriller':3,'Crime':4,'Action':5,'Adventure':6,'Horror':7,'Documentary':8,'Mystery':9,'Sci-Fi':10,'Fantasy':11,'Family':12,'Biography':13,'War':14,
# 'History':15,'Music':16,'Animation':17,'Musical':18,'Western':19,'Sport':20,'Short':21,'Film-Noir':22}


gens={'Drama':0,'Comedy':1,'Action':2,'Adventure':3,'Romance':4,'Crime':5,'Horror':6,'Thriller':7,'Biography':8,'Animation':9,'Family':10,'Mystery':11,'Fantasy':12,'Music':13,'History':14,
'Western':15,'Sci-Fi':16,'Musical':17,'Sport':18,'Short':19,'War':20,'Documentary':21,'Film-Noir':22}
k=23
enti={'Drama':0,'Comedy':1,'Action':2,'Adventure':3,'Romance':4,'Crime':5,'Horror':6,'Thriller':7,'Biography':8,'Animation':9,'Family':10,'Mystery':11,'Fantasy':12,'Music':13,'History':14,
'Western':15,'Sci-Fi':16,'Musical':17,'Sport':18,'Short':19,'War':20,'Documentary':21,'Film-Noir':22}


k=23
for i in trainli:
    with open(os.path.join('/data/ljq/imdb/dataset',i+'.json')) as f:
        data=json.load(f)
        #title
        enti[i]=k
        k+=1
        #director
        # print(i)
        if 'director' in data.keys():
            dir=data['director']
            if dir not in enti.keys():
                enti[dir]=k
                k+=1

            
        #actor
        # 
        if 'cast' in data.keys():
            for actor in data['cast']:
                
                if actor not in enti.keys():
                    enti[actor]=k
                    k+=1
# pdb.set_trace()
#entiti cons
with open('entity2id.txt','w') as f:
    for i in enti.keys():
        f.write(i+'\t'+str(enti[i])+'\n')

dirac={}
dirge={}
acge={}
with open('triple.txt','w') as f:
    for i in trainli:
        with open(os.path.join('/data/ljq/imdb/dataset',i+'.json')) as fa:
            data=json.load(fa)
            title_id=enti[i]
            if 'director' in data.keys():
                dir=data['director']
                f.write(str(title_id)+'\t'+str(enti[dir])+'\t'+'0'+'\n')
                if enti[dir] not in dirge.keys():
                    dirge[enti[dir]]=[enti[gen] for gen in data['genres'] if gen in gens.keys()]
                else:
                    for gen in data['genres']:
                        if gen in gens.keys() and enti[gen] not in dirge[enti[dir]]:
                            dirge[enti[dir]].append(enti[gen])
            if 'cast' in data.keys():
                for act in data['cast']:
                    f.write(str(title_id)+'\t'+str(enti[act])+'\t'+'1'+'\n')
                    if enti[act] not in acge.keys():
                        acge[enti[act]]=[enti[gen] for gen in data['genres'] if gen in gens.keys()]
                    else:
                        for gen in data['genres']:
                            if gen in gens.keys() and enti[gen] not in acge[enti[act]]:
                                acge[enti[act]].append(enti[gen])
                            
            for gen in data['genres']:
                if gen in gens.keys():
                    f.write(str(title_id)+'\t'+str(enti[gen])+'\t'+'2'+'\n')
            if 'cast' in data.keys() and 'director' in data.keys():
                dir=data['director']
                if enti[dir] not in dirac.keys():
                    dirac[enti[dir]]=[enti[act] for act in data['cast']]
                else:
                    for act in data['cast']:
                        if enti[act] not in dirac[enti[dir]]:
                            dirac[enti[dir]].append(enti[act])
            
    for dir in dirac.keys():
        for act in dirac[dir]:
            f.write(str(dir)+'\t'+str(act)+'\t'+'3'+'\n')
    for dir in dirge.keys():
        for gen in dirge[dir]:
            f.write(str(dir)+'\t'+str(gen)+'\t'+'4'+'\n')
    for act in acge.keys():
        for gen in acge[act]:
            f.write(str(act)+'\t'+str(gen)+'\t'+'5'+'\n')