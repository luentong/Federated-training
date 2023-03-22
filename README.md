This repo provides a collection of single-machine and federated simulation on single machine (not parallel) baselines for DGraphFin dataset. Please download the dataset from the [DGraph](http://dgraph.xinye.com) web and place it under the folder './dataset/DGraphFin/raw'.  

## Environments
Implementing environment:  
- numpy = 1.21.2  
- pytorch = 1.6.0  
- torch_geometric = 1.7.2  
- torch_scatter = 2.0.8  
- torch_sparse = 0.6.9  

- CPU: 2.5 GHz Intel Core i7


## Training

- **MLP**
- single-machine
```bash
python gnn.py --model mlp --dataset DGraphFin --epochs 200 --runs 10 --device 0
```
- federated simulation
```bash
python distributed.py --model mlp --dataset DGraphFin --epochs 200 --globalepoch 1 --datasplit random --delay 0 --runs 10 --device 0 —clients 3
```

- **GCN**
- single-machine
```bash
python gnn.py --model gcn --dataset DGraphFin --epochs 200 --runs 10 --device 0
```
- federated simulation
```bash
python distributed.py --model gcn --dataset DGraphFin --epochs 200 --globalepoch 1 --datasplit random --delay 0 --runs 10 --device 0 —clients 3
```

- **GraphSAGE**
- single-machine
```bash
python gnn.py --model sage --dataset DGraphFin --epochs 200 --runs 10 --device 0
```
- federated simulation
```bash
python distributed.py --model sage --dataset DGraphFin --epochs 200 --globalepoch 1 --datasplit random --delay 0 --runs 10 --device 0 —clients 3
```

- **GraphSAGE (NeighborSampler)**
- single-machine
```bash
python gnn_mini_batch.py --model sage_neighsampler --dataset DGraphFin --epochs 200 --runs 10 --device 0
```
- federated simulation
```bash
python distributed_mini_batch.py --model sage_neighsampler --dataset DGraphFin --epochs 200 --globalepoch 1 --datasplit random --delay 0 --runs 10 --device 0 —clients 3
```

- **GAT (NeighborSampler)**
- single-machine
```bash
python gnn_mini_batch.py --model gat_neighsampler --dataset DGraphFin --epochs 200 --runs 10 --device 0
```
- federated simulation
```bash
python distributed.py --model mlp --dataset DGraphFin --epochs 200 --globalepoch 1 --datasplit random --delay 0 --runs 10 --device 0 —clients 3
```

- **GATv2 (NeighborSampler)**
- single-machine
```bash
python gnn_mini_batch.py --model gatv2_neighsampler --dataset DGraphFin --epochs 200 --runs 10 --device 0
```
- federated simulation
```bash
python distributed_mini_batch.py --model gatv2_neighsampler --dataset DGraphFin --epochs 200 --globalepoch 1 --datasplit random --delay 0 --runs 10 --device 0 —clients 3
```

## Training Paramters

- model   模型.
- dataset 数据源.
epoch   若单机训练为每个run的迭代次数，若联邦学习则为每个globalepoch中client自身迭代次数.
globalepoch  若联邦学习为server和client权重分享次数，默认为1，若单机训练则无意义.
run     重复训练几次，这几次的准确率的统计数据会被打印出来.
device  训练设备序号.
clients 联邦节点数量, 默认为1是非联邦场景.
datasplit    节点数据分布模式, same 每个节点数据都为全量数据，even 每个节点数据从全量数据随机平均分配，random 每个节点数据从全量数据随机正态分配.
fedalgo      指定联邦算法, 暂时只支持FedAvg，默认FedAvg.
randomclient 每次迭代用随机还是全量节点的数据, 默认为0每次选取全量节点，-1为每次选取大于等于1的随机节点数量，N为每次选取N个节点，N为整数 > 0 且 N <=                  clients数. 
delay   设置每轮所有参与的联邦节点和中心服务器数据模拟传输耗时量，默认为0, 每次训练后会输出每个run的训练时间.

## Results:
Performance on **DGraphFin**:
