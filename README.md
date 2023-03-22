This repo provides a collection of single-machine and federated baselines for DGraphFin dataset. Please download the dataset from the [DGraph](http://dgraph.xinye.com) web and place it under the folder './dataset/DGraphFin/raw'.  

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
- federated 
```bash
python distributed.py --model mlp --dataset DGraphFin --epochs 200 --globalepoch 1 --datasplit random --delay 0 --runs 10 --device 0 —clients 3
```

- **GCN**
- single-machine
```bash
python gnn.py --model gcn --dataset DGraphFin --epochs 200 --runs 10 --device 0
```
- federated 
```bash
python distributed.py --model gcn --dataset DGraphFin --epochs 200 --globalepoch 1 --datasplit random --delay 0 --runs 10 --device 0 —clients 3
```

- **GraphSAGE**
- single-machine
```bash
python gnn.py --model sage --dataset DGraphFin --epochs 200 --runs 10 --device 0
```
- federated 
```bash
python distributed.py --model sage --dataset DGraphFin --epochs 200 --globalepoch 1 --datasplit random --delay 0 --runs 10 --device 0 —clients 3
```

- **GraphSAGE (NeighborSampler)**
- single-machine
```bash
python gnn_mini_batch.py --model sage_neighsampler --dataset DGraphFin --epochs 200 --runs 10 --device 0
```
- federated 
```bash
python distributed_mini_batch.py --model sage_neighsampler --dataset DGraphFin --epochs 200 --globalepoch 1 --datasplit random --delay 0 --runs 10 --device 0 —clients 3
```

- **GAT (NeighborSampler)**
```bash
python gnn_mini_batch.py --model gat_neighsampler --dataset DGraphFin --epochs 200 --runs 10 --device 0
```
- federated 
```bash
python distributed.py --model mlp --dataset DGraphFin --epochs 200 --globalepoch 1 --datasplit random --delay 0 --runs 10 --device 0 —clients 3
```

- **GATv2 (NeighborSampler)**
```bash
python gnn_mini_batch.py --model gatv2_neighsampler --dataset DGraphFin --epochs 200 --runs 10 --device 0
```
- federated 
```bash
python distributed_mini_batch.py --model gatv2_neighsampler --dataset DGraphFin --epochs 200 --globalepoch 1 --datasplit random --delay 0 --runs 10 --device 0 —clients 3
```

## Results:
Performance on **DGraphFin**:
