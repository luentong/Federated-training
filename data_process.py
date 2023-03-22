from utils import DGraphFin
import argparse
import torch_geometric.transforms as T

parser = argparse.ArgumentParser(description='gnn_models')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--dataset', type=str, default='DGraphFin')
parser.add_argument('--log_steps', type=int, default=10)
parser.add_argument('--model', type=str, default='mlp')
parser.add_argument('--use_embeddings', action='store_true')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--fold', type=int, default=0)

args = parser.parse_args()
print(args)

dataset = DGraphFin(root='./dataset/', name=args.dataset, transform=T.ToSparseTensor())

nlabels = dataset.num_classes
if args.dataset in ['DGraphFin']: nlabels = 2

data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()

if args.dataset in ['DGraphFin']:
    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x
if data.y.dim() == 2:
    data.y = data.y.squeeze(1)

print(data)
print(data.x)
print(data.y)
print(data.train_mask)
print(len(data.train_mask))