# dataset name: DGraphFin

from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from models import MLP, MLPLinear, GCN, SAGE, GAT, GATv2
from logger import Logger
import numpy as np
import tqdm
import argparse
import random
import time

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import pandas as pd

eval_metric = 'auc'

mlp_parameters = {'lr': 0.01
    , 'num_layers': 2
    , 'hidden_channels': 128
    , 'dropout': 0.0
    , 'batchnorm': False
    , 'l2': 5e-7
                  }

gcn_parameters = {'lr': 0.01
    , 'num_layers': 2
    , 'hidden_channels': 128
    , 'dropout': 0.0
    , 'batchnorm': False
    , 'l2': 5e-7
                  }

sage_parameters = {'lr': 0.01
    , 'num_layers': 2
    , 'hidden_channels': 128
    , 'dropout': 0
    , 'batchnorm': False
    , 'l2': 5e-7
                   }


def train(model, data, train_idx, optimizer, no_conv=False):
    # data.y is labels of shape (N, )
    model.train()

    optimizer.zero_grad()
    if no_conv:
        out = model(data.x[train_idx])
    else:
        out = model(data.x, data.adj_t)[train_idx]



    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()



@torch.no_grad()
def test(model, data, split_idx, evaluator, no_conv=False):
    # data.y is labels of shape (N, )
    model.eval()

    if no_conv:
        out = model(data.x)
    else:
        out = model(data.x, data.adj_t)

    y_pred = out.exp()  # (N,num_classes)

    losses, eval_results = dict(), dict()
    eval_results['train'] = 0
    eval_results['valid'] = 0
    eval_results['test'] = 0
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
        try:
            eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])[eval_metric]
        except ValueError:
            pass

    return eval_results, losses, y_pred


class Client:
     def __init__(self, num, data, train_idx):
         self.num = num
         self.data = data
         self.train_idx = train_idx
         print(f'client {self.num + 1:1d} initiated')

     def local_update(self, localEpoch, model, train_idx, split_idx, optimizer, global_parameters, run, args, no_conv):
         eval_metric = 'auc'
         evaluator = Evaluator(eval_metric)
         model.load_state_dict(global_parameters, strict=True)
         for epoch in range(1, localEpoch + 1):
             if epoch % args.log_steps == 0:
                 print(f'Run: {run + 1:02d}, '
                       f'Client: {self.num + 1:02d}, '
                       f'Epoch: {epoch:02d}, ')
             loss = train(model, self.data, self.train_idx, optimizer, no_conv)
             # eval_results, losses, out = test(model, self.data, split_idx, evaluator, True)
             # train_eval, valid_eval, test_eval = eval_results['train'], eval_results['valid'], eval_results['test']
             # train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']

             #           f'Loss: {loss:.4f}, '
             #           f'Train: {100 * train_eval:.3f}%, '
             #           f'Valid: {100 * valid_eval:.3f}% '
             #           f'Test: {100 * test_eval:.3f}%')

         return model.state_dict()


def main():
    parser = argparse.ArgumentParser(description='gnn_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='DGraphFin')
    parser.add_argument('--log_steps', type=int, default=5)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--use_embeddings', action='store_true')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--clients', type=int, default=1)
    parser.add_argument('--delay', type=float, default=0)
    parser.add_argument('--randomclient', type=int, default=0)
    parser.add_argument('--datasplit', type=str, default="random")
    parser.add_argument('--globalepoch', type=int, default=1)
    parser.add_argument('--fedalgo', type=str, default="FedAvg")

    args = parser.parse_args()
    print(args)

    no_conv = False
    if args.model in ['mlp']: no_conv = True

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

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

    print("data", data)
    print(data.x)
    print(data.y)

    split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}

    fold = args.fold
    if split_idx['train'].dim() > 1 and split_idx['train'].shape[1] > 1:
        kfolds = True
        print('There are {} folds of splits'.format(split_idx['train'].shape[1]))
        split_idx['train'] = split_idx['train'][:, fold]
        split_idx['valid'] = split_idx['valid'][:, fold]
        split_idx['test'] = split_idx['test'][:, fold]
    else:
        kfolds = False

    data = data.to(device)
    train_idx = split_idx['train'].to(device)

    result_dir = prepare_folder(args.dataset, args.model)
    print('result_dir:', result_dir)

    if args.model == 'mlp':
        para_dict = mlp_parameters
        model_para = mlp_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = MLP(in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)
    if args.model == 'gcn':
        para_dict = gcn_parameters
        model_para = gcn_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = GCN(in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)
    if args.model == 'sage':
        para_dict = sage_parameters
        model_para = sage_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = SAGE(in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)

    print(f'Model {args.model} initialized')
    global_parameters = model.state_dict()
    logger = Logger(args.runs, args)
    random.shuffle(train_idx)
    rand_list = []
    for i in range(0,args.clients-1):
        rand_list.append(random.randrange(0, len(train_idx)))
    rand_list.append(0)
    rand_list.append(len(train_idx))
    rand_list.sort()
    for run in range(args.runs):
        client_list = []
        for i in range(0, args.clients):
            if args.datasplit == "same":
                client_list.append(Client(i, data, train_idx))
            elif args.datasplit == "even":
                client_list.append(Client(i, data, train_idx[0 + i*(len(train_idx)//args.clients) : (1+i)*(len(train_idx)//args.clients)]))
            else:
                client_list.append(Client(i, data, train_idx[rand_list[i]: rand_list[i + 1]]))
        starttime = time.time()
        import gc
        gc.collect()
        #print(sum(p.numel() for p in model.parameters()))
        delay_count = 0
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
        for global_e in range(0, args.globalepoch):
            sum_parameters = None
            print(f'\nGlobal Epoch {global_e + 1:1d}:\n')
            client_range = list(range(0, args.clients))
            random.shuffle(client_range)
            if args.randomclient == -1:
                num = random.randint(1,args.clients)
                client_range = client_range[0:num]
            elif type(args.randomclient) == int and args.randomclient >= 1:
                client_range = client_range[0:args.randomclient]

            for client in client_range:

                # 获取当前Client训练得到的参数
                local_parameters = client_list[client].local_update(args.epochs, model, train_idx, split_idx, optimizer, global_parameters, run, args, no_conv)

                # 对所有的Client返回的参数累加（最后取平均值）
                if sum_parameters is None:
                    sum_parameters = local_parameters
                else:
                    for var in sum_parameters:
                        sum_parameters[var] = sum_parameters[var] + local_parameters[var]
            # 取平均值，得到本次通信中Server得到的更新后的模型参数
            for var in global_parameters:
                global_parameters[var] = (sum_parameters[var] / args.clients)
            model.load_state_dict(global_parameters, strict=True)
            delay_count += 1
        eval_metric = 'auc'
        evaluator = Evaluator(eval_metric)
        model.load_state_dict(global_parameters, strict=True)
        eval_results, losses, out = test(model, data, split_idx, evaluator, no_conv)
        train_eval, valid_eval, test_eval = eval_results['train'], eval_results['valid'], eval_results['test']
        train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']
        print(f'Run_global: {run + 1:02d}, '
                  f'Train: {100 * train_eval:.3f}%, '
                  f'Valid: {100 * valid_eval:.3f}% '
                  f'Test: {100 * test_eval:.3f}%')
        logger.add_result(run, [train_eval, valid_eval, test_eval])
        endtime = time.time()
        print(f'run {run:1d} time taken: {- starttime + endtime + delay_count * args.delay:2.3f} seconds')
    final_results = logger.print_statistics()
    print('final_results:', final_results)

if __name__ == "__main__":
    main()
