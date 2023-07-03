import argparse
import torch
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected

from large_model import Transformer
from sampler import LocalSampler, dataset_drive_url, load_twitch_gamer, rand_train_test_idx, even_quantile_labels


import time
import os
from os import path
import pandas as pd
import scipy.io
from google_drive_downloader import GoogleDriveDownloader as gdd


def train(model, loader, x, pos_enc, y, optimizer, device, conv_type):
    model.train()

    counter = 1
    total_loss, total_correct, total_count = 0, 0, 0

    if conv_type == 'global' : 

        for node_idx in loader :
            batch_size = len(node_idx)

            feat = x[node_idx] if torch.is_tensor(x) else x(node_idx)
            input = feat.to(device), pos_enc[node_idx].to(device), node_idx            
            
            optimizer.zero_grad()
            out = model.to(device).global_forward(*input)
            loss = F.cross_entropy(out, y[node_idx].to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*batch_size
            total_correct += out.argmax(dim=-1).cpu().eq(y[node_idx]).sum().item()
            total_count += batch_size
        
            counter += 1            

    else :

        for edge_index, node_idx, batch_size in loader:

            edge_index, edge_dist = edge_index[:2], edge_index[2:]

            feat = x[node_idx] if torch.is_tensor(x) else x(node_idx)
            input = feat.to(device), edge_index.to(device), edge_dist.to(device), pos_enc[node_idx[:batch_size]].to(device), node_idx[:batch_size]            
            
            optimizer.zero_grad()
            out = model.to(device)(*input)
            loss = F.cross_entropy(out, y[node_idx[:batch_size]].to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*batch_size
            total_correct += out.argmax(dim=-1).cpu().eq(y[node_idx[:batch_size]]).sum().item()
            total_count += batch_size
        
            # print(f'Train Progress: {counter}/{len(loader)}:{100*total_correct/total_count:.2f}%, Train Mem:{torch.cuda.max_memory_allocated(device=device)/1e6} MB')
            counter += 1

    # print(f'Train Mem:{torch.cuda.max_memory_allocated(device=device)/1e6} MB')

    return total_loss/total_count, total_correct/total_count


def test(model, loader, x, pos_enc, y, device, conv_type, fast_eval=False):
    model.eval()

    counter = 1
    total_correct, total_count = 0, 0

    if conv_type == 'global' : 

        for node_idx in loader:
            batch_size = len(node_idx)

            feat = x[node_idx] if torch.is_tensor(x) else x(node_idx)
            out = model.to(device).global_forward(feat.to(device), pos_enc[node_idx].to(device), node_idx)

            total_correct += out.argmax(dim=-1).cpu().eq(y[node_idx]).sum().item()
            total_count += batch_size
            
            if fast_eval and counter == len(loader)//10 :
                if total_correct/total_count < 0.8 :
                    return 0
            counter += 1           

    else :

        for edge_index, node_idx, batch_size in loader:

            # print('node size:', len(node_idx))
            # print('edge size:', edge_index.shape[1])

            edge_index, edge_dist = edge_index[:2], edge_index[2:]
            feat = x[node_idx] if torch.is_tensor(x) else x(node_idx)
            out = model.to(device)(feat.to(device), edge_index.to(device), edge_dist.to(device), pos_enc[node_idx[:batch_size]].to(device), node_idx[:batch_size])

            total_correct += out.argmax(dim=-1).cpu().eq(y[node_idx[:batch_size]]).sum().item()
            total_count += batch_size

            # print(f'Test Progress: {counter}/{len(loader)}:{100*total_correct/total_count:.2f}%, Test Mem:{torch.cuda.max_memory_allocated(device=device)/1e6} MB')
            
            if fast_eval and counter == len(loader)//10 :
                if total_correct/total_count < 0.8 :
                    return 0
            counter += 1

    # print(f'Test Mem:{torch.cuda.max_memory_allocated(device=device)/1e6} MB')
    return total_correct/total_count


def train_bi(model, loader, x, pos_enc, y, optimizer, device, conv_type):
    model.train()

    counter = 1
    total_loss, total_correct, total_count = 0, 0, 0

    if conv_type == 'global' : 

        for node_idx in loader :
            batch_size = len(node_idx)

            feat = x[node_idx] if torch.is_tensor(x) else x(node_idx)
            input = feat.to(device), pos_enc[node_idx].to(device), node_idx            
            
            optimizer.zero_grad()
            out = model.to(device)(*input).squeeze_()

            loss = F.binary_cross_entropy_with_logits(out, y[node_idx].float().to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*batch_size
            total_correct += (out>0).int().cpu().eq(y[node_idx]).sum().item()
            total_count += batch_size
            counter += 1     
    
    else :

        for edge_index, node_idx, batch_size in loader:

            edge_index, edge_dist = edge_index[:2], edge_index[2:]

            input = x[node_idx].to(device), edge_index.to(device), edge_dist.to(device), pos_enc[node_idx[:batch_size]].to(device), node_idx[:batch_size]

            optimizer.zero_grad()
            out = model.to(device)(*input).squeeze_()

            loss = F.binary_cross_entropy_with_logits(out, y[node_idx[:batch_size]].float().to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*batch_size
            total_correct += (out>0).int().cpu().eq(y[node_idx[:batch_size]]).sum().item()
            total_count += batch_size
        
            # print(f'Train Progress: {counter}/{len(loader)}:{100*total_correct/total_count:.2f}%, Train Mem:{torch.cuda.max_memory_allocated(device=device)/1e6} MB')
            counter += 1

    # print(f'Train Mem:{torch.cuda.max_memory_allocated(device=device)/1e6} MB')

    return total_loss/total_count, total_correct/total_count


def test_bi(model, loader, x, pos_enc, y, device, conv_type, fast_eval=False):
    model.eval()

    counter = 1
    total_correct, total_count = 0, 0

    if conv_type == 'global' : 

        for node_idx in loader:
            batch_size = len(node_idx)

            feat = x[node_idx] if torch.is_tensor(x) else x(node_idx)
            out = model.to(device).global_forward(feat.to(device), pos_enc[node_idx].to(device), node_idx).squeeze_()

            total_correct += (out>0).int().cpu().eq(y[node_idx]).sum().item()
            total_count += batch_size

            counter += 1
            if fast_eval and counter >= len(loader)//10 :
                break

    else :

        for edge_index, node_idx, batch_size in loader:

            # print('node size:', len(node_idx))
            # print('edge size:', edge_index.shape[1])

            edge_index, edge_dist = edge_index[:2], edge_index[2:]
            out = model.to(device)(x[node_idx].to(device), edge_index.to(device), edge_dist.to(device), pos_enc[node_idx[:batch_size]].to(device), node_idx[:batch_size]).squeeze_()

            total_correct += (out>0).int().cpu().eq(y[node_idx[:batch_size]]).sum().item()
            total_count += batch_size

            # print(f'Test Progress: {counter}/{len(loader)}:{100*total_correct/total_count:.2f}%, Test Mem:{torch.cuda.max_memory_allocated(device=device)/1e6} MB')
            
            # if counter >= 2 :
                # break

            counter += 1
            if fast_eval and counter >= len(loader)//10 :
                break

    # print(f'Test Mem:{torch.cuda.max_memory_allocated(device=device)/1e6} MB')
    return total_correct/total_count


def main():
    parser = argparse.ArgumentParser(description='large')

    # data loading
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv', choices=['ogbn-arxiv', 'ogbn-products', 'arxiv-year', 'twitch-gamers', 'pokec', 'genius', 'snap-patents']) 
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--linkx_data_root', type=str, default='linkx_data')
    parser.add_argument('--data_downloading_flag', action='store_true')

    # training
    parser.add_argument('--hetero_train_prop', type=float, default=0.5)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3) 
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1024)  #
    parser.add_argument('--test_batch_size', type=int, default=1024) #
    parser.add_argument('--sizes', type=int, nargs='+', default=[20,10,5]) 
    parser.add_argument('--test_sizes', type=int, nargs='+', default=[20,10,5]) 
    parser.add_argument('--test_freq', type=int, default=1)  #
    parser.add_argument('--num_workers', type=int, default=4) #

    # NN 
    parser.add_argument('--conv_type', type=str, default='local', choices=['local', 'global', 'full'])
    parser.add_argument('--hidden_dim', type=int, default=256) 
    parser.add_argument('--global_dim', type=int, default=64) 
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=1) 
    parser.add_argument('--attn_dropout', type=float, default=0)
    parser.add_argument('--ff_dropout', type=float, default=0.5)
    parser.add_argument('--skip', type=int, default=0)  
    parser.add_argument('--dist_count_norm', type=int, default=1)  
    parser.add_argument('--num_centroids', type=int, default=4096) 
    parser.add_argument('--no_bn', action='store_true')
    parser.add_argument('--norm_type', type=str, default='batch_norm')

    # eval
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--save_ckpt', action='store_true')

    args = parser.parse_args()

    print(args)

    # convert int to boolean:
    args.skip = args.skip > 0
    args.dist_count_norm = args.dist_count_norm > 0

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.eval :
        ckpt = torch.load(f'checkpoints/ckpt_epoch{args.eval_epoch}.pt', map_location=device)        

    data_root, linkx_data_root = args.data_root, args.linkx_data_root

    # snap-patents and arxiv-year DIRECTED! others undirected, CRITICAL!
    if args.dataset.startswith('ogbn') :
        dataset = PygNodePropPredDataset(name=args.dataset, root=data_root)
        num_classes = dataset.num_classes
        data = dataset[0]
        if args.dataset == 'ogbn-arxiv' :
            data.edge_index = to_undirected(data.edge_index, data.num_nodes)
        split_idx = dataset.get_idx_split()
        x = data.x
        y = data.y.squeeze()

        # Convert split indices to boolean masks and add them to `data`.
        for key, idx in split_idx.items():
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            data[f'{key}_mask'] = mask

        assert args.batch_size <= len(split_idx['train'])

    elif args.dataset == 'arxiv-year' :
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=data_root)
        num_classes = 5
        
        data = dataset[0]
        # data.edge_index = to_undirected(data.edge_index, data.num_nodes)
        x = data.x

        label = even_quantile_labels(data.node_year.numpy().flatten(), nclasses=num_classes, verbose=False)
        y = torch.as_tensor(label)

        if args.eval :
            split_idx = ckpt['split_idx']
        else :
            train_idx, valid_idx, test_idx = rand_train_test_idx(y, train_prop=args.hetero_train_prop)
            split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

        assert args.batch_size <= len(split_idx['train'])

        # Convert split indices to boolean masks and add them to `data`.
        for key, idx in split_idx.items():
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            data[f'{key}_mask'] = mask

    elif args.dataset == 'pokec' :
        if not path.exists(f'{linkx_data_root}/pokec.mat'):
            gdd.download_file_from_google_drive(
                file_id=dataset_drive_url['pokec'],
                dest_path=f'{linkx_data_root}/pokec.mat', 
                showsize=True
            )

        fulldata = scipy.io.loadmat(f'{linkx_data_root}/pokec.mat')

        edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
        node_feat = torch.tensor(fulldata['node_feat']).float()
        num_nodes = int(fulldata['num_nodes'])

        class MyObject:
            pass
        data = MyObject()
        x = data.x = node_feat
        y = data.y = torch.tensor(fulldata['label'].flatten(), dtype=torch.long)
        data.num_features = data.x.shape[-1]
        data.edge_index = edge_index
        data.num_nodes = num_nodes

        #################################################################
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)

        num_classes = 1
        train_idx, valid_idx, test_idx = rand_train_test_idx(y, train_prop=args.hetero_train_prop)
        split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    elif args.dataset == 'genius' :
        
        fulldata = scipy.io.loadmat(f'{linkx_data_root}/genius.mat')

        edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
        node_feat = torch.tensor(fulldata['node_feat'], dtype=torch.float)
        label = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()
        num_nodes = label.shape[0]

        class MyObject:
            pass
        data = MyObject()
        x = data.x = node_feat
        y = data.y = label
        data.num_features = data.x.shape[-1]
        data.edge_index = edge_index
        data.num_nodes = num_nodes

        #################################################################
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)

        num_classes = 1
        train_idx, valid_idx, test_idx = rand_train_test_idx(y, train_prop=args.hetero_train_prop)
        split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    elif args.dataset == 'snap-patents' :

        num_classes = 5
        fulldata = scipy.io.loadmat(f'{linkx_data_root}/snap_patents.mat')
        edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
        
        num_nodes = int(fulldata['num_nodes'])
        node_feat = torch.tensor(fulldata['node_feat'].todense(), dtype=torch.float)

        years = fulldata['years'].flatten()
        label = even_quantile_labels(years, num_classes, verbose=False)
        label = torch.tensor(label, dtype=torch.long)

        class MyObject:
            pass
        data = MyObject()
        x = data.x = node_feat
        y = data.y = label
        data.num_features = data.x.shape[-1]

        data.edge_index = edge_index
        data.num_nodes = num_nodes

        #################################################################
        # data.edge_index = to_undirected(data.edge_index, data.num_nodes)

        train_idx, valid_idx, test_idx = rand_train_test_idx(y, train_prop=args.hetero_train_prop)
        split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    elif args.dataset == 'twitch-gamers' :
        if not path.exists(f'{linkx_data_root}/twitch-gamer_feat.csv'):
            gdd.download_file_from_google_drive(
                file_id=dataset_drive_url['twitch-gamer_feat'],
                dest_path=f'{linkx_data_root}/twitch-gamer_feat.csv', 
                showsize=True
            )
        if not path.exists(f'{linkx_data_root}/twitch-gamer_edges.csv'):
            gdd.download_file_from_google_drive(
                file_id=dataset_drive_url['twitch-gamer_edges'],
                dest_path=f'{linkx_data_root}/twitch-gamer_edges.csv', 
                showsize=True
            )
        
        edges = pd.read_csv(f'{linkx_data_root}/twitch-gamer_edges.csv')
        nodes = pd.read_csv(f'{linkx_data_root}/twitch-gamer_feat.csv')
        edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
        num_nodes = len(nodes)

        label, features = load_twitch_gamer(nodes, "mature")
        node_feat = torch.tensor(features, dtype=torch.float)
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)
        
        class MyObject:
            pass
        data = MyObject()
        x = data.x = node_feat
        y = data.y = torch.tensor(label)
        data.num_features = data.x.shape[-1]
        data.edge_index = edge_index
        data.num_nodes = num_nodes

        #################################################################
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)
        
        num_classes = 1
        train_idx, valid_idx, test_idx = rand_train_test_idx(y, train_prop=args.hetero_train_prop)
        split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    if args.data_downloading_flag :
        print(f'Dataset {args.dataset} successfully downloaded.')
        exit(0)

    assert len(args.sizes) == len(args.test_sizes)
    model = Transformer(
        num_nodes=data.num_nodes,
        in_channels=data.num_features,
        hidden_channels=args.hidden_dim, 
        out_channels=num_classes,
        global_dim=args.global_dim,
        num_layers=args.num_layers,
        heads=args.num_heads,
        ff_dropout=args.ff_dropout,
        attn_dropout=args.attn_dropout,
        spatial_size=len(args.sizes),
        skip=args.skip,
        dist_count_norm=args.dist_count_norm,
        conv_type=args.conv_type,
        num_centroids=args.num_centroids,
        no_bn=args.no_bn,
        norm_type=args.norm_type
    )

    if args.conv_type == 'local' :
        pos_enc = x
    else :
        if args.dataset == 'arxiv-year' :
            dataset_name_input = 'ogbn-arxiv'
        else :
            dataset_name_input = args.dataset

        pos_enc = torch.load(f'pos_enc/{dataset_name_input}_embedding_{args.global_dim}.pt', map_location='cpu')

    if args.eval :
        valid_loader = LocalSampler(data.edge_index, node_idx=split_idx['valid'], 
                                    num_nodes=data.num_nodes, sizes=args.test_sizes, 
                                    batch_size=args.batch_size, shuffle=False, 
                                    num_workers=args.num_workers)
        test_loader = LocalSampler(data.edge_index, node_idx=split_idx['test'], 
                                    num_nodes=data.num_nodes, sizes=args.test_sizes, 
                                    batch_size=args.batch_size, shuffle=False, 
                                    num_workers=args.num_workers)

        model.load_state_dict(ckpt['model'])
        # valid_acc = test(model, valid_loader, x, y, device)
        test_acc = test(model, test_loader, x, y, device)
        print(f'Valid acc:{0}, Test acc:{test_acc}')

    else :
        if args.conv_type == 'global' :
            train_loader = torch.utils.data.DataLoader(split_idx['train'], batch_size=args.batch_size, 
                                                        shuffle=True, num_workers=args.num_workers)
            valid_loader = torch.utils.data.DataLoader(split_idx['valid'], batch_size=args.test_batch_size, 
                                                        shuffle=False, num_workers=args.num_workers)
            test_loader = torch.utils.data.DataLoader(split_idx['test'], batch_size=args.test_batch_size, 
                                                        shuffle=False, num_workers=args.num_workers)
        else :
            train_loader = LocalSampler(data.edge_index, node_idx=split_idx['train'], 
                                        num_nodes=data.num_nodes, sizes=args.sizes, 
                                        batch_size=args.batch_size, shuffle=True, 
                                        num_workers=args.num_workers)
            valid_loader = LocalSampler(data.edge_index, node_idx=split_idx['valid'], 
                                        num_nodes=data.num_nodes, sizes=args.test_sizes, 
                                        batch_size=args.test_batch_size, shuffle=False, 
                                        num_workers=args.num_workers)
            test_loader = LocalSampler(data.edge_index, node_idx=split_idx['test'], 
                                        num_nodes=data.num_nodes, sizes=args.test_sizes, 
                                        batch_size=args.test_batch_size, shuffle=False, 
                                        num_workers=args.num_workers)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if args.dataset in ['pokec', 'twitch-gamers', 'genius'] :
            train_f, test_f = train_bi, test_bi
        else :
            train_f, test_f = train, test

        if args.dataset == 'ogbn-products' :
            test_start_epoch = 200
            test_start_epoch = 0

            if args.save_ckpt :
                save_path = f'checkpoint/{time.time()}'
                os.mkdir(save_path)
        else :
            test_start_epoch = 0

        valid_acc_final, test_acc_final, test_acc_highest = 0, 0, 0

        whole_start = time.time()
        for epoch in range(1, 1 + args.epochs):

            # train_loss, train_acc, valid_acc, test_acc = 0, 0, 0, 0
            start = time.time()

            train_loss, train_acc = train_f(model, train_loader, x, pos_enc, y, optimizer, device, args.conv_type)
            train_time = time.time() - start
            print(f'Epoch: {epoch}, Train loss:{train_loss:.4f}, Train acc:{100*train_acc:.2f}, Epoch time: {train_time:.4f}, Train Mem:{torch.cuda.max_memory_allocated(device=device)/1e6:.0f} MB')

            if epoch > test_start_epoch and epoch % args.test_freq == 0 :
                if args.save_ckpt :
                    ckpt = {}
                    ckpt['model'] = model.state_dict()
                    if args.dataset == 'arxiv-year' :
                        ckpt['split_idx'] = split_idx
                    torch.save(ckpt, f'{save_path}/{args.dataset}_ckpt_epoch{epoch}.pt')
                    # ckpt = model.load_state_dict(torch.load('model.pt'))

                else :
                    start = time.time()
                    valid_acc = test_f(model, valid_loader, x, pos_enc, y, device, args.conv_type, False)

                    if args.dataset == 'ogbn-products' and  valid_acc < 0.0 :
                        pass
                    else :
                        fast_eval_flag = (args.dataset == 'ogbn-products')
                        fast_eval_flag = False

                        test_acc = test_f(model, test_loader, x, pos_enc, y, device, args.conv_type, fast_eval_flag)
                        test_time = time.time() - start
                        print(f'Test acc: {100 * test_acc:.2f}, Val+Test time used: {test_time:.4f}')

                        if valid_acc > valid_acc_final :
                            valid_acc_final = valid_acc
                            test_acc_final = test_acc
                        if test_acc > test_acc_highest :
                            test_acc_highest = test_acc


if __name__ == "__main__":
    main()
