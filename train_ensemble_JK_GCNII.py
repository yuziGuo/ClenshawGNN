from data.citation_dataloader import  citation_loader
from data.citation_full_dataloader import  citation_full_supervised_loader
from data.geom_dataloader import geom_dataloader
from data.linkx_dataloader import linkx_dataloader

from models.GCNJK import GCNJK
from models.GCNII import GCNII

from utils.grading_logger import get_logger
from utils.stopper import EarlyStopping
import argparse
import random
import time 

from torch_geometric.nn.conv.gcn_conv import gcn_norm

import numpy as np
import torch as th
import torch.nn.functional as F

def build_dataset(args):
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        loader = citation_loader(args.dataset, args.gpu, args.self_loop)
    elif args.dataset in ['corafull', 'citeseerfull', 'pubmedfull']:
        loader = citation_full_supervised_loader(args.dataset, args.gpu, args.self_loop, n_cv=args.n_cv)
    elif args.dataset.startswith('geom'):
        dataset = args.dataset.split('-')[1]
        loader = geom_dataloader(dataset, args.gpu, args.self_loop, digraph=not args.udgraph, n_cv=args.n_cv, cv_id=args.start_cv)
    elif args.dataset in ['twitch-gamer']:
        loader = linkx_dataloader(args.dataset, args.gpu, args.self_loop, n_cv=args.n_cv)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    loader.load_data()
    return loader

def build_model_GCNJK(args, edge_index, norm_A, in_feats, n_classes):
    model = GCNJK(
        edge_index,
        norm_A,
        in_feats,
        args.n_hidden,
        n_classes,
        args.n_layers_jk,
        args.dropout,
        jk_type=args.jk_type
        )
    model.to(args.gpu)
    return model


def build_model_GCNII(args, edge_index, norm_A, in_feats, n_classes):
    model = GCNII(
        edge_index,
        norm_A,
        in_feats,
        args.n_hidden,
        n_classes,
        args.n_layers_gcnii,
        args.dropout,
        args.alpha,
        args.lamda
        )
    model.to(args.gpu)
    return model


class META(th.nn.Module):
    def __init__(self, n_classes,dropout) -> None:
        super(META, self).__init__()
        self.fc = th.nn.Linear(n_classes * 2, n_classes)
        self.act_fc = F.relu
        self.dropout = dropout

    def forward(self, x_jk_gcnii):
        x = th.cat([x_jk_gcnii], dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x)
    

def build_model_META(args, n_classes):
    model = META(n_classes, args.dropout2)
    model.to(args.gpu)
    return model
    

def build_optimizer(args, model, model_name):
    if model_name == 'jk':
        param_groups = [{'params':model.parameters(), 'lr':args.lr1,'weight_decay':args.wd1}]
    elif model_name == 'gcnii':
        param_groups = [{'params':model.parameters(), 'lr':args.lr2,'weight_decay':args.wd2}]
    elif model_name == 'meta':
        param_groups = [{'params':model.parameters(), 'lr':args.lr3,'weight_decay':args.wd3}]
    else:
        assert False
    optimizer = th.optim.Adam(param_groups)
    return optimizer

def build_stopper(args):
    stopper = EarlyStopping(patience=args.patience, store_path=args.es_ckpt+'.pt')
    return stopper.step, stopper


def evaluate(model, loss_fcn, features, labels, mask, epoch, evaluator=None):
    model.eval()
    with th.no_grad():
        logits = model(features)
        if not th.is_tensor(logits):
            logits = logits[0]
        logits = logits[mask]
        labels = labels[mask]
        loss = loss_fcn(logits, labels)

        if evaluator is not None:
            acc = evaluator.eval({"y_pred": logits.argmax(dim=-1, keepdim=True),
                                  "y_true": labels})["acc"]
            return acc, loss
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        acc = correct.item() * 1.0 / len(labels)
        return acc, loss


def split_level01_nodes(data, level1_data_prop=0.2):
    # Set aside a portion for trainning the meta model (to ensemble two models)
    n_train_nodes = data.train_mask.sum()
    n_train_nodes_level1 = int(n_train_nodes * level1_data_prop)
    train_nodes_idx = data.train_mask.nonzero()
    train_nodes_idx_level1 = train_nodes_idx[:n_train_nodes_level1]
    train_nodes_idx_level0 = train_nodes_idx[n_train_nodes_level1+1:]

    data.train_mask_level1 = th.zeros_like(data.train_mask)
    data.train_mask_level1[train_nodes_idx_level1] = True
    data.train_mask_level0 = th.zeros_like(data.train_mask)
    data.train_mask_level0[train_nodes_idx_level0] = True

    # It is better to also set aside a portion of validation nodes 
    n_val_nodes = data.val_mask.sum()
    n_val_nodes_level1 = int(n_val_nodes * level1_data_prop)
    val_nodes_idx = data.val_mask.nonzero()
    val_nodes_idx_level1 = val_nodes_idx[:n_val_nodes_level1]
    val_nodes_idx_level0 = val_nodes_idx[n_val_nodes_level1+1:]

    data.val_mask_level1 = th.zeros_like(data.val_mask)
    data.val_mask_level1[val_nodes_idx_level1] = True
    data.val_mask_level0 = th.zeros_like(data.val_mask)
    data.val_mask_level0[val_nodes_idx_level0] = True


def train_level0_jk(edge_index, data, norm_A, features, labels, model_seed):
    dur = []
    # 1. train_model_jk
    reset_random_seeds(model_seed)
    loss_fcn = th.nn.NLLLoss()
    data.in_feats = features.shape[-1] 
    model_jk = build_model_GCNJK(args, edge_index, norm_A, data.in_feats, data.n_classes)
    optimizer = build_optimizer(args, model_jk, 'jk')
    stopper_step_jk, stopper_jk = build_stopper(args)
    rec_val_loss = []
    rec_val_accs = []
    for epoch in range(args.n_epochs): 
        t0 = time.time()
        model_jk.train()
        optimizer.zero_grad()
        logits = model_jk(features)
        loss = loss_fcn(logits[data.train_mask_level0], labels[data.train_mask_level0])
        loss.backward()
        optimizer.step()

        train_acc, train_loss = evaluate(model_jk, loss_fcn, features, labels, data.train_mask_level0, epoch, evaluator=None)
        val_acc, val_loss = evaluate(model_jk, loss_fcn, features, labels, data.val_mask_level0, epoch, evaluator=None)
        rec_val_loss.append(val_loss.item())
        rec_val_accs.append(val_acc)
        dur.append(time.time() - t0)
        if args.log_detail and (epoch+1) % 50 == 0 :
            logger.info("Epoch {:05d} | Time(s) {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} |  Train Acc {:.4f} | "
                        "ETputs(KTEPS) {:.2f}". format(epoch+1, np.mean(dur), val_loss.item(),
                                                        val_acc, train_acc, 
                                                        data.n_edges / np.mean(dur) / 100)
                        )
        if args.early_stop and epoch >= 0:
            if stopper_step_jk(val_loss, model_jk):
                break   
    model_jk.load_state_dict(th.load(stopper_jk.store_path))
    logger.debug('Model Saved by Early Stopper is Loaded!')
    return model_jk


def train_level0_gcnii(edge_index, data, norm_A, features, labels, model_seed):
    dur = []
    reset_random_seeds(model_seed)
    loss_fcn = th.nn.NLLLoss()
    data.in_feats = features.shape[-1] 
    model_gcnii = build_model_GCNII(args, edge_index, norm_A, data.in_feats, data.n_classes)
    optimizer = build_optimizer(args, model_gcnii, 'gcnii')
    stopper_step_gcnii, stopper_gcnii = build_stopper(args)
    rec_val_loss = []
    rec_val_accs = []
    for epoch in range(args.n_epochs): 
        t0 = time.time()
        model_gcnii.train()
        optimizer.zero_grad()
        logits = model_gcnii(features)
        loss = loss_fcn(logits[data.train_mask_level0], labels[data.train_mask_level0])
        loss.backward()
        optimizer.step()

        train_acc, train_loss = evaluate(model_gcnii, loss_fcn, features, labels, data.train_mask_level0, epoch, evaluator=None)
        val_acc, val_loss = evaluate(model_gcnii, loss_fcn, features, labels, data.val_mask_level0, epoch, evaluator=None)
        rec_val_loss.append(val_loss.item())
        rec_val_accs.append(val_acc)
        dur.append(time.time() - t0)
        if args.log_detail and (epoch+1) % 50 == 0 :
            logger.info("Epoch {:05d} | Time(s) {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} |  Train Acc {:.4f} | "
                        "ETputs(KTEPS) {:.2f}". format(epoch+1, np.mean(dur), val_loss.item(),
                                                        val_acc, train_acc, 
                                                        data.n_edges / np.mean(dur) / 100)
                        )
        if args.early_stop and epoch >= 0:
            if stopper_step_gcnii(val_loss, model_gcnii):
                break   
    model_gcnii.load_state_dict(th.load(stopper_gcnii.store_path))
    logger.debug('Model Saved by Early Stopper is Loaded!')
    return model_gcnii

def train_level1_model(data, cv_id, features, labels, model_seed):
    dur = []
    reset_random_seeds(model_seed)
    loss_fcn = th.nn.NLLLoss()
    model_meta = build_model_META(args, data.n_classes)
    optimizer = build_optimizer(args, model_meta, 'meta')
    stopper_step_meta, stopper_meta = build_stopper(args)
    rec_val_loss = []
    rec_val_accs = []
    for epoch in range(args.n_epochs): 
        t0 = time.time()
        model_meta.train()
        optimizer.zero_grad()
        logits = model_meta(features)
        loss = loss_fcn(logits[data.train_mask_level1], labels[data.train_mask_level1])
        loss.backward()
        optimizer.step()

        train_acc, train_loss = evaluate(model_meta, loss_fcn, features, labels, data.train_mask_level1, epoch, evaluator=None)
        val_acc, val_loss = evaluate(model_meta, loss_fcn, features, labels, data.val_mask_level1, epoch, evaluator=None)
        rec_val_loss.append(val_loss.item())
        rec_val_accs.append(val_acc)
        dur.append(time.time() - t0)
        if args.log_detail and (epoch+1) % 50 == 0 :
            logger.info("Epoch {:05d} | Time(s) {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} |  Train Acc {:.4f} | "
                        "ETputs(KTEPS) {:.2f}". format(epoch+1, np.mean(dur), val_loss.item(),
                                                        val_acc, train_acc, 
                                                        data.n_edges / np.mean(dur) / 100)
                        )
        if args.early_stop and epoch >= 0:
            if stopper_step_meta(val_loss, model_meta):
                break   
    model_meta.load_state_dict(th.load(stopper_meta.store_path))
    logger.debug('Model Saved by Early Stopper is Loaded!')

    val_acc, val_loss = evaluate(model_meta, loss_fcn, features, labels, data.val_mask, epoch, evaluator=None)
    logger.info("[FINAL MODEL] Run {} .\Val accuracy {:.2%} \Val loss: {:.2}".format(cv_id+args.start_cv, val_acc, val_loss))
    test_acc, test_loss = evaluate(model_meta, loss_fcn, features, labels, data.test_mask, epoch, evaluator=None)
    logger.info("[FINAL MODEL] Run {} .\tTest accuracy {:.2%} \Test loss: {:.2}".format(cv_id+args.start_cv, test_acc, test_loss))
    return model_meta, val_acc, test_acc

def run(args, cv_id, edge_index, data, norm_A, features, labels, model_seed):    
    if args.dataset in ['twitch-gamer']: # encouraged to use fixed splits
        data.load_mask()
    else:
        data.load_mask(p=(0.6,0.2,0.2))

    split_level01_nodes(data)
    # -> data.val_mask_level0 / data.val_mask_level1 / data.train_mask_level0 ...

    model_jk = train_level0_jk(edge_index, data, norm_A, features, labels, model_seed)
    output_jk = model_jk.predict(features)

    model_gcnii = train_level0_gcnii(edge_index, data, norm_A, features, labels, model_seed)
    output_gcnii = model_gcnii.predict(features)

    features = th.cat([output_jk, output_gcnii], dim=1)
    model, val_acc, test_acc = train_level1_model(data, cv_id, features, labels, model_seed)
    return model, val_acc, test_acc
    

def main(args):
    reset_random_seeds(args.seed)
    data  = build_dataset(args) 
    data.seeds = [random.randint(0,10000) for i in range(args.n_cv)]
    model_seeds = [random.randint(0,10000) for i in range(args.n_cv)]
    logger.info('Split_seeds:{:s}'.format(str(data.seeds)))
    logger.info('Model_seeds:{:s}'.format(str(model_seeds)))

    edge_index = data.edge_index
    _, norm_A = gcn_norm(edge_index, add_self_loops=False)
    features = data.features
    labels = data.labels
    
    accs = []
    val_accs = []
    
    for cv_id in range(args.n_cv):
        model, val_acc, test_acc = run(args, cv_id, edge_index, data, norm_A,  features, labels, model_seed=model_seeds[cv_id])
        accs.append(test_acc)
        val_accs.append(val_acc)
    
    logger.info("Mean Acc For Cross Validation: {:.4f}, STDV: {:.4f}".format(np.array(accs).mean(), np.array(accs).std()))
    logger.info(accs)
    
def set_args():
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name ('cora', 'citeseer', 'pubmed').")

    parser.add_argument("--n-hidden", type=int, default=64, help="number of hidden gcn units")
    parser.add_argument("--n-layers-gcnii", type=int, default=4, help="number of hidden gcnii layers")
    parser.add_argument("--n-layers-jk", type=int, default=4, help="number of hidden jk-gcn layers")

    # for training
    parser.add_argument("--wd1", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--wd2", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--wd3", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--lr1",  type=float, default=1e-2, help="learning rate")
    parser.add_argument("--lr2",  type=float, default=1e-2, help="learning rate")
    parser.add_argument("--lr3",  type=float, default=1e-2, help="learning rate")
    parser.add_argument("--dropout",  type=float, default=0.6, help="learning rate")
    parser.add_argument("--dropout2",  type=float, default=0.6, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=2000, help="number of training epochs")

    parser.add_argument("--loss", type=str, default='nll')
    parser.add_argument("--self-loop", action='store_true', default=False, help="graph self-loop (default=False)")
    parser.add_argument("--udgraph", action='store_true', default=False, help="undirected graph (default=False)")
        
    # for gcnii
    parser.add_argument("--alpha", type=float, default=0.5, help="GCNII alpha")
    parser.add_argument("--lamda", type=float, default=1.0, help="GCNII lambda")
    # for jknet+gcn
    parser.add_argument("--jk-type", type=str, default='max')

    # for experiment running
    parser.add_argument("--early-stop", action='store_true', default=False, help="early stop (default=False)")
    parser.add_argument("--patience", type=int, default=300, help="patience for early stop")
    parser.add_argument("--es-ckpt", type=str, default="es_checkpoint", help="EarlyStopper checkpoint. Saving directory for early stop checkpoint")
    parser.add_argument("--n-cv", type=int, default=1)
    parser.add_argument("--start-cv", type=int, default=0)
    parser.add_argument("--logging", action='store_true', default=False, help="log results and details to files (default=False)")
    parser.add_argument("--log-detail", action='store_true', default=False)
    parser.add_argument("--log-detailedCh", action='store_true', default=False)
    parser.add_argument("--id-log", type=int, default=0)

    args = parser.parse_args()

    if args.gpu < 0:
        args.gpu = 'cpu'

    if args.es_ckpt == 'es_checkpoint':
        args.es_ckpt = '_'.join([args.es_ckpt, 'device='+str(args.gpu)])
    return args


def reset_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed) 


def set_logger(args):
    if args.id_log > 0:
        log_d = 'runs/Logs'+str(args.id_log)
        logger = get_logger(file_mode=args.logging, dir_name=log_d)
    else:
        logger = get_logger(file_mode=args.logging, detailedConsoleHandler=args.log_detailedCh)
    return logger


if __name__=='__main__':
    args = set_args()
    logger = set_logger(args)
    logger.info(args)
    main(args)