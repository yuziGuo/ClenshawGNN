from ast import Raise
import optuna
from optuna.trial import TrialState
import optuna.study
import matplotlib.pyplot as plt
from data.citation_dataloader import  citation_loader
from data.citation_full_dataloader import  citation_full_supervised_loader
from data.amazon_dataloader import amazon_full_supervised_loader
from data.geom_dataloader import geom_dataloader
from data.linkx_dataloader import linkx_dataloader

from models.GCNIIJK import GCNIIJK

import logging
from utils.grading_logger import get_logger
from utils.stopper import EarlyStopping
import argparse
import random
import time 

import numpy as np
import os

import torch as th
import torch.nn.functional as F


def build_dataset(args):
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        loader = citation_loader(args.dataset, args.gpu, args.self_loop)
    elif args.dataset in ['corafull', 'citeseerfull', 'pubmedfull']:
        loader = citation_full_supervised_loader(args.dataset, args.gpu, args.self_loop, n_cv=args.n_cv)
    elif args.dataset in ['computersfull', 'photofull']:
        loader = amazon_full_supervised_loader(args.dataset, args.gpu, args.self_loop, n_cv=args.n_cv)
    elif args.dataset.startswith('geom'):
        dataset = args.dataset.split('-')[1]
        loader = geom_dataloader(dataset, args.gpu, args.self_loop, digraph=not args.udgraph, n_cv=args.n_cv, cv_id=args.start_cv)
    elif args.dataset in ['twitch-gamer', 'genius']:
        loader = linkx_dataloader(args.dataset, args.gpu, args.self_loop, n_cv=args.n_cv)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    loader.load_data()
    return loader


def build_model(args, edge_index, norm_A, in_feats, n_classes):
    if args.model == 'GCNIIJK':
        model = GCNIIJK(
                    edge_index,
                    norm_A,
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers,
                    args.dropout,
                    args.dropout2,
                    args.alpha,
                    args.lamda,
                    jk_type=args.jk_type
                    )
        model.to(args.gpu)
        return model


def build_optimizer(args, model):
    param_groups = [
        {'params':model.params1, 'lr':args.lr1,'weight_decay':args.wd1}, # convs
        {'params':model.params2, 'lr':args.lr2,'weight_decay':args.wd2}, # fcs
        # {'params':model.params3, 'lr':args.lr3,'weight_decay':args.wd3}, # bns
    ]
    optimizer = th.optim.Adam(param_groups)
    return optimizer



def build_stopper(args):
    stopper = EarlyStopping(patience=args.patience, store_path=args.es_ckpt+'.pt')
    step = stopper.step
    return step, stopper

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


def run(args, logger, trial, cv_id, edge_index, data, norm_A, features, labels, model_seed):
    dur = []

    if args.dataset in ['twitch-gamer', 'genius', 'Penn94']: # encouraged to use fixed splits
        data.load_mask()
    else:
        data.load_mask(p=(0.6,0.2,0.2))
    
    reset_random_seeds(model_seed)
    loss_fcn = th.nn.NLLLoss()

    data.in_feats = features.shape[-1] # ----
    model = build_model(args, edge_index, norm_A, data.in_feats, data.n_classes)
    optimizer = build_optimizer(args, model)
    stopper_step, stopper = build_stopper(args)
    
    rec_val_loss = []
    rec_val_accs = []

    for epoch in range(args.n_epochs): 
        t0 = time.time()
        
        model.train()
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fcn(logits[data.train_mask], labels[data.train_mask])
        loss.backward()

        optimizer.step()

        val_acc, val_loss = evaluate(model, loss_fcn, features, labels, data.val_mask, epoch, evaluator=None)

        trial.report(val_acc, cv_id) # TODO: fix when val_acc is not a list
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        rec_val_loss.append(val_loss.item())
        rec_val_accs.append(val_acc)
        dur.append(time.time() - t0)
        if args.log_detail and (epoch+1) % 50 == 0 :
            logger.info("Epoch {:05d} | Time(s) {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | "
                        "ETputs(KTEPS) {:.2f}". format(epoch+1, np.mean(dur), val_loss.item(),
                                                        val_acc, 
                                                        data.n_edges / np.mean(dur) / 100)
                        )
        if args.early_stop and epoch >= 0:
            if stopper_step(val_loss, model):
                break   
        
    # end for


    if args.early_stop:
        model.load_state_dict(th.load(stopper.store_path))
        logger.debug('Model Saved by Early Stopper is Loaded!')
    val_acc, val_loss = evaluate(model, loss_fcn, features, labels, data.val_mask, epoch, evaluator=None)
    logger.info("[FINAL MODEL] Run {} .\Val accuracy {:.2%} \Val loss: {:.2}".format(cv_id+args.start_cv, val_acc, val_loss))
    test_acc, test_loss = evaluate(model, loss_fcn, features, labels, data.test_mask, epoch, evaluator=None)
    logger.info("[FINAL MODEL] Run {} .\tTest accuracy {:.2%} \Test loss: {:.2}".format(cv_id+args.start_cv, test_acc, test_loss))
    return model, val_acc, test_acc
    

def main(args, logger, trial):
    reset_random_seeds(args.seed)
    data  = build_dataset(args)
    data.seeds = [random.randint(0,10000) for i in range(args.n_cv)]
    model_seeds = [random.randint(0,10000) for i in range(args.n_cv)]
    logger.info('Split_seeds:{:s}'.format(str(data.seeds)))
    logger.info('Model_seeds:{:s}'.format(str(model_seeds)))

    edge_index = data.edge_index
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    # Always set add_self_loops=False here!
    # self-loops are set in loader before.
    _, norm_A = gcn_norm(edge_index, add_self_loops=False)
    features = data.features
    labels = data.labels

    test_accs = []
    val_accs = []
    
    cv_id = 0
    model, val_acc, test_acc = run(args, logger, trial, cv_id, edge_index, data, norm_A,  features, labels, model_seed=model_seeds[cv_id])
    test_accs.append(test_acc)
    val_accs.append(val_acc)

    
    logger.info("Mean Acc For Cross Validation (Validation Set): {:.4f}, STDV: {:.4f}".format(np.array(val_accs).mean(), np.array(val_accs).std()))
    logger.info(val_accs)
    logger.info("Mean Acc For Cross Validation (Test Set): {:.4f}, STDV: {:.4f}".format(np.array(test_accs).mean(), np.array(test_accs).std()))
    logger.info(test_accs)
    return val_accs, test_accs


# https://github.com/optuna/optuna/issues/862
def set_args():
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--split-seed', type=str, default='random', help='(predefined, random)')
    parser.add_argument("--model", type=str, default='GCNIIJK')
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument("--ds-split", type=str, default="standard", help="split by ('standard', 'mg', 'random').")

    parser.add_argument("--optuna-n-trials", type=int, default=100)
    parser.add_argument("--kw", type=str)


    parser.add_argument("--n-hidden", type=int, default=64, help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=4, help="number of hidden gcn layers")

    # for training
    parser.add_argument("--wd1", type=float, default=1e-2, help="Weight for L2 loss")
    parser.add_argument("--wd2", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--lr1",  type=float, default=1e-2, help="learning rate")
    parser.add_argument("--lr2",  type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=2000, help="number of training epochs")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--dropout2", type=float, default=0.7, help="dropout probability")

    # for gcnii
    parser.add_argument("--alpha", type=float, default=0.5, help="GCNII alpha")
    parser.add_argument("--lamda", type=float, default=1.0, help="GCNII lambda")
    # for jknet
    parser.add_argument("--jk-type", type=str, default='max')

    parser.add_argument("--loss", type=str, default='nll')
    parser.add_argument("--self-loop", action='store_true', default=False, help="graph self-loop (default=False)")
    parser.add_argument("--udgraph", action='store_true', default=False, help="undirected graph (default=False)")
    parser.add_argument("--l-norm", type=str, default='sym')

    # for experiment running
    parser.add_argument("--early-stop", action='store_true', default=False, help="early stop (default=False)")
    parser.add_argument("--patience", type=int, default=300, help="patience for early stop")
    parser.add_argument("--es-ckpt", type=str, default="es_checkpoint", help="Saving directory for early stop checkpoint")
    parser.add_argument("--n-cv", type=int, default=1)
    parser.add_argument("--start-cv", type=int, default=0)
    parser.add_argument("--logging", action='store_true', default=False, help="log results and details to files (default=False)")
    parser.add_argument("--log-detail", action='store_true', default=False)
    parser.add_argument("--id-log", type=int, default=0)

    args = parser.parse_args()

    if args.gpu < 0:
        args.gpu = 'cpu'
    return args


def reset_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)  # 其实现在不需要这一句了


def set_logger(args):
    if args.id_log > 0:
        log_d = 'runs/Logs'+str(args.id_log)
        logger = get_logger(file_mode=args.logging, dir_name=log_d)
    else:
        logger = get_logger(file_mode=args.logging)
    return logger


def update_args_(args, params):
  """updates args in-place"""
  dargs = vars(args)
  dargs.update(params)

def np_arg_max_last_occurrence(val_acc_means):
    _ = val_acc_means[::-1]
    i = len(_) - np.argmax(_) - 1 
    return i

def get_es_ckpt_fname(study, trial):
    return  '_'.join([study.study_name, 'trialNo{}'.format(trial.number)])
    

def suggest_args(trial):
    args = set_args()
    # args <--> suggest number
    if trial is not None:
        params = {
            'early_stop': True,
            'udgraph': True,
            'self_loop': True,
            'loss': 'nll',
            'logging': True,
            'log_detail': True,
            # 'log_detailedCh': True,

            'lr1': trial.suggest_float('lr1', -0.02, 0.05, step=0.01), # -0.01 --> 0.001  0.-->0.005
            'lr2': trial.suggest_float('lr2', -0.02, 0.05, step=0.01),
            'wd1': trial.suggest_int('wd1', -8, -3),
            'wd2': trial.suggest_int('wd2', -8, -3),
            'dropout': trial.suggest_float('dropout', 0.0, 0.7, step=0.1),
            'dropout2': trial.suggest_float('dropout2', 0.0, 0.7, step=0.1),
            'lamda': trial.suggest_float('lamda', 0.5, 2.0, step=0.5),
            'alpha': trial.suggest_float('alpha', 0.1, 0.9, step=0.1),
            'jk_type': trial.suggest_categorical('jk_type', ['max', 'cat']),
            'n_layers': trial.suggest_int('n_layers', 0, 16, step=4),
            }
        for k in params.keys():
            if k.startswith('wd'):
                params[k] = float('1e'+str(params[k]))
                continue
            if k.startswith('lr'):
                if params[k] == -0.02:
                    params[k] = 0.0005
                    continue
                if params[k] == -0.01:
                    params[k] = 0.001
                    continue
                if params[k] == 0.:
                    params[k] = 0.005
                continue
            if k.startswith('n_layers'):
                if params[k] == 0:
                    params[k] = 2
        params['es_ckpt'] = get_es_ckpt_fname(trial.study, trial)
        update_args_(args, params)
    return args

def pruneDuplicate(trial):
    trials = trial.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    numbers=np.array([t.number for t in trials])
    bool_params= np.array([trial.params==t.params for t in trials]).astype(bool)
    #Don´t evaluate function if another with same params has been/is being evaluated before this one
    if np.sum(bool_params)>1:
        if trial.number>np.min(numbers[bool_params]):
            logging.getLogger('optuna.pruners').info('[YH INFO] Prune duplicated args!')
            raise optuna.exceptions.TrialPruned()
    return 


def objective(trial):
    args = suggest_args(trial)
    logger = set_logger(args)
    logger.info(args)

    # might prune
    pruneDuplicate(trial)
    val_accs ,test_accs= main(args, logger, trial)

    # set user attrs
    trial.set_user_attr("repeat_times", args.n_cv)
    trial.set_user_attr("test_acc", np.array(val_accs).mean())
    return np.array(val_accs).mean()


class myCallBack:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def __call__(self, study, Frozentrial) -> None:
        # clean logger
        logger=logging.getLogger("detail")
        logger.handlers = []
        # clean ckpt
        for suffix in ['', '-0', '-1', '-2', '-3', '-4']:
            _p = os.path.join('cache','ckpts', get_es_ckpt_fname(study, Frozentrial)  +'.pt' + suffix)
            if os.path.exists(_p):
                os.remove(_p)


def study_main(n_trials, study):
    study.optimize(objective, 
                n_trials=n_trials,
                catch=(RuntimeError,), 
                callbacks=(myCallBack(), )
    )
    

if __name__ == '__main__':
    args = set_args()
    dataset = args.dataset
    n_trials = args.optuna_n_trials
    
    kw = args.kw
    study = optuna.create_study(
        study_name="GCNIIJK-{}-{}".format(dataset, kw),
        direction="maximize", 
        storage = optuna.storages.RDBStorage(url='sqlite:///{}/GCNIIJK-{}.db'.format('cache/OptunaTrials', kw), 
                engine_kwargs={"connect_args": {"timeout": 20000}}),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=3,interval_steps=1,n_min_trials=5),
        load_if_exists=True
        )
    study.set_system_attr('kw', kw)

    num_completed = len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]))
    num_pruned = len(study.get_trials(deepcopy=False, states=[TrialState.PRUNED]))

    # while num_completed + num_pruned < n_trials:
    while num_completed < n_trials:
        print('{} trials to go!'.format(n_trials - num_completed))
        study_main(n_trials-num_completed, study)
        FailedTrials = study.get_trials(deepcopy=False, states=[TrialState.FAIL])
        num_failed = len(FailedTrials)
        num_completed = len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]))
        num_pruned = len(study.get_trials(deepcopy=False, states=[TrialState.PRUNED]))
        if num_pruned > 2000:
            break


    ##################################
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print("Study statistics this: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print("  Yuhe's record: ")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))