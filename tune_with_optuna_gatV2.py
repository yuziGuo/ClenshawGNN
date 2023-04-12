import optuna
from optuna.trial import TrialState
import optuna.study
from data.citation_dataloader import  citation_loader
from data.citation_full_dataloader import  citation_full_supervised_loader
from data.geom_dataloader import geom_dataloader
from data.linkx_dataloader import linkx_dataloader

from models.GATV2 import GATV2

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
    if args.dataset in ['twitch-gamer', 'Penn94', 'genius']:
        loader = linkx_dataloader(args.dataset, args.gpu, args.self_loop, n_cv=1)
    elif args.dataset in ['citeseerfull', 'pubmedfull', 'corafull']:
        # For full-supervised  
        loader = citation_full_supervised_loader(args.dataset, args.gpu, args.self_loop, n_cv=args.n_cv)
    elif args.dataset.startswith('geom'):
        dataset = args.dataset.split('-')[1]
        loader = geom_dataloader(dataset, args.gpu, args.self_loop, digraph=not args.udgraph, n_cv=args.n_cv, cv_id=args.start_cv)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    loader.load_data()
    return loader


def build_model(args, edge_index, norm_A, in_feats, n_classes):
    if args.model == 'GAT':
        model = GATV2( 
                    edge_index,
                    norm_A,
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers,
                    args.heads,
                    args.out_heads,
                    args.dropout,
                    args.dropout2,
                    args.with_negative_residual,
                    args.with_initial_residual,
                    args.bn
                    )
        model.to(args.gpu)
        return model

def build_optimizers(args, model):
    param_groups = [
        {'params': model.fcs.parameters(), 'lr':args.lr1,'weight_decay':args.wd1},
        {'params': model.convs.parameters(), 'lr':args.lr2,'weight_decay':args.wd2}
    ]
    if args.bn:
        param_groups.append(
            {'params': model.bns.parameters(), 'lr':args.lr2,'weight_decay':args.wd2}
        )
    optimizer_adam = th.optim.Adam(param_groups)
    if args.with_initial_residual:
        param_groups = [
            {'params':[model.alphas], 'lr':args.lr3,'weight_decay':args.wd3}
        ]
        optimizer_sgd = th.optim.SGD(param_groups, momentum=args.momentum)
        return [optimizer_adam, optimizer_sgd]
    return [optimizer_adam]


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

        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        acc = correct.item() * 1.0 / len(labels)
        return acc, loss

def run(args, logger, trial, cv_id, edge_index, data, norm_A, features, labels, model_seed):
    dur = []

    if args.dataset in ['twitch-gamer', 'Penn94', 'genius']: # encouraged to use fixed splits
        data.load_mask()
    else:
        data.load_mask(p=(0.6,0.2,0.2))
    
    reset_random_seeds(model_seed)
    if args.dataset != 'genius':    
        loss_fcn = th.nn.NLLLoss()
        evaluator = 'acc'
    else:
        loss_fcn = th.nn.BCEWithLogitsLoss()
        evaluator = 'rocauc'
        labels = F.one_hot(labels, labels.max()+1).float()


    data.in_feats = features.shape[-1] # ----
    model = build_model(args, edge_index, norm_A, data.in_feats, data.n_classes)
    optimizers = build_optimizers(args, model)
    if args.early_stop:
        stopper_step, stopper = build_stopper(args)
    

    if args.dataset in ['citeseerfull', 'pubmedfull', 'Penn94'] or args.dataset.startswith('geom'):
        log_step = 50
    elif args.dataset:
        log_step = 5

    for epoch in range(args.n_epochs): 
        t0 = time.time()
        
        model.train()
        for _ in optimizers:
            _.zero_grad()

        logits = model(features)
        loss = loss_fcn(logits[data.train_mask], labels[data.train_mask])
        loss.backward()

        for _ in optimizers:
            _.step()

        val_acc, val_loss = evaluate(model, loss_fcn, features, labels, data.val_mask, epoch, evaluator)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        dur.append(time.time() - t0)
        if args.log_detail and (epoch+1) % log_step == 0 :
            logger.info("Epoch {:05d} | Time(s) {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | "
                        "ETputs(KTEPS) {:.2f}". format(epoch+1, np.mean(dur), val_loss.item(),
                                                        val_acc, 
                                                        data.n_edges / np.mean(dur) / 100)
                        )
        if args.early_stop and epoch >= 0:
            score = val_loss
            if stopper_step(score, model):
                break   
    # end for  
    if args.early_stop:
        model.load_state_dict(th.load(stopper.store_path))
        logger.debug('Model Saved by Early Stopper is Loaded!')
    val_acc, val_loss = evaluate(model, loss_fcn, features, labels, data.val_mask, epoch, evaluator)
    logger.info("[FINAL MODEL] Val accuracy {:.2%} \Val loss: {:.2}".format(val_acc, val_loss))
    test_acc, test_loss = evaluate(model, loss_fcn, features, labels, data.test_mask, epoch, evaluator)
    logger.info("[FINAL MODEL] Test accuracy {:.2%} \Test loss: {:.2}".format(test_acc, test_loss))
    return val_acc, test_acc
    

def main(args, logger, trial):
    reset_random_seeds(args.seed)
    data  = build_dataset(args)
    data.seeds = [random.randint(0,10000) for _ in range(args.n_cv)]
    model_seeds = [random.randint(0,10000) for _ in range(args.n_cv)]
    logger.info('Split_seeds:{:s}'.format(str(data.seeds)))
    logger.info('Model_seeds:{:s}'.format(str(model_seeds)))

    edge_index = data.edge_index
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    _, norm_A = gcn_norm(edge_index, add_self_loops=False)
    features = data.features
    labels = data.labels

    cv_id = 0
    val_acc, test_acc = run(args, logger, trial, cv_id, edge_index, data, norm_A,  features, labels, model_seed=model_seeds[cv_id])

    logger.info("Acc on the first split (Validation Set): {:.4f}".format(val_acc))
    logger.info("Acc on the first split (Test Set): {:.4f}".format(test_acc))
    return val_acc, test_acc


# https://github.com/optuna/optuna/issues/862
def set_args():
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--model", type=str, default='GAT')
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name ('cora', 'citeseer', 'pubmed').")

    parser.add_argument("--n-hidden", type=int, default=64, help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=4, help="number of hidden gcn layers")
    parser.add_argument("--heads", type=int, default=8, help="attention heads")
    parser.add_argument("--out-heads", type=int, default=1, help="number of output attention heads")
    parser.add_argument("--with-negative-residual", action='store_true', default=False, help="")
    parser.add_argument("--with-initial-residual", action='store_true', default=False, help="")
    parser.add_argument("--bn", action='store_true', default=False, help="")


    # for training
    parser.add_argument("--wd1", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--wd2", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--lr1",  type=float, default=1e-2, help="learning rate")
    parser.add_argument("--lr2",  type=float, default=1e-2, help="learning rate")
    parser.add_argument("--momentum",  type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--dropout",  type=float, default=0.6)
    parser.add_argument("--dropout2",  type=float, default=0.6)
    parser.add_argument("--n-epochs", type=int, default=2000, help="number of training epochs")

    parser.add_argument("--loss", type=str, default='nll')
    parser.add_argument("--self-loop", action='store_true', default=False, help="graph self-loop (default=False)")
    parser.add_argument("--udgraph", action='store_true', default=False, help="undirected graph (default=False)")
    
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

    parser.add_argument("--optuna-n-trials", type=int, default=202)
    parser.add_argument("--kw", type=str)

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


def update_args_(args, params):
  """updates args in-place"""
  dargs = vars(args)
  dargs.update(params)

def np_arg_max_last_occurrence(val_acc_means):
    _ = val_acc_means[::-1]
    i = len(_) - np.argmax(_) - 1 
    return i

def get_es_ckpt_fname(study, trial):
    return  '_'.join([study.study_name, study.system_attrs['kw'],'trialNo{}'.format(trial.number)])
    

fixed_params = {
        'early_stop': True,
        'udgraph': True,
        'self_loop': True,

        'out_heads': 1,
        'bn': True,
        'logging': True,
        'log_detail': True,
        'log_detailedCh': False,
    }

def suggest_args(trial):
    args = set_args()
    if args.with_initial_residual or args.with_negative_residual:
        use_clenshaw_for_gat = True
    else:
        use_clenshaw_for_gat = False
    # args <--> suggest number
    if trial is not None:  
        suggested_params = {
            'lr1': trial.suggest_float('lr1', -0.02, 0.05, step=0.01),
            'wd1': trial.suggest_int('wd1', -8, -3),
            'lr2': trial.suggest_float('lr2', -0.02, 0.05, step=0.01),
            'wd2': trial.suggest_int('wd2', -8, -3),
            'dropout': trial.suggest_float('dropout', 0.0, 0.9, step=0.1),
            'dropout2': trial.suggest_float('dropout2', 0.0, 0.9, step=0.1)
            } 
        if use_clenshaw_for_gat:
            suggested_params['n_layers'] = trial.suggest_int("n_layers", 2, 12, step=2)
        if args.with_initial_residual:
            suggested_params['lr3'] = trial.suggest_float('lr3', -0.02, 0.05, step=0.01)
            suggested_params['wd3'] =  trial.suggest_int('wd3', -8, -3)
            suggested_params['momentum'] = trial.suggest_float("momentum", 0.8, 0.95, step=0.05)

        # II. postprocess of params
        for k in suggested_params.keys():
            if k.startswith('wd'):
                suggested_params[k] = float('1e'+str(suggested_params[k]))
                continue
            if k.startswith('lr'):
                if suggested_params[k] == -0.02:
                    suggested_params[k] = 0.0005
                    continue
                if suggested_params[k] == -0.01:
                    suggested_params[k] = 0.001
                    continue
                if suggested_params[k] == 0.:
                    suggested_params[k] = 0.005
                continue
        suggested_params['es_ckpt'] = get_es_ckpt_fname(trial.study, trial)
        
        # III. params --> args
        update_args_(args, fixed_params)
        update_args_(args, suggested_params)
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
    val_acc, test_acc = main(args, logger, trial)
    trial.set_user_attr("repeat_times", 1)
    trial.set_user_attr("test_acc", test_acc)
    trial.set_user_attr("val_acc", val_acc)
    return val_acc


class CkptsAndHandlersClearerCallBack:
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
                callbacks=(CkptsAndHandlersClearerCallBack(), )
    )
    
def get_complete_and_pruned_trial_nums(study):
    num_completed = len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]))
    num_pruned = len(study.get_trials(deepcopy=False, states=[TrialState.PRUNED]))
    return num_completed, num_pruned

if __name__ == '__main__':
    args = set_args()
    dataset = args.dataset
    n_trials = args.optuna_n_trials

    kw = args.kw
    study = optuna.create_study(
        study_name="GATV2-{}-{}".format(dataset, kw),
        direction="maximize", 
        storage = optuna.storages.RDBStorage(url='sqlite:///{}/GATV2-{}.db'.format('cache/OptunaTrials', kw), 
                engine_kwargs={"connect_args": {"timeout": 20000}}),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=10,interval_steps=1,n_min_trials=5),
        load_if_exists=True
        )
    study.set_system_attr('kw', kw)

    # set fixed params
    for k, v in fixed_params.items():
        study.set_system_attr(k, v)

    num_completed, num_pruned = get_complete_and_pruned_trial_nums(study)
    while num_completed + num_pruned < n_trials:
        print('{} trials to go!'.format(n_trials - num_completed - num_pruned))
        study_main(1, study)
        num_completed, num_pruned = get_complete_and_pruned_trial_nums(study)
        if num_pruned > 1000:
            break
    
    ##################################
    num_completed, num_pruned = get_complete_and_pruned_trial_nums(study)

    print("Study statistics this: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", num_pruned)
    print("  Number of complete trials: ", num_completed)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print("  Yuhe's record: ")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))