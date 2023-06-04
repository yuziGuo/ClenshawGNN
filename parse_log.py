import os

def remap(k,v):
    if k.startswith('wd'):
        v = float('1e'+str(v))
    if k.startswith('lr'):
        v = float(v)
        if v == -0.02:
            v = 0.0005
        elif v == -0.01:
            v = 0.001
        elif v == 0.:
            v = 0.005
        else:
            pass
    if len(k.split('_'))>1:
        k = '-'.join(k.split('_'))
    if k=='n-layers':
        v = int(v)
        if v == 0:
            v = 2
        else:
            pass
    return k,v

def parse_line(line):
    keysOfInterests = ['dropout', 'dropout2', 
                       'lamda', 'alpha','jk_type',
                       'lr1', 'lr2', 'lr3', 'wd1', 'wd2', 'wd3','momentum', 
                       'n_layers', 'n_layers_gcnii', 'n_layers_jk',
                       ] 
    matchedKey = None
    for k in keysOfInterests:
        if len(line.split(f'{k}:'))>1:
            matchedKey = k
            break
    if matchedKey is None:
        return None
    v = line.rstrip().split(f'{matchedKey}: ')[-1]

    if v.startswith('0.'):
        v = round(float(v),2)
    k,v=remap(matchedKey, v)
    return f'--{k} {v}'


def parse_file(name):
    with open(name, 'r') as f:
        kvs = []
        for line in f.readlines():
            # print(line)
            kv = parse_line(line)
            if kv is not None:
                kvs.append(kv)
    return ' '.join(kvs)


def parse_example():
    root_dir = './optunalogKlayers'
    prefixes = {'pubmedfull': 'pub', 'geom-squirrel': 'sq'}
    for ds,abbr in prefixes.items():
        for K in range(8,36,4): # 8, 12, ..., 32
            line_prefix = f'python train_clenshaw.py --dataset {ds} --n-layers {K} --early-stop  --udgraph --self-loop  --loss nll --n-cv 20 --log-detail --log-detailedCh'
            name = f"{root_dir}/{abbr}-clenshaw{K}.log"
            # print(name)
            params_str = parse_file(name)
            print(line_prefix+' '+params_str)


def parse1():
    root_dir = './optunalogJK'
    prefixes = {'pubmedfull': 'pub', 'geom-squirrel': 'sq', 'corafull':'cora', 'geom-chameleon':'cham'}
    for ds,abbr in prefixes.items():
        line_prefix = f'python train_GCNIIJK.py --dataset {ds}  --early-stop  --udgraph --self-loop  --loss nll --n-cv 20 --log-detail --log-detailedCh'
        name = f"{root_dir}/{abbr}.log"
        # print(name)
        params_str = parse_file(name)
        output_tail = f'--es-ckpt es{abbr}  1>gcniijk_{abbr}.log 2>gcniijk_{abbr}.err&'
        print(line_prefix+' '+params_str+' '+output_tail)

def parse2():
    root_dir = './optunalogJK'
    prefixes = {'pubmedfull': 'pub', 'geom-squirrel': 'sq', 'corafull':'cora', 'geom-chameleon':'cham'}
    for ds,abbr in prefixes.items():
        line_prefix = f'python train_ensemble_JK_GCNII.py --dataset {ds}  --early-stop  --udgraph --self-loop  --loss nll --n-cv 20 --log-detail --log-detailedCh'
        name = f"{root_dir}/{abbr}.log"
        # print(name)
        params_str = parse_file(name)
        output_tail = f'--es-ckpt es{abbr}  1>gcniijk_{abbr}.log 2>gcniijk_{abbr}.err&'
        print(line_prefix+' '+params_str+' '+output_tail)


def parse3():
    root_dir = './logs'
    prefixes = {'pubmedfull': 'pub', 'geom-squirrel': 'sq', 'geom-chameleon':'cham'}
    for ds,abbr in prefixes.items():
        line_prefix = f'python train_gatJK.py --dataset {ds}  --early-stop  --udgraph --self-loop  --loss nll --n-cv 20 --log-detail --gpu 1 --log-detailedCh'
        name = f"{root_dir}/gatJK-{abbr}.log"
        # print(name)
        params_str = parse_file(name)
        output_tail = f'--es-ckpt es{abbr}  1>gatjk_{abbr}.log 2>gatjk_{abbr}.err&'
        print(line_prefix+' '+params_str+' '+output_tail)


if __name__ == '__main__':
    parse3()
