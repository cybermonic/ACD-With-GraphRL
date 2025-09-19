from argparse import ArgumentParser

import torch
from tqdm import tqdm
from joblib import Parallel, delayed

from env.yt_env import build_graph, YTEnv, BlueAgent, seed
from model.doorman_et_al import load_doorman
from train_doorman import simulate

# TODO make these args
ap = ArgumentParser()
ap.add_argument('n', nargs=1, type=int)
ap.add_argument('-s', '--seed', default=0, type=int)
ap.add_argument('-e', '--eps', default=0, type=float)
ap.add_argument('--dir', default='')
ap.add_argument('--model-n', type=int, default=-1)

args = ap.parse_args()

N = args.n[0]
MODEL_N = args.model_n if args.model_n > 0 else N
SEED = args.seed

#MODEL = f'dqn_40N_{SEED}_last'
FNAME = f'dqn_{MODEL_N}N_{SEED}_last'
model = load_doorman('saved_models/doorman_lr0001_N1_gamma1/' + FNAME + '.pt')

model.eval()
torch.no_grad()

def eval_one_graph(model, i):
    torch.set_num_threads(16)

    # Repeatability, +10,000 there's no collisions w graphs from training
    seed(10_000 + i)

    rews,lens = [],[]
    g = build_graph(N)
    env = YTEnv(*g)

    blue = BlueAgent(env, model, deterministic=True)
    for _ in tqdm(range(10)):
        l,r,_ = simulate(env, blue, args.eps, N)
        rews.append(r)
        lens.append(l)

    print(sum(rews) / 10)
    return rews,lens

ret = Parallel(prefer='processes', n_jobs=50)(
    delayed(eval_one_graph)(model, i)
    for i in range(50)
)
rews, lens = zip(*ret)
rews = sum([list(r) for r in rews], [])
lens = sum([list(l) for l in lens], [])

rews = torch.tensor(rews, dtype=torch.float)
lens = torch.tensor(lens, dtype=torch.float)

torch.save(
    {'rews': rews, 'lens': lens},
    f'results/{FNAME}_eval.pt'
)

print(f"R Mean: {rews.mean().item()}, L Mean: {lens.mean().item()}")
print(f"R Std: {rews.mean().item()}, L Std: {lens.mean().item()}")
