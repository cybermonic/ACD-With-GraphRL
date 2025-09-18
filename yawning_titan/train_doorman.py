from argparse import ArgumentParser
from copy import deepcopy
from math import ceil 
import random 

import torch 
from torch.optim import Adam
from joblib import Parallel, delayed

from env.yt_env import YTEnv, BlueAgent, build_graph, seed
from model.doorman_et_al import DoormanAgent, DQN_Buffer

MAX_STEPS = 5e6

# Stable baselines hyperparams
EPOCHS = 1
BATCH_SIZE=64
GAMMA = 0.95
N = 5
C_LR = 0.001
A_LR = 0.0003
CLIP = 0.2

SEED = 0
seed(SEED)

torch.set_num_threads(16)


@torch.no_grad()
def simulate(env: YTEnv, agent: BlueAgent, eps: float, graph_size: int):
    torch.set_num_threads(1)
    agent.model.eval()
    s = env.reset()
    tot_r = 0
    t = False
    mem = DQN_Buffer(graph_size)

    while not t:
        values = agent.model(*s, graph_size)
        
        if random.random() < eps: 
            idx = random.randint(0,values.size(1)-1)
        else: 
            idx = values.argmax().item()

        act,target = agent.num_to_action_inductive(idx)
        r_true,r_shaped,t,next_s = env.step(act,target)
        
        mem.push(s,idx,r_true,next_s)
        s = next_s

        tot_r += r_true

    return env.ts, tot_r, mem

def experiment(env: YTEnv, agent: BlueAgent, domain_randomization=False):
    tr_steps = 0
    ep_lens, ep_rews, ep_steps = [],[],[]

    model = agent.model
    tgt_model = deepcopy(model)
    opt = Adam(model.parameters(), lr=0.0001)

    updates = 0 
    main_mem = DQN_Buffer(GRAPH_SIZE, BATCH_SIZE)
    while tr_steps < MAX_STEPS:
        # Same as paper linear decay from 1 to 0.1
        eps = 0.1 + (0.9 * max(0, ((MAX_STEPS/3)-tr_steps)/(MAX_STEPS/3)) )
        ret = Parallel(n_jobs=N, prefer='processes')(
            delayed(simulate)(env, agent, eps if n != N else 0, GRAPH_SIZE) for n in range(N+1)
        )
        ret, test = ret[:-1], ret[-1]

        steps, rew, mems = zip(*ret)
        tr_steps += sum(steps)

        avg_r = test[1] #sum(test) / len(rew)
        avg_l = test[0] #sum(steps) / len(steps)

        ep_rews += [avg_r]
        ep_lens += [avg_l]
        ep_steps.append(tr_steps)

        memory = DQN_Buffer(GRAPH_SIZE, BATCH_SIZE).combine(mems)
        main_mem.update(memory)
        model.train()
        tgt_model.eval()
        main_mem.loss(model, tgt_model, GAMMA, opt)

        updates += 1 

        print(f'[{tr_steps}] Avg r: {avg_r:0.2f}, Avg l: {avg_l:0.2f}, Buffer: {len(main_mem.s)}, Eps: {eps:0.4f}')
        torch.save({'rews': ep_rews, 'lens': ep_lens, 'steps': ep_steps}, f'logs/dqn_{GRAPH_SIZE}N_{SEED}.pt')
        
        if updates % 50 == 0:
            model.save(f'saved_models/dqn_{GRAPH_SIZE}N_{SEED}_last.pt')
            tgt_model = deepcopy(model)

        if domain_randomization:
            x,ei = build_graph(GRAPH_SIZE, verbose=False)
            env = YTEnv(x,ei)
            agent = BlueAgent(env, agent.model, inductive=agent.inductive)

    model.save(f'saved_models/dqn_{GRAPH_SIZE}N_{SEED}_last.pt')

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('num_nodes', nargs=1, type=int)
    ap.add_argument('--attn', action='store_true')
    ap.add_argument('--hidden', default=64, type=int)

    args = ap.parse_args()
    #args.num_nodes = [10]
    GRAPH_SIZE = args.num_nodes[0]

    x,ei = build_graph(GRAPH_SIZE)
    env = YTEnv(x,ei)

    blue = DoormanAgent(3, args.hidden, 2, 2)
    agent = BlueAgent(env, blue, inductive=True)

    experiment(env, agent, domain_randomization=False)