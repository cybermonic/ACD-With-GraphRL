from argparse import ArgumentParser

from joblib import Parallel, delayed
import torch

from env.yt_env import YTEnv, BlueAgent, build_graph, seed
from model.inductive_ppo import InductiveGraphPPO, PPOMemory

MAX_STEPS = 5e6

# Stable baselines hyperparams
EPOCHS = 10 #10
BATCH_SIZE=64
N = 5
C_LR = 0.001
A_LR = 0.0003
CLIP = 0.2

SEED = 0
seed(SEED)

torch.set_num_threads(16)


@torch.no_grad()
def simulate(env: YTEnv, agent: BlueAgent):
    agent.model.eval()
    s = env.reset()
    tot_r = 0
    t = False
    mem = PPOMemory(0)

    while not t:
        (act,target), a,v,p = agent.select_action(*s)
        r_true,r_shaped,t,next_s = env.step(act,target)
        mem.remember(s,a,v,p,r_shaped,t)
        s = next_s

        tot_r += r_true

    return env.ts, tot_r, mem

def experiment(env: YTEnv, agent: BlueAgent, domain_randomization=False):
    tr_steps = 0
    ep_lens, ep_rews = [],[]

    while tr_steps < MAX_STEPS:
        ret = Parallel(n_jobs=N, prefer='processes')(
            delayed(simulate)(env, agent) for _ in range(N)
        )
        steps, rew, mems = zip(*ret)
        ep_rews += list(rew)
        ep_lens += list(steps)
        tr_steps += sum(steps)

        avg_r = sum(rew) / len(rew)
        avg_l = sum(steps) / len(steps)

        agent.model.memory = PPOMemory(BATCH_SIZE).combine(mems)
        agent.model.learn(EPOCHS, verbose=False)

        print(f'[{tr_steps}] Avg r: {avg_r:0.2f}, Avg l: {avg_l:0.2f}')
        torch.save({'rews': ep_rews, 'lens': ep_lens}, f'logs/{GRAPH_SIZE}N_{SEED}.pt')

        if domain_randomization:
            x,ei = build_graph(GRAPH_SIZE, verbose=False)
            env = YTEnv(x,ei)
            agent = BlueAgent(env, agent.model, inductive=agent.inductive)

    agent.model.save(f'saved_models/ppo_{GRAPH_SIZE}N_{SEED}_last.pt')

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('num_nodes', nargs=1, type=int)
    ap.add_argument('--attn', action='store_true')

    args = ap.parse_args()
    GRAPH_SIZE = args.num_nodes[0]

    x,ei = build_graph(GRAPH_SIZE)
    env = YTEnv(x,ei)

    blue = InductiveGraphPPO(x.size(1), 2, BATCH_SIZE, alr=A_LR, clr=C_LR, clip=CLIP, attn=args.attn)
    agent = BlueAgent(env, blue, inductive=True)

    experiment(env, agent, domain_randomization=False)