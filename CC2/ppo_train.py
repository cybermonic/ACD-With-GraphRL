from argparse import ArgumentParser 
import inspect
import random 
from types import SimpleNamespace

from joblib import Parallel, delayed
import torch 
from tqdm import tqdm 

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from CybORG.Agents.SimpleAgents.GreenAgent import GreenAgent

from models.ppo import GraphPPOAgent, PPOMemory
from models.gnn_type_ablation import InductiveGraphPPOAgent

from wrapper.graph_wrapper import GraphWrapper
from wrapper.observation_graph import ObservationGraph

DEFAULT_DIM = ObservationGraph.DIM + 4

torch.set_num_threads(16)
HYPER_PARAMS = SimpleNamespace(
    N = 100,     # How many episodes before training
    bs = 2500,   # How many steps to learn from at a time
    episode_len = 100,
    training_episodes = 10_001,
    epochs = 4,
)
GENERATE_VIZ_DATA=False

@torch.no_grad()
def generate_episode(agent, hp): 
    torch.set_num_threads(1)

    red_agent = random.choice([RedMeanderAgent, B_lineAgent])
    wrapped = GraphWrapper('Blue', CybORG(path, 'sim', agents={'Red': red_agent})) 
    mem = PPOMemory(0)

    state = wrapped.reset()
    for step_cnt in range(hp.episode_len): 
        action,value,prob = agent.get_action(state)
        next_state, reward, _,_ = wrapped.step(action)
        mem.remember(state, action, value, prob, reward, 0 if step_cnt < hp.episode_len-1 else 1)
        state = next_state

    return mem

def train(agent, hp):
    agent.train()

    with open(f'logs/{hp.fnames}.txt', 'w+') as f: 
        f.write('epoch,avg_r,loss\n')

    for e in range(1,hp.training_episodes):
        mems = Parallel(n_jobs=hp.N, prefer='processes')(
            delayed(generate_episode)(agent, hp) for _ in range(hp.N)
        )

        mem = PPOMemory(hp.bs).load(mems)
        agent.memory = mem 
        avg_r = sum(mem.r) / hp.N 
        
        torch.set_num_threads(16)
        print(f"[{e}] Average reward: {avg_r}")
        loss = agent.learn()

        with open(f'logs/{hp.fnames}.txt', 'a') as f:
            f.write(f'{e},{avg_r},{loss}\n')

        agent.save(outf=f'checkpoints/{hp.fnames}.pt')
        if e % 1000 == 0: 
            agent.save(outf=f'checkpoints/{hp.fnames}_{e//1000}.pt')


if __name__ == '__main__':
    scenario = 'Scenario2'
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    ap = ArgumentParser()
    ap.add_argument('name', metavar='Name', action='store', type=str, nargs=1, help='Filename to use for output logs/weights')
    ap.add_argument('--inductive', action='store_true', help='Use naive inductive model')
    ap.add_argument('--attn', action='store_true', help='Use self-attention-based inductive model')
    ap.add_argument('--globalnode', action='store_true', help='Use global node inductive aggregation')
    ap.add_argument('--hidden', action='store', type=int, default=256, help='Dimension of middle layer for actor/critic')
    ap.add_argument('--embedding', action='store', type=int, default=64, help='Dimension of node representation for actor/critic')
    ap.add_argument('--layers', action='store', type=int, default=2, help='Number of layers for GCNs')
    ap.add_argument('--gnn', default='gcn')
    ap.add_argument('--epochs', default=10_000, type=int)

    args = ap.parse_args()
    args.name = args.name[0]
    print(args)

    HYPER_PARAMS.training_episodes = args.epochs

    if args.inductive or args.attn or args.globalnode:
        agent = InductiveGraphPPOAgent(
            DEFAULT_DIM, 
            args.gnn,
            bs=HYPER_PARAMS.bs,
            a_kwargs={'lr': 0.0003, 'hidden1': args.hidden, 'hidden2': args.embedding}, 
            c_kwargs={'lr': 0.001, 'hidden1': args.hidden, 'hidden2': args.embedding},
            clip=0.2,
            epochs=HYPER_PARAMS.epochs, 
            naive= not args.attn, 
            globalnode= args.globalnode
        )
    else:
        agent = GraphPPOAgent(
            DEFAULT_DIM, 
            bs=HYPER_PARAMS.bs,
            a_kwargs={'lr': 0.0003, 'hidden1': args.hidden, 'hidden2': args.embedding}, 
            c_kwargs={'lr': 0.001, 'hidden1': args.hidden, 'hidden2': args.embedding},
            clip=0.2,
            epochs=HYPER_PARAMS.epochs
        )
    
    HYPER_PARAMS.fnames = args.name
    train(agent, HYPER_PARAMS)