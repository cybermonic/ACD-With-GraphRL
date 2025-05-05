from argparse import ArgumentParser
import inspect
import pickle 
from statistics import mean, stdev
import subprocess
import sys 

from joblib import Parallel, delayed
import torch 
from tqdm import tqdm 

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.SimpleAgents.GreenAgent import GreenAgent
from custom_red_agents.sleepy import SleepyBLine, SleepyMeander

sys.path.append('CardiffUni')
from Agents.MainAgent import MainAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

from models.ppo import load_ppo
from models.inductive_ppo import load_inductive_ppo
from models.gnn_type_ablation import load_inductive_ppo as load_gnn_ablation
from wrapper.graph_wrapper import GraphWrapper

#from viz.create_viz_data import create_viz_data


MAX_EPS = 100
SAVE_ACTION_DISTRO = False
SHUFFLE_NIDS = False
agent_name = 'Blue'

'''
Copied from CybORG directory 
'''

def wrap(env):
    return GraphWrapper('Blue', env)

def wrap_cardiff(env):
    return ChallengeWrapper2('Blue', env)

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

@torch.no_grad()
def generate_episode(agent, red_agent): 
    cyborg = CybORG(path, 'sim', agents={'Red': red_agent})

    wrapped_cyborg = wrap(cyborg)
    observation = wrapped_cyborg.reset()

    if SHUFFLE_NIDS:
        x,ei = observation 
        n_hosts = int(x[:,0].sum().item())
        
        host_ids = (x[:,0] == 1).nonzero().squeeze(-1)
        shuffle = torch.randperm(host_ids.size(0))
        
        edge_map = {
            host_ids[i].item() : host_ids[shuffle][i].item()
            for i in range(shuffle.size(0))
            if host_ids[i].item() != host_ids[shuffle][i].item()
        }
        shuf_host_to_og = {
            shuffle[i].item() : i
            for i in range(shuffle.size(0))
        }

    r = []
    a = []
    
    for j in range(num_steps):
        if SHUFFLE_NIDS:
            x,ei = observation 
            new_x = x.clone()
            new_ei = ei.clone()

            new_x[host_ids[shuffle]] = new_x[host_ids]
            for i in range(ei.size(1)):
                src = ei[0,i].item()
                dst = ei[1,i].item()

                if src in edge_map:
                    new_ei[0,i] = edge_map[src]
                if dst in edge_map:
                    new_ei[1,i] = edge_map[dst]

            observation = (new_x,new_ei)

        action = agent.get_action(observation)

        # Remap actions back to original host number 
        if SHUFFLE_NIDS:
            # Sleep/monitor don't need to be changed
            # Those are actions 0 and 1 
            if action > 1: 
                action -= 2 
                fn = action // n_hosts
                target = action % n_hosts 

                # Translate here: 
                target = shuf_host_to_og[target]

                # Then rebuild action 
                action = int((fn*n_hosts) + target + 2)

        observation, rew, done, info = wrapped_cyborg.step(action)
        
        #action_log[red_agent][str(cyborg.get_last_action('Blue'))] += 1
        #true_state = cyborg.get_agent_state('True')

        r.append(rew)
        a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))

    agent.end_episode()
    # observation = cyborg.reset().observation

    observation = wrapped_cyborg.reset()
    return sum(r)

def dim_ablation(): 
    global fname, path, num_steps
    fname = 'dim_ablation.txt'
    path = 'scenarios/cage2.yaml'

    for weight in ['32_8', '64_16', '128_32', '512_128', '1024_256']: 
        with open(fname, 'a') as f: 
            f.write(f"{weight}\n")

        agent = load_inductive_ppo(
            f'model_weights/ablation/dim/{weight}.pt', 
            globalnode=True
        )
        agent.set_deterministic(True)

        all_r = []
        for num_steps in [30, 50, 100]:
            for red_agent in [B_lineAgent, RedMeanderAgent]: # , SleepAgent]:
                total_reward = Parallel(n_jobs=MAX_EPS, prefer='processes')(
                    delayed(generate_episode)(agent, red_agent) for _ in range(MAX_EPS)
                )

                print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
                with open(fname, 'a') as f:
                    f.write(f'{red_agent.__name__},{num_steps},{mean(total_reward)},{stdev(total_reward)}\n')

                all_r.append(mean(total_reward))

        with open(fname, 'a') as f: 
            f.write(f"Total,{sum(all_r)}\n\n")

def gnn_ablation(): 
    global fname, path, num_steps
    fname = 'gnn_ablation.txt'
    path = 'scenarios/cage2.yaml'

    for weight in ['gcn']: #, 'sage', 'gat', 'gin']: 
        with open(fname, 'a') as f: 
            f.write(f"{weight}\n")

        agent = load_gnn_ablation(
            f'model_weights/ablation/gnn/{weight}.pt', 
            globalnode=True
        )
        agent.set_deterministic(True)

        all_r = []
        for num_steps in [30, 50, 100]:
            for red_agent in [B_lineAgent, RedMeanderAgent]: # , SleepAgent]:
                total_reward = Parallel(n_jobs=MAX_EPS, prefer='processes')(
                    delayed(generate_episode)(agent, red_agent) for _ in range(MAX_EPS)
                )

                print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
                with open(fname, 'a') as f:
                    f.write(f'{red_agent.__name__},{num_steps},{mean(total_reward)},{stdev(total_reward)}\n')

                all_r.append(mean(total_reward))

        with open(fname, 'a') as f: 
            f.write(f"Total,{sum(all_r)}\n\n")

if __name__ == "__main__":
    #gnn_ablation()
    #dim_ablation()
    #exit()

    ap = ArgumentParser()
    ap.add_argument('-s', '--scenario', default='scenarios/cage2.yaml')
    ap.add_argument('--ni', action='store_true')
    ap.add_argument('--self-attn', action='store_true')
    ap.add_argument('--globalnode', action='store_true')
    ap.add_argument('--cardiff', action='store_true')
    ap.add_argument('--num_nodes', default=None, type=int)
    args = ap.parse_args()

    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]
    
    # PPO 
    if args.ni:
        agent = load_ppo('model_weights/noninductive.pt', shared=False)
        agent.set_deterministic(True)
        fname = 'ni_results.txt'

    # Global node
    elif args.globalnode:
        agent = load_inductive_ppo('model_weights/inductive_global-attn.pt', globalnode=True, num_nodes=args.num_nodes) # Tmp file. Still running as of 03/12
        agent.set_deterministic(True)
        fname = 'global_node.txt'

    # Inductive
    else:
        agent = load_inductive_ppo('model_weights/inductive_simple.pt')
        agent.set_deterministic(True, num_nodes=args.num_nodes)
        fname = 'ind_results.txt'

    # Self attention
    if args.self_attn:
        agent = load_inductive_ppo('model_weights/inductive_self-attn.pt')
        agent.set_deterministic(True)
        fname = 'self-attn_results.txt'
    
    # Cardiff
    if args.cardiff:
        agent = MainAgent()
        wrap = wrap_cardiff
        fname = 'cardiff.txt'

    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    path = args.scenario # 'scenarios/wolk_et_al/scenario3.yaml'
    with open(fname, 'a') as f:
        f.write(path + '\n')

    
    all_r = []
    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    for num_steps in [30, 50, 100]:
        for red_agent in [B_lineAgent, RedMeanderAgent]: # , SleepAgent]:
            total_reward = Parallel(n_jobs=MAX_EPS, prefer='processes')(
                delayed(generate_episode)(agent, red_agent) for _ in range(MAX_EPS)
            )

            print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            with open(fname, 'a') as f:
                f.write(f'{red_agent.__name__},{num_steps},{mean(total_reward)},{stdev(total_reward)}\n')

            all_r.append(mean(total_reward))

    with open(fname, 'a') as f: 
        f.write(f"Total,{sum(all_r)}\n\n")