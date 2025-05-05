from flask import Flask, request, render_template
import subprocess
import inspect
import time
from statistics import mean, stdev
import sys

import torch 
from tqdm import tqdm 

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent

from models.ppo import load_ppo
from models.inductive_ppo import load_inductive_ppo
from wrapper.graph_wrapper import GraphWrapper

from viz.create_viz_data import create_viz_data
import json
import os

MAX_EPS = 100
GENERATE_VIZ_DATA = True
agent_name = 'Blue'

NUM_NODES = {
	'wolk_et_al/scenario3.yaml': 13,
	'wolk_et_al/scenario4.yaml': 13,
	'wolk_et_al/scenario5.yaml': 13,
	'cage2.yaml': 13,
	'extra_enterprise.yaml': 14,
	'extra_user.yaml': 14,
	'max_users.yaml': 21,
	'min_users.yaml': 10,
	'minus_enterprise.yaml': 12,
	'minus_user.yaml': 12
}




def wrap(env):
    return GraphWrapper('Blue', env)


# Change this line to load your agent
#args,weights = torch.load('tracking.pt')['sd']
#agent = EGreedyPolicy(0.0, SimpleGNN, qnet_args=args)
#agent.dqn.load_state_dict(weights)
#agent.dqn.eval()

mode="live"
current_historic_step = 0
current_historic_agent = "bline"

# agent = load_ppo('model_weights/ppo_both-100.pt', shared=False)
# agent.set_deterministic(True)
blue_agent_name = 'transductive'
blue_agent = load_ppo('model_weights/noninductive.pt')
blue_agent.set_deterministic(True)

environment_file='cage2.yaml'

path = str(inspect.getfile(CybORG))
path = path[:-10] + f'/Shared/Scenarios/Scenario2.yaml'
path = 'scenarios/' + environment_file

red_agent = B_lineAgent
cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
wrapped_cyborg = wrap(cyborg)

# initial observation
observation = wrapped_cyborg.reset()

def generate_historic():
    red_agent = B_lineAgent
    cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
    wrapped_cyborg = wrap(cyborg)
    for step in tqdm(range(1000), desc='BLine'):
        if step % 100 == 0: 
            observation = wrapped_cyborg.reset()

        os.makedirs('vizdata/bline/%d'%step, exist_ok=True)
        try:
            action = blue_agent.get_action(observation)
        except:
            print("ERROR getting blue agent action")
            action = None
        (feats,edges,names), rew, done, info, succ = wrapped_cyborg.step(action, include_names=True, include_success=True)
        observation = (feats,edges)

        red = cyborg.get_last_action('Red')
        red_suc = 'Success' if str(cyborg.get_observation('Red')['success']) == 'TRUE' else 'Failed'
        if hasattr(red, 'ip_address'):
            red_str = str(red) + f' ({wrapped_cyborg.graph.ip_map.get(red.ip_address, "UNK. IP")})'
        else:
            red_str = str(red)
        red_str += f':{red_suc}'

        succ = 'Success' if succ == 'TRUE' else 'Failed'
        a = (str(cyborg.get_last_action('Blue'))+':'+succ, red_str)
        
        # Features from ObservationGraph
        feat_map = wrapped_cyborg.graph.FEAT_MAP
        # Other features
        blueTableFeats = ['is scanned', 'is exploited', 'admin compromise', 'user compromise']
        feat_map = feat_map + blueTableFeats

        create_viz_data(names, edges, feats, feat_map, a, 'vizdata/bline/%d/viz.json'%step, rew)  
    
    red_agent = RedMeanderAgent
    cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
    wrapped_cyborg = wrap(cyborg)
    for step in tqdm(range(1000), desc='Meander'):
        if step % 100 == 0:
            observation = wrapped_cyborg.reset()

        os.makedirs('vizdata/meander/%d'%step, exist_ok=True)
        action = blue_agent.get_action(observation)
        (feats,edges,names), rew, done, info, succ = wrapped_cyborg.step(action, include_names=True, include_success=True)
        observation = (feats,edges)

        red = cyborg.get_last_action('Red')
        red_suc = 'Success' if str(cyborg.get_observation('Red')['success']) == 'TRUE' else 'Failed'
        if hasattr(red, 'ip_address'):
            red_str = str(red) + f' ({wrapped_cyborg.graph.ip_map.get(red.ip_address, "UNK. IP")})'
        else:
            red_str = str(red)
        red_str += f':{red_suc}'

        succ = 'Success' if succ == 'TRUE' else 'Failed'
        a = (str(cyborg.get_last_action('Blue'))+':'+succ, red_str)
        
        # Features from ObservationGraph
        feat_map = wrapped_cyborg.graph.FEAT_MAP
        # Other features
        blueTableFeats = ['is scanned', 'is exploited', 'admin compromise', 'user compromise']
        feat_map = feat_map + blueTableFeats

        create_viz_data(names, edges, feats, feat_map, a, 'vizdata/meander/%d/viz.json'%step, rew)  


def change_to_bline():
    global red_agent
    global cyborg
    global wrapped_cyborg
    global observation
    global current_historic_agent

    current_historic_agent = "bline"

    red_agent = B_lineAgent
    cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
    wrapped_cyborg = wrap(cyborg)

    observation = wrapped_cyborg.reset()

def change_to_meander():
    global red_agent
    global cyborg
    global wrapped_cyborg
    global observation
    global current_historic_agent

    current_historic_agent = "meander"

    red_agent = RedMeanderAgent
    cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
    wrapped_cyborg = wrap(cyborg)

    observation = wrapped_cyborg.reset()

def change_to_blue_agent(new_blue_agent_name):
    global blue_agent
    global blue_agent_name
    global environment_file
    global NUM_NODES

    blue_agent_name = new_blue_agent_name
    # transductive, inductiveSimple, inductiveSelfAttn
    print("Changing blue agent to: %s" % blue_agent_name)
    if blue_agent_name == 'transductive':
        blue_agent = load_ppo('model_weights/noninductive.pt')
        blue_agent.set_deterministic(True)
    elif blue_agent_name== 'inductiveSimple':
        blue_agent = load_inductive_ppo('model_weights/inductive_simple.pt',
            num_nodes=NUM_NODES[environment_file], 
            naive=True
        ) 
        # blue_agent = load_inductive_ppo('model_weights/inductive_simple.pt')
        blue_agent.set_deterministic(True)
    elif blue_agent_name == 'inductiveSelfAttn':
        blue_agent = load_inductive_ppo('model_weights/inductive_self-attn.pt',
            num_nodes=NUM_NODES[environment_file], 
            naive=False
        ) 
        #blue_agent = load_inductive_ppo('model_weights/inductive_self-attn.pt')
        blue_agent.set_deterministic(True)
    else:
        print("ERROR: Unknown blue agent: %s" % blue_agent_name)

def change_to_environment(envfile):
    global red_agent
    global blue_agent_name
    global cyborg
    global wrapped_cyborg
    global observation
    global current_historic_agent
    global environment_file

    environment_file = envfile
    print("Changing environment to: %s" % environment_file)
    path = 'scenarios/' + environment_file
    cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
    wrapped_cyborg = wrap(cyborg)
    observation = wrapped_cyborg.reset()

    # also need to update blue agent
    # just reload current blue agent
    change_to_blue_agent(blue_agent_name)


def change_to_live():
    global mode
    mode="live"

def change_to_historic():
    global mode
    mode="historic"


def reset_environment():
    global observation, current_historic_step
    observation = wrapped_cyborg.reset()
    current_historic_step = 0
    return {"nodes":[], "links":[], "action":""}

def next_step():
    global observation
    global cyborg

    try:
        action = blue_agent.get_action(observation)
    except:
        print("ERROR getting blue agent action")
        action = None
  
    (feats,edges,names), rew, done, info, succ = wrapped_cyborg.step(action, include_names=True, include_success=True)
    observation = (feats,edges)

    # TODO Return this and display in UI?
    rewards = cyborg.get_rewards()
    print(rewards)

    red = cyborg.get_last_action('Red')
    red_suc = 'Success' if str(cyborg.get_observation('Red')['success']) == 'TRUE' else 'Failed'
    if hasattr(red, 'ip_address'):
        red_str = str(red) + f' ({wrapped_cyborg.graph.ip_map.get(red.ip_address, "UNK. IP")})'
    else:
        red_str = str(red)
    red_str += f':{red_suc}'

    succ = 'Success' if succ == 'TRUE' else 'Failed'
    a = (str(cyborg.get_last_action('Blue')) + ':' + succ, red_str)
    
    # Features from ObservationGraph
    feat_map = wrapped_cyborg.graph.FEAT_MAP
    # Other features
    blueTableFeats = ['is scanned', 'is exploited', 'admin compromise', 'user compromise']
    feat_map = feat_map + blueTableFeats

    data = create_viz_data(names, edges, feats, feat_map, a, None, rewards)  
    return data

def next_historic_step(jumpStep = None):
    print("opening historic data!")
    global current_historic_agent, current_historic_step

    if jumpStep:
        current_historic_step = jumpStep

    with open('./vizdata/%s/%d/viz.json'%(current_historic_agent, current_historic_step)) as fp:
        data = json.load(fp)
        current_historic_step+=1
        return data


app = Flask(__name__)

@app.route("/", methods = ['GET','POST'])
def index():
    return render_template('viz.html')

@app.get('/api/episode')
def get_episode():
    global observation

    observation = wrapped_cyborg.reset()
    steps = [next_step() for _ in range(30)]
    wrapped_cyborg.end_episode()
    return steps 


@app.get('/api/step')
def get_step():
    global mode

    if mode == "live":
        return next_step()
    else :
        return next_historic_step()

@app.get('/api/step/<jumpstep>')
def get_jump_step(jumpstep):
    global mode

    print(mode)

    if mode == "live":
        return next_step()
    else :
        return next_historic_step(int(jumpstep))
    

@app.get('/api/reset')
def reset():
    return reset_environment()

@app.get('/api/redagent/bline')
def bline():
    change_to_bline()
    return "OK"

@app.get('/api/redagent/meander')
def meander():
    change_to_meander()
    return "OK"

@app.get('/api/environment/<envfile>')
def environemnt(envfile):
    change_to_environment(envfile)
    return "OK"

@app.get('/api/blueagent/<blueAgentName>')
def blueAgent(blueAgentName):
    change_to_blue_agent(blueAgentName)
    return "OK"


@app.get('/api/mode/live')
def live():
    change_to_live()
    return "OK"

@app.get('/api/mode/historic')
def historic():
    change_to_historic()
    return "OK"




if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--generate_historic':
        print("Generating historic data!")
        generate_historic()
    else:
        app.run(port=5252)