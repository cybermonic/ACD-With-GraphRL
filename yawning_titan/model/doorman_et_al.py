import torch 
from torch import nn 
from torch import functional as F 
from torch_geometric.nn import MessagePassing 

'''
https://proceedings.mlr.press/v198/doorman22a/doorman22a.pdf
'''

class DoormanAgent(nn.Module): 
    def __init__(self, in_dim, hidden_dim, out_dim, layers): 
        super().__init__() 
        self.args = (in_dim, hidden_dim, out_dim, layers)

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), 
            nn.ReLU()
        )

        self.aggr_layers = nn.ParameterList(
            [
                nn.Linear(hidden_dim, hidden_dim)
                for _ in range(layers)
            ]
        )

        self.layers = nn.ParameterList(
            [
                nn.Linear(hidden_dim, hidden_dim)
                for _ in range(layers)
            ]
        )

        self.final = nn.Sequential(
            nn.BatchNorm1d(hidden_dim*2),
            nn.Linear(hidden_dim*2, out_dim)
        )

        self.mp = MessagePassing(aggr='sum')


    def forward(self, x,ei, n_nodes): 
        x = self.proj(x)

        for i in range(len(self.layers)): 
            x_i = self.layers[i](x)
            u_i = self.aggr_layers[i](x)
            u = self.mp.propagate(ei, size=None, x=u_i)
            
            x = torch.relu(x_i + u)

        u_g = u.unfold(0, size=n_nodes, step=n_nodes)
        u_g = u_g.sum(dim=-1)
        u_g = u_g.repeat_interleave(n_nodes,0)

        out = torch.cat([x,u_g], dim=1)
        out = self.final(out).reshape(x.size(0) // n_nodes, -1)
        return out 
    
    def save(self, fname): 
        torch.save((self.args, self.state_dict()), fname)

def load_doorman(fname): 
    args, sd = torch.load(fname) 
    model = DoormanAgent(*args)
    model.load_state_dict(sd) 

    return model 
    
from math import ceil 

import torch 
from torch import nn 

from model.doorman_et_al import DoormanAgent

class DQN_Buffer: 
    def __init__(self, num_nodes, bs=2048, max_size=12_000):
        self.bs = bs  
        self.num_nodes = num_nodes 
        self.max_size = max_size
        self.clear() 

    def clear(self): 
        self.s = []
        self.a = []
        self.r = []
        self.s_prime = []

    def push(self, s,a,r,s_prime):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.s_prime.append(s_prime)

    def update(self, other): 
        self.s += other.s
        self.a += other.a
        self.r += other.r 
        self.s_prime += other.s_prime 

        self.s = self.s[-self.max_size:]
        self.a = self.a[-self.max_size:]
        self.r = self.r[-self.max_size:]
        self.s_prime = self.s_prime[-self.max_size:]

    def sample(self): 
        idx = torch.randperm(len(self.s))

        for i in range(ceil(len(self.s) / self.bs)): 
            batch = idx[i*self.bs : (i+1)*self.bs]

            s = [self.s[b] for b in batch]
            a = [self.a[b] for b in batch]
            r = [self.r[b] for b in batch]
            s_prime = [self.s_prime[b] for b in batch]

            s = self.combine_states(s) 
            a = torch.tensor(a)
            r = torch.tensor(r) 
            s_prime = self.combine_states(s_prime)

            yield s,a,r,s_prime 

    def combine_states(self, states): 
        xs = []
        eis = []
        for i,s in enumerate(states): 
            xs.append(s[0])
            eis.append(s[1] + self.num_nodes*i)

        return torch.cat(xs, dim=0), torch.cat(eis, dim=1)
    
    def loss(self, value_net, target_net, gamma, opt): 
        opt.zero_grad()
        criterion = nn.SmoothL1Loss()

        for i,(s,a,r,s_prime) in enumerate(self.sample()): 
            q = value_net(*s, self.num_nodes)
            v = q[torch.arange(q.size(0)), a]

            with torch.no_grad(): 
                tgt = target_net(*s_prime, self.num_nodes).max(dim=1).values
                tgt = r + gamma*tgt 

            loss = criterion(v, tgt)
            loss.backward()
            opt.step()

        

    def combine(self, others): 
        self.s = sum([o.s for o in others], [])
        self.a = sum([o.a for o in others], [])
        self.r = sum([o.r for o in others], [])
        self.s_prime = sum([o.s_prime for o in others], [])

        return self 