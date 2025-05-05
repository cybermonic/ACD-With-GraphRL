import torch 

def combine_subgraphs(states):
    xs,eis = zip(*states)
    
    # ei we need to update each node idx to be
    # ei[i] += len(ei[i-1])
    offset=0
    new_eis=[]
    for i in range(len(eis)):
        new_eis.append(eis[i]+offset)
        offset += xs[i].size(0)

    # X is easy, just cat
    xs = torch.cat(xs, dim=0)
    eis = torch.cat(new_eis, dim=1)

    return xs,eis