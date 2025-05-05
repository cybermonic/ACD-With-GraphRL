import json
import os

def create_viz_data(names, edges, feats, feat_map,action, fname, r):
    res = {"nodes":[], "links":[], "action":str(action)}
    for idx,n in enumerate(names):
        label = "<div>"
        
        if feats[idx][0]:
            label += "<p>SystemNode</p>"
            group=1
        elif feats[idx][1]:
            label += "<p>SubnetNode</p>"
            group=2
        elif feats[idx][2]:
            label += "<p>ConnectionNode</p>"
            group=3
        elif feats[idx][3]:
            label += "<p>FileNode</p>"
            group=4
        label += "<p>%s</p>"%n

        label += "</hr>"
        label += "<p>Features:</p>"

        user_compromise=False
        admin_compromise=False
        crown_jewel=False
        
        for idx2, val in enumerate(feats[idx]):
            if idx2 < 4: # node type features from above
                continue
            if val > 0:
                if type(feat_map[idx2]) == tuple:
                    label += "<p>%s:%f</p>" % (str(feat_map[idx2][0]), float(val))
                else:
                    label += "<p>%s:%f</p>" % (str(feat_map[idx2]), float(val))

                if str(feat_map[idx2]) == "user compromise":
                    user_compromise=True
                if str(feat_map[idx2]) == "admin compromise":
                    admin_compromise=True
                if str(feat_map[idx2]) == "crown_jewel":
                    crown_jewel=True
        label += "</div>"

        res['nodes'].append({"id":n, "label":label, 'group': group, 'user_compromise':user_compromise, 'admin_compromise':admin_compromise, 'crown_jewel':crown_jewel})
    for idx in range(len(edges[0])):
        res['links'].append({"source":names[edges[0][idx]], "target":names[edges[1][idx]]})
    
    res['reward'] = r
    if fname:
        with open(fname, 'w') as f:
            json.dump(res, f)
    else:
        return res