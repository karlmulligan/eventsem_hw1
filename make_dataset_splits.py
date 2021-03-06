#!/usr/bin/env python
# coding: utf-8

# # EN.601.769 Assignment 1: Semantic Role Labeling
# 
# #### Karl Mulligan 
# 
# February 28th 2021 

# ## Part 1: Data Collection

# In[1]:


import os
from collections import defaultdict
import json


# In[2]:


from decomp import UDSCorpus

splits = {"train" : UDSCorpus(split="train"),
          "dev" : UDSCorpus(split="dev"),
          "test" : UDSCorpus(split="test")}


# In[13]:


# SEMANTIC ROLES
ROLES = {"agent": lambda p: ((p["volition"]["value"] > 0) or (p["instigation"]["value"] > 0)) and (p["existed_before"]["value"] > 0),
         "patient": lambda p: (p["change_of_state"]["value"] > 0) and (p["existed_after"]["value"] > 0),
         "theme": lambda p: ((p["change_of_possession"]["value"] > 0) or (p["change_of_location"]["value"])) and (p["instigation"]["value"] < 0),
         "recipient": lambda p: ((p["awareness"]["value"] > 0) and p["sentient"]["value"] > 0) and (p["existed_before"]["value"] > 0) and (p["volition"]["value"] < 0),
         "experiencer": lambda p: ((p["awareness"]["value"] > 0) and (p["change_of_state"]["value"] > 0) and (p["sentient"]["value"] > 0))                                   
        }


# In[14]:


def parse_edge_name(edge):
    
    def parse_node_name(node):
        typ, idx = node.split('-')[-2:] #['arg', 'x'] or ['pred', 'x']
        return typ, idx
    
    predicate_head_idx = None
    argument_head_idx = None
    
    typ, idx = parse_node_name(edge[0])
    if typ == "pred":
        predicate_head_idx = idx
    elif typ == "arg":
        argument_head_idx = idx
    else:
        raise ValueError(f"{edge[0]}, {typ}, {idx}")

    typ, idx = parse_node_name(edge[1])
    if typ == "pred":
        predicate_head_idx = idx
    elif typ == "arg":
        argument_head_idx = idx
    else:
        raise ValueError(f"{edge[0]}, {typ}, {idx}")
        
    assert (predicate_head_idx != None) and (argument_head_idx != None)
    
    return predicate_head_idx, argument_head_idx


# In[15]:


def process_split(split, role, criteria):
    pos_counter = 0
    neg_counter = 0
    dataset = {}
    for graphid, graph in split.items():
        tokens = tuple(graph.sentence.split())
        
        semantics_edges = graph.semantics_edges()
        for edge, properties in semantics_edges.items():
            if "protoroles" in properties:
                try:
                    predicate_head_idx, argument_head_idx = parse_edge_name(edge)
                except:
                    import pdb; pdb.set_trace()

                try:
                    role_applies = criteria(properties["protoroles"])
                    if role_applies:
                        label = "positive"
                        pos_counter += 1
                    else:
                        label = "negative"
                        neg_counter += 1

                    item_id = "|".join([graphid, predicate_head_idx, argument_head_idx])
                    dataset[item_id] = {"graphid": graphid,
                                        "tokens": tokens,
                                        "predicate_head_idx": predicate_head_idx,
                                        "argument_head_idx": argument_head_idx,
                                        "label": label
                                       }
                except: 
                    continue
    print(f"{pos_counter} positive examples of {role}.")
    print(f"{neg_counter} negative examples of {role}.")
    return dataset


# In[16]:


DATA_PATH = "/Users/karlmulligan/Documents/jhu/event_sem/decomp/data"


# In[17]:


datasets = defaultdict(dict)
for role, criteria in ROLES.items():
    for split in ['train', 'dev', 'test']:
        datasets[role][split] = process_split(splits[split], role, criteria)

for role in datasets.keys():
    role_path = os.path.join(DATA_PATH, role)
    if not os.path.exists(role_path):
        os.mkdir(role_path)
    
    for split in datasets[role].keys():
        with open(os.path.join(role_path, f"{split}.json"), "w") as f:
            json.dump(datasets[role][split], f, indent=2)


