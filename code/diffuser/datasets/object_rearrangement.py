import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from collections import namedtuple
import copy


x_range = [0,5]
y_range = [0,5]
s_range = [0.3,1]
theta_range = [0,2*np.pi] #radians
circle = [1,0,0]
triangle = [0,1,0]
sqaure = [0,0,1]
shapes_text = ['circle', 'triangle', 'square']

def get_task(cond):
    SHAPE_TEXT = ['circle', 'triangle', 'square']
    RELATIONS = ['right of', 'above']
    conds_map_vals = [SHAPE_TEXT[0]+' '+RELATIONS[0]+' '+SHAPE_TEXT[1], SHAPE_TEXT[0]+' '+RELATIONS[0]+' '+SHAPE_TEXT[2],\
                      SHAPE_TEXT[1]+' '+RELATIONS[0]+' '+SHAPE_TEXT[0], SHAPE_TEXT[1]+' '+RELATIONS[0]+' '+SHAPE_TEXT[2],\
                      SHAPE_TEXT[2]+' '+RELATIONS[0]+' '+SHAPE_TEXT[0], SHAPE_TEXT[2]+' '+RELATIONS[0]+' '+SHAPE_TEXT[1],\
                      SHAPE_TEXT[0]+' '+RELATIONS[1]+' '+SHAPE_TEXT[1], SHAPE_TEXT[0]+' '+RELATIONS[1]+' '+SHAPE_TEXT[2],\
                      SHAPE_TEXT[1]+' '+RELATIONS[1]+' '+SHAPE_TEXT[0], SHAPE_TEXT[1]+' '+RELATIONS[1]+' '+SHAPE_TEXT[2],\
                      SHAPE_TEXT[2]+' '+RELATIONS[1]+' '+SHAPE_TEXT[0], SHAPE_TEXT[2]+' '+RELATIONS[1]+' '+SHAPE_TEXT[1],\
                      ]
    conds_map_keys = np.array([-0.06979119, -0.06053966, -0.06120787, -0.05390831, -0.04821796,
                               -0.06482547, -0.07765396, -0.06771035, -0.06354303, -0.06717855,
                               -0.05195947, -0.0559589 ])
    cond_text = conds_map_vals[np.abs(conds_map_keys-float(cond[0])).argmin()] 
    return cond_text

def circles_intersect(x1,y1,r1,x2,y2,r2):
    d = np.sqrt((x1-x2)**2+(y1-y2)**2)
    return d <= r1 + r2

def isRightOf(shape1, shape2):
    """shape1 right of shape2"""
    return shape1[0] >= shape2[0] and (np.abs(shape1[1] - shape2[1]) <= min(shape1[2], shape2[2]))

def isAbove(shape1, shape2):
    """shape1 above shape2"""
    return shape1[1] >= shape2[1] and (np.abs(shape1[0] - shape2[0]) <= min(shape1[2], shape2[2]))

def get_shape(shapes, shape_type):
    #order circle, triangle, square
    shape_type_idx = 3
    if shape_type == 'circle':
        shape = shapes[:7]
        shape_type = np.argmax(shape[-shape_type_idx:]) #argmax last 3 entries
        if shape_type == 0: return shape #confirm one-hot matches order
    elif shape_type == 'triangle':
        shape = shapes[7:14]
        shape_type = np.argmax(shape[-shape_type_idx:]) 
        if shape_type == 1: return shape        
    elif shape_type == 'square':
        shape = shapes[14:]
        shape_type = np.argmax(shape[-shape_type_idx:])
        if shape_type == 2: return shape        
    return None

def create_shape_from_cond(shape1_type, shape2_type, cond_relation):
    """create shape vector s.t. shape1_type cond_relation shape2_type
       verify 3rd shape does not have a defined realation with the other shapes"""
    shape3_type = list(set(shapes_text) - set([shape1_type, shape2_type]))[0]     
    sampled_shapes = {}
    #shape 2
    #sample vars [x,y,size,angle]
    s = np.random.uniform(s_range[0],s_range[1])
    th = np.random.uniform(theta_range[0],theta_range[1])    
    if cond_relation == 'right of':
        shape2 = [np.random.uniform(x_range[0],x_range[1]-s_range[1]),
                  np.random.uniform(y_range[0],y_range[1]),
                  s,
                  th,
                ]
    elif cond_relation == 'above':
        shape2 = [np.random.uniform(x_range[0],x_range[1]),
                  np.random.uniform(y_range[0],y_range[1]-s_range[1]),
                  s,
                  th,
                ]
    sampled_shapes[shape2_type] = shape2
    #shape 1
    s = np.random.uniform(s_range[0],s_range[1])
    th = np.random.uniform(theta_range[0],theta_range[1])
    rel_s = min(s, shape2[3])
    if cond_relation == 'right of':
        shape1 = [np.random.uniform(shape2[0],x_range[1]),
                  np.random.uniform(max(shape2[1]-rel_s, y_range[0]), min(shape2[1]+rel_s, y_range[1])),
                  s,
                  th,
                ]
    elif cond_relation == 'above':
        shape1 = [np.random.uniform(max(shape2[0]-rel_s, x_range[0]), min(shape2[0]+rel_s, x_range[1])),
                  np.random.uniform(shape2[1],y_range[1]),
                  s,
                  th,
                ]
    if circles_intersect(shape1[0],shape1[1],shape1[2],shape2[0],shape2[1],shape2[2]):
        return None
    sampled_shapes[shape1_type] = shape1
    #shape 3
    shape3 = [np.random.uniform(x_range[0],x_range[1]),
              np.random.uniform(y_range[0],y_range[1]),
              np.random.uniform(s_range[0],s_range[1]),
              np.random.uniform(theta_range[0],theta_range[1])
            ]
    sampled_shapes[shape3_type] = shape3
    #no intersection shape 3
    if circles_intersect(shape1[0],shape1[1],shape1[2],shape3[0],shape3[1],shape3[2]):
        return None
    if circles_intersect(shape2[0],shape2[1],shape2[2],shape3[0],shape3[1],shape3[2]):
        return None
    #shape 3 not introducing more relations
    if isRightOf(shape1, shape3) or isRightOf(shape3, shape1) or isAbove(shape1, shape3) or isAbove(shape3, shape1):
        return None
    if isRightOf(shape2, shape3) or isRightOf(shape3, shape2) or isAbove(shape2, shape3) or isAbove(shape3, shape2):
        return None
    
    sampled_shapes['circle'][-1] = 0 #circle no angle
    x_circ, x_tri, x_sq = sampled_shapes['circle'], sampled_shapes['triangle'], sampled_shapes['square']
    return np.array(x_circ + circle + x_tri + triangle + x_sq + sqaure)    

def generate_shape(shape2, relation):
    s = np.random.uniform(s_range[0],s_range[1])
    th = np.random.uniform(theta_range[0],theta_range[1])
    rel_s = min(s, shape2[3])
    if relation == 'right of':
        shape1 = [np.random.uniform(shape2[0],x_range[1]),
                  np.random.uniform(max(shape2[1]-rel_s, y_range[0]), min(shape2[1]+rel_s, y_range[1])),
                  s,
                  th,
                ]
    elif relation == 'above':
        shape1 = [np.random.uniform(max(shape2[0]-rel_s, x_range[0]), min(shape2[0]+rel_s, x_range[1])),
                  np.random.uniform(shape2[1],y_range[1]),
                  s,
                  th,
                ]   
    return shape1

def create_shape_from_conds(shape1_type, shape2_type, shape3_type, cond_relation1, cond_relation2):
    """create shape vector s.t. (shape1_type cond_relation1 shape2_type) and (shape3_type cond_relation2 shape2_type)"""
    circle = [1,0,0]
    triangle = [0,1,0]
    sqaure = [0,0,1]
    shapes_text = ['circle', 'triangle', 'square']
    sampled_shapes = {}  #sample vars [x,y,size,angle]
    #shape 2   
    s = np.random.uniform(s_range[0],s_range[1])
    th = np.random.uniform(theta_range[0],theta_range[1])  
    shape2 = [np.random.uniform(x_range[0],x_range[1]-s_range[1]),
              np.random.uniform(y_range[0],y_range[1]-s_range[1]),
              s,
              th,
            ] #limit to support right of and above relative to shape2
    sampled_shapes[shape2_type] = shape2
    #shape 1
    shape1 = generate_shape(shape2, cond_relation1)
    if circles_intersect(shape1[0],shape1[1],shape1[2],shape2[0],shape2[1],shape2[2]):
        return None
    sampled_shapes[shape1_type] = shape1
    #shape 3
    shape3 = generate_shape(shape2, cond_relation2)
    sampled_shapes[shape3_type] = shape3
    if circles_intersect(shape1[0],shape1[1],shape1[2],shape3[0],shape3[1],shape3[2]):
        return None
    if circles_intersect(shape2[0],shape2[1],shape2[2],shape3[0],shape3[1],shape3[2]):
        return None
    #verify 1&3 have no relation
    if isRightOf(shape1, shape3) or isRightOf(shape3, shape1) or isAbove(shape1, shape3) or isAbove(shape3, shape1):
        return None    
    
    sampled_shapes['circle'][-1] = 0 #circle no angle
    x_circ, x_tri, x_sq = sampled_shapes['circle'], sampled_shapes['triangle'], sampled_shapes['square']
    return np.array(x_circ + circle + x_tri + triangle + x_sq + sqaure)

def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    https://stackoverflow.com/questions/28910718/give-3-points-and-a-plot-circle
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    
    if abs(det) < 1.0e-6:
        return (None, np.inf)
    
    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return (cx, cy), radius

def get_train_acc(all_samples, all_cond_text):
    n_succ = []
    for samp, cond in zip(all_samples, all_cond_text):
        shape1_type, shape2_type = cond.split(' ')[0], cond.split(' ')[-1]
        #confirm shape order and type
        shape1 = get_shape(samp, shape1_type)
        shape2 = get_shape(samp, shape2_type)
        if shape1 is None or shape2 is None: n_succ.append(0); continue
        if 'right of' in cond: n_succ.append(1) if isRightOf(shape1, shape2) else n_succ.append(0)
        elif 'above' in cond: n_succ.append(1) if isAbove(shape1, shape2) else n_succ.append(0)   
    print(f'training acc {np.sum(n_succ)/len(all_samples)}')
    return n_succ 

def isRightOf_eval(shape1, shape2):
    """shape1 right of shape2"""
    return shape1[0] >= shape2[0] and (np.abs(shape1[1] - shape2[1]) <= 2*max(shape1[2], shape2[2]))

def isAbove_eval(shape1, shape2):
    """shape1 above shape2"""
    return shape1[1] >= shape2[1] and (np.abs(shape1[0] - shape2[0]) <= 2*max(shape1[2], shape2[2]))

def get_learned_concept_acc(all_samples, conds_text, dataset_name):
    EPS = 0.3
    X_RANGE = [0,5]
    if 'compos' in dataset_name: # 2 conditions
        n_succ = []
        for samp in all_samples:
            cond1, cond2 = conds_text.split(' \n and ')[0], conds_text.split(' \n and ')[1]
            n_curr_succ = []
            for c in [cond1, cond2]:
                shape1_type, shape2_type = c.split(' ')[0], c.split(' ')[-1]
                shape1 = get_shape(samp, shape1_type)
                shape2 = get_shape(samp, shape2_type)
                if shape1 is None or shape2 is None: n_curr_succ.append(0); continue
                if 'right of' in c: n_curr_succ.append(1) if isRightOf_eval(shape1, shape2) else n_curr_succ.append(0)
                elif 'above' in c: n_curr_succ.append(1) if isAbove_eval(shape1, shape2) else n_curr_succ.append(0)
            n_succ.append(n_curr_succ[0]*n_curr_succ[1])
    elif 'new' in dataset_name: #circle or diagonal
        n_succ = []
        for samp in all_samples:
            if 'circle: ' in conds_text:
                _, r = define_circle(get_shape(samp, 'circle')[:2], get_shape(samp, 'triangle')[:2], get_shape(samp, 'square')[:2])
                n_succ.append(1) if np.abs(r - (X_RANGE[1]/3)) < EPS else n_succ.append(0)
            elif 'diag: ' in conds_text:
                shape1_type = conds_text.split(' above ')[0].split('\n and ')[-1]
                shape2_type = conds_text.split(' above ')[1]
                shape1 = get_shape(samp, shape1_type)
                shape2 = get_shape(samp, shape2_type)
                if shape1 is None or shape2 is None: n_succ.append(0); continue
                EPS = 2*max([shape1[2], shape2[2]]) #max size
                n_succ.append(1) if (shape2[1]<shape1[1]) and (shape2[0]<shape1[0]) and (np.abs((shape2[1]-shape1[1])-(shape2[0]-shape1[0])) < EPS) else n_succ.append(0)
    print(f'acc: {np.sum(n_succ)/len(all_samples)}')

def get_compos_learned_acc(all_samples, conds_text, seen_cond_text):
    n_succ = []
    for samp in all_samples:
        #diag
        shape1_type = conds_text.split(' above ')[0].split('\n and ')[-1]
        shape2_type = conds_text.split(' above ')[1]
        shape1 = get_shape(samp, shape1_type)
        shape2 = get_shape(samp, shape2_type)
        if shape1 is None or shape2 is None: n_succ.append(0); continue
        EPS = 2*max([shape1[2], shape2[2]]) #max size
        is_diag = (shape2[1]<shape1[1]) and (shape2[0]<shape1[0]) and (np.abs((shape2[1]-shape1[1])-(shape2[0]-shape1[0])) < EPS)
        #seen cond
        shape1_type, shape2_type = seen_cond_text.split(' ')[0], seen_cond_text.split(' ')[-1]
        shape1 = get_shape(samp, shape1_type)
        shape2 = get_shape(samp, shape2_type)
        if shape1 is None or shape2 is None: n_succ.append(0); continue
        if 'right of' in seen_cond_text: seen_cond = True if isRightOf_eval(shape1, shape2) else False
        elif 'above' in seen_cond_text: seen_cond = True if isAbove_eval(shape1, shape2) else False
        #
        n_succ.append(1) if is_diag and seen_cond else n_succ.append(0)
    print(f'acc: {np.sum(n_succ)/len(all_samples)}')    

def to_tensor(x, dtype=torch.float, device='cpu'):
    return torch.tensor(x, dtype=dtype, device=device)

Batch = namedtuple('Batch', 'trajectories conditions dummy_cond')


class ObjectRearrangement(Dataset):

    def __init__(self, dataset_path=None, *args, **kwargs):
        with open(dataset_path, "rb") as input_file:
            self.observations, self.conditions, self.dummy_cond = pickle.load(input_file)
        self.original_obs = copy.deepcopy(self.observations)
        self.observation_dim = self.observations.shape[1] # 7*3
        self.action_dim = 0
        self.cond_dim = self.conditions.shape[1] # 768 T5
        self.obs_cond_dim = len(self.observations[0])
        self.mins = self.observations.min(axis=0)
        self.maxs = self.observations.max(axis=0)
        self.normalize()

        print(f'[ TampDataset ] observations: {self.observations.shape}')


    def normalize(self):
        '''
            normalizes to [-1, 1]
        '''
        self.observations = (self.observations - self.mins) / (self.maxs - self.mins + 1e-5) # [ 0, 1 ]
        self.observations = self.observations * 2 - 1 # [ -1, 1 ]


    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ 0, 1 ]
        '''
        assert x.max() <= 1.0 + eps and x.min() >= -1.0 - eps, f'x range: ({x.min():.4f}, {x.max():.4f})'
        mins, maxs = self.mins, self.maxs
        ret = x + 1 #[-1,1]-->[0,2]
        ret /= 2 #[0,2]-->[0,1]
        return ret * (maxs - mins + 1e-5) + mins #[min,max]


    def __len__(self):
        return len(self.observations)


    def __getitem__(self, idx, eps=1e-7):
        observations = self.observations[idx]
        assert observations.max() <= 1.0 + eps and observations.min() >= -1.0 - eps, f'observations range: ({observations.min():.4f}, {observations.max():.4f})' #check normalized
        cond = self.conditions[idx]
        observations = to_tensor(observations)
        batch = Batch(observations, cond, self.dummy_cond)
        return batch
