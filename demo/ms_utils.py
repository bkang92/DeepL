import pickle
import torch
from collections import OrderedDict, defaultdict
actions = {
    1: "drink water",
    2: "eat meal/snack",
    3: "brushing teeth",
    4: "brushing hair",
    5: "drop",
    6: "pickup",
    7: "throw",
    8: "sitting down",
    9: "standing up (from sitting position)",
    10: "clapping",
    11: "reading",
    12: "writing",
    13: "tear up paper",
    14: "wear jacket",
    15: "take off jacket",
    16: "wear a shoe",
    17: "take off a shoe",
    18: "wear on glasses",
    19: "take off glasses",
    20: "put on a hat/cap",
    21: "take off a hat/cap",
    22: "cheer up",
    23: "hand waving",
    24: "kicking something",
    25: "reach into pocket",
    26: "hopping (one foot jumping)",
    27: "jump up",
    28: "make a phone call/answer phone",
    29: "playing with phone/tablet",
    30: "typing on a keyboard",
    31: "pointing to something with finger",
    32: "taking a selfie",
    33: "check time (from watch)",
    34: "rub two hands together",
    35: "nod head/bow",
    36: "shake head",
    37: "wipe face",
    38: "salute",
    39: "put the palms together",
    40: "cross hands in front (say stop)",
    41: "sneeze/cough",
    42: "staggering",
    43: "falling",
    44: "touch head (headache)",
    45: "touch chest (stomachache/heart pain)",
    46: "touch back (backache)",
    47: "touch neck (neckache)",
    48: "nausea or vomiting condition",
    49: "use a fan (with hand or paper)/feeling warm",
    50: "punching/slapping other person",
    51: "kicking other person",
    52: "pushing other person",
    53: "pat on back of other person",
    54: "point finger at the other person",
    55: "hugging other person",
    56: "giving something to other person",
    57: "touch other person's pocket",
    58: "handshaking",
    59: "walking towards each other",
    60: "walking apart from each other",
}
def import_class(name):
    components = name.split('.')
    print(components)
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def start(args):
#     if not args.test_feeder_args['debug']:
#         wf = os.path.join(args.work_dir, 'wrong-samples.txt')
#         rf = os.path.join(args.work_dir, 'right-samples.txt')
#     else:
#         wf = rf = None
#     if args.weights is None:
#         raise ValueError('Please appoint --weights.')

#     print(f'Model :   {args.model}')
#     print(f'Weights : {self.weights}')

#     return eval()

def load_msg3d(args, model):
    
    print(f'Loading weights from {args.weights}')
    if '.pkl' in args.weights:
        with open(args.weights, 'r') as f:
            weights = pickle.load(f)
    else:
        weights = torch.load(args.weights)

    weights = OrderedDict(
        [[k.split('module.')[-1],
            v.cuda()] for k, v in weights.items()])

    for w in args.ignore_weights:
        if weights.pop(w, None) is not None:
            print(f'Sucessfully Remove Weights : {w}')
        else:
            print(f'Can Not Remove Weights: {w}')
    
    try:
        model.load_state_dict(weights)
    except:
        state = model.state_dict()
        diff = list(set(state.keys()).difference(set(weights.keys())))
        print('Can not find these weights: ')
        for d in diff:
            print('    ' + d)
    
        state.update(weights)
        model.load_state_dict(state)

    return model

# def load_param_groups(model):
#     param_groups = defaultdict(list)

#     for name, params in model.named_parameters():
#         param_groups['other']append(params)
    
#     optim_param_groups = {
#             'other': {'params': param_groups['other']}
#         }
