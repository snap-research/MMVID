import argparse
from pathlib import Path
import os
import shutil
import sys

curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(curr_path).parent))
from tqdm import tqdm
import pickle
import numpy as np

import pdb

st = pdb.set_trace

parser = argparse.ArgumentParser()

parser.add_argument('--num_videos', type=int, default=100)
parser.add_argument('--num_frames', type=int, default=10)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--threshold_quantile', type=float, default=0.5)
parser.add_argument('--quantile', action='store_true')
parser.add_argument('--label_dir',
                    type=str,
                    default='../data/mmvoxceleb/label')
parser.add_argument('--text_dir', type=str, default='../data/mmvoxceleb/text')

args = parser.parse_args()

classnames = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
    'Wearing_Necktie', 'Young'
]
classnames = [c.lower() for c in classnames]
class2index = {c: i for i, c in enumerate(classnames)}

label_root = args.label_dir
os.makedirs(label_root, exist_ok=True)

from pcfg import generate

with open('face-attributes-2_parse_json.txt', 'r') as f:
    lines = [l.rstrip() for l in f.readlines()]

for line in tqdm(lines):
    name = line.split(',')[0]

    prediction = np.zeros(40) > 1
    for classname in line.split(',')[1:]:
        classname_ = classname.lower().replace(' ', '_')
        i = class2index[classname_]
        prediction[i] = True

    label = ','.join(['1' if p else '0' for p in prediction])
    with open(os.path.join(label_root, name + '.txt'), 'w') as f:
        f.write(label)
