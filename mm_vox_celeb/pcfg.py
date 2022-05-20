import random
import numpy as np
import torch
import pdb

st = pdb.set_trace

random.seed(0)
np.random.seed(0)

ATTR = [
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
ATTR_NP = np.array(ATTR)
NAME = [
    a.replace('No_', '').replace('Wearing_', '').replace('_', ' ').lower()
    for a in ATTR
]
NAME[0] = '5 o\'clock shadow'
NAME = np.array(NAME)

GET_NAME = {a: NAME[i] for i, a in enumerate(ATTR)}

ATTR_VERB = {
    '5_o_Clock_Shadow': 'has',
    'Arched_Eyebrows': 'has',
    'Attractive': 'is',
    'Bags_Under_Eyes': 'has',
    'Bald': 'is',
    'Bangs': 'has',
    'Big_Lips': 'has',
    'Big_Nose': 'has',
    'Black_Hair': 'has',
    'Blond_Hair': 'has',
    'Blurry': 'is',
    'Brown_Hair': 'has',
    'Bushy_Eyebrows': 'has',
    'Chubby': 'is',
    'Double_Chin': 'has',
    'Eyeglasses': 'wear',
    'Goatee': 'wear',
    'Gray_Hair': 'has',
    'Heavy_Makeup': 'has',
    'High_Cheekbones': 'has',
    'Male': 'is',
    'Mouth_Slightly_Open': 'na',
    'Mustache': 'has',
    'Narrow_Eyes': 'has',
    'No_Beard': 'has',
    'Oval_Face': 'has',
    'Pale_Skin': 'has',
    'Pointy_Nose': 'has',
    'Receding_Hairline': 'has',
    'Rosy_Cheeks': 'has',
    'Sideburns': 'has',
    'Smiling': 'is',
    'Straight_Hair': 'has',
    'Wavy_Hair': 'has',
    'Wearing_Earrings': 'wear',
    'Wearing_Hat': 'wear',
    'Wearing_Lipstick': 'wear',
    'Wearing_Necklace': 'wear',
    'Wearing_Necktie': 'wear',
    'Young': 'is',
}

NEGATE_IDX = [ATTR.index(a) for a in ATTR if a.startswith('No_')]
GENDER_IDX = ATTR.index('Male')


def generate(pred, n=10):
    # pred = pred.squeeze().detach().cpu().numpy()

    # negate
    pred[NEGATE_IDX] = ~pred[NEGATE_IDX]

    # detect attributes
    attr = ATTR_NP[pred]
    random.shuffle(attr)
    wear_list = [GET_NAME[a] for a in attr if ATTR_VERB[a] == 'wear']
    has_list = [GET_NAME[a] for a in attr if ATTR_VERB[a] == 'has']
    is_list = [
        GET_NAME[a] for a in attr if ATTR_VERB[a] == 'is' and a != 'Male'
    ]
    na_attr = [GET_NAME[a] for a in attr if ATTR_VERB[a] == 'na']

    attr_tuple = []
    while sum([len(wear_list), len(has_list), len(is_list)]) > 0:
        p = np.array([len(wear_list), len(has_list), len(is_list)])
        c = np.random.choice([1, 2, 3], p=p / p.sum())
        if c == 1:
            this_attr = ('wear', merge_and_pop(wear_list))
        elif c == 2:
            this_attr = ('has', merge_and_pop(has_list))
        elif c == 3:
            this_attr = ('is', merge_and_pop(is_list))
        attr_tuple.append(this_attr)
    sentences = []
    for i in range(n):
        phrases = []
        first = True
        for attr_t in attr_tuple:
            male = (pred[GENDER_IDX], 0.5) if first else (pred[GENDER_IDX],
                                                          0.85)
            first = False
            phrases.append(generate_phrase(male, attr_t))
        sentence = ' '.join(phrases)
        sentences.append(sentence)
    return sentences


def merge_and_pop(attr_list, p2=0.9, p3=0.85):
    this_attr = [attr_list.pop(0)]
    if len(attr_list) > 0 and random.random() < p2:
        this_attr += [attr_list.pop(0)]
    if len(attr_list) > 0 and random.random() < p3:
        this_attr += [attr_list.pop(0)]
    if len(this_attr) == 1:
        this_attr = this_attr[0]
    elif len(this_attr) == 2:
        this_attr = f"{this_attr[0]} and {this_attr[1]}"
    elif len(this_attr) == 3:
        this_attr = f"{this_attr[0]}, {this_attr[1]} and {this_attr[2]}"
    return this_attr


def generate_phrase(male=(True, 0.5), attr=('is', 'male')):
    PN = 'he' if male[0] else 'she'

    # NP
    # c = np.random.choice([1, 2])
    if random.random() > male[1]:
        # NP -> Det Gender
        cc = np.random.choice([1, 2])
        if cc == 1:
            Det = 'a'
        elif cc == 2:
            Det = 'this'
        if random.random() < 0.75:
            if male[0]:
                Gender = random.choice(['male', 'man'])
            else:
                Gender = random.choice(['female', 'woman'])
        else:
            Gender = 'person'
        NP = f"{Det} {Gender}"
    else:
        # NP -> PN
        NP = PN

    # VP
    if attr[0] == 'is':
        # VP -> NP Are
        IsVerb = 'is'
        IsAttributes = attr[1]
        VP = f"{NP} {IsVerb} {IsAttributes}"
    elif attr[0] == 'has':
        # VP -> NP HaveWith
        HaveVerb = 'has'
        HaveAttributes = attr[1]
        VP = f"{NP} {HaveVerb} {HaveAttributes}"
    elif attr[0] == 'wear':
        # VP -> NP Wearing
        c = np.random.choice([1, 2])
        if c == 1:
            WearVerb = 'wears'
        elif c == 2:
            WearVerb = 'is wearing'
        WearAttributes = attr[1]
        VP = f"{NP} {WearVerb} {WearAttributes}"
    VP = VP[0].upper() + VP[1:] + '.'
    return VP


def mutual_exclusive(pred, subset):
    num = 0
    for a in subset:
        num += pred[ATTR.index(a)]
    if num > 1:
        idx = random.randint(0, len(subset) - 1)
        for i, a in enumerate(subset):
            if i != idx:
                pred[ATTR.index(a)] = False
            else:
                pred[ATTR.index(a)] = True
    return pred


def generate_random_sentences(n_attr=8, n_sent=16):
    sentences = []
    for _ in range(n_sent):
        pred = torch.rand(40) < (n_attr / 40)
        pred = mutual_exclusive(
            pred, ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'])
        pred[GENDER_IDX] = random.random() < 0.5
        pred[ATTR.index('Attractive')] = False
        pred[ATTR.index('Brown_Hair')] = False
        pred[ATTR.index('Mouth_Slightly_Open')] = False
        pred[ATTR.index('Blurry')] = False
        pred[ATTR.index('Smiling')] = False
        pred = pred.detach().cpu().numpy()
        sentence = generate(pred, 1)
        sentences += sentence
    return sentences


if __name__ == "__main__":
    import torch
    pred = torch.rand(40) > 0.5
    labels = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]
    pred = pred.detach().cpu().numpy()
    pred[labels.index('No_Beard')] = False
    print(np.array(labels)[pred])
    sentence = generate(pred, 1)[0]
    print(sentence)
