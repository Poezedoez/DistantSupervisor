import click
import random

SEED = 25

def annotate(dataset, types, output_path='annotated_dataset.json'):
    entities = []
    relations = []
    annotated_dataset = []

    sequences_annotated = 0
    sequences_discarded = 0

    actions = {
        "redo": _redo,
        "back": _redo,
        "next": _next,
    }



    # sequences = range(0, len(dataset))
    random.Random(SEED).shuffle(dataset)

    print("Hello, welcome to the labeling program")
    for sequence in dataset:
        print(' '.join(sequence["tokens"]))
        q = "Is this is proper natural language sentence? (yes/no)"
        proper_sentence = _ask_proper_sentence(q)
        if not proper_sentence:
            print("skipping sentence")
            sequences_discarded += 1
            print()
            continue
        
        _action_proper_sentence(a)

        print("Please label entities. If there are no more entities in the sentence, type 'next'")
        q = "Entity start:"
        start, action = _ask_position(q, actions.keys)
        if action:
            actions.get(action)()
        
        q = "Entity end:"
        end, action = _ask_position(q, actions.keys)
        
def _print_numbered_tokens(tokens):
    for i, token in enumerate(tokens):
            print("[{}] {} ".format(i, token), end='')

def _print_status(entities, relations):
    pass

def _ask_yes_no(q, actions={}):
    yes_no = {
        "yes": True,
        "y": True,
        "no": False,
        "n": False
    }

    input_ = str(input(q)).lower()
    answer = yes_no.get(input_, None)
    while answer==None:
        print("Cannot understand, tell me {}".format(yes_no.keys))
        print("Or give me an action {}".format(actions))
        a = str(input(q)).lower()
        if a in actions:
            return None, a
        answer = yes_no.get(a, None)

    return answer, None

def _ask_position(q, actions={}):
    def _ask():
        a = input(q)
        try:
            a = int(a)
        except:
            a = a.lower()
        return a

    a = _ask()
    if a in actions:
        return None, a

    while not isinstance(a, int):
        print("Cannot understand, give me a position (int)")
        print("Or give me an action {}".format(actions))
        a = _ask()
        if a in actions:
            return None, a

    return a, None