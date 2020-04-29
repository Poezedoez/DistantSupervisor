import json
import random
import utils
from read import read_ontology_types

SEED = 25

def annotate(dataset_path, types_path, output_path='annotated_dataset.json'):
    types = json.load(open(types_path))
    add_annotation_examples(types)
    dataset = json.load(open(dataset_path))
    annotated_dataset = []

    actions = {
        "done": "go to the next step/sequence",
        "back": "go back to previous annotation",
        "redo": "redo annotations for the whole sequence",
        "help": "show possible types and commands"
    }

    sequences_annotated = 0
    sequences_discarded = 0

    random.Random(SEED).shuffle(dataset)

    print("Hello, welcome to the labeling program.")
    print("-"*50)
    print()

    for sequence in dataset:
        
        # Validate sentence properness
        character_length = sum([len(t) for t in sequence["tokens"]])+len(sequence["tokens"])-1
        print("*"*character_length)
        print(' '.join(sequence["tokens"]))
        print("*"*character_length)
        q = "Is this is proper natural language sentence? (yes/no) \n >>"
        proper_sentence = _ask_yes_no(q)
        print(proper_sentence)
        if not proper_sentence:
            print("skipping sentence")
            sequences_discarded += 1
            print()
            continue
        
        # Annotate sentence
        sequence["entities"], sequence["relations"] = annotate_sequence(sequence, types, actions)
        annotated_dataset.append(sequence)
        sequences_annotated += 1

        # Continue / exit ?
        q = "You annotated {} sentences, continue? (yes/no) \n >>".format(sequences_annotated)
        continuing = _ask_yes_no(q)
        if not continuing:
            _save(annotated_dataset, output_path)
            exit()

def _save(dataset, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f)


def add_annotation_examples(types):
    entity_examples = {
        "application_area": "computer vision, NLP, QA, object boundary detection ...",
        "learning_paradigm": "reinforcement learning, supervised learning, transfer learning ...",
        "dataset": "SQuAD, TREC, CoNLL2003, IMDB-reviews ...",
        "architectural_element": "Convolutional layer, softmax, hidden layer, ReLu ...",
        "software_framework": "PyTorch, TensorFlow, Keras, CAFFE ...",
        "machine_learning_algorithm": "convolutional neural network, k-NN, SVM, Expectation Maximization, BERT ...",
        "hardware": "GeForce GTX 1080 Ti, Intel, TPU v3 ...",
        "metric": "F1-score, AUC, BLEU, ROUGE, average precision, Kappa score ...",
        "technique": "regularization, drop-out, masking, adversarial training ..."
    }

    relation_examples = {
        "hasMetric": "We obtain a new SOTA with a 92.3 |F1-score|[tail] on the |SQuAD|[head] dataset",
        "trainedWith": "|BERT|[head] was trained on a multiple of |Cloud TPU's|[tail]",
        "hasProperty": "We use a |MLP|[head] with only one |hidden layer|[tail]",
        "builtWith": "Our |k-nn language model|[head] is implemented with |Faiss|[tail]",
        "usedFor": "|Convolutional neural networks|[head] are widely used for |computer vision|[tail] tasks",
        "scoresOn": "|ElMo|[head] improves SOTA on all |GLUE|[tail] benchmarks"
    }

    for type_, properties in types["entities"].items():
        properties["example"] = entity_examples.get(type_, "")

    for type_, properties in types["relations"].items():
        properties["example"] = relation_examples.get(type_, "")


def annotate_sequence(sequence, types, actions, skip_relations=False):
    tokens = sequence["tokens"]
    entities = _annotate_entities(sequence, [], actions, types)
    if len(entities) < 2 or skip_relations:
        relations = []
    else:
        relations = _annotate_relations(sequence, entities, [], actions, types)

    return entities, relations
    
def _annotate_entities(sequence, entities, actions={}, types={}):
    def _check_action(action, annotations):
        if action in {"done", "d"}:
            return annotations, True
        if action in {"back", "b"}:
            return _annotate_entities(sequence, annotations, actions, types), True
        if action in {"redo", "r"}:
            return annotate_sequence(sequence, types, actions, True)[0], True
        if action in {"help", "h"}:
            print("commands: ")
            for action, effect in actions.items():
                print("\t {} -> {} ".format(action, effect))
            print()
            print("entity types: ")
            for type_, properties in types["entities"].items():
                print("\t {} -> {} ".format(type_, properties["example"]))
            print()
            return _annotate_entities(sequence, annotations, actions, types), True

        return annotations, False

    print("\nPlease label entities. If there are no more entities in the sentence, type 'done' \n")
    annotated_entities = entities
    tokens = sequence["tokens"]
    stop = False
    while not stop:
        print('"'+' '.join(tokens)+'"')
        _print_numbered_list(tokens)
        print("annotated_entities:", annotated_entities)
        print()

        start, action = _ask_position("Entity start: \n >>", len(tokens), actions.keys())
        annotated_entities, stop = _check_action(action, annotated_entities)

        if stop:
            break

        end, action = _ask_position("Entity end: \n >>", len(tokens), actions.keys())
        annotated_entities, stop = _check_action(action, annotated_entities)

        if stop:
            break

        entity_tokens = ' '.join(tokens[start:end])
        type_, action = _ask_type("Entity type of |{}|: \n >>".format(entity_tokens), 
                                  actions.keys(), types["entities"].keys())
        annotated_entities, stop = _check_action(action, annotated_entities)

        if utils.check_nones([start, end, type_]):
            annotated_entities.append({"start": start, "end": end, "type": type_})

    return annotated_entities

def _annotate_relations(sequence, entities, relations, actions={}, types={}):
    def _check_action(action, annotations):
        if action in {"done", "d"}:
            return annotations, True
        if action in {"back", "b"}:
            return _annotate_relations(sequence, entities, annotations, actions, types), True
        if action in {"redo", "r"}:
            return annotate_sequence(sequence, types, actions)[0], True
        if action in {"help", "h"}:
            print("commands: ")
            for action, effect in actions.items():
                print("\t {} -> {} ".format(action, effect))
            print()
            print("relation types: ")
            for type_, properties in types["relations"].items():
                print("\t {} -> {} ".format(type_, properties["example"]))
            print()
            return _annotate_relations(sequence, entities, annotations, actions, types), True

        return annotations, False

    print("\nPlease label relations. If there are no more relations in the sentence, type 'done' \n")
    annotated_relations = relations
    tokens = sequence["tokens"]
    stop = False
    while not stop:
        print('"'+' '.join(tokens)+'"')
        _print_numbered_list(['_'.join(tokens[e["start"]:e["end"]]) for e in entities])
        print("annotated relations:", annotated_relations)
        print()

        head, action = _ask_position("Relation head: \n >>", len(entities)-1, actions.keys())
        annotated_relations, stop = _check_action(action, annotated_relations)

        if stop:
            break

        tail, action = _ask_position("Relation tail: \n >>", len(entities)-1, actions.keys())
        annotated_relations, stop = _check_action(action, annotated_relations)

        if stop:
            break

        head_tokens = ' '.join(tokens[entities[head]["start"]:entities[head]["end"]])
        tail_tokens = ' '.join(tokens[entities[tail]["start"]:entities[tail]["end"]])
        type_, action = _ask_type("Relation type of |{}| |?| |{}| : \n >>".format(head_tokens, tail_tokens), 
                                  actions.keys(), types["relations"].keys())
        annotated_relations, stop = _check_action(action, annotated_relations)

        if utils.check_nones([head, tail, type_]):
            annotated_relations.append({"head": head, "tail": tail, "type": type_})

    return annotated_relations


def _print_numbered_list(l):
    for i, token in enumerate(l):
            print("[{}]{} ".format(i, token), end='')
    print()

def _print_status(entities, relations):
    pass

def _ask_yes_no(q):
    yes_no = {
        "yes": True,
        "y": True,
        "no": False,
        "n": False
    }

    input_ = str(input(q)).lower()
    answer = yes_no.get(input_, None)
    while answer==None:
        print("Cannot understand, tell me {}".format(yes_no.keys()))
        input_ = str(input(q)).lower()
        answer = yes_no.get(input_, None)

    return answer

def _ask_position(q, max_i, actions={}):
    def _ask():
        a = input(q)
        try:
            a = int(a)
        except:
            a = a.lower()
        return a

    def _check_valid_int(i):
        try:
            i = int(i)
            return (0 <= i <= max_i) 
        except:
            return False

    a = _ask()
    if a in list(actions):
        return None, a

    while not _check_valid_int(a):
        print("I need a position (int) between 0 and {}".format(max_i))
        print("Or give me an action {}".format(actions))
        a = _ask()
        if a in actions:
            return None, a

    return a, None

def _ask_type(q, actions={}, types={}):
    a = str(input(q)).lower()
    if a in list(actions):
        return None, a

    while a not in {t.lower() for t in types}:
        print("Cannot understand, tell me {}".format(list(types)))
        print("Or give me an action {}".format(list(actions)))
        a = str(input(q)).lower()
        if a in actions:
            return None, a

    return a, None


if __name__ == "__main__":
    dataset_path = "data/DistantlySupervisedDatasets/ontology_v5/02_04_2020_16_22_09/train/combined_labeling/dataset.json"
    types_path = "data/DistantlySupervisedDatasets/ontology_v5/02_04_2020_16_22_09/train/ontology_types.json"
    annotate(dataset_path, types_path)