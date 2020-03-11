import json
import argparse

def pretty_print(path):
    with open(path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    for i, sentence in enumerate(dataset):
        tokens = sentence["tokens"]
        print("Original sentence id:", sentence["orig_id"])
        print("[{}] \t".format(i), " ".join(tokens))
        for entity in sentence["entities"]:
            entity_tokens = tokens[entity["start"]:entity["end"]]
            print("[{}] \t".format(i), " ".join(entity_tokens) + "\t {}".format(entity["type"]))
        for relation in sentence["relations"]:
            entities = sentence["entities"]
            head_entity = entities[relation["head"]]
            tail_entity = entities[relation["tail"]]
            head_tokens = tokens[head_entity["start"]:head_entity["end"]]
            tail_tokens = tokens[tail_entity["start"]:tail_entity["end"]]
            print("[{}] \t".format(i), " ".join(head_tokens) + "\t {} \t".format(relation["type"]), " ".join(tail_tokens))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print spert json format in a readable way')
    parser.add_argument('path', type=str, help='path to the dataset.json file')
    args = parser.parse_args()
    pretty_print(args.path)