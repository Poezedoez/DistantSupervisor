import json
import argparse

def pretty_print(path):
    with open(path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    for i, sentence in enumerate(dataset):
        tokens = sentence["tokens"]
        print("Original sentence id: {} \n".format(sentence["orig_id"]))
        print("[{}] \t {} \n".format(i, " ".join(tokens)))
        for entity in sentence["entities"]:
            entity_tokens = tokens[entity["start"]:entity["end"]]
            print("[{}] \t {} \t {} \n".format(i, " ".join(entity_tokens), entity["type"]))
        for relation in sentence["relations"]:
            entities = sentence["entities"]
            head_entity = entities[relation["head"]]
            tail_entity = entities[relation["tail"]]
            head_tokens = tokens[head_entity["start"]:head_entity["end"]]
            tail_tokens = tokens[tail_entity["start"]:tail_entity["end"]]
            print("[{}] \t {} \t |{}| \t {} \n".format(i, " ".join(head_tokens), relation["type"], " ".join(tail_tokens)))
        print()

def pretty_write(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(dataset):
            tokens = sentence["tokens"]
            f.write("Original sentence id: {} \n".format(sentence["orig_id"]))
            f.write("[{}] \t {} \n".format(i, " ".join(tokens)))
            for entity in sentence["entities"]:
                entity_tokens = tokens[entity["start"]:entity["end"]]
                f.write("[{}] \t {} \t {} \n".format(i, " ".join(entity_tokens), entity["type"]))
            for relation in sentence["relations"]:
                entities = sentence["entities"]
                head_entity = entities[relation["head"]]
                tail_entity = entities[relation["tail"]]
                head_tokens = tokens[head_entity["start"]:head_entity["end"]]
                tail_tokens = tokens[tail_entity["start"]:tail_entity["end"]]
                f.write("[{}] \t {} \t |{}| \t {} \n".format(i, " ".join(head_tokens), relation["type"], " ".join(tail_tokens)))
            f.write('\n')    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print spert json format in a readable way')
    parser.add_argument('path', type=str, help='path to the dataset.json file')
    args = parser.parse_args()
    pretty_print(args.path)