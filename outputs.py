import json
import argparse
from collections import Counter

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
                f.write("[{}] \t {} \t |{}| \t {} \n".format(i, " ".join(head_tokens), relation["type"],
                                                             " ".join(tail_tokens)))
            f.write('\n')


def print_sentences(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in dataset:
            f.write("{} \n".format(" ".join(sentence["tokens"])))


def print_statistics(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        stats = json.load(json_file)

    print("--- STATISTICS ---")
    label_functions = {0: "string_labeling", 1: "embedding_similarity_labeling",
                       2: "string labeling in conjunction with embedding similarity"}
    print("Label function: {}".format(label_functions[stats["label_function"]]))
    print("Threshold on embedding cosine similarity (cos_theta):", stats["cos_theta"])
    print("Processed {} sentences of which {} contained at least one entity".format(
        stats["sentences_processed"], stats["entity_sentences"]
    ))
    print("Time taken: {} seconds".format(stats["time_taken"]))
    print("--- Entities ---")
    tokens_per_entity = stats["tokens_total"] / stats["entities_total"] if(
        stats["entities_total"] != 0
    ) else 0
    print("Every {} tokens an entity occurs".format(tokens_per_entity))
    print("Entities were found in the following classes:")
    for type_, instance_counter in stats["entities"].items():
        count = sum([count for _, count in instance_counter.items()])
        print(type_, count)
    print("The most frequently labeled entities per class are:")
    for type_, instance_counter in stats["entities"].items():
        print("{} \t".format(type_), Counter(instance_counter).most_common(5))
    print("--- Relations ---")
    relations_per_sentence = stats["relation_candidates"]/stats["relations_total"] if(
        stats["relations_total"] != 0
    ) else 0
    print("Every {} sentences with at least two entities a relation occurs".format(relations_per_sentence))
    print("Relations were found in the following classes:")
    for relation, count in stats["relations"].items():
        print(relation, count)


def compare_datasets(path1, path2):
    with open(path1, 'r', encoding='utf-8') as f:
        dataset1 = json.load(f)

    with open(path2, 'r', encoding='utf-8') as f:
        dataset2 = json.load(f)

    # with open(output_path, 'w', encoding='utf-8') as csv_file:
    #     writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow("sentence_id", "sentence", "entity_x", "label_x")
    for i, sentence1 in enumerate(dataset1):
        tokens = sentence1["tokens"]
        print()
        for entity in sentence1["entities"]:
            entity_tokens = tokens[entity["start"]:entity["end"]]
            line += ", {}, {}".format(" ".join(entity_tokens), entity["type"])
        line += ' \n'
        f.write(line)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print spert json format in a readable way')
    parser.add_argument('path', type=str, help='path to the dataset.json file')
    args = parser.parse_args()
    pretty_print(args.path)
    # in_path = 'data/DistantlySupervisedDatasets/train/string/20200317-132432/dataset.json'
    # out_path = "sentences.txt"
    # print_sentences(in_path, out_path)

