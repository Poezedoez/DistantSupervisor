import utils
import faiss
import json
import os

def init(size):
    return faiss.IndexFlatIP(size)

def save(index, table, token_pooling, mention_pooling, save_path="data/save/"):
    print("Saving Faiss index and table...")
    utils.create_dir(save_path)
    index_name = "entities_T|{}|_M|{}|_index".format(token_pooling, mention_pooling)
    table_name = "entities_T|{}|_M|{}|_table".format(token_pooling, mention_pooling)
    index_path = os.path.join(save_path, index_name)
    table_path = os.path.join(save_path, table_name)
    faiss.write_index(index, index_path)
    with open(table_path, 'w') as json_file:
        json.dump(table, json_file)

    print("Indexed {} entities with their labels".format(len(table)))


def load(path, token_pooling, mention_pooling, device="cpu"):
    index_name = "entities_T|{}|_M|{}|_index".format(token_pooling, mention_pooling)
    table_name = "entities_T|{}|_M|{}|_table".format(token_pooling, mention_pooling)
    index_path = os.path.join(path, index_name)
    table_path = os.path.join(path, table_name)
    if not os.path.exists(index_path) or not os.path.exists(table_path):
        return None, None

    # Load index
    index = faiss.read_index(index_path)
    if device == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    # Load table
    with open(table_path, 'r', encoding='utf-8') as json_table:
        table = json.load(json_table)

    print("Loaded index and table with {} entities from {}".format(len(table), path))

    return index, table
