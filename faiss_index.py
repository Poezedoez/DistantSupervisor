import utils
import faiss
import json
import os

def init(size):
    return faiss.IndexFlatIP(size)

def save(index, table, name, save_path="data/save/"):
    print("Saving {} index...".format(name))
    utils.create_dir(save_path)
    index_path = os.path.join(save_path, "{}_index".format(name))
    faiss.write_index(index, index_path)
    table_path = os.path.join(save_path, "{}_table.json".format(name))
    with open(table_path, 'w') as json_file:
        json.dump(table, json_file)
    print("Indexed {} {} with their labels".format(len(table), name))


def load(path, name, device="cpu"):
    index_path = os.path.join(path, "{}_index".format(name))
    if not os.path.exists(index_path):
        return None, None
    index = faiss.read_index(index_path)
    if device == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    table_path = os.path.join(path, "{}_table.json".format(name))
    with open(table_path, 'r', encoding='utf-8') as json_table:
        table = json.load(json_table)

    return index, table
