import os
import glob
import json

def read_full_texts(path):
    flist = os.listdir(path)
    with open("full_texts.txt", 'w', encoding='utf-8') as out_file:
        for folder in flist:
            text_path = glob.glob(path + "{}/representations/".format(folder) + "text|*")[0]
            with open(text_path, 'r', encoding='utf-8') as text_json:
                text = json.load(text_json)["value"]
                out_file.write(text)
                out_file.write('\n')


if __name__ == "__main__":
    document_path = "data/ScientificDocuments/"
    read_full_texts(document_path)




