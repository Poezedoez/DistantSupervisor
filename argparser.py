import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Create a distantly supervised dataset of scientific documents')
    parser.add_argument('--ontology_entities_path', type=str, default="data/ontology/ontology_entities.csv",
                        help="path to the ontology entities file")
    parser.add_argument('--ontology_relations_path', type=str, default="data/ontology/ontology_relations.csv",
                        help="path to the ontology relations file")
    parser.add_argument('--data_path', type=str, default="data/ScientificDocuments/",
                        help='path to the folder containing scientific documents/zeta objects')
    parser.add_argument('--output_path', type=str, default="data/DistantlySupervisedDatasets/", help="output path")
    parser.add_argument('--entity_embedding_path', type=str, default="data/ontology/entity_embeddings.json",
                        help="path to file of precalculated lexical embeddings of the entities")
    parser.add_argument('--selection', type=int, nargs=2, default=None,
                        help="start and end of file range for train/test split")
    parser.add_argument('--label_strategy', type=int, default=2, choices=range(0, 3),
                        help="0 = string, 1 = embedding, 2 = string + embedding")
    parser.add_argument('--timestamp_given', default=False, action="store_true")
    parser.add_argument('--cos_theta', type=float, default=0.83,
                        help="similarity threshold for embedding based labeling")    
    parser.add_argument('--filter_sentences', default=False, action="store_true")
    parser.add_argument('--f_reduce', type=str, default="mean")
    
    return parser    