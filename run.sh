COS_THETA=0.83
COMBINED_LABELING=2
V=4
ENT_PATH="data/ontology/v${V}/ontology_entities.csv" 
REL_PATH="data/ontology/v${V}/ontology_relations.csv" 
REDUCE="none"
EMB_PATH="data/ontology/v${V}/faiss/"
DATE=`date +%d_%m_%Y_%H_%M_%S`
RUN_NAME="${REDUCE}_filtered_$DATE";

python DistantSupervisor.py  \
    --selection 0 2 \
    --output_path "data/DistantlySupervisedDatasets/ontology_v$V/$RUN_NAME/train/" \
    --timestamp_given \
    --label_strategy $COMBINED_LABELING \
    --cos_theta $COS_THETA \
    --ontology_version $V \
    --f_reduce $REDUCE \
    --filter_sentences

python DistantSupervisor.py  \
    --selection 2 3 \
    --output_path "data/DistantlySupervisedDatasets/ontology_v$V/$RUN_NAME/dev/" \
    --timestamp_given \
    --label_strategy $COMBINED_LABELING \
    --cos_theta $COS_THETA \
    --ontology_version $V \
    --f_reduce $REDUCE \
    --filter_sentences

# python DistantSupervisor.py  \
#     --selection 700 900 \
#     --output_path "data/DistantlySupervisedDatasets/ontology_v$V/$RUN_NAME/test/" \
#     --timestamp_given \
#     --label_strategy $COMBINED_LABELING \
#     --cos_theta $COS_THETA \
#     --ontology_entities_path $ENT_PATH \
#     --ontology_relations_path $REL_PATH \
#     --entity_embedding_path $EMB_PATH \
#     --f_reduce $REDUCE \
#     --filter_sentences