COS_THETA=0.83
COMBINED_LABELING=2
V=4
ONTOLOGY_PATH="data/ontology/v$V/"
TOKEN_POOLING="none"
MENTION_POOLING="none"
DATE=`date +%d_%m_%Y_%H_%M_%S`
RUN_NAME="${REDUCE}_filtered_$DATE";

python DistantSupervisor.py  \
    --selection 0 2 \
    --output_path "data/DistantlySupervisedDatasets/ontology_v$V/$RUN_NAME/train/" \
    --timestamp_given \
    --label_strategy $COMBINED_LABELING \
    --cos_theta $COS_THETA \
    --ontology_path $ONTOLOGY_PATH \
    --token_pooling $TOKEN_POOLING \
    --mention_pooling $MENTION_POOLING \ 
    --filter_sentences

python DistantSupervisor.py  \
    --selection 2 3 \
    --output_path "data/DistantlySupervisedDatasets/ontology_v$V/$RUN_NAME/dev/" \
    --timestamp_given \
    --label_strategy $COMBINED_LABELING \
    --cos_theta $COS_THETA \
    --ontology_path $ONTOLOGY_PATH \
    --token_pooling $TOKEN_POOLING \
    --mention_pooling $MENTION_POOLING \ 
    --filter_sentences

# python DistantSupervisor.py  \
#     --selection 700 900 \
#     --output_path "data/DistantlySupervisedDatasets/ontology_v$V/$RUN_NAME/test/" \
#     --timestamp_given \
#     --label_strategy $COMBINED_LABELING \
#     --cos_theta $COS_THETA \
#     --token_pooling $TOKEN_POOLING \
#     --mention_pooling $MENTION_POOLING \ 
#     --filter_sentences