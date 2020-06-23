COS_THETA=0.9
LABELING_STRATEGY=2 # combined labeling
V=4
ONTOLOGY="data/ontology/v${V}/"
TOKEN_POOLING="mean"
MENTION_POOLING="none"
DATE=`date +%d_%m_%Y_%H_%M_%S`
RUN_NAME="T|${TOKEN_POOLING}|_M|${MENTION_POOLING}|_$DATE";

python DistantSupervisor.py  \
    --selection 0 2 \
    --output_path "data/DistantlySupervisedDatasets/ontology_v$V/$RUN_NAME/train/" \
    --timestamp_given \
    --label_strategy $LABELING_STRATEGY \
    --cos_theta $COS_THETA \
    --ontology_path $ONTOLOGY \
    --filter_sentences \
    --token_pooling ${TOKEN_POOLING} \
    --mention_pooling ${MENTION_POOLING}
    

python DistantSupervisor.py  \
    --selection 2 3 \
    --output_path "data/DistantlySupervisedDatasets/ontology_v$V/$RUN_NAME/test/" \
    --timestamp_given \
    --label_strategy $LABELING_STRATEGY \
    --cos_theta $COS_THETA \
    --ontology_path $ONTOLOGY \
    --filter_sentences \
    --token_pooling ${TOKEN_POOLING} \
    --mention_pooling ${MENTION_POOLING}