DATE=`date +%d_%m_%Y_%H_%M_%S`
LABELING_STRATEGY=2 # combined labeling
V=2
ONTOLOGY_PATH="data/ontology/v${V}/"


TOKEN_POOLING="none"
MENTION_POOLING="none"
COS_THETA=0.9
RUN_NAME="$DATE/T|${TOKEN_POOLING}|_M|${MENTION_POOLING}|";

python DistantSupervisor.py  \
    --selection 0 2 \
    --output_path "data/DistantlySupervisedDatasets/ontology_v$V/$RUN_NAME/train/" \
    --timestamp_given \
    --label_strategy $LABELING_STRATEGY \
    --cos_theta $COS_THETA \
    --filter_sentences \
    --ontology_path $ONTOLOGY_PATH \
    --token_pooling ${TOKEN_POOLING} \
    --mention_pooling ${MENTION_POOLING}
    
python DistantSupervisor.py  \
    --selection 2 3 \
    --output_path "data/DistantlySupervisedDatasets/ontology_v$V/$RUN_NAME/test/" \
    --timestamp_given \
    --filter_sentences \
    --label_strategy $LABELING_STRATEGY \
    --cos_theta $COS_THETA \
    --ontology_path $ONTOLOGY_PATH \
    --token_pooling ${TOKEN_POOLING} \
    --mention_pooling ${MENTION_POOLING}


TOKEN_POOLING="mean"
MENTION_POOLING="mean"
COS_THETA=0.88
RUN_NAME="$DATE/T|${TOKEN_POOLING}|_M|${MENTION_POOLING}|";

python DistantSupervisor.py  \
--selection 0 2 \
--output_path "data/DistantlySupervisedDatasets/ontology_v$V/$RUN_NAME/train/" \
--timestamp_given \
--label_strategy $LABELING_STRATEGY \
--cos_theta $COS_THETA \
--filter_sentences \
--ontology_path $ONTOLOGY_PATH \
--token_pooling ${TOKEN_POOLING} \
--mention_pooling ${MENTION_POOLING}

python DistantSupervisor.py  \
--selection 2 3 \
--output_path "data/DistantlySupervisedDatasets/ontology_v$V/$RUN_NAME/test/" \
--timestamp_given \
--filter_sentences \
--label_strategy $LABELING_STRATEGY \
--cos_theta $COS_THETA \
--ontology_path $ONTOLOGY_PATH \
--token_pooling ${TOKEN_POOLING} \
--mention_pooling ${MENTION_POOLING}



TOKEN_POOLING="mean"
MENTION_POOLING="none"
COS_THETA=0.9
RUN_NAME="$DATE/T|${TOKEN_POOLING}|_M|${MENTION_POOLING}|";

python DistantSupervisor.py  \
--selection 0 2 \
--output_path "data/DistantlySupervisedDatasets/ontology_v$V/$RUN_NAME/train/" \
--timestamp_given \
--label_strategy $LABELING_STRATEGY \
--cos_theta $COS_THETA \
--filter_sentences \
--ontology_path $ONTOLOGY_PATH \
--token_pooling ${TOKEN_POOLING} \
--mention_pooling ${MENTION_POOLING}

python DistantSupervisor.py  \
--selection 2 3 \
--output_path "data/DistantlySupervisedDatasets/ontology_v$V/$RUN_NAME/test/" \
--timestamp_given \
--filter_sentences \
--label_strategy $LABELING_STRATEGY \
--cos_theta $COS_THETA \
--ontology_path $ONTOLOGY_PATH \
--token_pooling ${TOKEN_POOLING} \
--mention_pooling ${MENTION_POOLING}