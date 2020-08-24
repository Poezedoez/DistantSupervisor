DATE=`date +%d_%m_%Y_%H_%M_%S`
LABELING_STRATEGY=2 # combined labeling
V=7
ONTOLOGY_PATH="data/ontology/v${V}/"
TOKEN_POOLING="none"
MENTION_POOLING="none"
ENTITY_FRACTION=1
COS_THETA=0.87
RUN_NAME="$DATE/T|${TOKEN_POOLING}|_M|${MENTION_POOLING}|_F|${ENTITY_FRACTION}|";


python DistantSupervisor.py  \
--selection 0 1000 \
--output_path "data/DistantlySupervisedDatasets/ontology_v$V/${RUN_NAME}/train/" \
--timestamp_given \
--label_strategy $LABELING_STRATEGY \
--cos_theta $COS_THETA \
--filter_sentences \
--ontology_path $ONTOLOGY_PATH \
--token_pooling ${TOKEN_POOLING} \
--mention_pooling ${MENTION_POOLING} \
--entity_fraction $ENTITY_FRACTION
