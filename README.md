# DistantSupervisor

Distant Supervisor is the main class that uses:

* Ontology class located in Ontology.py for the schema of entity and relation types,
   as well as calculating (an) representation(s) for an entity instance.
  
* DataIterator located in read.py for reading the local document objects that include precalculated BERT embeddings
 
* Labeling functions in heuristics.py for distant supervision


Example script:

```sh
$ chmod +x scripts/run.sh
$ ./scripts/run.sh
```

