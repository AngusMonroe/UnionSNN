#!/bin/sh

model="configs/MUTAG/TUs_graph_classification_GIN_MUTAG_100k.json"
echo ${model}
python3 main.py --config  $model --gpu_id 0 --preprocess shortest_path_graph
