#!/bin/bash

#####################################################################
# OBJECT REARRANGEMENT
# TRAIN
python scripts/train.py
# EVAL TRAIN
python scripts/eval_train.py --condition_guidance_w 1.2
# LEARN AND EVAL NEW CONCEPTS
for i in {0..4} # new concepts training comops
do
   # learn weights
   python scripts/learn_concepts.py --n_concepts 2 --dataset_name "data/object_rearrangement/compos_demo$i.pkl"
   # fixed weights
   python scripts/learn_concepts.py --n_concepts 2 --dataset_name "data/object_rearrangement/compos_demo$i.pkl" --learn_weights 0 --condition_guidance_w 1.2
done
for i in 0 2 3 4 # new concepts
do
   python scripts/learn_concepts.py --n_concepts 2 --dataset_name "data/object_rearrangement/new_demo$i.pkl"
done
# compose learned concept with training concepts
python scripts/learn_concepts.py --n_concepts 2 --dataset_name "data/object_rearrangement/new_demo1.pkl" --compose_dataset "data/object_rearrangement/eval_compos_learned_train.pkl"


#####################################################################
# AGENT
python scripts/train.py
python scripts/eval_train.py --condition_guidance_w 1.6
for i in {1..5}
do
    python scripts/learn_concepts.py --n_concepts 2 --dataset_name "data/AGENT/compos_demo$i.pkl" --new_init_dataset "data/AGENT/eval_learned_new_init_$i.pkl"
    python scripts/learn_concepts.py --n_concepts 2 --dataset_name "data/AGENT/compos_demo$i.pkl" --new_init_dataset "data/AGENT/eval_learned_new_init_$i.pkl" --learn_weights 0 --condition_guidance_w 1.6
done


#####################################################################
# MOCAP
python scripts/train.py
python scripts/eval_train.py --condition_guidance_w 1.8
declare -a test_data_names=("jumping_jacks" "breaststroke" "chop_wood")
for data_name in ${test_data_names[@]};
    do
        python scripts/learn_concepts.py --n_concepts 2 --dataset_name "data/mocap/test_dataset_$data_name.pkl"
        python scripts/learn_concepts.py --n_concepts 2 --dataset_name "data/mocap/test_dataset_$data_name.pkl" --learn_weights 0 --condition_guidance_w 1.8
    done


#####################################################################
# HIGHWAY
python scripts/train.py
python scripts/eval_train.py --condition_guidance_w 1.8
python scripts/learn_concepts.py --n_concepts 2 --dataset_name "data/highway/test_dataset.pkl"
python scripts/learn_concepts.py --n_concepts 2 --dataset_name "data/highway/test_dataset.pkl" --learn_weights 0 --condition_guidance_w 1.8


#####################################################################
# ROBOT
python scripts/train.py
python scripts/eval_train.py --condition_guidance_w 1.8
python scripts/learn_concepts.py
python scripts/learn_concepts.py --learn_weights 0 --condition_guidance_w 1.8
