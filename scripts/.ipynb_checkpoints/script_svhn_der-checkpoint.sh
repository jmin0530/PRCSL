#!/bin/bash

if [ "$1" != "" ]; then
    echo "Running approach: $1"
else
    echo "No approach has been assigned."
fi
if [ "$2" != "" ]; then
    echo "Running on gpu: $2"
else
    echo "No gpu has been assigned."
fi

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd .. && pwd )"
SRC_DIR="$PROJECT_DIR/src"
echo "Project dir: $PROJECT_DIR"
echo "Sources dir: $SRC_DIR"

RESULTS_DIR="$PROJECT_DIR/results"
if [ "$4" != "" ]; then
    RESULTS_DIR=$4
else
    echo "No results dir is given. Default will be used."
fi
echo "Results dir: $RESULTS_DIR"

for SEED in 0 1 2 3 4
do
    if [ "$3" = "base_cl" ]; then
          PYTHONPATH=$SRC_DIR python -u $SRC_DIR/main.py --exp-name base_cl_${SEED} \
                --datasets svhn --num-tasks 1 --network resnet32 --seed $SEED \
                --nepochs 100 --batch-size 128 --results-path $RESULTS_DIR --opt adam \
                --approach $1 --gpu $2 --lr 0.001 --lr-patience 15 \
                --lamb-distill-ewc 5000 --lamb-distill-mas 1 --lamb-distill 4 \
                
        
    elif [ "$3" = "reservoir_cl" ]; then
          PYTHONPATH=$SRC_DIR python -u $SRC_DIR/main.py --exp-name base_cl_${SEED} \
                --datasets svhn --num-tasks 5 --network resnet32 --seed $SEED \
                --nepochs 50 --batch-size 32 --results-path $RESULTS_DIR --opt sgd \
                --approach $1 --gpu $2 --lr 0.03 --lr-factor 10 \
                --lamb-distill-ewc 5000 --lamb-distill-mas 1 --lamb-distill 4 \
                --num-exemplars 200 --exemplar-selection herding \
                --keep-classifier --alpha 0.3
        
    elif [ "$3" = "base_csl" ]; then
          PYTHONPATH=$SRC_DIR python -u $SRC_DIR/main_split.py --exp-name base_csl_${SEED} \
                --datasets svhn --num-tasks 5 --network resnet32 --seed $SEED \
                --nepochs 100 --batch-size 128 --results-path $RESULTS_DIR --opt adam \
                --approach $1 --gpu $2 --lr 0.001 --lr-patience 15 --nclients 10 \
                --lamb-distill-ewc 5000 --lamb-distill-mas 1 --lamb-distill 4

    elif [ "$3" = "fixd_csl" ]; then
          PYTHONPATH=$SRC_DIR python -u $SRC_DIR/main_split.py --exp-name fixd_csl_${SEED} \
                --datasets svhn --num-tasks 5 --network resnet32 --seed $SEED \
                --nepochs 50 --batch-size 32 --exem-batch-size 32 --results-path $RESULTS_DIR --opt sgd \
                --approach $1 --gpu $2 --lr 0.03 --lr-factor 10 --nclients 10 \
                --num-exemplars 600 --exemplar-selection herding --dp-mean-batch 3 --epsilon 10 \
                --lamb-distill 4 \
                --keep-classifier --alpha 0.3
#                 --lr-finetune-factor 0.1
    else
          echo "No scenario provided."
    fi
done
