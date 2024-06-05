#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0 python gen.py --input-image app/examples/anya.png --remove-bg --do-refine --expansion-weight 0.1 --init-type std --seed 10 --render-video --debug 
