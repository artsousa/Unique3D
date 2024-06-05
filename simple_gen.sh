#!/usr/bin/bash

IMAGES=$(ls app/examples/ | grep -E "\.png$|\.jpg$")

for FILE in $IMAGES
do
	echo "GENERATING... ", $FILE
	CUDA_VISIBLE_DEVICES=0 python gen.py --input-image app/examples/$FILE --remove-bg --do-refine --expansion-weight 0.15 --init-type thin --seed 73 --render-video --debug 
done

