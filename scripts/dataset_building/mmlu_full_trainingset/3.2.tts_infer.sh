conda activate whisper

# cd scripts/dataset_building/mmlu_full_trainingset

# for i in $(seq 0 7); do
#     RANK=$i WORLD_SIZE=8 CUDA_VISIBLE_DEVICES=$i python 3.tts.py &
# done

# wait

accelerate launch 3.tts.py