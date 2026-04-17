conda activate fish

# cd scripts/dataset_building/mmlu_full_trainingset/openaudio-s1-mini

# for i in {0..7}; do
#   GRADIO_SERVER_PORT=$((7860 + i)) CUDA_VISIBLE_DEVICES=$i python app.py &
# done

# wait

cp ../app.py .
accelerate launch app.py 