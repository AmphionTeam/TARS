code_dir=/scratch/amlt_code/scripts/dataset_building/mmlu_full_trainingset

conda activate whisper

accelerate launch \
--num_processes=$8 \
--main_process_port 29503 3.tts.py
