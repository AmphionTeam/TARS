code_dir=/scratch/amlt_code/scripts/dataset_building/mmlu_full_trainingset

conda activate cosy
cd CosyVoice

accelerate launch \
--num_processes=8 \
--main_process_port 29502 cosy_app.py

cd ..