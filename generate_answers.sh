# if use_fewshot set --use_fewshot

input_path='。/data/fewshot1_task4/Psych-Exam-a-psychAiD.json' # Clinical 提炼主诉 0-shot

task_name='PsychClinical' 


model_id="psychAiD" # which model to evaluate


mkdir -p logs/${task_name}/

accelerate launch \
    --gpu_ids='0' \
    --main_process_port 27274 \
    --config_file ./configs/accelerate_config.yaml  \
    ./src/generate_answers.py \
    --use_fewshot \
    --model_id=$model_id \
    --all_gather_freq=20 \
    --use_input_path \
    --input_path=$input_path \
    --output_path=./result/${task_name}/${model_id}/modelans_psychAiD_no_sample.json \
    --batch_size 1 \
    --model_config_path="./configs/model_config.yaml" \
    --start 1
