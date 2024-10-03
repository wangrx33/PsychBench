
model_id="psychAiD"

n_shot=1


# Clinical
test_path=data/psych/task3.jsonl
val_path=data/psych/task3.jsonl


# save path
output_dir=data/fewshot1_task3

python ./src/generate_fewshot_psych.py \
    --n_shot=$n_shot \
    --model_id=$model_id \
    --output_dir=$output_dir  \
    --val_path=$val_path \
    --test_path=$test_path 