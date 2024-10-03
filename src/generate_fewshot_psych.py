import argparse
import json
import random
import os

from constants import id2worker_class
from tqdm import tqdm
import pdb

query_prompt_1 = "{question_type}，请分析并最后给出答案。\n{question}\n{option_str}"
query_prompt_2 = "{question_type}，不需要做任何分析和解释，直接输出答案。\n{question}\n{option_str}"

global data_val 
data_val = []

def get_output_path(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    cot_or_a = 'cot' if args.use_cot else 'a'
    f_name = f'Psych-Exam-{cot_or_a}-{args.model_id}.json'
    output_path = os.path.join(args.output_dir, f_name)
    return output_path

def select_items(question_type=None, num=0):
    if question_type is not None:
        # Filter the data based on the given exam_type and exam_class
        filtered_data = [
            item
            for item in data_val
            if item["question_type"] == question_type
        ]
    else:
        filtered_data = data_val
    # Check if we have enough items
    if len(filtered_data) < num:
        raise ValueError(f"Not enough items matching the given criteria: {question_type}  \nfiltered data:{filtered_data}")

    # Select a random sample of items
    selected_items = random.sample(filtered_data, num)

    return selected_items


def get_query(da, use_cot):
    da["option_str"] = "\n".join(
        [f"{k}. {v}" for k, v in da["option"].items() if len(v) > 0 and v!=" "]
    )
    if use_cot:
        query = query_prompt_1.format_map(da)
    else:
        query = query_prompt_2.format_map(da)

    return query


def main(args):
    global data_val
    output_path = get_output_path(args)
    data_test = []
    with open(args.test_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                # 将每行转换为JSON对象并添加到列表中
                json_object = json.loads(line)
                data_test.append(json_object)
                ## 先只要选择题
                # if '选择题' in json_object['question_type']:
                #     data_test.append(json_object)
                    # if args.use_cot:
                    #     if len(json_object['explanation']) > 3:
                    #         data_test.append(json_object)
                    # else:                            
                    #     data_test.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from line: {line}")
    
    data_test = data_test[:args.max_test_num] 
    print('oringinal test dataset length:{}, random split to test:val=9:1:'.format(len(data_test)))
    
    if args.val_path:
        data_val = data_val
    else:
        import random
        random.seed(123)
        random.shuffle(data_test)
        data_val = data_test[int(len(data_test)*0.9):]
        data_test = data_test[:int(len(data_test)*0.9)]
        
    
    print("test data num:",len(data_test))
    print("val data num:",len(data_val))

    # initialize a worker
    from omegaconf import OmegaConf

    worker = id2worker_class[args.model_id].from_config(
        OmegaConf.load(args.model_config_path)[
            args.model_id
        ],
        generate_fewshot_examples_only=True,
    )
    for id,item in tqdm(enumerate(data_test)):
        if 'question_type' in item.keys(): # psych exam
            if '选择题' in item['question_type']:
                # select examples
                samples = select_items(
                    item["question_type"], args.n_shot
                )
                # one step to generate formatted few-shot examples
                item["fewshot_examples"] = worker.generate_fewshot_examples(
                    data=samples, use_cot=args.use_cot
                )
            else:
                item["fewshot_examples"] = ''
        else: ## real-world clinical case
            samples = select_items(num=args.n_shot)
            item["fewshot_examples"] = worker.generate_fewshot_examples(
                    data=samples, use_cot=args.use_cot
                )
            item["id"] = 'clinical-{}'.format(id)
            

    directory = os.path.dirname(output_path)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(data_test, output_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_shot", type=int, help="number of shots", default=5)
    parser.add_argument(
        "--model_config_path",
        type=str,
        help="path to the model configuration file",
        default='configs/model_config.yaml'
    )
    parser.add_argument(
        "--model_id",
        type=str,
        help="model id",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="path to the val json",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        help="path of the val file",
        default = None 
    )
    parser.add_argument(
        "--test_path",
        type=str,
        help="path of the test file",
    )
    parser.add_argument(
        "--use_cot",
        action="store_true",
        help="whether to use cot(action: True)",
    )
    parser.add_argument(
        "--max_test_num",
        type=int,
        help="case number of test data",
        default = 2000
    )

    args = parser.parse_args()

    
    if args.val_path:
        with open(args.val_path, "r", encoding="utf-8") as f:
            # data_val = json.load(f)
            for line in f:
                try:
                    json_object = json.loads(line)
                    data_val.append(json_object)
                    
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from line: {line}")
    main(args)

