import sys

sys.path.append("./src")

from transformers import AutoModel, AutoModelForCausalLM
from transformers import GenerationConfig
import re

import pdb
import json


from collections import defaultdict
from constants import id2worker_class


def get_runner_class(model_id):
    cls = id2worker_class.get(model_id, None)
    if cls is None:
        print(f"{model_id} is not registered in src/constants.py")
        exit(1)
    return cls


def match_choice(text, cot_flag):
    if cot_flag:
        option = ["A", "B", "C", "D", "E", "F", "G"]
        res = re.search(r"(答案|正确选项)(?:是|：|为|应该是|应该为)(.*?)(。|\.|$)", text, re.S)
        if res:
            return "".join([x for x in res.group(2) if x in option])
        return "".join([i for i in text if i in option])
    else:
        option = ["A", "B", "C", "D", "E", "F", "G"]
        res = re.search(r"(答案|正确选项)(?:是|：|为|应该是|应该为)(.*?)(。|\.|$)", text, re.S)
        if res:
            return "".join([x for x in res.group(2) if x in option])
        return "".join([i for i in text if i in option])


def extract_ans(ans_num, output_path, cot_flag):
    datas = []
    with open(output_path, encoding="utf-8") as f:
        for l in f:
            datas.append(json.loads(l))

    for da in datas:
        if "question_type" in da.keys():
            ty = da["question_type"]
        else:
            ty = 'clinical'
        ress = defaultdict(int)
        if "选择题" in ty:
            for ind in range(ans_num):
                res = da[f"answer_{ind}"]
                choice = match_choice(res, cot_flag)
                if len(choice) > 1 and ty != "多项选择题":
                    choice = choice[0]
                if len(choice) > 0:
                    ress[choice] += 1
            if len(ress) > 0:
                model_ans = sorted(ress.items(), key=lambda x: x[1], reverse=True)[0][0]
            else:
                model_ans = ""
            da["model_answer"] = model_ans
        else:
            ind = 0
            res = da[f"answer_{ind}"]
            da["model_answer"] = res



    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)


def make_output_dir(fp):
    import os

    dir = "/".join(fp.split("/")[:-1])
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)
    print(f"output to {dir}")
