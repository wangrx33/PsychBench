# import evaluate 
import evaluate
import io
import json
import numpy as np
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import tqdm
import csv
import jieba

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from rouge import Rouge

import argparse

CALC_REDUNDANT = False # re-calculate, even if scores already exist 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="psych exam", default=5)
    parser.add_argument(
        "--ans_path",
        type=str,
        help="path to the model generated ans file",
        default='./result/API/task5_qwen.json'
    )

    parser.add_argument(
        "--dir_out",
        type=str,
        help="path to the eval matrics",
        default='./result/PsychClinical/Qwen_api/task5'

    )
    
    # parse arguments, set data paths
    # args = parser.get_parser()
    args = parser.parse_args()
    is_cxr = True if args.dataset in ['cxr', 'opi'] else False
    
    os.makedirs(args.dir_out,exist_ok=True)
   
    # load data
    lst_tgt = []
    lst_out = []
    lst_idx = []
    option_qa = [[],[]]
    with open(args.ans_path, "r", encoding="utf-8") as f:
        answers = json.load(f)
    idx = 0
    for ans in answers:
        if 'question_type' not in ans.keys():
            ans['question_type'] = 'clinical'
        if '选择题' in ans['question_type']:
            option_qa[0].append(ans['answer'])
            option_qa[1].append(ans['model_answer'])
            idx += 1
        else:
            if ans['question_type'] == 'clinical':
                if ans['conversations'][1]['from']=='gpt':
                    lst_tgt.append(ans['conversations'][1]['value'])
                else:
                    lst_tgt.append(ans['conversations'][2]['value'])
            else:
                lst_tgt.append(ans['answer'])
            # lst_out.append(ans['model_answer'])
            if ans['answer_0'] == 'API调用失败':
                continue
            lst_out.append(ans['answer_0'])
            
            lst_idx.append(idx)
            idx += 1


    # load metrics
    # bleu = evaluate.load('bleu')
    # rouge = evaluate.load('rouge')
    bleu = sentence_bleu
    rouge = Rouge()
    bertscore = evaluate.load('./evaluate/metrics/bertscore/bertscore.py')
    
    ## Download the metrics directory to the local path
    # git clone https://github.com/huggingface/evaluate.git
    # metric = evaluate.load("evaluate/metrics/accuracy/accuracy.py")
    
    metrics = (bleu, rouge, bertscore)

    # compute scores of each sample across entire dataset
    scores_all = {}
    for tgt, out, idx in tqdm.tqdm(zip(lst_tgt, lst_out, lst_idx)):

        # get sub-dict containing scores for each metric
        scores = compute_scores(tgt, out, metrics, is_cxr)

        # append to master dict, dataset object
        scores_all[idx] = scores

    # save averaged scores across entire dataset
    write_all_scores(args, scores_all)
    
    score_list = []
    for i in range(len(option_qa[0])):
        gt = option_qa[0][i]
        pred = option_qa[1][i]
        if isinstance(gt,list):
            hit = 0
            for apred in pred:
                if apred in gt:
                    hit += 1
            score = hit/len(gt)
        else:
            if gt == pred:
                score = 1
            else:
                score = 0
        score_list.append(score)
    acc = np.mean(score_list)
    print('accuracy:',acc) 


def compute_scores(tgt, out, metrics, is_cxr):
    ''' given output(s), target(s), and a tuple of metrics
        return a scores dict ''' 
    
    # unpack tuple of pre-loaded metrics
    bleu, rouge, bertscore = metrics
    smooth = SmoothingFunction()

    # convert single sample to list
    tgt, out = wrap_str_in_lst(tgt), wrap_str_in_lst(out)

    # compute hugging face scores
    try:
        # scores_bleu = bleu.compute(predictions=out, references=tgt)
        sentence_bleu_score_4 = sentence_bleu([tgt], out, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
        scores_bleu = {'bleu':sentence_bleu_score_4}
    except: # division by zero
        print('bleu not computed correctly')
        scores_bleu = {'bleu': 0}
    # scores_rouge = rouge.compute(predictions=out, references=tgt)
    try:
        scores_rouge = rouge.get_scores(' '.join(out), ' '.join(tgt))[0]
    except: # division by zero
        print('rouge not computed correctly')
        scores_rouge = {'rouge-1': {'f':0},'rouge-2': {'f':0},'rouge-l': {'f':0}}
    scores_bert = bertscore.compute(model_type="path_to/bert-base-chinese/",predictions=[''.join(out)], references=[''.join(tgt)], lang='zh') #
    ########################################################################################################
    ### make change to source code!
    # if num_layers is None:
    #         try:
    #             num_layers = bert_score.utils.model2layers[model_type]
    #         except:
    #             num_layers = bert_score.utils.model2layers[bert_score.utils.lang2model[lang.lower()]]
    #########################################################################################################
    

    # compute f1-radgraph, f1-chexbert

    scores = {
        'BLEU': scores_bleu['bleu'],
        'ROUGE-1': scores_rouge['rouge-1']['f'],
        'ROUGE-2': scores_rouge['rouge-2']['f'],
        'ROUGE-L': scores_rouge['rouge-l']['f'],
        'BERT': np.mean(scores_bert['f1']), 
    }

    # scale scores to be on [0,100] instead of [0,1]
    for key in scores:
        scores[key] *= 100.
        scores[key] = round(scores[key], 2)

    return scores


def write_all_scores(args, scores_all): 
    ''' write all scores across dataset to json file 
        redundantly write to txt for copy-paste into overleaf '''

    validate_keys(scores_all) # sanity check

    # compute avg, std across all samples. write to json
    scores_avg_std = avg_across_samples(scores_all)
    fn_scores_json = os.path.join(args.dir_out, 'metrics.json')
    with open(fn_scores_json, 'w') as f:
        f.write(json.dumps(scores_avg_std))

    # extract avg, write to txt file
    scores_avg = extract_avg_only(scores_avg_std)
    ss = scores_avg
    txt_out = []
    for key, val in scores_avg.items():
        ss[key] = round(ss[key], 1)
    header = 'BLEU & ROUGE-L & BERT'
    txt_out.append(header)
    str_txt = f'{ss["BLEU"]} & {ss["ROUGE-L"]} & {ss["BERT"]}'
    txt_out.append(str_txt)
    fn_scores_txt = os.path.join(args.dir_out, 'metrics.txt')
    write_list_to_csv(fn_scores_txt, txt_out)

    return


def avg_across_samples(scores_all):
    ''' average across individual sample scores (sub-dicts) '''

    scores_avg_std = {} 
    keys_to_avg = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L','BERT']

    for key in keys_to_avg:
        values = [sub_dict[key] for sub_dict in scores_all.values()]
        avg_std = {'avg': round(np.mean(values), 2),
                   'std': round(np.std(values), 2)}
        scores_avg_std[key] = avg_std

    return scores_avg_std


def extract_avg_only(scores_avg_std):
    ''' extract only values from sub-dict key avg '''
    scores_avg = {}
    for idx in scores_avg_std:
        scores_avg[idx] = scores_avg_std[idx]['avg']
    return scores_avg


def validate_keys(my_dict):
    ''' given dict w sub-dict, validate all sub-dicts have same keys '''
    
    sub_dict_keys = None
    for sub_dict in my_dict.values():
        if sub_dict_keys is None:
            sub_dict_keys = set(sub_dict.keys())
        else:
            msg = 'sub-dicts do not contain same keys'
            assert set(sub_dict.keys()) == sub_dict_keys, msg 

    return
   

# def wrap_str_in_lst(var):
#     if isinstance(var, str):
#         return [var]
#     return var

def wrap_str_in_lst(text):
    # 使用 jieba 进行精确分词
    segmented_text = jieba.cut(text, cut_all=False)
    # 直接返回分词结果的列表
    return list(segmented_text)


def write_list_to_csv(fn_csv, list_, csv_action='w'):
    ''' write each element of 1d list to csv 
        can also append to existing file w csv_action="a" '''

    with open(fn_csv, csv_action) as f:
        writer = csv.writer(f, delimiter='\n')
        writer.writerow(list_)

    return


if __name__ == '__main__':
    main()