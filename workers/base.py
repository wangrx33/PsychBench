import pdb
import os
import torch
from torch.utils.data import Dataset, DataLoader
import json
from accelerate import Accelerator
from dataclasses import dataclass
from accelerate import Accelerator
from copy import deepcopy

from peft import PeftModel
from transformers import (
    AutoModel, AutoModelForCausalLM, 
    AutoTokenizer, LlamaTokenizer,
    AutoConfig,
    
)


### task1_1 提炼主诉以及四个诊断标准
TASK1 = '''请扮演一位专业的精神科临床医生，根据提供的患者信息，提取总结出患者的主诉，并根据ICD-10的相应标准，总结该患者的诊断标准（包括病程标准、症状学标准、严重程度标准和排除标准）。提取总结的主诉应当简明扼要地总结疾病的主要症状及病程，控制在20字以内，各项诊断标准应与ICD-10相对应。
病程标准：包括病程的时长，明确指出症状出现和持续的时间；及病程的模式，描述症状是否有急性、慢性或间歇性发作的特点。
症状学标准：症状学标准指的是与特定精神疾病相关的关键症状（核心症状和附加症状）。核心症状：列出符合诊断标准的主要症状，如“患者表现出明显的抑郁心境，兴趣和愉悦感丧失，精力减退”。附加症状：列举支持诊断的其他相关症状，例如，“伴有失眠、体重减轻和自我评价下降”。在书写病历时，应详细描述患者的具体症状，包括它们是如何表现、何时出现以及它们对患者的日常生活有何影响。
严重程度标准：严重程度标注通常涉及对患者的功能水平进行评估。这可能包括对工作、社交、家庭生活等方面的能力进行评估，根据症状的强度和功能损害程度进行分级。例如，患者是否因为症状而无法工作或维持人际关系。
排除标准：排除标准是指在做出精神科诊断时需要排除的其他可能的诊断。例如，某些症状可能是由于药物副作用或其他医疗条件引起的，因此在做出最终诊断前需要排除这些可能性。
{question}
'''

### task1_2
TASK1_2 = '''请扮演一位专业的精神心理临床医生，分析提供的血液检查结果（全血细胞分析），并总结哪些指标异常。对于异常指标，请参照正常参考范围，进行标注，以识别和报告异常值。同时，结合提供的患者信息对异常指标进行分析，以帮助诊断以及治疗策略的调整。\n{question}'''

### task2 主要诊断
TASK2 = '''请扮演一位专业的精神科临床医生，根据下述患者信息按照ICD-10诊断标准给出主要诊断以及共病诊断（若有）的ICD-10代码以及疾病名称（精确到亚型）。仅需要给出诊断的ICD代码及疾病名称，无需进行分析。
输出格式为：
主要诊断：ICD10代码及疾病名称
精神科共病诊断：ICD10代码及疾病名称，若无则填“无”
（可选的ICD10代码及其对应的诊断为：
F20.0 偏执型精神分裂症 
F20.1 青春型精神分裂症 
F20.2 紧张型精神分裂症 
F20.3 未分化型精神分裂症 
F20.4 精神分裂症后抑郁 
F20.5 残留型精神分裂症 
F20.6 单纯型精神分裂症 
F20.8 其它精神分裂症 
F20.9 精神分裂症，未特定 
F30.001 轻躁狂 
F30.101 不伴有精神病性症状的躁狂发作 
F30.201 伴有精神病性症状的躁狂发作 
F30.802 兴奋状态 
F30.901 躁狂发作 
F30.902 躁狂状态 
F31.001 双相情感障碍,目前为轻躁狂发作 
F31.201 双相情感障碍,目前为伴有精神病性症状的躁狂发作 
F31.101 双相情感障碍,目前为不伴有精神病性症状的躁狂发作 
F31.302 双相情感障碍,目前为轻度抑郁发作 
F31.303 双相情感障碍,目前为不伴有躯体症状的轻度抑郁发作 
F31.304 双相情感障碍,目前为中度抑郁发作 
F31.305 双相情感障碍,目前为不伴有躯体症状的中度抑郁发作 
F31.311 双相情感障碍,目前为伴有躯体症状的轻度抑郁发作 
F31.312 双相情感障碍,目前为伴有躯体症状的中度抑郁发作 
F31.401 双相情感障碍,目前为不伴有精神病性症状的重度抑郁发作 
F31.501 双相情感障碍,目前为伴有精神病性症状的重度抑郁发作 
F31.601 双相情感障碍,目前为混合性发作 
F31.701 双相情感障碍,目前为缓解状态 
F31.901 双相情感障碍 
F32.001 轻度抑郁发作 
F32.002 不伴有躯体症状的轻度抑郁发作 
F32.011 伴有躯体症状的轻度抑郁发作 
F32.101 中度抑郁发作 
F32.102 不伴有躯体症状的中度抑郁发作 
F32.111 伴有躯体症状的中度抑郁发作 
F32.201 不伴有精神病性症状的重度抑郁发作 
F32.301 伴有精神病性症状的重度抑郁发作 
F32.901 抑郁发作 
F32.902 抑郁状态 
F33.001 复发性抑郁障碍,目前为轻度发作 
F33.002 复发性抑郁障碍,目前为伴有躯体症状的轻度发作 
F33.011 复发性抑郁障碍,目前为不伴有躯体症状的轻度发作 
F33.101 复发性抑郁障碍,目前为中度发作 
F33.102 复发性抑郁障碍,目前为伴有躯体症状的中度发作 
F33.111 复发性抑郁障碍,目前为不伴有躯体症状的中度发作 
F33.201 复发性抑郁障碍,目前为不伴有精神病性症状的重度发作 
F33.301 复发性抑郁障碍,目前为伴有精神病性症状的重度发作 
F33.401 复发性抑郁障碍,目前为缓解状态 
F33.901 复发性抑郁障碍 
F00.- 阿尔茨海默病性痴呆 
F06.7 轻度认知障碍 
F10.- 酒精所致的精神和行为障碍 
F13.- 使用镇静催眠剂所致的精神和行为障碍 
F34.0 环性心境 
F34.1 恶劣心境 
F34.8 其它持续性心境（情感）障碍 
F34.9 持续性心境（情感）障碍，未特定 
F41.1 广泛性焦虑障碍 
F42.- 强迫性障碍 
F44.- 分离性障碍 
F70 轻度精神发育迟滞 
F71 中度精神发育迟滞 
F72 重度精神发育迟滞 
F73 极重度精神发育迟滞 
F78 其它精神发育迟滞 
F79 未特定的精神发育迟滞 
F90 注意缺陷与多动障碍 
F95 抽动障碍 ）
\n{question}'''

### task3 分析诊断及鉴别诊断
TASK3 = '''要求：请扮演一位专业的精神科临床医生，根据以下患者信息，进行精神心理疾病之间的临床鉴别诊断分析，给出1个主要诊断和2个鉴别诊断。您的回答应当包括以下部分：“
鉴别诊断分析：
1. 患者性别年龄、病前性格等基本信息概要。
2. 起病诱因及病程。分析症状的发作频率、持续时间、病程的波动性，区分急性、慢性或间歇性发作的模式。
3. 既往史、家族史。
4. 症状分析，详细列出患者的主要临床表现，症状和特征以及严重程度。
诊断：
1. 主要诊断：按照ICD-10诊断标准给出主要诊断的ICD-10代码和疾病名称，格式为[ICD10代码]-[疾病名称]。
2. 鉴别诊断：给出2个鉴别诊断的ICD-10代码和疾病名称，说明与主要诊断的鉴别点，结合提供的患者信息，通过症状的特异性进行区分。格式为 <鉴别诊断1>[ICD10代码]-[疾病名称],鉴别点：\n<鉴别诊断2>[ICD10代码]-[疾病名称],鉴别点：”
请以专业、清晰和结构化的方式回答，确保输出的信息完整、准确，并且适合作为临床参考。
患者信息：
{question}
'''

TASK3_1 = '''要求：请扮演一位专业的精神科临床医生，根据以下患者信息，进行精神心理疾病之间的临床鉴别诊断分析，给出1个主要诊断和最需要与之鉴别的2个鉴别诊断。
可供选择的诊断和鉴别诊断疾病名称包括：“
人格障碍
双相情感障碍
器质性精神障碍
复发性抑郁障碍
妄想性障碍
广泛性焦虑障碍
强迫性障碍
心境障碍
急性而短暂的精神病性障碍
抑郁发作
抑郁障碍
焦虑障碍
环性心境
精神分裂症
精神障碍
脑器质性精神病
躁狂发作
躯体症状障碍
酒精所致的精神行为障碍
”

您的回答应当包括以下部分：“
鉴别诊断分析：
1. 患者性别年龄、病前性格等基本信息概要。
2. 起病诱因及病程。分析症状的发作频率、持续时间、病程的波动性，区分急性、慢性或间歇性发作的模式。
3. 既往史、家族史。
4. 症状分析，详细列出患者的主要临床表现，症状和特征以及严重程度。
诊断：
1. 主要诊断：按照ICD-10诊断标准给出主要诊断的疾病名称，格式为[疾病名称]。
2. 鉴别诊断：给出2个鉴别诊断的疾病名称，说明与主要诊断的鉴别点，结合提供的患者信息，通过症状的特异性进行区分。格式为 <鉴别诊断1>[疾病名称],鉴别点：\n<鉴别诊断2>[疾病名称],鉴别点：”
请以专业、清晰和结构化的方式回答，确保输出的信息完整、准确，并且适合作为临床参考。
患者信息：
{question}
'''

### task4 用药建议
TASK4 = '''请扮演一位专业的精神科临床医生，根据患者的简要病史、查体及精神检查结果、诊断及鉴别诊断给出用药建议。在选择药物及调整剂量时综合考量多方面因素，采取个体化、循证医学为基础的策略，确保治疗方案既能有效控制症状，又能最大程度减少副作用，提高患者的生活质量；注意考虑患者的诊断与检查结果，遵循ICD-10、DSM5等治疗指南，严谨考虑该患者可能的不良反应和药物相互作用信息。
可选的药物包括：['丙戊酸','伏硫西汀','利培酮','劳拉西泮','唑吡坦','喹硫平','地西泮','奋乃静','奥氮平','奥沙西泮','帕利哌酮','帕罗西汀','度洛西汀','拉莫三嗪','文拉法辛','曲唑酮','氟伏沙明','氟哌啶醇','氟西汀','氨磺必利','氯丙嗪','氯氮平','硝西泮','碳酸锂','米氮平','美金刚','舍曲林','艾司西酞普兰','阿戈美拉汀','阿立哌唑','鲁拉西酮','齐拉西酮']
您的回答应当包括以下部分：“
分析：根据提供的信息分析之前用药的疗效情况以及是否需要调整用药等
推荐药物：仅按照推荐顺序给出药物名称即可，多个药物之间用逗号隔开。
”
{question}'''

### task5 长期记忆能力以及用药分析
TASK5 = '''请扮演一位专业的精神科临床医生，详细阅读分析下面的患者多期病程及检查结果记录，并回答问题：
{question}
'''




@dataclass
class BaseWorker():
    """
    The base class of each model worker.
    """
    cfg: dict
    input_pth: str
    output_pth: str
    batch_size: int 
    use_cot: bool = False
    use_qa: bool = False
    generate_fewshot_examples_only: bool = False
    use_fewshot: bool = False

    def __post_init__(self):
        if self.generate_fewshot_examples_only: # no need to do post_init if we only need to generate fewshot examples
            return
        self.print_in_main(f'loading config: {self.cfg.load}')
        self.model, self.tokenizer = self.load_model_and_tokenizer(self.cfg.load)
        self.device = self.cfg.load.device
        self.accelerator = Accelerator()
        self.prompt_wrapper = PromptWrapper(
            self.tokenizer,
            self.instruction_template_with_fewshot if self.use_fewshot else self.instruction_template,
            conv_collater=self.collate_conv,
            use_cot=self.use_cot,
        )
        self.wrap_model()
        print('initiatig generation config...')
        self.init_generation_config(self.cfg)
        self.init_dataloader(self.input_pth, self.batch_size)
        self.init_writer(self.output_pth)
    

    @classmethod
    def from_config(
        cls, 
        cfg, 
        input_pth: str = '',
        output_pth: str = '',
        batch_size = 1,
        use_qa = False, 
        use_cot = False,
        generate_fewshot_examples_only = False,
        use_fewshot = False,
        ):
        assert cfg.get('load', None) is not None
        
        return cls(
            cfg,             
            input_pth,
            output_pth,
            batch_size, 
            use_cot = use_cot,
            use_qa = use_qa, 
            generate_fewshot_examples_only = generate_fewshot_examples_only,
            use_fewshot = use_fewshot,
        )


    def load_model_and_tokenizer(self, load_config):
        # model = ...
        # return model, tokenizer,
        raise NotImplementedError


    @property
    def system_prompt(self,):
        return "" # no placeholder is needed
    @property
    def instruction_template(self,):
        return "" # with role and placeholder
    @property
    def instruction_template_with_fewshot(self,):
        return "" # with role and placeholder
    
    @property
    def query_prompt_1(self):
        return "以下是中国精神医学专业阶段性考试的一道{question_type}，请分析每个选项，并最后给出答案。\n{question}\n{option_str}"
    @property
    def query_prompt_2(self):
        return "以下是中国精神医学专业阶段性考试的一道{question_type}，不需要做任何分析和解释，直接输出答案选项。\n{question}\n{option_str}"
    @property
    def query_prompt_3(self):
        return "以下是中国精神医学专业阶段性考试的一道{question_type}，请根据你的知识给出专业详细的回答。\n{question}"
    @property
    def query_prompt_4(self):
        # return "扮演一名专业的精神心理临床医生进行诊疗，请根据你的知识完成下面的问题。\n{question}"
        
        return TASK4

    @property
    def fewshot_prompt(self, ):
        '''the string that starts the fewshot example.
        default: `""`'''
        return ""
    @property
    def fewshot_separator(self, ):
        '''the string that separates fewshot examples.
        default: `""`'''
        return ""
    
    @property
    def fewshot_template(self):
        raise NotImplementedError
    
    def collate_conv(self, data, convs):
        raise NotImplementedError
    
    def init_generation_config(self, config):
        if config.generation_config.get('num_return_sequences', None) is None:
            config.generation_config['num_return_sequences'] = 1
        elif config.generation_config.get('num_return_sequences', 1) > 1 and config.generation_config.get('do_sample', False):
            self.print_in_main('`num_return_sequences` must be 1 when using `do_sample=True`. Setting `num_return_sequences=1`')
            config.generation_config['num_return_sequences'] = 1

        if self.use_qa:
            config.generation_config['repetition_penalty'] = 1.1
        
        self.generation_config = config.generation_config

        if (self.tokenizer.pad_token_id is None) and (self.tokenizer.eos_token_id is not None):
            self.print_in_main('warning: No pad_token in the config file. Setting pad_token_id to eos_token_id')
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            assert self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        self.print_in_main(f'Generation config: {self.generation_config}')

    def init_dataloader(self, input_pth, batch_size):
        dataset = MyDataset(input_pth)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=dataset.collate_fn,
        )
        self.dataloader = dataloader 
        self.wrap_dataloader()

    def wrap_dataloader(self):
        self.dataloader = self.accelerator.prepare(self.dataloader)

    def wrap_model(self,):
        self.model = self.accelerator.prepare(self.model)

    def unwrap_model(self,): # this is NOT inplace
        return self.accelerator.unwrap_model(self.model)
    
    def init_writer(self, output_pth):
        if self.is_main_process:
            lines = []
            ## if unfinished generation exist
            if os.path.exists(output_pth):
                # 读取现有文件内容
                with open(output_pth, 'r', encoding="utf-8") as f:
                    for line in f:
                        lines.append(json.loads(line))
                # with open(output_pth, 'w', encoding="utf-8") as f:
                #     json.dump(lines, f, ensure_ascii=False)
            else:
                print('{} not existing, generate it.'.format(output_pth))
            self.writer = open(output_pth, "w", encoding='utf-8')
            for line in lines:
                self.writer.write(json.dumps(line, ensure_ascii=False) + "\n")
            print('continue gen ans to file :{}'.format(output_pth))
    def is_main_process(self):
        return self.accelerator.is_main_process
    def print_in_main(self, *args, **kwargs):
        if self.is_main_process:
            print(*args, **kwargs)

    def close(self,):
        if self.accelerator.is_main_process:
            self.writer.close()
    
    @torch.no_grad()
    def generate_batch(self, batch: list):
        r"""
        Args:
            batch (`List[dict]`):
                a list of raw data.
        Returns:
            outputs (`List[str]`):
                a list of generated output from the model.
        Usage:
            runner.generate(prompts)
        """

        if self.use_qa:
            batch, lines = self.prompt_wrapper.wrap_conv(batch) 
        else:
            batch, lines = self.prompt_wrapper.wrap(batch) 
        
        inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
        self.prompt_wrapper.lengths = inputs.input_ids.shape[1]
        print('batch:',batch,'\nlength:',inputs.input_ids.shape)
        outputs = self.unwrap_model().generate( **inputs, **self.generation_config)
        outputs = self.prompt_wrapper.unwrap(outputs, self.generation_config.get('num_return_sequences', 1))
        
        return outputs, lines


    def get_single_query(self, datum, use_cot):
        if 'question_type' in datum.keys():
            if '选择题' in datum['question_type']:
                datum["option_str"] = "\n".join(
                    [f"{k}. {v}" for k, v in datum["option"].items() if len(v) > 1]
                )
            if use_cot:
                query = self.query_prompt_1.format_map(datum)
            else:
                if '选择题' in datum['question_type']:
                    query = self.query_prompt_2.format_map(datum)
                else:
                    query = self.query_prompt_3.format_map(datum)
        else:
            datum['question'] = datum['conversations'][0]['value']
            query = self.query_prompt_4.format_map(datum)

        return query
    
    def format_fewshot_user_and_gpt(self, item: dict, use_cot):
        if "answer" in item.keys():
            user = self.get_single_query(item, use_cot)
            explanation = item["explanation"]
            answer = item["answer"]
        else:
            if not item["conversations"][0]['from'] == 'human':
                item["conversations"] = item["conversations"][1:]
            # user = item["conversations"][0]['value']
            user = self.get_single_query(item, use_cot)
            assert item["conversations"][1]['from'] == 'gpt'
            answer = item["conversations"][1]['value']
            

        if use_cot:
            # gpt = '[答案]\n' + "\n".join([f'{answer_item}. ' + item["option"][answer_item] for answer_item in answer]) + '\n'
            # gpt += f"[思考过程]\n{explanation}\n[结论]\n所以答案是{answer}。"
            gpt = f"{explanation}所以答案是{answer}。"
        else:
            if "answer" in item.keys():
                gpt = f'答案是{answer}。'
            else:
                gpt = f'{answer}'

        ''' zero-shot
        {role1}:
        以下是考试题，请给答案。
        李时珍写了啥？
        B. 黄帝内经
        C. 本草纲目
        {role2}
        本草纲目是李时珍写的。所以答案是C。
        '''

        ''' few-shot
        {role1}:
        以下是考试题，请给答案。
        李时珍写了啥？
        B. 黄帝内经
        C. 本草纲目
        {role2}
        答案是C。
        '''
        return user, gpt
    
    def generate_fewshot_examples(self, data: list[dict], use_cot=False):
        """
        Generate a fewshot prompt given a list of data.
        Note that the roles are already in the fewshot examples.
        Be careful not to add any extra roles in the final query that is input to an LLM.
        """
        prompt = self.fewshot_prompt
        for item in data:
            user, gpt = self.format_fewshot_user_and_gpt(item, use_cot)
            prompt += self.fewshot_template.format(user=user, gpt=gpt) + self.fewshot_separator
        return prompt
    


class PromptWrapper():
    def __init__(
            self, 
            tokenizer, 
            instruction_template, 
            conv_collater,
            use_cot=False
    ):

        self.instruction_template = instruction_template

        self.question_template_option,self.question_template_nonoption = self.get_question_template(use_cot=use_cot)

        if '{fewshot_examples}' in self.instruction_template:
            # use fewshot examples
            # keep the fewshot placeholder, since examples are sample-specific 
            self.input_template_option = self.instruction_template.format(instruction=self.question_template_option, fewshot_examples='{fewshot_examples}') 
            self.input_template_nonoption = self.instruction_template.format(instruction=self.question_template_nonoption, fewshot_examples='{fewshot_examples}') 

        else:
            self.input_template_option = self.instruction_template.format(instruction=self.question_template_option)
            self.input_template_nonoption = self.instruction_template.format(instruction=self.question_template_nonoption)

        self.conv_collater = conv_collater # for multi-turn QA only, implemented for each model
        self.tokenizer = tokenizer


    def get_system_template(self, t):
        if t.strip() == '':
            return '{instruction}'
        else:
            try:
                t.format(instruction='')
            except:
                raise Exception('there must be a {instruction} placeholder in the system template')
        return t
    
    def get_question_template(self, use_cot):
        if use_cot:
            return ["以下是中国精神医学专业阶段性考试的一道{question_type}，请分析每个选项，并最后给出答案。\n{question}\n{option_str}",\
                    "以下是中国精神医学专业阶段性考试的一道{question_type}，请根据你的知识给出专业详细的回答。\n{question}"]
        else:
            return ["以下是中国精神医学专业阶段性考试的一道{question_type}，不需要做任何分析和解释，直接输出答案选项。\n{question}\n{option_str}",\
                    "以下是中国精神医学专业阶段性考试的一道{question_type}，请根据你的知识给出专业详细的回答。\n{question}"]

    def wrap(self, data: list[dict]):
        '''
        data.keys(): ['id', 'exam_type', 'exam_class', 'question_type', 'question', 'option']. These are the raw data.
        We still need 'option_str'.
        '''
        res = []
        lines = []
        for line in data:
            if 'question_type' in line.keys():
                if '选择题' in line['question_type']:
                    line["option_str"] = "\n".join(
                        [f"{k}. {v}" for k, v in line["option"].items() if len(v) > 1]
                    )
                    query = self.input_template_option.format_map(line)
                else:
                    query = self.input_template_nonoption.format_map(line)
            else:
                if not line["conversations"][0]['from'] == 'human':
                    line["conversations"] = line["conversations"][1:]
                line['question'] = line['conversations'][0]['value']
                input_template_nonoption_clinical = self.input_template_nonoption.replace('以下是中国精神医学专业阶段性考试的一道{question_type}','扮演一名专业的精神心理临床医生进行诊疗')
                query = input_template_nonoption_clinical.format_map(line)
            line['query'] = query

            res.append(query)
            lines.append(line)
        
        return res, lines
    
    def wrap_conv(self, data: list[dict]): # add
        lines = []
        res = []
        for line in data:
            # print(line)
            collated, partial_qa = self.conv_collater(line)
            # collated: ['Q', 'QAQ', 'QAQAQ', ...]
            # partial_qa: [
            #   [{'q': 'q'}], 
            #   [{'q': 'q', 'a': 'a'}, {'q'}], 
            #   [{'q': 'q', 'a': 'a'}, {'q': 'q', 'a': 'a'}, {'q': 'q'}]
            # ]
            res.extend(collated) # 1d list
            lines.extend(partial_qa)           
        return res, lines

    def unwrap(self, outputs, num_return_sequences):        
        batch_return = []
        responses_list = []
        for i in range(len(outputs)):
            # sample_idx = i // num_return_sequences
            output = outputs[i][self.lengths: ] # slicing on token level
            output = self.tokenizer.decode(output, skip_special_tokens=True)

            batch_return.append(output)
            if i % num_return_sequences == num_return_sequences - 1:
                responses_list.append(batch_return)
                batch_return = []
        return responses_list


class MyDataset(Dataset):
    def __init__(self, input_path):
        # data = []
        with open(input_path) as f:
            data = json.load(f)
        print(f"loading {len(data)} data from {input_path}")
        self.data = data


    def __getitem__(self, index):
        item: dict = self.data[index]
        return item

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        # print(batch); exit()
        '''
        [id: '', title: '', description: '', QA_pairs: [
            {question: '', answer: ''},
            {question: '', answer: ''},
        ]]
        '''
        return batch


