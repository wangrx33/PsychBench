### load data
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from itertools import islice

from zhipuai import ZhipuAI
import time

class PromptWrapper():
    def __init__(
            self, 
            tokenizer, 
            instruction_template, 
            conv_collater,
            use_cot=False,
            prompt_prefix = '扮演一名专业的精神心理临床医生进行诊疗'
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
        self.prompt_prefix = prompt_prefix


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
            return ["{question_type}，请分析并最后给出答案。\n{question}\n{option_str}",\
                    "{question_type}，请根据你的知识给出专业详细的回答。\n{question}"]
        else:
            return ["{question_type}，不需要做任何分析和解释，直接输出答案。\n{question}\n{option_str}",\
                    "{question_type}，请根据你的知识给出专业详细的回答。\n{question}"]

    def wrap(self, data):
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
                input_template_nonoption_clinical = self.input_template_nonoption.replace('以下是中国精神医学专业阶段性考试的一道{question_type}',self.prompt_prefix)
                query = input_template_nonoption_clinical.format_map(line)
            line['query'] = query

            res.append(query)
            lines.append(line)
        
        return res, lines
    
    def wrap_conv(self, data): # add
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
        with open(input_path,encoding='utf-8') as f:
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

def get_dataloader_iterator(dataset_path,batch_size=1,start=1):
    dataset = MyDataset(dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )
    
    dataloader_iterator = (tqdm(dataloader, total=len(dataloader)))

    dataloader_iterator = islice(dataloader_iterator, start - 1, None)
    return dataloader_iterator

class Prompt():
    def __init__(self):
        print('\ndefining prompt')
    
    @property
    def system_prompt(self):
        return ""
    @property
    def instruction_template(self):
        return self.system_prompt + '问：{instruction}\n答：'
    @property
    def instruction_template_with_fewshot(self):
        return self.system_prompt + '{fewshot_examples}[Question]\n问：{instruction}\n答：'
        # return self.system_prompt + '{fewshot_examples}[Question]\n问：{instruction}\n用中文回答：'

    @property
    def fewshot_template(self,):
        return "[Example {round}]\n问：{user}\n答：{gpt}\n"

### Qwen
# https://help.aliyun.com/zh/dashscope/developer-reference/api-details#3317c9d00bnmh 
import random
from http import HTTPStatus
import dashscope
from dashscope import Generation  # dashscope SDK >= 1.14.0
API_KEY = '' 

dashscope.api_key=API_KEY

def call_with_messages(input_message,model):
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': input_message}]
    response = Generation.call(model=model,
                               messages=messages,
                               temperature=0.1,
                               seed=random.randint(1, 10000),
                               result_format='message')
    if response.status_code == HTTPStatus.OK:
        # print(response)
        return response
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        return None

def gen_with_Qwen(output_pth,dataloader_iterator,prompt_wrapper,max_num=10000,model='qwen-max'):
    '''
    model [qwen-max,qwen-plus]
    '''
    
    writer = open(output_pth, "a", encoding='utf-8')
    print('using model ',model)
    for batch_idx, batch in enumerate(dataloader_iterator, start=1):
        batch, lines = prompt_wrapper.wrap(batch)
        input_message = batch[0]
        if batch_idx > max_num: return
        
        print(batch_idx)
        attempt = 0
        success = False

        while attempt < 3 and not success:  
            try:
                response = call_with_messages(input_message,model)
                response = [choice.message.content for choice in response['output']['choices']]
                for idx, _r in enumerate(response):
                    line[f"answer_{idx}"] = _r
                # print(response)
                # print(line)
                writer.write(json.dumps(line, ensure_ascii=False) + "\n")
                success = True
            except Exception as e:  
                print(f"An error occurred: {e}")
                time.sleep(1)  
                attempt += 1  

        if not success:
            line = lines[0]
            print(f"Failed to process item {batch_idx} after 3 attempts.")
            line[f"answer_{idx}"] = 'API call failed'
            writer.write(json.dumps(line, ensure_ascii=False) + "\n")
            # break
    

def gen_with_glm(output_pth,dataloader_iterator,prompt_wrapper,max_num=10000):
    client = ZhipuAI(api_key="") # fill in your APIKey
    

    # record_enhanced = []
    # diag_enhanced = []
    # continue
    # output_pth = './result/API/task3_1_glm.json'
    writer = open(output_pth, "a", encoding='utf-8')
    

    for batch_idx, batch in enumerate(dataloader_iterator, start=1):
        batch, lines = prompt_wrapper.wrap(batch)
        input_message = batch[0]
        if batch_idx >= max_num: return
        # if batch_idx < 20: continue

        print(batch_idx)
        # print(batch[0])
        attempt = 0
        success = False

        while attempt < 3 and not success:  
            try:
                response = client.chat.completions.create(
                    model="glm-4",  
                    messages=[
                        {"role": "user", "content": input_message}
                    ],
                    temperature=0.1
                )
                
                line = lines[0]
                response = [choice.message.content for choice in response.choices]
                for idx, _r in enumerate(response):
                    line[f"answer_{idx}"] = _r
                
                writer.write(json.dumps(line, ensure_ascii=False) + "\n")
                success = True  
            except Exception as e:  
                print(f"An error occurred: {e}")
                time.sleep(1)  
                attempt += 1  

        if not success:
            line = lines[0]
            print(f"Failed to process item {batch_idx} after 3 attempts.")
            # break
            line[f"answer_{idx}"] = 'API call failed'
            
            writer.write(json.dumps(line, ensure_ascii=False) + "\n")



import openai
OPENAI_API_KEY = ''
openai.api_key  = OPENAI_API_KEY 
# !pip install openai==0.27.4 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
import os
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890" 

def get_completion(prompt, model="gpt-3.5-turbo"):
    client = OpenAI(api_key="")
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1, # this is the degree of randomness of the model's output
    )
    return response

def gen_with_gpt(output_pth,dataloader_iterator,prompt_wrapper,max_num=10000,start=0, model="gpt-3.5-turbo"):
    '''
    model [gpt-3.5-turbo,gpt-4,gpt-4o-mini]
    https://platform.openai.com/docs/concepts 
    '''
    # output_pth = './result/API/task3_1_glm.json'
    writer = open(output_pth, "a", encoding='utf-8')
    print('using model ',model)

    for batch_idx, batch in enumerate(dataloader_iterator, start=1):
        batch, lines = prompt_wrapper.wrap(batch)
        input_message = batch[0]
        if batch_idx >= max_num: return
        if batch_idx < start: continue
        print('\n',batch_idx)
        # print(batch[0])
        attempt = 0
        success = False

        while attempt < 3 and not success:  
            try:
                response = get_completion(input_message,model)
                # print(response.choices[0].message.content)
                # record_enhanced.append(response.choices[0].message.content)
                line = lines[0]
                response = [choice.message.content for choice in response.choices]
                for idx, _r in enumerate(response):
                    line[f"answer_{idx}"] = _r
                # print(response)
                # print(line)
                writer.write(json.dumps(line, ensure_ascii=False) + "\n")
                success = True  
            except Exception as e:  
                print(f"An error occurred: {e}")
                time.sleep(1)  
                attempt += 1  

        if not success:
            line = lines[0]
            print(f"Failed to process item {batch_idx} after 3 attempts.")
            # break
            line[f"answer_{idx}"] = 'API call failed'
            # print(response)
            # print(line)
            writer.write(json.dumps(line, ensure_ascii=False) + "\n")



### doubao
from volcenginesdkarkruntime import Ark
DOUBAO_API = ''

def gen_with_doubao(output_pth,dataloader_iterator,prompt_wrapper,max_num=10000,model='Doubao-pro-32k'):
    '''
    model [Doubao-pro-32k,Doubao-pro-128k]
    '''
    
    client = Ark(api_key=DOUBAO_API)
    writer = open(output_pth, "a", encoding='utf-8')
    
    if model == "Doubao-pro-32k":
        print('using model ',model)
        model_id = ""
    if model == "Doubao-pro-128k":
        print('using model ',model)
        model_id = ''
    
    for batch_idx, batch in enumerate(dataloader_iterator, start=1):
        batch, lines = prompt_wrapper.wrap(batch)
        input_message = batch[0]
        if batch_idx >= max_num: return
        print(batch_idx)
        # print(batch[0])
        attempt = 0
        success = False

        while attempt < 3 and not success:  
            try:
                response = client.chat.completions.create(
                    model=model_id, 

                    messages = [
                        {"role": "user", "content": input_message},
                    ],
                    temperature=0.1,
                    max_tokens = 2048,
                    n=1
                )
                # print(response.choices[0].message.content)
                # record_enhanced.append(response.choices[0].message.content)
                line = lines[0]
                response = [choice.message.content for choice in response.choices]
                for idx, _r in enumerate(response):
                    line[f"answer_{idx}"] = _r
                # print(response)
                # print(line)
                writer.write(json.dumps(line, ensure_ascii=False) + "\n")
                success = True  
            except Exception as e:  
                print(f"An error occurred: {e}")
                time.sleep(1)  
                attempt += 1  

        if not success:
            line = lines[0]
            print(f"Failed to process item {batch_idx} after 3 attempts.")
            # break
            line[f"answer_{idx}"] = 'API call failed'
            # print(response)
            # print(line)
            writer.write(json.dumps(line, ensure_ascii=False) + "\n")
    

### ERNIE
import qianfan
os.environ["QIANFAN_ACCESS_KEY"] = ''
os.environ["QIANFAN_SECRET_KEY"] = ''

def gen_with_ernie(output_pth,dataloader_iterator,prompt_wrapper,max_num=1000,start=1,model="ERNIE-4.0-8K"):
    '''
    model [ERNIE-4.0-8k,ERNIE-3.5-8k,ERNIE-3.5-128k]
    '''
    print('using model ',model)
    chat_comp = qianfan.ChatCompletion()
    writer = open(output_pth, "a", encoding='utf-8')
    
    for batch_idx, batch in enumerate(dataloader_iterator, start=1):
        batch, lines = prompt_wrapper.wrap(batch)
        input_message = batch[0]
        if batch_idx >= max_num: return
        if batch_idx < start: continue
        print(batch_idx)
        # print(batch[0])
        attempt = 0
        success = False

        while attempt < 3 and not success:  
            try:
                response = chat_comp.do(model=model, messages=[{
                    "role": "user",
                    "content": input_message
                }],temperature=0.1)
                
                line = lines[0]
                response = [response["body"]['result']]
                for idx, _r in enumerate(response):
                    line[f"answer_{idx}"] = _r
                # print(response)
                # print(line)
                writer.write(json.dumps(line, ensure_ascii=False) + "\n")
                success = True  
            except Exception as e:  
                print(f"An error occurred: {e}")
                time.sleep(1)  
                attempt += 1  

        if not success:
            line = lines[0]
            print(f"Failed to process item {batch_idx} after 3 attempts.")
            # break
            line[f"answer_{idx}"] = 'API call failed'
            # print(response)
            # print(line)
            writer.write(json.dumps(line, ensure_ascii=False) + "\n")

### moonshot Kimi
from openai import OpenAI
KIMI_API = ''
import json

def gen_with_kimi(output_pth, dataloader_iterator, prompt_wrapper, max_num=10000, model='moonshot-v1-32k'):
    '''
    model [moonshot-v1-32k,moonshot-v1-8k,moonshot-v1-128k]
    '''
    print('using model ', model)
    client = OpenAI(
        api_key=KIMI_API,
        base_url="https://api.moonshot.cn/v1",
    )
    target_indices = []
    if len(target_indices) > 0:
        writer = open(output_pth, "r+", encoding='utf-8')  
        original_lines = writer.readlines()  
        writer.seek(0)  
        writer.truncate()  
    else:
        writer = open(output_pth, "a", encoding='utf-8')

    
    
    for batch_idx, batch in enumerate(dataloader_iterator, start=1):
        if batch_idx >= max_num:
            break
        # if batch_idx < 47: continue
        batch, lines = prompt_wrapper.wrap(batch)
        input_message = batch[0]

        if len(target_indices) > 0:
            if batch_idx not in target_indices:
                writer.write(original_lines[batch_idx-1])
                continue

        print('\n', batch_idx)
        attempt = 0
        success = False

        while attempt < 3 and not success:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": input_message}
                    ],
                    temperature=0.1,
                )
                line = lines[0]
                response = [choice.message.content for choice in response.choices]
                for idx, _r in enumerate(response):
                    line[f"answer_{idx}"] = _r
                writer.write(json.dumps(line, ensure_ascii=False) + "\n")
                success = True
            except Exception as e:
                print(f"An error occurred: {e}")
                error_message = str(e)
                
                if "high risk" in error_message:
                    attempt=3
                time.sleep(15)
                attempt += 1

        if not success:
            line = lines[0]
            print(f"Failed to process item {batch_idx} after 3 attempts.")
            line[f"answer_{idx}"] = 'API call failed'
            writer.write(json.dumps(line, ensure_ascii=False) + "\n")

    if len(target_indices) > 0:
        for line in original_lines[batch_idx:]:
            writer.write(line)

    writer.close()
 





### spark
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

#（https://www.xfyun.cn/doc/spark/Web.html）
# SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.5/chat'
SPARKAI_URL = 'wss://spark-api.xf-yun.com/v4.0/chat'
SPARKAI_APP_ID = ''
SPARKAI_API_SECRET = ''
SPARKAI_API_KEY = ''
# SPARKAI_DOMAIN = 'generalv3.5'
SPARKAI_DOMAIN = '4.0Ultra'


def gen_with_sparkle(output_pth,dataloader_iterator,prompt_wrapper,max_num=10000,model="spark-4.0Ultra"):
    '''
    model:[spark-4.0Ultra,spark-max,spark-pro,spark-pro-128k]
    '''

    SPARKAI_APP_ID = ''
    SPARKAI_API_SECRET = ''
    SPARKAI_API_KEY = ''


    if model == "spark-4.0Ultra":
        SPARKAI_URL = 'wss://spark-api.xf-yun.com/v4.0/chat'
        SPARKAI_DOMAIN = '4.0Ultra'
    
    if model=='spark-max':
        SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.5/chat'
        SPARKAI_DOMAIN = 'generalv3.5'

        
    if model=='spark-pro':
        SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.1/chat'
        SPARKAI_DOMAIN = 'generalv3'

    
    if model=='spark-pro-128k':
        SPARKAI_URL = 'wss://spark-api.xf-yun.com/pro-128k'
        SPARKAI_DOMAIN = 'pro-128k'

        
    print("using model ",model)
        
    

    target_indices = []
    if len(target_indices) > 0:
        writer = open(output_pth, "r+", encoding='utf-8')  
        original_lines = writer.readlines()  
        
        os.makedirs(os.path.dirname(output_pth.replace('result','result_retry')),exist_ok=True)
        print('retrying in file:',output_pth.replace('result','result_retry'))
        writer = open(output_pth.replace('result','result_retry'), "w", encoding='utf-8')  
        writer.seek(0)  
        writer.truncate()  
    else:
        writer = open(output_pth, "a", encoding='utf-8')



    for batch_idx, batch in enumerate(dataloader_iterator, start=1):
        batch, lines = prompt_wrapper.wrap(batch)
        input_message = batch[0]
        if batch_idx >= max_num: return
        
        if len(target_indices) > 0:
            if batch_idx not in target_indices:
                writer.write(original_lines[batch_idx-1])
                continue

        print('\n',batch_idx)
        # print(batch[0])
        attempt = 0
        success = False

        idx = 0
        spark = ChatSparkLLM(
            spark_api_url=SPARKAI_URL,
            spark_app_id=SPARKAI_APP_ID,
            spark_api_key=SPARKAI_API_KEY,
            spark_api_secret=SPARKAI_API_SECRET,
            spark_llm_domain=SPARKAI_DOMAIN,
            streaming=False,
            max_tokens=2048,
            temperature=0.1
        )
        while attempt < 3 and not success: 
            try:
                messages = [ChatMessage(
                    role="user",
                    content=input_message
                )]
                handler = ChunkPrintHandler()
                response = spark.generate([messages], callbacks=[handler])
               
                line = lines[0]
                response = [response.generations[0][0].text]
                for idx, _r in enumerate(response):
                    line[f"answer_{idx}"] = _r
                
                writer.write(json.dumps(line, ensure_ascii=False) + "\n")
                success = True  
            except Exception as e:  
                print(f"An error occurred: {e}")
                error_message = str(e)
                
                if "根据相关法律法规，有关信息不予显示" in error_message:
                    attempt=3
                time.sleep(10)  
                attempt += 1  # 

        if not success:
            line = lines[0]
            print(f"Failed to process item {batch_idx} after 3 attempts.")
            # break
            line[f"answer_{idx}"] = 'API call failed'
            # print(response)
            # print(line)
            writer.write(json.dumps(line, ensure_ascii=False) + "\n")
    
    try:
        if len(target_indices) > 0:
            for line in original_lines[batch_idx+1:]:
                writer.write(line)
    except:
        print('done')

    writer.close()

### baichuan
BAICHUAN_API = ''

def gen_with_baichuan(output_pth,dataloader_iterator,prompt_wrapper,max_num=10000,model="Baichuan4"):
    '''
    model [Baichuan4,Baichuan3-turbo,Baichuan3-turbo-128k]
    '''
    print("using model ",model)
    client = OpenAI(
        api_key=BAICHUAN_API,
        base_url="https://api.baichuan-ai.com/v1/",
    )


    target_indices = []
    if len(target_indices) > 0:
        writer = open(output_pth, "r+", encoding='utf-8')  
        original_lines = writer.readlines()  
    
        os.makedirs(os.path.dirname(output_pth.replace('result','result_retry')),exist_ok=True)
        print('retrying in file:',output_pth.replace('result','result_retry'))
        writer = open(output_pth.replace('result','result_retry'), "w", encoding='utf-8')  
        writer.seek(0)  
        writer.truncate()  
    else:
        writer = open(output_pth, "a", encoding='utf-8')
    

    for batch_idx, batch in enumerate(dataloader_iterator, start=1):
        batch, lines = prompt_wrapper.wrap(batch)
        input_message = batch[0]
        if batch_idx >= max_num: return
        
        if len(target_indices) > 0:
            if batch_idx not in target_indices:
                writer.write(original_lines[batch_idx-1])
                continue

        print('\n',batch_idx)
        # print(batch[0])
        attempt = 0
        success = False

        while attempt < 3 and not success:  
            try:
                response = client.chat.completions.create(
                model = model,
                messages = [
                    {"role": "user", "content": input_message}
                ],
                temperature = 0.1,
            )
                
                line = lines[0]
                response = [choice.message.content for choice in response.choices]
                for idx, _r in enumerate(response):
                    line[f"answer_{idx}"] = _r
                
                writer.write(json.dumps(line, ensure_ascii=False) + "\n")
                success = True  
            except Exception as e:  
                print(f"An error occurred: {e}")
                time.sleep(10)  
                attempt += 1  

        if not success:
            line = lines[0]
            print(f"Failed to process item {batch_idx} after 3 attempts.")
            # break
            line[f"answer_{idx}"] = 'API call failed'
            # print(response)
            # print(line)
            writer.write(json.dumps(line, ensure_ascii=False) + "\n")
    
    try:
        if len(target_indices) > 0:
            for line in original_lines[batch_idx+1:]:
                writer.write(line)
    except:
        print('done')

    writer.close()
    
### Yi
API_BASE = "https://api.lingyiwanwu.com/v1"
YI_API = ''


def gen_with_yi(output_pth,dataloader_iterator,prompt_wrapper,max_num=10000,model='yi-large'):
    '''
    model [yi-large,yi-medium,yi-medium-200k]
    '''
    print('using model ',model)
    client = OpenAI(
        api_key=YI_API,
        base_url=API_BASE
    )
    writer = open(output_pth, "a", encoding='utf-8')

    for batch_idx, batch in enumerate(dataloader_iterator, start=1):
        batch, lines = prompt_wrapper.wrap(batch)
        input_message = batch[0]
        if batch_idx >= max_num: return

        print('\n',batch_idx)
        attempt = 0
        success = False

        while attempt < 3 and not success:  
            try:
                response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": input_message}],
                temperature = 0.1,
            )
                
                # print(response.choices[0].message.content)
                # record_enhanced.append(response.choices[0].message.content)
                line = lines[0]
                response = [choice.message.content for choice in response.choices]
                for idx, _r in enumerate(response):
                    line[f"answer_{idx}"] = _r
                # print(response)
                # print(line)
                writer.write(json.dumps(line, ensure_ascii=False) + "\n")
                success = True  
            except Exception as e:  
                print(f"An error occurred: {e}")
                time.sleep(1)  
                attempt += 1  

        if not success:
            line = lines[0]
            print(f"Failed to process item {batch_idx} after 3 attempts.")
            # break
            line[f"answer_{idx}"] = 'API call failed'
            # print(response)
            # print(line)
            writer.write(json.dumps(line, ensure_ascii=False) + "\n")


### hunyuan
# https://console.cloud.tencent.com/cam/capi
# pip install --upgrade tencentcloud-sdk-python
import json
import types
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models

TENCENT_SecretId=''
TENCENT_SecretKey = ''

def gen_with_hunyuan(output_pth,dataloader_iterator,prompt_wrapper,max_num=10000,model="hunyuan-pro"):
    '''
    model [hunyuan-pro,hunyuan-standard-32k,hunyuan-standard-256k]
    '''
    print('using model ', model)
    writer = open(output_pth, "a", encoding='utf-8')
    
    for batch_idx, batch in enumerate(dataloader_iterator, start=1):
        batch, lines = prompt_wrapper.wrap(batch)
        input_message = batch[0]
        if batch_idx >= max_num: return
        print(batch_idx)
        # print(batch[0])
        attempt = 0
        success = False

        while attempt < 3 and not success:  
            try:
        
                cred = credential.Credential(TENCENT_SecretId, TENCENT_SecretKey)
            
                httpProfile = HttpProfile()
                httpProfile.endpoint = "hunyuan.tencentcloudapi.com"

                
                clientProfile = ClientProfile()
                clientProfile.httpProfile = httpProfile
                
                client = hunyuan_client.HunyuanClient(cred, "", clientProfile)

                
                req = models.ChatCompletionsRequest()
                params = {
                    "Model": model,
                    "Messages": [
                        {
                            "Role": "user",
                            "Content": input_message,
                        }
                    ],
                    "Stream": False,
                    "Temperature": 0.1,
                }
                req.from_json_string(json.dumps(params))

                
                resp = client.ChatCompletions(req)
                
                if isinstance(resp, types.GeneratorType):  
                    for event in resp:
                        print(event)
                else:  
                    # print(resp)
                    response = resp
                    # print(resp.Choices[0].Message.Content)
                    line = lines[0]
                    response = [choice.Message.Content for choice in response.Choices]
                    for idx, _r in enumerate(response):
                        line[f"answer_{idx}"] = _r
                    # print(response)
                    # print(line)
                    writer.write(json.dumps(line, ensure_ascii=False) + "\n")
                    success = True  
            except Exception as e:  
                print(f"An error occurred: {e}")
                time.sleep(1)  
                attempt += 1  
        
        if not success:
            line = lines[0]
            print(f"Failed to process item {batch_idx} after 3 attempts.")
            
            line[f"answer_{idx}"] = 'API call failed'
            
            writer.write(json.dumps(line, ensure_ascii=False) + "\n")


### DeepSeek
DEEPSEEK_API = ''
def gen_with_deepseek(output_pth,dataloader_iterator,prompt_wrapper,max_num=10000,model="deepseek-chat"):
    '''
    model [deepseek-chat]
    '''
    print('using model ',model)
    client = OpenAI(api_key=DEEPSEEK_API, base_url="https://api.deepseek.com")
    writer = open(output_pth, "a", encoding='utf-8')

    for batch_idx, batch in enumerate(dataloader_iterator, start=1):
        batch, lines = prompt_wrapper.wrap(batch)
        input_message = batch[0]
        if batch_idx >= max_num: return
        print(batch_idx)
        # print(batch[0])
        attempt = 0
        success = False

        while attempt < 3 and not success:  
            try:
                response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": input_message}],
                temperature = 0.1,
                max_tokens=2048,
            )
                
                # print(response.choices[0].message.content)
                # record_enhanced.append(response.choices[0].message.content)
                line = lines[0]
                response = [choice.message.content for choice in response.choices]
                for idx, _r in enumerate(response):
                    line[f"answer_{idx}"] = _r
                # print(response)
                # print(line)
                writer.write(json.dumps(line, ensure_ascii=False) + "\n")
                success = True  
            except Exception as e: 
                print(f"An error occurred: {e}")
                time.sleep(1)  
                attempt += 1  

        if not success:
            line = lines[0]
            print(f"Failed to process item {batch_idx} after 3 attempts.")
            # break
            line[f"answer_{idx}"] = 'API call failed'
            # print(response)
            # print(line)
            writer.write(json.dumps(line, ensure_ascii=False) + "\n")



### minimax
### MiniMax
import requests
import readline

def gen_with_minimax(output_pth,dataloader_iterator,prompt_wrapper,max_num=10000,model="abab6.5s-chat"):
    group_id = ""
    api_key = ""

    url = f"https://api.minimax.chat/v1/text/chatcompletion_pro?GroupId={group_id}"
    headers = {"Authorization":f"Bearer {api_key}", "Content-Type":"application/json"}

    # tokens_to_generate/bot_setting/reply_constraints可自行修改
    request_body = payload = {
        "model":model,
        "tokens_to_generate":2048*2,
        "reply_constraints":{"sender_type":"BOT", "sender_name":"MM智能助理"},
        "messages":[{'place_holder':0}],
        "bot_setting":[
            {
                "bot_name":"MM智能助理",
                "content":"MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。",
            }
        ],
    }
    
    print('using model ',model)
    writer = open(output_pth, "a", encoding='utf-8')

    for batch_idx, batch in enumerate(dataloader_iterator, start=1):
        batch, lines = prompt_wrapper.wrap(batch)
        input_message = batch[0]
        if batch_idx >= max_num: return
        # if batch_idx < 13: continue
        print(batch_idx)
        # print(batch[0])
        attempt = 0
        success = False

        while attempt < 3 and not success:  
            try:
                request_body["messages"][0]={"sender_type":"USER", "sender_name":"小明", "text":input_message}
                
                response = requests.post(url, headers=headers, json=request_body).json()
                
                line = lines[0]
                response = [response["choices"][0]["messages"][0]['text']]
                for idx, _r in enumerate(response):
                    line[f"answer_{idx}"] = _r
                # print(response)
                # print(line)
                writer.write(json.dumps(line, ensure_ascii=False) + "\n")
                success = True  
            except Exception as e:  
                print(f"An error occurred: {e}")
                time.sleep(10)  
                attempt += 1  

        if not success:
            line = lines[0]
            print(f"Failed to process item {batch_idx} after 3 attempts.")
            # break
            line[f"answer_{idx}"] = 'API call failed'
            # print(response)
            # print(line)
            writer.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    
    ### task1
    task1 = '''请扮演一位专业的精神科临床医生，根据提供的患者信息，提取总结出患者的主诉，并根据ICD-10的相应标准，总结该患者的诊断标准（包括病程标准、症状学标准、严重程度标准和排除标准）。提取总结的主诉应当简明扼要地总结疾病的主要症状及病程，控制在20字以内，各项诊断标准应与ICD-10相对应。
病程标准：包括病程的时长，明确指出症状出现和持续的时间；及病程的模式，描述症状是否有急性、慢性或间歇性发作的特点。
症状学标准：症状学标准指的是与特定精神疾病相关的关键症状（核心症状和附加症状）。核心症状：列出符合诊断标准的主要症状，如“患者表现出明显的抑郁心境，兴趣和愉悦感丧失，精力减退”。附加症状：列举支持诊断的其他相关症状，例如，“伴有失眠、体重减轻和自我评价下降”。在书写病历时，应详细描述患者的具体症状，包括它们是如何表现、何时出现以及它们对患者的日常生活有何影响。
严重程度标准：严重程度标注通常涉及对患者的功能水平进行评估。这可能包括对工作、社交、家庭生活等方面的能力进行评估，根据症状的强度和功能损害程度进行分级。例如，患者是否因为症状而无法工作或维持人际关系。
排除标准：排除标准是指在做出精神科诊断时需要排除的其他可能的诊断。例如，某些症状可能是由于药物副作用或其他医疗条件引起的，因此在做出最终诊断前需要排除这些可能性。
'''

    ### task2
    task2 = '''请扮演一位专业的精神科临床医生，根据下述患者信息按照ICD-10诊断标准给出主要诊断以及共病诊断（若有）的ICD-10代码以及疾病名称（精确到亚型）。仅需要给出诊断的ICD代码及疾病名称，无需进行分析。
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
'''

    ### task3 
    task3 = '''要求：请扮演一位专业的精神科临床医生，根据以下患者信息，进行精神心理疾病之间的临床鉴别诊断分析，给出1个主要诊断和最需要与之鉴别的2个鉴别诊断。
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
    '''

    ### task4 
    task4 = '''请扮演一位专业的精神科临床医生，根据患者的简要病史、查体及精神检查结果、诊断及鉴别诊断给出用药建议。在选择药物及调整剂量时综合考量多方面因素，采取个体化、循证医学为基础的策略，确保治疗方案既能有效控制症状，又能最大程度减少副作用，提高患者的生活质量；注意考虑患者的诊断与检查结果，遵循ICD-10、DSM5等治疗指南，严谨考虑该患者可能的不良反应和药物相互作用信息。
可选的药物包括：['丙戊酸','伏硫西汀','利培酮','劳拉西泮','唑吡坦','喹硫平','地西泮','奋乃静','奥氮平','奥沙西泮','帕利哌酮','帕罗西汀','度洛西汀','拉莫三嗪','文拉法辛','曲唑酮','氟伏沙明','氟哌啶醇','氟西汀','氨磺必利','氯丙嗪','氯氮平','硝西泮','碳酸锂','米氮平','美金刚','舍曲林','艾司西酞普兰','阿戈美拉汀','阿立哌唑','鲁拉西酮','齐拉西酮']
您的回答应当包括以下部分：“
分析：根据提供的信息分析之前用药的疗效情况以及是否需要调整用药等
推荐药物：仅按照推荐顺序给出药物名称即可，多个药物之间用逗号隔开。
”
'''

    ### task5
    task5 = '''请扮演一位专业的精神科临床医生，详细阅读分析下面的患者多期病程及检查结果记录，并回答问题。
'''

    task2_CoT = '''请扮演一位专业的精神科临床医生，根据下述患者信息，首先按照ICD-10诊断标准进行分析，进一步给出主要诊断以及共病诊断（若有）的ICD-10代码以及疾病名称（精确到亚型）。
输出格式为：
1.分析：按照ICD-10诊断标准进行分析\n
2.主要诊断：ICD-10代码及疾病名称\n
3.精神科共病诊断：ICD-10代码及疾病名称，若无则填“无”
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
'''

    task4_CoT = '''请扮演一位专业的精神科临床医生，根据患者的简要病史、查体及精神检查结果、诊断及鉴别诊断给出用药建议。回答时只需给出推荐药物名称，无需进分析。
可选的药物包括：['丙戊酸','伏硫西汀','利培酮','劳拉西泮','唑吡坦','喹硫平','地西泮','奋乃静','奥氮平','奥沙西泮','帕利哌酮','帕罗西汀','度洛西汀','拉莫三嗪','文拉法辛','曲唑酮','氟伏沙明','氟哌啶醇','氟西汀','氨磺必利','氯丙嗪','氯氮平','硝西泮','碳酸锂','米氮平','美金刚','舍曲林','艾司西酞普兰','阿戈美拉汀','阿立哌唑','鲁拉西酮','齐拉西酮']
回答格式为：“
推荐药物：仅按照推荐顺序给出药物名称即可，多个药物之间用逗号隔开。
”
'''

    USE_COT = False

    for TASK_ID in ['1','2','3','4','5']:

        SHOT_NUM = 0
        max_num = 101
        
        dataset_path = './data/fewshot{}_task{}/CMB-Exam-a-psychAiD.json'.format(SHOT_NUM,TASK_ID)

        
        print('-'*25,'Testing task{}'.format(TASK_ID),'-'*25)
        task_prefix = 'task_prefix'
        if USE_COT:
            
            exec('task_prefix = task{}'.format(TASK_ID+ '_CoT'))
        else:
            exec('task_prefix = task{}'.format(TASK_ID))
        print('using prompt:')
        print(task_prefix)

        dataloader_iterator = get_dataloader_iterator(dataset_path)
        prompt_template = Prompt()
        prompt_wrapper = PromptWrapper(
            None,
            prompt_template.instruction_template_with_fewshot,
            conv_collater=None,
            use_cot=False,
            prompt_prefix = task_prefix
        )
        
        ## 
        # print('GLM')
        # gen_with_glm('./result/API/{}shot/task{}_glm4.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)
        
        # print('Qwen')
        # gen_with_Qwen('./result/API/{}shot/task{}_qwen-max.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)
        
        # print('GPT')    
        # # gen_with_gpt('./result/API/{}shot/task{}_gpt-4.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num,start=0,model='gpt-4')
        # # gen_with_gpt('./result/API/{}shot/task{}_gpt-3.5-turbo.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num,start=0,model='gpt-3.5-turbo')
        # # gen_with_gpt('./result/API/{}shot/task{}_gpt-4o-mini.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num,start=0,model='gpt-4o-mini')

        
        # print('Baichuan')
        # gen_with_baichuan('./result/API/{}shot/task{}_baichuan4.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)
        
        # print('ERNIE')
        # gen_with_ernie('./result/API/{}shot/task{}_ernie-4-8k.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)
        
        print('KIMI')
        gen_with_kimi('./result/API/{}shot/task{}_moonshot-v1-128k.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num,model='moonshot-v1-128k')
        
        # print('SPARKLE')
        # gen_with_sparkle('./result/API/{}shot/task{}_spark-4ultra.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)
        
        # print('Doubao')
        # gen_with_doubao('./result/API/{}shot/task{}_doubao-pro-32k.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)
        
        # print('Yi')
        # gen_with_yi('./result/API/{}shot/task{}_yi-large.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)
        
        # print('Hunyuan')
        # gen_with_hunyuan('./result/API/{}shot/task{}_hunyuan-lite.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num,model='hunyuan-lite')
        # # gen_with_hunyuan('./result/API/{}shot/task{}_hunyuan-pro.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num,model='hunyuan-pro')
        
        # print('Deepseek')
        # gen_with_deepseek('./result/API/{}shot/task{}_deepseek.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)
        
        # print('Minimax')
        # gen_with_minimax('./result/API/{}shot/task{}_minimax.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)


        # print("llama3")
        # # Mixtral-8x7B-Instruct\Meta-Llama-3-8B\Meta-Llama-3-70B
        # # gen_with_ernie('./result/API/{}shot/task{}_Meta-Llama-3-8B.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num,start=64,model='Meta-Llama-3-8B')
        # gen_with_ernie('./result/API/{}shot/task{}_Meta-Llama-3-70B.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num,start=79,model='Meta-Llama-3-70B')
        
        # print("Mixtral-8x7B-Instruct")
        # gen_with_ernie('./result/API/{}shot/task{}_Mixtral-8x7B-Instruct.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num,start=0,model='Mixtral-8x7B-Instruct')



    ###### CoT exp  #########################
        # print('GLM')
        # gen_with_glm('./result/API/{}shot_CoT/task{}_glm4.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)
        
        # print('Qwen')
        # gen_with_Qwen('./result/API/{}shot_CoT/task{}_qwen-max.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)
        
        # print('GPT')
        # gen_with_gpt('./result/API/{}shot_CoT/task{}_gpt-4.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num,model='gpt-4')
        # gen_with_gpt('./result/API/{}shot_CoT/task{}_gpt-3.5-turbo.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num,model='gpt-3.5-turbo')
        # gen_with_gpt('./result/API/{}shot_CoT/task{}_gpt-4o-mini.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num,model='gpt-4o-mini')


        
        # print('Baichuan')
        # gen_with_baichuan('./result/API/{}shot_CoT/task{}_baichuan4.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)
        
        # print('ERNIE')
        # gen_with_ernie('./result/API/{}shot_CoT/task{}_ernie-4-8k.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)
        
        # print('KIMI')
        # gen_with_kimi('./result/API/{}shot_CoT/task{}_moonshot-v1-32k.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)
        
        # print('SPARKLE')
        # gen_with_sparkle('./result/API/{}shot_CoT/task{}_spark-4ultra.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)
        
        # print('Doubao')
        # gen_with_doubao('./result/API/{}shot_CoT/task{}_doubao-pro-32k.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)
        
        # print('Yi')
        # gen_with_yi('./result/API/{}shot_CoT/task{}_yi-large.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)
        
        # print('Hunyuan')
        # gen_with_hunyuan('./result/API/{}shot_CoT/task{}_hunyuan-lite.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num,model='hunyuan-lite')
        # # gen_with_hunyuan('./result/API/{}shot_CoT/task{}_hunyuan-pro.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num,model='hunyuan-pro')

        
        # print('Deepseek')
        # gen_with_deepseek('./result/API/{}shot_CoT/task{}_deepseek.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)
        
        # print('Minimax')
        # gen_with_minimax('./result/API/{}shot_CoT/task{}_minimax.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num)
    