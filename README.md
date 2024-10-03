# PsychBench


## ℹ️ How to evaluate your LLM

### Modify model configuration file
<details><summary>Click to expand</summary>

`configs/model_config.yaml`：
```
my_model:
    model_id: 'my_model'
    load:
        # # HuggingFace model weights
        config_dir: "path/to/full/model"

        # # load with Peft
        # llama_dir: "path/to/base"
        # lora_dir: "path/to/lora"

        device: 'cuda'          # only support cuda
        precision: 'fp16'       # 

    # supports all parameters in transformers.GenerationConfig
    generation_config: 
        max_new_tokens: 4096 
        min_new_tokens: 1          
        do_sample: False         
```
</details>

### Modify model worker
<details><summary>Click to expand</summary>

In `workers/mymodel.py`:
1. load model and tokenizer to cpu
   ```
   def load_model_and_tokenizer(self, load_config):
        '''
        Params: 
            load_config: the `load` key in `configs/model_config.yaml`
        Returns:
            model, tokenizer: both on cpu
        '''
        hf_model_config = {"pretrained_model_name_or_path": load_config['config_dir'],'trust_remote_code': True, 'low_cpu_mem_usage': True}
        hf_tokenizer_config = {"pretrained_model_name_or_path": load_config['config_dir'], 'padding_side': 'left', 'trust_remote_code': True}
        precision = load_config.get('precision', 'fp16')
        device = load_config.get('device', 'cuda')

        if precision == 'fp16':
            hf_model_config.update({"torch_dtype": torch.float16})

        model = AutoModelForCausalLM.from_pretrained(**hf_model_config)
        tokenizer = AutoTokenizer.from_pretrained(**hf_tokenizer_config)

        model.eval()
        return model, tokenizer # cpu
   ```

2. system prompt
    ```
    @property
    def system_prompt(self):
        '''
        The prompt that is prepended to every input.
        '''
        return "你是一位专业的精神科临床医生。"
    ```

3. instruction template
    ```
    @property
    def instruction_template(self):
        '''
        The template for instruction input. An '{instruction}' placeholder must be contained.
        '''
        return self.system_prompt + '问：{instruction}\n答：'
    ```

4. instruction template with fewshot examples
    ```
    @property
    def instruction_template_with_fewshot(self,):
        '''
        The template for instruction input. There must be an '{instruction}' placeholder in this template.
        '''
        return self.system_prompt + '{fewshot_examples}问：{instruction}\n答：'  
    ```
    
5. template for each fewshot example
    ```
    @property
    def fewshot_template(self):
        '''
        The template for each fewshot example. Each fewshot example is concatenated and put in the `{fewshot_examples}` placeholder above.
        There must be a `{user}` and `{gpt}` placeholder in this template.
        '''
        return "问：{user}\n答：{gpt}\n" 
    ```
</details>


### Modify /src/constants.py
<details><summary>Click to expand</summary>

```python
from workers.mymodel import MyModelWorker # modify here
id2worker_class = {
"my_model": MyModelWorker,  # modify here
}
```
</details> 



## ℹ️ How to evaluate open source LLMs using API

Set API Keys properly and run API_call.py



### Generate fewshot examples (required if using fewshot)
<details><summary>Click to expand</summary>

Modify `generate_fewshot.sh`:
```bash
model_id="PsychAiD"
n_shot=1

test_path='data/psych/task4.jsonl'
val_path='data/psych/task4.jsonl'
output_dir=data/fewshot1_task4
python ./src/generate_fewshot_psych.py \
--n_shot=$n_shot \
--model_id=$model_id \
--output_dir=$output_dir  \
--val_path=$val_path \
--test_path=$test_path 
```

and run:
```bash
bash generate_fewshot.sh

```

</details>


### Modify the main script 
<details><summary>Click to expand</summary>

`generate_answers.sh`:
```
task_name='Zero-test-cot'   
port_id=27272

model_id="my_model"                                                      # the same as in `configs/model_config.yaml` 

accelerate launch \
    --gpu_ids='all' \                                                   
    --main_process_port 12345 \                                      
    --config_file ./configs/accelerate_config.yaml  \                   # /path/to/accelerate_config
    ./src/generate_answers.py \                                         # main program
    --model_id=$model_id \                                              # model id
    --use_cot \                                                         # whether to use CoT template   
    --use_fewshot \                                                     # whether to use fewshot
    --batch_size 3  \                                                                                   
    --input_path=$test_data_path \                                      # input path
    --output_path=./result/${task_name}/${model_id}/answers.json \      # output path
    --model_config_path="./configs/model_config.yaml"                   # /path/to/model_config
```
</details>




### Run evaluation

Use jupyter notebook to extract and evaluate answers
```
./run_PsychBench.ipynb
```


