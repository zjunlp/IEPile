# IEPile: A Large-Scale Information Extraction Instruction Corpus

<p align="left">
    <b> English | <a href="">Chinese</a> </b>
</p>


- [IEPile: Unearthing Large-Scale Schema-Based Information Extraction Corpus](#iepile-unearthing-large-scale-schema-based-information-extraction-corpus)
  - [🎯1.Introduction](#1introduction)
  - [📊2.Data](#2data)
    - [2.1Construction of IEPILE](#21construction-of-iepile)
    - [2.2Statistics of IEPILE](#22statistics-of-iepile)
  - [🚴3Using IEPILE to Train Models](#3using-iepile-to-train-models)
    - [3.1Environment](#31environment)
    - [3.2Download Data](#32download-data)
    - [3.3Models](#33models)
    - [3.4LoRA Fine-tuning](#34lora-fine-tuning)
      - [3.4.1Fine-tuning LLaMA2 with LoRA](#341fine-tuning-llama2-with-lora)
      - [3.4.3LoRA Fine-tuning Baichuan2](#343lora-fine-tuning-baichuan2)
      - [3.4.3LoRA Fine-tuning Other Models](#343lora-fine-tuning-other-models)
    - [3.5Continued Model Training](#35continued-model-training)
      - [3.5.1Training Data Conversion](#351training-data-conversion)
      - [3.5.2Continued Training](#352continued-training)
  - [4.Prediction](#4prediction)
    - [4.1Test Data Conversion](#41test-data-conversion)
    - [4.2IE-Specific Model Prediction](#42ie-specific-model-prediction)
    - [4.3Basic Model + LoRA Prediction](#43basic-model--lora-prediction)
  - [5.Evaluation](#5evaluation)
- [6. Statement and License](#6-statement-and-license)
- [7. Limitations](#7-limitations)


## 🎯1.Introduction


**`IEPILE`** dataset download links: [Google Drive](https://drive.google.com/file/d/1jPdvXOTTxlAmHkn5XkeaaCFXQkYJk5Ng/view?usp=sharing) | [Hugging Face](https://huggingface.co/datasets/zjunlp/IEPILE)


> Please be aware that the data contained in the dataset links provided above has already excluded any part related to the ACE2005 dataset. Should you require access to the unfiltered, complete dataset and have successfully obtained the necessary permissions, please do not hesitate to contact us via email at guihonghao@zju.edu.cn or zhangningyu@zju.edu.cn. We will provide the complete dataset resources for your use.


Model download links for **`LLaMA2-IEPILE`** | **`Baichuan2-IEPILE`** | **`KnowLM-IE-v2`**: [zjunlp/llama2-13b-iepile-lora](https://huggingface.co/zjunlp/llama2-13b-iepile-lora/tree/main) | [zjunlp/baichuan2-13b-iepile-lora](https://huggingface.co/zjunlp/baichuan2-13b-iepile-lora) | [zjunlp/KnowLM-IE-v2]()


**Large Language Models (LLMs)** demonstrate remarkable potential across various domains; however, they exhibit a significant performance gap in **Information Extraction (IE)**. Note that high-quality instruction data is the vital key for enhancing the specific capabilities of LLMs, while current IE datasets tend to be small in scale, fragmented, and lack standardized schema. To this end, we introduce **IEPILE**, a comprehensive bilingual (English and Chinese) IE instruction corpus, which contains approximately **0.32B** tokens. We construct IEPILE by collecting and cleaning 33 existing IE datasets, and introduce schema-based instruction generation to unearth a large-scale corpus. Experimental results on LLaMA and Baichuan demonstrate that using IEPILE can enhance the performance of LLMs for IE, especially the zero-shot generalization. We open-source the resource and pre-trained models, hoping to provide valuable support to the NLP community.



![statistic](./assets/statistic.jpg)


We collected a total of 15 English NER datasets, 3 Chinese NER datasets, 8 English RE datasets, 2 Chinese RE datasets, as well as 3 English EE datasets and 2 Chinese EE datasets. Figure 1 shows the statistical information of these datasets, covering a wide range of fields including **general**, **medicine**, **financial**, and more. We not only standardized the data format across various tasks but also conducted a meticulous audit for each dataset, creating detailed **data records** including quantity, domain, schema, and other important information.



Based on **IEPILE**, we fine-tuned the `Baichuan2-13B-Chat` and `LLaMA2-13B-Chat` models using `Lora` technology. The experimental results showed that the fine-tuned models `Baichuan2-IEPILE` and `LLaMA2-IEPILE` not only achieved comparable results on fully supervised training sets but also saw significant improvements in **zero-shot information extraction**.


![zero_en](./assets/zero_en.jpg)

![zero_zh](./assets/zero_zh.jpg)


<details>
  <summary><b>Supervision Results</b></summary>

![supervision_ner](./assets/supervision_ner.jpg)

![supervision_re](./assets/supervision_re.jpg)

![supervision_ee](./assets/supervision_ee.jpg)

</details>


## 📊2.Data


### 2.1Construction of IEPILE

We concentrate on schema-based IE, thus the construction of schema within the instructions is crucial. This is because they reflect the specific extraction requirements and are dynamically variable. Previous approaches with existing IE datasets often employ a rather extensive schema processing strategy when constructing instructions, utilizing all schemas within a label set for instruction building, raising two potential issues: 
1. **Inconsistency in the number of schema queries within instruction between training and evaluation**. For example, the model's performance will decrease if it is trained on about 20 schema queries but tested with either 10 or 30, even if the training and evaluation schemas are similar in content.
2. **Inadequate differentiation among schemas in the instructions**. For example, semantically similar schemas like "layoffs", "depart" and "dismissals", may present co-occurrence ambiguities that could confuse the LLMs. Such schemas should co-occur more frequently within the instruction.

Therefore, we introduce the following solutions: 1) Hard Negative Schema; and 2) Batched Instruction Generation.


![iepile](./assets/iepile.jpg)


<details>
  <summary><b>Hard Negative Schema</b></summary>

Assuming that dataset $\mathcal{D}$ possesses a full label set $L$. For a given text $S$, the schemas present in its annotation constitute the positive schema set $Pos\_L$, while others form the negative schema set $Neg\_L$. In our analysis, we discover that the primary cause of model misjudgment stems from the semantic ambiguity of the schema.  In traditional approaches, the $Neg\_L$ is simply defined as $L - Pos\_L$. However, they overlook a critical aspect: it is important to pay special attention to negative schemas that are semantically close to positive schemas. Inspired by the theory of contrastive learning, we construct a hard negative schema dictionary $\mathcal{K}$, where each key represents a unique schema and the associated value is a collection of schemas that are semantically similar to the key schema. Based on this, we define the hard negative schema set as $Hard\_L = \mathcal{K}[Pos\_L]$, and the other negative schema set as $Other\_L = L - Pos\_L - Hard\_L$. The final $Neg\_L$ is constituted by $Hard\_L$ and a small subset of $Other\_L$. Through this strategy, we not only present semantically similar schemas more frequently within the instruction but also reduce the number of training instances without sacrificing model performance.

</details>


<details>
  <summary><b>Batched Instruction Generation</b></summary>

Subsequently, we obtain the final schema set $L' = Pos\_L + Neg\_L$. We employ a batched instruction generation method, limiting the number of schemas inquired in each instruction to the number of $split\_num$, which ranges between 4 to 6. Therefore, $L'$ will be divided into $|L'|/split\_num$ batches for querying, with each batch querying $split\_num$ schemas. Consequently, even if the number of schemas inquired during the evaluation phase differs from that of training, the batched mechanism allows us to distribute the inquiries across $split\_num$ schemas, thereby mitigating the decline in generalization performance.

</details>


**Instruction Format**

The **instruction** format of `IEPILE` adopts a JSON-like string structure, which is essentially a dictionary-type string composed of the following three main components:
(1) **`'instruction'`**, which is the task description outlining the execution goal of the instruction;
(2) **`'schema'`**, a list of labels to be extracted, clearly indicating the key fields of the information to be extracted;
(3) **`'input'`**, referring to the source text used for information extraction.


Here is an example of an instruction for executing a NER task:
```json
{
    "instruction": "You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.",
    "schema": ["location", "else", "organization", "person"],
    "input": "The objective of the Basic Course on War is to provide for combatants of the EPR basic military knowledge for the armed conflict against the police and military apparatus of the bourgeoisie."
}
```

Please note that the above dictionary should be in the format of a JSON string. For the sake of clear demonstration, it has been modified into a dictionary format.

<details>
  <summary><b>More Tasks</b></summary>


```json
{
    "instruction": "You are an expert in relationship extraction. Please extract relationship triples that match the schema definition from the input. Return an empty list for relationships that do not exist. Please respond in the format of a JSON string.", 
    "schema": ["children", "country capital", "country of administrative divisions", "company"], 
    "input": "Born on May 1 , 1927 , in Brichevo , Bessarabia in the present-day Republic of Moldova , Mr. Bertini emigrated to Palestine with his family as a child and pursued musical studies there , in Milan , and in Paris , where he worked with Nadia Boulanger and Arthur Honegger."
}

{
    "instruction": "You are an expert in event extraction. Please extract events from the input that conform to the schema definition. Return an empty list for events that do not exist, and return NAN for arguments that do not exist. If an argument has multiple values, please return a list. Respond in the format of a JSON string.", 
    "schema": [{"event_type": "pardon", "trigger": true, "arguments": ["defendant"]},{"event_type": "extradite", "trigger": true, "arguments": ["person", "agent", "destination", "origin"]}, {"event_type": "sue", "trigger": true, "arguments": ["place", "plaintiff"]}, {"event_type": "start organization", "trigger": true, "arguments": ["organization", "agent", "place"]}], 
    "input": "Ethical and legal issues in hiring Marinello"
}
```

</details>

The file [instruction.py](./ie2instruction/convert/utils/instruction.py) provides instructions for various tasks.



### 2.2Statistics of IEPILE
Based on the aforementioned methods, we obtain a high-quality information extraction instruction dataset, known as **`IEPILE`**. This dataset contains approximately **over 2 million** instruction entries, each comprising of `instruction` and `output` fields, which can be directly used for supervised fine-tuning of models. In terms of storage size, IEPILE occupies about **3GB** of disk space and contains roughly **0.32B** tokens (using the baichuan2 tokenizer).


```json
{
    "task": "NER", 
    "source": "CoNLL2003", 
    "instruction": "{\"instruction\": \"You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.\", \"schema\": [\"person\", \"organization\", \"else\", \"location\"], \"input\": \"284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )\"}", 
    "output": "{\"person\": [\"Robert Allenby\", \"Allenby\", \"Miguel Angel Martin\"], \"organization\": [], \"else\": [], \"location\": [\"Australia\", \"Spain\"]}"
}
```

<details>
  <summary><b>More Tasks</b></summary>

```json
{
  "task": "EE", 
  "source": "PHEE", 
  "instruction": "{\"instruction\": \"You are an expert in event extraction. Please extract events from the input that conform to the schema definition. Return an empty list for events that do not exist, and return NAN for arguments that do not exist. If an argument has multiple values, please return a list. Respond in the format of a JSON string.\", \"schema\": [{\"event_type\": \"potential therapeutic event\", \"trigger\": true, \"arguments\": [\"Treatment.Time_elapsed\", \"Treatment.Route\", \"Treatment.Freq\", \"Treatment\", \"Subject.Race\", \"Treatment.Disorder\", \"Effect\", \"Subject.Age\", \"Combination.Drug\", \"Treatment.Duration\", \"Subject.Population\", \"Subject.Disorder\", \"Treatment.Dosage\", \"Treatment.Drug\"]}, {\"event_type\": \"adverse event\", \"trigger\": true, \"arguments\": [\"Subject.Population\", \"Subject.Age\", \"Effect\", \"Treatment.Drug\", \"Treatment.Dosage\", \"Treatment.Freq\", \"Subject.Gender\", \"Treatment.Disorder\", \"Subject\", \"Treatment\", \"Treatment.Time_elapsed\", \"Treatment.Duration\", \"Subject.Disorder\", \"Subject.Race\", \"Combination.Drug\"]}], \"input\": \"Our findings reveal that even in patients without a history of seizures, pregabalin can cause a cortical negative myoclonus.\"}", 
  "output": "{\"potential therapeutic event\": [], \"adverse event\": [{\"trigger\": \"cause \", \"arguments\": {\"Subject.Population\": \"NAN\", \"Subject.Age\": \"NAN\", \"Effect\": \"cortical negative myoclonus\", \"Treatment.Drug\": \"pregabalin\", \"Treatment.Dosage\": \"NAN\", \"Treatment.Freq\": \"NAN\", \"Subject.Gender\": \"NAN\", \"Treatment.Disorder\": \"NAN\", \"Subject\": \"patients without a history of seizures\", \"Treatment\": \"pregabalin\", \"Treatment.Time_elapsed\": \"NAN\", \"Treatment.Duration\": \"NAN\", \"Subject.Disorder\": \"NAN\", \"Subject.Race\": \"NAN\", \"Combination.Drug\": \"NAN\"}}]}"
}

{
  "task": "RE", 
  "source": "NYT11", 
  "instruction": "{\"instruction\": \"You are an expert in relationship extraction. Please extract relationship triples that match the schema definition from the input. Return an empty list for relationships that do not exist. Please respond in the format of a JSON string.\", \"schema\": [\"neighborhood of\", \"nationality\", \"children\", \"place of death\"], \"input\": \" In the way New Jersey students know that Thomas Edison 's laboratory is in West Orange , the people of Colma know that Wyatt Earp 's ashes are buried at Hills of Eternity , a Jewish cemetery he was n't ; his wife was , and that Joe DiMaggio is at Holy Cross Cemetery , where visitors often lean bats against his gravestone . \"}", 
  "output": "{\"neighborhood of\": [], \"nationality\": [], \"children\": [], \"place of death\": [{\"subject\": \"Thomas Edison\", \"object\": \"West Orange\"}]}"
}
```

</details>


Descriptions of the fields:

|Field|Description|
|:---:|:---:|
|task|The task demonstrated by the instance, one of the five types (NER, RE, EE, EET, EEA).                           |
|source|The dataset demonstrated by the instance.|
|instruction|The instruction inputted into the model, processed into a JSON string by json.dumps, includes `"instruction"`, `"schema"`, and `"input"` fields.|
|output|The model's output, formatted as a dictionary's JSON string, where the key is the schema and the value is the content extracted.|


## 🚴3Using IEPILE to Train Models

### 3.1Environment

Before you begin, make sure to create an appropriate **virtual environment** following the instructions below:

```bash
conda create -n IEPILE python=3.9   # Create a virtual environment
conda activate IEPILE               # Activate the environment
pip install -r requirements.txt     # Install dependencies
```


### 3.2Download Data

**`IEPILE`** dataset download links: [Google Drive](https://drive.google.com/file/d/1jPdvXOTTxlAmHkn5XkeaaCFXQkYJk5Ng/view?usp=sharing) | [Hugging Face](https://huggingface.co/datasets/zjunlp/IEPILE)

```bash
mkdir results
mkdir lora
mkdir data
```


Data should be placed in the `./data` directory.



### 3.3Models

Here are some of the models supported by the code in this repository:
["`llama`", "`alpaca`", "`vicuna`", "`zhixi`", "`falcon`", "`baichuan`", "`chatglm`", "`qwen`", "`moss`", "`openba`"]

Model download links for **`LLaMA2-IEPILE`** | **`Baichuan2-IEPILE`** | **`KnowLM-IE-v2`**: [zjunlp/llama2-13b-iepile-lora](https://huggingface.co/zjunlp/llama2-13b-iepile-lora/tree/main) | [zjunlp/baichuan2-13b-iepile-lora](https://huggingface.co/zjunlp/baichuan2-13b-iepile-lora) | [zjunlp/KnowLM-IE-v2]()

### 3.4LoRA Fine-tuning


#### 3.4.1Fine-tuning LLaMA2 with LoRA

> Important Note: All the commands below should be executed within the IEPILE directory. For example, if you want to run the fine-tuning script, you should use the following command: `bash ft_scripts/fine_llama.bash`. Please ensure your current working directory is correct.



```bash
output_dir='lora/llama2-13b-chat-v1'
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node=8 --master_port=1287 src/test_finetune.py \
    --do_train --do_eval \
    --overwrite_output_dir \
    --model_name_or_path 'models/llama2-13b-chat' \
    --stage 'sft' \
    --model_name 'llama' \
    --template 'llama2' \
    --train_file 'data/train.json' \
    --valid_file 'data/dev.json' \
    --output_dir=${output_dir} \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --gradient_accumulation_steps 4 \
    --preprocessing_num_workers 16 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --max_grad_norm 0.5 \
    --optim "adamw_torch" \
    --max_source_length 400 \
    --cutoff_len 700 \
    --max_target_length 300 \
    --report_to tensorboard \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --bf16 
```


* `--model_name`: Specifies the model name you wish to use. The current list of supported models includes: ["`llama`", "`alpaca`", "`vicuna`", "`zhixi`", "`falcon`", "`baichuan`", "`chatglm`", "`qwen`", "`moss`", "`openba`"]. Please note that this parameter should be distinguished from `--model_name_or_path`.
* `--model_name_or_path`: The path to the model parameters. Please download the corresponding model from HuggingFace.
* `--template`: The template name being used, which includes: `alpaca`, `baichuan`, `baichuan2`, `chatglm3`, and others. Refer to [src/datamodule/template.py](./src/datamodule/template.py) to see all supported template names, with `alpaca` being the default template.
* `--train_file`, `--valid_file` (optional): The file paths for the training and validation sets, respectively. If valid_file is not provided, the system will by default allocate a number of samples specified by val_set_size from the file designated by train_file to be the validation set. You can also change the number of samples in the validation set by adjusting the val_set_size parameter. Note: The current format for training, validation, and test files only supports JSON format.
* `--output_dir`: Sets the path to save the weights after LoRA fine-tuning.
* `--val_set_size`: Defines the number of samples in the validation set, with the default being 1000.
* `per_device_train_batch_size`, per_device_eval_batch_size: The batch size per GPU device, with 2-4 being recommended for RTX 3090.
* `max_source_length`, `max_target_length`, `cutoff_len`: The maximum input length, maximum output length, and cutoff length. The cutoff length can be simply considered as the maximum input length + maximum output length, and should be set to an appropriate value based on specific needs and memory size.

To learn more about parameter configuration, please refer to the [src/utils/args](./src/args). 

The specific script for fine-tuning the LLaMA2 model can be found in [ft_scripts/fine_llama.bash](./ft_scripts/fine_llama.bash).



#### 3.4.3LoRA Fine-tuning Baichuan2

The specific script for fine-tuning the Baichuan2 model can be found in [ft_scripts/fine_baichuan.bash](./ft_scripts/fine_baichuan.bash).bash.


#### 3.4.3LoRA Fine-tuning Other Models

To fine-tune other models, you just need to adjust the `--model_name`, `--template` parameters. For example, for the `alpaca` model, set `--model_name alpaca`, `--template alpaca`, and for the `chatglm3` model, set `--model_name chatglm`, `--template chatglm3`.


### 3.5Continued Model Training

Although the `Baichuan2-IEPILE` and `LLaMA2-IEPILE` models have undergone extensive instruction fine-tuning on multiple general datasets and thus possess a degree of general information extraction capability, they may still exhibit certain limitations when processing data in specific domains (such as `law`, `education`, `science`, `telecommunications`). To address this challenge, it is recommended to conduct secondary training of these models on datasets specific to these domains. This will help the models better adapt to the semantic and structural characteristics of the specific domains, significantly enhancing their information extraction capability within those domains.



#### 3.5.1Training Data Conversion

Firstly, it's necessary to **format the data** to include `instruction` and `output` fields. For this purpose, we provide a script [convert_func.py](./ie2instruction/convert_func.py), which can batch convert data into a format that can be directly used by the model.


> Before using the [convert_func.py](./ie2instruction/convert_func.py) script, please make sure to refer to the [data](./data) directory. This directory provides detailed instructions on the data format required for each task. Refer to `sample.json` to understand the format of the data before conversion, `schema.json` to see the organization of the schema, and `train.json` to describe the data format after conversion.


```bash
python ie2instruction/convert_func.py \
    --src_path data/NER/sample.json \
    --tgt_path data/NER/train.json \
    --schema_path data/NER/schema.json \
    --language zh \
    --task NER \
    --split_num 6 \       
    --random_sort \
    --split train
```

* `--language`: Supports two languages, `zh` (Chinese) and `en` (English), with different instruction templates used for different languages.
* `--task`: Currently supports five types of tasks: ['RE', 'NER', 'EE', 'EET', 'EEA'].
* `--split_num`: The maximum number of schemas in a single instruction. The default is 4; -1 means no splitting. Recommended splitting numbers vary by task: NER: 6, RE: 4, EE: 4, EET: 4, EEA: 4.
* `--random_sort`: Whether to randomly sort the schemas in the instruction. The default is False, meaning schemas are sorted alphabetically.
* `--split`: Indicates the type of data constructed, ['train', 'test']. `train` will include the converted `output`, while `test` will be the `label`.

After conversion, the training data will have four fields: `task`, `source`, `instruction`, and `output`. You can refer to [Statistics of IEPILE](./README.md#22statistics-of-iepile) to view the format and function of each field.


#### 3.5.2Continued Training


* If you are continuing training from the fine-tuned LoRA weights, you only need to set the `--checkpoint_dir` parameter to the path of the fine-tuned LoRA weights, for example, `'zjunlp/llama2-13b-iepile-lora'`.

* If you are continuing training from the fine-tuned model weights, you only need to set the `--model_name_or_path` parameter to the path of the fine-tuned model weights, for example, `'zjunlp/KnowLM-IE-v2'`.

The specific script for continuing training from the fine-tuned LoRA weights can be found in [ft_scripts/fine_continue.bash](./ft_scripts/fine_continue.bash).




## 4.Prediction

### 4.1Test Data Conversion


Firstly, it's necessary to **format the data** to include `instruction` and `output` fields. For this purpose, we provide a script [convert_func.py](./ie2instruction/convert_func.py), which can batch convert data into a format that can be directly used by the model.


> Before using the [convert_func.py](./ie2instruction/convert_func.py) script, please make sure to refer to the [data](./data) directory. This directory provides detailed instructions on the data format required for each task. Refer to `sample.json` to understand the format of the data before conversion, `schema.json` to see the organization of the schema, and `train.json` to describe the data format after conversion.

```bash
python ie2instruction/convert_func.py \
    --src_path data/NER/sample.json \
    --tgt_path data/NER/test.json \
    --schema_path data/NER/schema.json \
    --language zh \
    --task NER \
    --split_num 6 \
    --split test
```


* `--language`: Supports two languages, `zh` (Chinese) and `en` (English), with different instruction templates used for different languages.
* `--task`: Currently supports five types of tasks: ['RE', 'NER', 'EE', 'EET', 'EEA'].
* `--split_num`: The maximum number of schemas in a single instruction. The default is 4; -1 means no splitting. Recommended splitting numbers vary by task: NER: 6, RE: 4, EE: 4, EET: 4, EEA: 4.
* `--random_sort`: Whether to randomly sort the schemas in the instruction. The default is False, meaning schemas are sorted alphabetically.
* `--split`: Indicates the type of data constructed, ['train', 'test']. `train` will include the converted `output`, while `test` will be the `label`.

After conversion, the test data will have three fields: `id`, `instruction`, `label`. 



### 4.2IE-Specific Model Prediction

```bash
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --stage sft \
    --model_name_or_path 'zjunlp/KnowLM-IE-v2' \
    --model_name 'baichuan' \
    --template 'baichuan2' \
    --do_predict \
    --input_file 'data/input.json' \
    --output_file 'results/KnowLM-IE-v2_output.json' \
    --output_dir 'lora/test' \
    --predict_with_generate \
    --max_source_length 512 \
    --bf16 \
    --max_new_tokens 300
```

* `--model_name`, `--template`, `--bf16` should be consistent with the settings during training.
* `--output_dir`: This can be set to any path as it doesn't carry significance for inference.
* `--input_file`, `--output_file`: The path to the input test file and the path for the prediction output file.
* `--max_source_length`, `--max_new_tokens`: Maximum input and output lengths, adjusted according to device capabilities.


### 4.3Basic Model + LoRA Prediction

```bash
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --stage sft \
    --model_name_or_path 'models/llama2-13B-Chat' \
    --checkpoint_dir 'zjunlp/llama2-13b-iepile-lora' \
    --model_name 'llama' \
    --template 'llama2' \
    --do_predict \
    --input_file 'data/input.json' \
    --output_file 'results/llama2-13b-iepile-lora_output.json' \
    --finetuning_type lora \
    --output_dir 'lora/test' \
    --predict_with_generate \
    --max_source_length 512 \
    --bf16 \
    --max_new_tokens 300
```

* `--checkpoint_dir`: Path to the trained LoRA weights.



## 5.Evaluation

We provide scripts for evaluating the F1 scores for various tasks.

```bash
python ie2instruction/eval_func.py \
  --path1 data/NER/processed.json \
  --task NER 
```

* `--task`: Currently supports five types of tasks: ['RE', 'NER', 'EE', 'EET', 'EEA'].


# 6. Statement and License
We believe that annotated data contains the wisdom of humanity, and its existence is to promote the benefit of all humankind and help enhance our quality of life. We strongly urge all users not to use our corpus for any actions that may harm national or public security or violate legal regulations.
We have done our best to ensure the quality and legality of the data provided. However, we also recognize that despite our efforts, there may still be some unforeseen issues, such as concerns about data protection and risks and problems caused by data misuse. We will not be responsible for these potential problems.
For original data that is subject to usage permissions stricter than the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) agreement, IEPILE will adhere to those stricter terms. In all other cases, our operations will be based on the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) license agreement.



# 7. Limitations

From the data perspective, our study primarily focuses on schema-based IE, which limits our ability to generalize to human instructions that do not follow our specific format requirements. 
Additionally, we do not explore the field of Open Information Extraction (Open IE); however, if we remove schema constraints, our dataset would be suitable for Open IE scenarios.
Besides, IEPILE is confined to data in English and Chinese, and in the future, we hope to include data in more languages.

From the model perspective, due to computational resource limitations, our research only assessed two models: Baichuan and LLaMA, along with some baseline models. Our dataset can be applied to any other large language models (LLMs), such as Qwen, ChatGLM, Gemma.
