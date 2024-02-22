# IEPile: Unearthing Large-Scale Schema-Based Information Extraction Corpus

<p align="left">
    <b> <a href="">English</a> | 简体中文 </b>
</p>


- [IEPile: Unearthing Large-Scale Schema-Based Information Extraction Corpus](#iepile-unearthing-large-scale-schema-based-information-extraction-corpus)
  - [🎯1.介绍](#1介绍)
  - [📊2.数据](#2数据)
    - [2.1IEPILE的构造](#21iepile的构造)
    - [2.2IEPILE的数据统计分析](#22iepile的数据统计分析)
  - [🚴3使用IEPILE训练模型](#3使用iepile训练模型)
    - [3.1环境](#31环境)
    - [3.2下载数据](#32下载数据)
    - [3.3模型](#33模型)
    - [3.4LoRA微调训练](#34lora微调训练)
      - [3.4.1LoRA微调LLaMA2](#341lora微调llama2)
      - [3.4.3LoRA微调Baichuan2](#343lora微调baichuan2)
      - [3.4.3LoRA微调其他模型](#343lora微调其他模型)
    - [3.5模型继续训练](#35模型继续训练)
      - [3.5.1训练数据转换](#351训练数据转换)
      - [3.5.2继续训练](#352继续训练)
  - [4.预测](#4预测)
    - [4.1测试数据转换](#41测试数据转换)
    - [4.2IE专用模型预测](#42ie专用模型预测)
    - [4.3基础模型+Lora预测](#43基础模型lora预测)
  - [5.评估](#5评估)
- [6.声明和许可](#6声明和许可)
- [7.局限](#7局限)


## 🎯1.介绍


**`IEPILE`** 数据集下载链接：[Google Drive](https://drive.google.com/file/d/1jPdvXOTTxlAmHkn5XkeaaCFXQkYJk5Ng/view?usp=sharing) | [Hugging Face](https://huggingface.co/datasets/zjunlp/IEPILE)


> 请注意，以上提供的数据集链接中所含数据已经排除了与ACE2005数据集相关的部分。若您需要访问未经过滤的完整数据集，并且已成功获取所需的权限，敬请通过电子邮件方式联系 guihonghao@zju.edu.cn 或 zhangningyu@zju.edu.cn。我们将提供完整数据集资源。


**`LLaMA2-IEPILE`** | **`Baichuan2-IEPILE`** | **`KnowLM-IE-v2`** 模型下载链接：[zjunlp/llama2-13b-iepile-lora](https://huggingface.co/zjunlp/llama2-13b-iepile-lora/tree/main) | [zjunlp/baichuan2-13b-iepile-lora](https://huggingface.co/zjunlp/baichuan2-13b-iepile-lora) | [zjunlp/KnowLM-IE-v2]()


**大型语言模型(LLMs)** 在各种领域中表现出了显著的潜力，然而，在 **信息提取(IE)** 方面，LLM存在显著的性能差距。当前IE数据集往往规模较小，分布散乱，且schema不规范。我们通过收集和清洗现有的IE数据，并采取本研究所提出的 `基于schema的指令构造方法`，成功创建了一个名为 **IEPILE** 的综合性包含约 `0.32B tokens` 的IE指令微调数据集。实验结果表明，IEPILE显著提高了LLMs在基于schema的IE上的**零样本泛化**能力。我们开源了自己的数据集和代码，为学术界提供了宝贵的支持。

![statistic](./assets/statistic.jpg)

我们总共收集了15个英文NER数据集，3个中文NER数据集，8个英文RE数据集，2个中文RE数据集，以及3个英文EE数据集和2个中文EE数据集。图1展示了这些数据集的统计信息, 覆盖了**通用**、**生物**、**金融**等众多领域。我们不仅统一了各类任务上的数据格式，而且对每个数据集进行了仔细的审计，为每个数据集创建了详细的**数据记录**，包括数据量、领域、模式等重要信息。


基于**IEPILE**，我们使用`Lora`技术对`Baichuan2-13B-Chat`、`LLaMA2-13B-Chat`模型进行了微调。实验结果显示, 微调后的模型`Baichuan2-IEPILE`, `LLaMA2-IEPILE` 不仅在全监督训练集上取得了可比的结果, 还在**零样本信息抽取**上取得了显著提升。


![zero_en](./assets/zero_en.jpg)

![zero_zh](./assets/zero_zh.jpg)


<details>
  <summary><b>全监督数据集结果</b></summary>

![supervision_ner](./assets/supervision_ner.jpg)

![supervision_re](./assets/supervision_re.jpg)

![supervision_ee](./assets/supervision_ee.jpg)

</details>


## 📊2.数据


### 2.1IEPILE的构造

我们专注于基于schema的信息抽取，因此指令中的schema的构造至关重要，因为它反映着具体抽取需求，是动态可变的。然而，现有研究在构造指令时往往采取一种较为**粗放的schema处理策略**，即利用标签集内全部schema进行指令构建。这种方法潜在地存在2个重要的问题：
1. **训练和评估阶段schema询问的数量不一致，即使这些schema在内容上相似，可能损害模型的泛化能力**。若训练过程中每次询问的schema数量大约是20个，而评估时询问的是10个或30个schema，即使这些schema在内容上与训练阶段相似，模型性能仍可能受到影响。
2. **指令中的schema之间的对比性不足**。语义近似的schema，如“裁员”、“离职”与“解雇”，它们的语义模糊性可能造成模型混淆。这类易混淆的模式应当在指令集中更为频繁地出现。
   
因此，我们提出如下解决方案：1、`轮询式的指令生成`；2、`构造难负样本字典`。

![iepile](./assets/iepile.jpg)

<details>
  <summary><b>难负样本</b></summary>


假设数据集 $\mathcal{D}$ 有其全量标签集 $L$，$\mathcal{D}$ 中某一文本 $S$，$S$ 中真实存在的标签构成**正例标签集** $Pos\_L$，而不存在的标签则形成**负例标签集** $Neg\_L$。在我们的分析中，我们发现模型误判的主要原因在于schema的**语义模糊**，导致了模型的混淆。传统方法中，负例标签 $Neg\_L$通常简单地定义为 $L - Pos\_L$。然而，这种方法忽视了一个重要方面：需要特别注意那些与正例标签**语义相近**的负例标签。受**对比学习**理论的启发。我们构造了一个**难负样本字典** $\mathcal{D}$，其键值对应的是Schema及其语义上相近的Schema集。因此**难负样本集** $Hard\_L = \mathcal{D}[Pos\_L]$。然而，若 $Neg\_L$ 仅由 $Hard\_L$ 构成会缺少足够的负例让模型学习。因此，我们定义其他负样本 $Other\_L = L - Hard\_L$，最终，负例标签 $Neg\_L$ 由 $Hard\_L$ 和少量的 $Other\_L$ 组成。这种难负样本的构建旨在促进语义近似的模式更频繁地出现在指令中，同时也能在不牺牲性能的情况下**减少训练样本量**（例如，原本需12个指令集的49个schema可减至3个）。

</details>


<details>
  <summary><b>轮询式的指令生成</b></summary>

在完成了上述步骤后，我们得到了最终的schema集合 $L'=Pos\_L + Neg\_L$。在基于schema的信息抽取（IE）指令构造中，schema的作用至关重要，它直接决定了模型需要抽取的信息类型，并且反映了用户的具体需求。传统做法通常将完整的schema一次性整合入指令中，然而，在本研究中，我们采纳了一种**轮询式方法**，限制每次询问的schema数量为 $split\_num$ 个，取值范围在4至6之间。因此 $L'$ 将被分为 $|L'|/split\_num$ 个批次进行询问，每批次询问 $split\_num$ 个schema。即使在评估阶段询问的schema数目与训练时不同，通过轮询机制，我们可以将询问数量平均分散至 $split\_num$ 个，从而缓解泛化性能下降的问题。

</details>



**指令格式**
`IEPILE` 的**指令**格式采纳了类JSON字符串的结构，实质上是一种字典型字符串，它由以下三个主要部分构成：
(1) **`'instruction'`**，即任务描述，它概述了指令的执行目标；
(2) **`'schema'`**，这是一份需提取的标签列表，明确指出了待抽取信息的关键字段；
(3) **`'input'`**，指的是用于信息抽取的源文本。


以下是一条执行NER任务的指令示例：
```json
{
    "instruction": "You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.",
    "schema": ["location", "else", "organization", "person"],
    "input": "The objective of the Basic Course on War is to provide for combatants of the EPR basic military knowledge for the armed conflict against the police and military apparatus of the bourgeoisie."
}
```

注意上面的字典应该是json字符串的格式, 这里为了直观展示, 改造成了字典格式。

<details>
  <summary><b>更多任务</b></summary>


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

[instruction.py](./ie2instruction/convert/utils/instruction.py) 中提供了各个任务的指令



### 2.2IEPILE的数据统计分析
我们基于上述方法，形成了一个高质量的信息抽取指令数据集，即 **`IEPILE`**。此数据集大约蕴含**200多万**条指令数据，每一条指令数据均包含`instruction`及`output`两个字段，可直接用于监督式微调模型的训练。就存储量而言，IEPILE占据大约**3GB**的磁盘空间，包含约**3.2亿**个token（使用baichuan2 tokenizer）。


```json
{
    "task": "NER", 
    "source": "CoNLL2003", 
    "instruction": "{\"instruction\": \"You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.\", \"schema\": [\"person\", \"organization\", \"else\", \"location\"], \"input\": \"284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )\"}", 
    "output": "{\"person\": [\"Robert Allenby\", \"Allenby\", \"Miguel Angel Martin\"], \"organization\": [], \"else\": [], \"location\": [\"Australia\", \"Spain\"]}"
}
```

<details>
  <summary><b>更多任务</b></summary>


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


各字段的说明:

|字段|说明|
|:---------:|:----------------------------------------------------------:|
|task|该实例所示的任务, (NER、RE、EE、EET、EEA) 5种任务之一。|
|source| 该实例所示的数据集|
|instruction|输入模型的指令,经过json.dumps处理成JSON字符串,包括`"instruction"`, `"schema"`, `"input"`三个字段|
|output|模型的输出,采用字典的json字符串的格式,key是schema,value是抽取出的内容|




## 🚴3使用IEPILE训练模型

### 3.1环境

在开始之前，请确保按照下面的指导创建适当的**虚拟环境**：

```bash
conda create -n IEPILE python=3.9   # 创建虚拟环境
conda activate IEPILE               # 激活环境
pip install -r requirements.txt     # 安装依赖 
```

### 3.2下载数据

**`IEPILE`** 数据集下载链接：[Google Drive](https://drive.google.com/file/d/1jPdvXOTTxlAmHkn5XkeaaCFXQkYJk5Ng/view?usp=sharing) | [Hugging Face](https://huggingface.co/datasets/zjunlp/IEPILE)

```bash
mkdir results
mkdir lora
mkdir data
```

数据放在目录 `./data` 中。


### 3.3模型

以下是本仓库代码支持的一些模型：
["`llama`", "`alpaca`", "`vicuna`", "`zhixi`", "`falcon`", "`baichuan`", "`chatglm`", "`qwen`", "`moss`", "`openba`"]

**`LLaMA2-IEPILE`** | **`Baichuan2-IEPILE`** | **`KnowLM-IE-v2`** 模型下载链接：[zjunlp/llama2-13b-iepile-lora](https://huggingface.co/zjunlp/llama2-13b-iepile-lora/tree/main) | [zjunlp/baichuan2-13b-iepile-lora](https://huggingface.co/zjunlp/baichuan2-13b-iepile-lora) | [zjunlp/KnowLM-IE-v2]()


### 3.4LoRA微调训练

#### 3.4.1LoRA微调LLaMA2

> 重要提示：以下的所有命令均应在IEPILE目录下执行。例如，如果您想运行微调脚本，您应该使用如下命令：bash ft_scripts/fine_llama.bash。请确保您的当前工作目录正确。


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


* `--model_name`: 指定您想要使用的**模型名称**。当前支持的模型列表包括：["`llama`", "`alpaca`", "`vicuna`", "`zhixi`", "`falcon`", "`baichuan`", "`chatglm`", "`qwen`", "`moss`", "`openba`"]。**请注意**，此参数应与 `--model_name_or_path` 区分。
* `--model_name_or_path`: 模型参数路径, 请到[HuggingFace](https://huggingface.co/models)下载相应模型。
* `--template`: 使用的**模板名称**，包括：`alpaca`, `baichuan`, `baichuan2`, `chatglm3`等, 请参考[src/datamodule/template.py](./src/datamodule/template.py)查看所有支持的模版名称，默认使用的是`alpaca`模板。
* `--train_file`, `--valid_file`（可选）: 分别指向训练集和验证集的**文件路径**。如果未提供 valid_file，系统将默认从 train_file 指定的文件中划分出 val_set_size 指定数量的样本作为验证集。您也可以通过调整 val_set_size 参数来改变**验证集的样本数量**。注意：目前的训练、验证、测试文件的格式只支持**json格式**。
* `--output_dir``: 设置LoRA微调后的**权重参数保存路径**。
* `--val_set_size`: 定义**验证集的样本数量**，默认为1000。
* `per_device_train_batch_size`, `per_device_eval_batch_size`: 每台GPU设备上的batch_size, RTX3090建议设置2～4。
* `max_source_length`, `max_target_length`, `cutoff_len`: 最大输入长度、最大输出长度、截断长度, 截断长度可以简单地视作最大输入长度 + 最大输出长度, 需根据具体需求和显存大小设置合适值。

* 要了解更多关于**参数配置**的信息，请参考 [src/utils/args](./src/args) 目录。


微调LLaMA2模型的具体脚本可以在 [ft_scripts/fine_llama.bash](./ft_scripts/fine_llama.bash) 中找到。


#### 3.4.3LoRA微调Baichuan2

微调Baichuan2模型的具体脚本可以在 [ft_scripts/fine_baichuan.bash](./ft_scripts/fine_baichuan.bash) 中找到。


#### 3.4.3LoRA微调其他模型

微调其他模型只需调整`--model_name`, `--template`两个参数, 例如: 对于`alpaca`模型设置`--model_name alpaca`, `--template alpaca`, 对于`chatglm3`模型设置`--model_name chatglm`, `--template chatglm3`。


### 3.5模型继续训练

尽管 `Baichuan2-IEPILE` 和 `LLaMA2-IEPILE` 模型已在多个通用数据集上接受了广泛的指令微调，并因此获得了一定的通用信息抽取能力，但它们在特定领域(如`法律`、`教育`、`科学`、`电信`)的数据处理上可能仍显示出一定的局限性。针对这一挑战，建议对这些模型在特定领域的数据集上进行二次训练。这将有助于模型更好地适应特定领域的语义和结构特征，从而显著增强其在该领域内的信息抽取能力。


#### 3.5.1训练数据转换

首先, 需要将**数据格式化**以包含`instruction`、`output`字段。为此，我们提供了一个脚本 [convert_func.py](./ie2instruction/convert_func.py)，它可以将数据批量转换成模型可以直接使用的格式。

> 在使用 [convert_func.py](./ie2instruction/convert_func.py) 脚本之前，请确保参考了 [data](./data) 目录。该目录详细说明了每种任务所需的数据格式要求。请参考 `sample.json` 以了解转换前数据的格式，`schema.json` 则展示了 schema 的组织结构，而 `train.json` 则描述了转换后的数据格式。



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

* `--language`: 支持`zh`, `en`两种语言, 不同语言使用的指令模版不同。
* `--task`: 目前支持['RE', 'NER', 'EE', 'EET', 'EEA']五类任务。
* `--split_num`: 单个指令中最大schema数量。默认为4, -1表示不切分, 各个任务推荐的切分数量不同: NER:6, RE:4, EE:4, EET:4, EEA:4。
* `--random_sort`: 是否对指令中的schema随机排序, 默认为False, 即按字母顺序排序。
* `--split`: 表示构造的数据类型, ['train', 'test'], `train`会包含转换后的`output`, `test`则是`label`。

转换后的训练数据将具有 `task`, `source`, `instruction`, `output` 4个字段, 可参考[2.2IEPILE的数据统计分析](./README.md#22iepile的数据统计分析)查看各字段格式和作用。


#### 3.5.2继续训练

* 如果是从微调后的lora权重继续训练则只需要设置 `--checkpoint_dir` 参数为微调后lora权重路径, 例如`'zjunlp/llama2-13b-iepile-lora'`。

* 如果是从微调后的模型权重继续训练只需要设置 `--model_name_or_path` 参数为微调后模型权重路径, 例如`'zjunlp/KnowLM-IE-v2'`。


从微调后的lora权重继续训练的具体脚本可以在 [ft_scripts/fine_continue.bash](./ft_scripts/fine_continue.bash) 中找到。


## 4.预测

### 4.1测试数据转换

在对模型进行数据输入之前，需要将**数据格式化**以包含`instruction`字段。为此，我们提供了一个脚本 [kg2instruction/convert_func.py](./kg2instruction/convert_func.py)，它可以将数据批量转换成模型可以直接使用的格式。

> 在使用 [kg2instruction/convert_func.py](./kg2instruction/convert_func.py) 脚本之前，请确保参考了 [data](./data) 目录。该目录详细说明了每种任务所需的数据格式要求。请参考 sample.json 以了解转换前数据的格式，schema.json 则展示了 schema 的组织结构，而 test.json 则描述了转换后的数据格式。


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

* `--language`: 支持`zh`, `en`两种语言, 不同语言使用的指令模版不同。
* `--task`: 目前支持['RE', 'NER', 'EE', 'EET', 'EEA']五类任务。
* `--split_num`: 单个指令中最大schema数量。默认为4, -1表示不切分, 各个任务推荐的切分数量不同: NER:6, RE:4, EE:4, EET:4, EEA:4。
* `--random_sort`: 是否对指令中的schema随机排序, 默认为False, 即按字母顺序排序。
* `--split`: 表示构造的数据类型, ['train', 'test'], `train`会包含转换后的`output`, `test`则是`label`。

转换后的测试数据将具有`id`, `instruction`, `label` 3个字段。


### 4.2IE专用模型预测

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

* `--model_name`, `--template`, `--bf16`应该与训练时保持一致。
* `--output_dir`: 无意义随便设置一个路径。
* `--input_file`, `--output_file`: 输入的测试文件路径, 预测输出文件路径。
* `--max_source_length`, `--max_new_tokens`: 最大输入、输出长度, 根据设备条件调整。


### 4.3基础模型+Lora预测

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

* `--checkpoint_dir`: 训练后lora权重路径。


## 5.评估

我们提供了评估各个任务F1分数的脚本。

```bash
python ie2instruction/eval_func.py \
  --path1 data/NER/processed.json \
  --task NER 
```

* `--task`: 目前支持['RE', 'NER', 'EE', 'EET', 'EEA']五类任务。


# 6.声明和许可

我们认为标注数据蕴含着人类的智慧宝库，它的存在是为了促进全人类的利益，并有助于提升我们的生活质量。我们强烈敦促所有的用户不要将我们的语料库用于任何可能对国家或公共安全造成伤害、违反法律法规的行为。

我们竭尽所能地保证所提供数据的质量与其合法性。但我们也意识到，尽管如此，可能还是存在一些不可预见的问题，诸如数据保护的担忧以及数据被滥用可能引起的风险和问题。对于这些潜在的问题，我们将不承担责任。

对于那些受限于比[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)协议更为严格的使用许可的原始数据，IEPILE将恪守那些较为严格的条款。在其他所有情形下，我们的操作将基于[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)许可协议。


# 7.局限

从数据角度来看，我们的研究主要集中在基于schema的信息提取（IE）上，这限制了我们将研究成果推广至不遵循我们特定格式要求的人类指令的能力。此外，我们没有探索开放信息提取（Open IE）领域；然而，如果我们去除schema约束，我们的数据集将适用于开放信息提取场景。此外，我们的数据集目前仅包含英语和中文数据，在未来，我们希望能够包含更多语言的数据。

从模型的角度来看，由于计算资源的限制，我们的研究仅评估了两个模型：Baichuan和LLaMA，以及一些基线模型。我们的数据集可以应用于任何其他的大型语言模型（LLMs），如Qwen、ChatGLM。

