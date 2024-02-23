<p align="left">
    <b> <a href="https://github.com/zjunlp/IEPile/tree/main">English</a> | 简体中文 </b>
</p>

# IEPile：大规模信息提取语料库

这是论文 [IEPile: Unearthing Large-Scale Schema-Based Information Extraction Corpus](https://arxiv.org/abs/2402.14710) 的官方仓库


[**数据集**](https://huggingface.co/datasets/zjunlp/iepie) |
[**论文**](https://huggingface.co/papers/2402.14710) |
[**使用方法**](./README_CN.md#🚴3使用IEPile训练模型) |
[**局限性**](./README_CN.md#7局限) |
[**声明和许可**](./README_CN.md#6声明和许可) |
[**引用**](./README_CN.md#)

> 请注意，我们的IEPile可能会进行**更新**（一旦发布更新，我们将通知您）。建议使用最新版本。


- [IEPile：大规模信息提取语料库](#iepile大规模信息提取语料库)
  - [🎯1.介绍](#1介绍)
  - [📊2.数据](#2数据)
    - [2.1IEPile的构造](#21iepile的构造)
    - [2.2IEPile的数据格式](#22iepile的数据格式)
  - [🚴3.使用IEPile训练模型](#3使用iepile训练模型)
    - [3.1环境](#31环境)
    - [3.2下载数据和模型](#32下载数据和模型)
    - [3.3LoRA微调](#33lora微调)
  - [🔁4.领域内数据继续训练](#4领域内数据继续训练)
      - [4.1训练数据转换](#41训练数据转换)
      - [4.2继续训练](#42继续训练)
  - [👓5.预测](#5预测)
    - [5.1测试数据转换](#51测试数据转换)
    - [5.2基础模型+Lora预测](#52基础模型lora预测)
    - [5.3IE专用模型预测](#53ie专用模型预测)
  - [6.评估](#6评估)
  - [7.声明和许可](#7声明和许可)
  - [8.局限](#8局限)
  - [9.引用](#9引用)
  - [10. 致谢](#10-致谢)


## 🎯1.介绍


**`IEPile`** 数据集下载链接：[Google Drive](https://drive.google.com/file/d/1jPdvXOTTxlAmHkn5XkeaaCFXQkYJk5Ng/view?usp=sharing) | [Hugging Face](https://huggingface.co/datasets/zjunlp/iepile)


> 请注意，以上提供的数据集链接中所含数据已经排除了与ACE2005数据集相关的部分。若您需要访问未经过滤的完整数据集，并且已成功获取所需的权限，敬请通过电子邮件方式联系 guihonghao@zju.edu.cn 或 zhangningyu@zju.edu.cn。我们将提供完整数据集资源。


**`LLaMA2-IEPile`** | **`Baichuan2-IEPile`** | **`knowlm-ie-v2(基于Baichuan2)`** 模型下载链接：[zjunlp/llama2-13b-iepile-lora](https://huggingface.co/zjunlp/llama2-13b-iepile-lora/tree/main) | [zjunlp/baichuan2-13b-iepile-lora](https://huggingface.co/zjunlp/baichuan2-13b-iepile-lora) | [zjunlp/knowlm-ie-v2](https://huggingface.co/zjunlp/knowlm-ie-v2)


![statistic](./assets/statistic.jpg)

我们精心收集并清洗了现有的信息提取（IE）数据，共整合了`26`个英文IE数据集和`7`个中文IE数据集。如图1所示，这些数据集覆盖了包括**通用**、**医学**、**金融**等多个领域。

本研究采用了所提出的“`基于schema的轮询指令构造方法`”，成功创建了一个名为 **IEPile** 的大规模高质量IE微调数据集，包含约`0.32B` tokens。

基于**IEPile**，我们对 `Baichuan2-13B-Chat` 和 `LLaMA2-13B-Chat` 模型应用了 `Lora` 技术进行了微调。实验证明，微调后的 `Baichuan2-IEPile` 和 `LLaMA2-IEPile` 模型在全监督训练集上成绩斐然，并且在**零样本信息提取任务**中取得了显著进步。


![zero_en](./assets/zero_en.jpg)

![zero_zh](./assets/zero_zh.jpg)


<details>
  <summary><b>全监督数据集结果</b></summary>

![supervision_ner](./assets/supervision_ner.jpg)

![supervision_re](./assets/supervision_re.jpg)

![supervision_ee](./assets/supervision_ee.jpg)

</details>


## 📊2.数据


### 2.1IEPile的构造

我们专注于**基于指令的信息抽取**，因此指令中的schema的构造至关重要，因为它反映着具体抽取需求，是动态可变的。然而，现有研究在构造指令时往往采取一种**较为粗放的schema处理策略**，即利用标签集内**全部schema**进行指令构建。这种方法潜在地存在2个重要的问题：
1. **训练和评估阶段schema询问的数量不一致，即使这些schema在内容上相似，可能损害模型的泛化能力**。若训练过程中每次询问的schema数量大约是20个，而评估时询问的是10个或30个schema，即使这些schema在内容上与训练阶段相似，模型性能仍可能受到影响。
2. **指令中的schema之间的对比性不足**。语义近似的schema，如“裁员”、“离职”与“解雇”，它们的语义模糊性可能造成模型混淆。这类易混淆的模式应当在指令集中更为频繁地出现。
   
因此，我们提出如下解决方案：1、`构造难负样本字典`；2、`轮询式的指令生成`。

![iepile](./assets/iepile.jpg)


<details>
  <summary><b>难负样本</b></summary>


假设数据集 $\mathcal{D}$ 有其全量标签集 $L$，$\mathcal{D}$ 中某一文本 $S$，$S$ 中真实存在的标签构成**正例标签集** $Pos\_L$，而不存在的标签则形成**负例标签集** $Neg\_L$。在我们的分析中，我们发现模型误判的主要原因在于schema的**语义模糊**，导致了模型的混淆。传统方法中，负例标签 $Neg\_L$通常简单地定义为 $L - Pos\_L$。然而，这种方法忽视了一个重要方面：需要特别注意那些与正例标签**语义相近**的负例标签。受**对比学习**理论的启发。我们构造了一个**难负样本字典** $\mathcal{D}$，其键值对应的是Schema及其语义上相近的Schema集。因此**难负样本集** $Hard\_L = \mathcal{D}[Pos\_L]$。然而，若 $Neg\_L$ 仅由 $Hard\_L$ 构成会缺少足够的负例让模型学习。因此，我们定义其他负样本 $Other\_L = L - Hard\_L$，最终，负例标签 $Neg\_L$ 由 $Hard\_L$ 和少量的 $Other\_L$ 组成。这种难负样本的构建旨在促进语义近似的模式更频繁地出现在指令中，同时也能在不牺牲性能的情况下**减少训练样本量**（例如，原本需12个指令集的49个schema可减至3个）。

</details>


<details>
  <summary><b>轮询式的指令生成</b></summary>

在完成了上述步骤后，我们得到了最终的schema集合 $L'=Pos\_L + Neg\_L$。在基于schema的信息抽取（IE）指令构造中，schema的作用至关重要，它直接决定了模型需要抽取的信息类型，并且反映了用户的具体需求。传统做法通常将完整的schema一次性整合入指令中，然而，在本研究中，我们采纳了一种**轮询式方法**，限制每次询问的schema数量为 $split\_num$ 个，取值范围在4至6之间。因此 $L'$ 将被分为 $|L'|/split\_num$ 个批次进行询问，每批次询问 $split\_num$ 个schema。即使在评估阶段询问的schema数目与训练时不同，通过轮询机制，我们可以将询问数量平均分散至 $split\_num$ 个，从而缓解泛化性能下降的问题。

</details>



### 2.2IEPile的数据格式

`IEPile` 中的每条数据均包含 `task`, `source`, `instruction`, `output` 4个字段, 以下是各字段的说明

| 字段 | 说明 |
| :---: | :---: |
| task | 该实例所属的任务, (`NER`、`RE`、`EE`、`EET`、`EEA`) 5种任务之一。 |
| source | 该实例所属的数据集 |
| instruction | 输入模型的指令, 经过json.dumps处理成JSON字符串, 包括`"instruction"`, `"schema"`, `"input"`三个字段 |
| output | 输出, 采用字典的json字符串的格式, key是schema, value是抽取出的内容 |


在`IEPile`中, **`instruction`** 的格式采纳了类JSON字符串的结构，实质上是一种字典型字符串，它由以下三个主要部分构成：
(1) **`'instruction'`**: 任务描述, 它概述了指令的执行任务(`NER`、`RE`、`EE`、`EET`、`EEA`之一)。
(2) **`'schema'`**: 待抽取的schema(`实体类型`, `关系类型`, `事件类型`)列表。
(3) **`'input'`**: 待抽取的文本。


[instruction.py](./ie2instruction/convert/utils/instruction.py) 中提供了各个任务的指令模版。


以下是一条**数据实例**：

```json
{
    "task": "NER", 
    "source": "CoNLL2003", 
    "instruction": "{\"instruction\": \"You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.\", \"schema\": [\"person\", \"organization\", \"else\", \"location\"], \"input\": \"284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )\"}", 
    "output": "{\"person\": [\"Robert Allenby\", \"Allenby\", \"Miguel Angel Martin\"], \"organization\": [], \"else\": [], \"location\": [\"Australia\", \"Spain\"]}"
}
```

该数据实例所属任务是 `NER`, 所属数据集是 `CoNLL2003`, 待抽取的schema列表是 ["`person`", "`organization`", "`else`", "`location`"], 待抽取的文本是 "*284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )*", 输出是 `{"person": ["Robert Allenby", "Allenby", "Miguel Angel Martin"], "organization": [], "else": [], "location": ["Australia", "Spain"]}`

> 注意输出中的 schema 顺序与 instruction 中的 schema 顺序一致


<details>
  <summary><b>更多任务的数据实例</b></summary>

```json
{
  "task": "RE", 
  "source": "NYT11", 
  "instruction": "{\"instruction\": \"You are an expert in relationship extraction. Please extract relationship triples that match the schema definition from the input. Return an empty list for relationships that do not exist. Please respond in the format of a JSON string.\", \"schema\": [\"neighborhood of\", \"nationality\", \"children\", \"place of death\"], \"input\": \" In the way New Jersey students know that Thomas Edison 's laboratory is in West Orange , the people of Colma know that Wyatt Earp 's ashes are buried at Hills of Eternity , a Jewish cemetery he was n't ; his wife was , and that Joe DiMaggio is at Holy Cross Cemetery , where visitors often lean bats against his gravestone . \"}", 
  "output": "{\"neighborhood of\": [], \"nationality\": [], \"children\": [], \"place of death\": [{\"subject\": \"Thomas Edison\", \"object\": \"West Orange\"}]}"
}

{
  "task": "EE", 
  "source": "PHEE", 
  "instruction": "{\"instruction\": \"You are an expert in event extraction. Please extract events from the input that conform to the schema definition. Return an empty list for events that do not exist, and return NAN for arguments that do not exist. If an argument has multiple values, please return a list. Respond in the format of a JSON string.\", \"schema\": [{\"event_type\": \"potential therapeutic event\", \"trigger\": true, \"arguments\": [\"Treatment.Time_elapsed\", \"Treatment.Route\", \"Treatment.Freq\", \"Treatment\", \"Subject.Race\", \"Treatment.Disorder\", \"Effect\", \"Subject.Age\", \"Combination.Drug\", \"Treatment.Duration\", \"Subject.Population\", \"Subject.Disorder\", \"Treatment.Dosage\", \"Treatment.Drug\"]}, {\"event_type\": \"adverse event\", \"trigger\": true, \"arguments\": [\"Subject.Population\", \"Subject.Age\", \"Effect\", \"Treatment.Drug\", \"Treatment.Dosage\", \"Treatment.Freq\", \"Subject.Gender\", \"Treatment.Disorder\", \"Subject\", \"Treatment\", \"Treatment.Time_elapsed\", \"Treatment.Duration\", \"Subject.Disorder\", \"Subject.Race\", \"Combination.Drug\"]}], \"input\": \"Our findings reveal that even in patients without a history of seizures, pregabalin can cause a cortical negative myoclonus.\"}", 
  "output": "{\"potential therapeutic event\": [], \"adverse event\": [{\"trigger\": \"cause \", \"arguments\": {\"Subject.Population\": \"NAN\", \"Subject.Age\": \"NAN\", \"Effect\": \"cortical negative myoclonus\", \"Treatment.Drug\": \"pregabalin\", \"Treatment.Dosage\": \"NAN\", \"Treatment.Freq\": \"NAN\", \"Subject.Gender\": \"NAN\", \"Treatment.Disorder\": \"NAN\", \"Subject\": \"patients without a history of seizures\", \"Treatment\": \"pregabalin\", \"Treatment.Time_elapsed\": \"NAN\", \"Treatment.Duration\": \"NAN\", \"Subject.Disorder\": \"NAN\", \"Subject.Race\": \"NAN\", \"Combination.Drug\": \"NAN\"}}]}"
}
```

</details>




## 🚴3.使用IEPile训练模型

### 3.1环境

在开始之前，请确保按照下面的指导创建适当的**虚拟环境**：

```bash
conda create -n IEPile python=3.9   # 创建虚拟环境
conda activate IEPile               # 激活环境
pip install -r requirements.txt     # 安装依赖 
```

### 3.2下载数据和模型

**`IEPile`** 数据集下载链接：[Google Drive](https://drive.google.com/file/d/1jPdvXOTTxlAmHkn5XkeaaCFXQkYJk5Ng/view?usp=sharing) | [Hugging Face](https://huggingface.co/datasets/zjunlp/IEPile)

```python
IEPile
├── train.json    # 训练集
└── dev.json      # 验证集
```

以下是本仓库代码支持的一些基础模型：[[llama](https://huggingface.co/meta-llama), [alpaca](https://github.com/tloen/alpaca-lora), [vicuna](https://huggingface.co/lmsys), [zhixi](https://github.com/zjunlp/KnowLM), [falcon](https://huggingface.co/tiiuae), [baichuan](https://huggingface.co/baichuan-inc), [chatglm](https://huggingface.co/THUDM), [qwen](https://huggingface.co/Qwen), [moss](https://huggingface.co/fnlp), [openba](https://huggingface.co/OpenBA)]


**`LLaMA2-IEPile`** | **`Baichuan2-IEPile`** | **`knowlm-ie-v2(基于Baichuan2)`** 模型下载链接：[zjunlp/llama2-13b-IEPile-lora](https://huggingface.co/zjunlp/llama2-13b-IEPile-lora/tree/main) | [zjunlp/baichuan2-13b-IEPile-lora](https://huggingface.co/zjunlp/baichuan2-13b-IEPile-lora) | [zjunlp/knowlm-ie-v2](https://huggingface.co/zjunlp/knowlm-ie-v2)


**`LLaMA2-IEPile`**, **`Baichuan2-IEPile`** 是IEPILE论文中涉及的在 `LLaMA2-13B-Chat`, `Baichuan2-13B-Chat` 上Lora微调得到的两个模型。



```bash
mkdir data         # 数据放这
mkdir mdoels       # 基础模型放这
mkdir results      # 预测结果放这
mkdir lora         # lora微调结果放这
```


### 3.3LoRA微调

> 重要提示：以下的所有命令均应在`IEPile`目录下执行。例如，如果您想运行微调脚本，您应该使用如下命令：bash ft_scripts/fine_llama.bash。请确保您的当前工作目录正确。


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
    --bf16 \
    --deepspeed configs/ds_config_bf16.json
```

* `model_name`: 指定所需的**模型架构名称**(7B、13B、Base、Chat属于同一模型架构)。当前支持的模型包括：["`llama`", "`alpaca`", "`vicuna`", "`zhixi`", "`falcon`", "`baichuan`", "`chatglm`", "`qwen`", "`moss`", "`openba`"]。**请注意**，此参数应与 `--model_name_or_path` 区分。
* `model_name_or_path`: 模型路径, 请到 [HuggingFace](https://huggingface.co/models) 下载相应模型。
* `template`: 使用的**模板名称**，包括：`alpaca`, `baichuan`, `baichuan2`, `chatglm3`等, 请参考 [src/datamodule/template.py](./src/datamodule/template.py) 查看所有支持的模版名称, 默认使用的是`alpaca`模板, **`Chat`版本的模型建议使用配套的模版, Base版本模型可默认使用`alpaca`**。
* `train_file`, `valid_file（可选）`: 训练集和验证集的**文件路径**。注意：目前仅支持json格式的文件。
* `output_dir`: LoRA微调后的**权重参数保存路径**。
* `val_set_size`: **验证集的样本数量**, 默认为1000。
* `per_device_train_batch_size`, `per_device_eval_batch_size`: 每台GPU设备上的`batch_size`, 根据显存大小调整。
* `max_source_length`, `max_target_length`, `cutoff_len`: 最大输入、输出长度、截断长度, 截断长度可以简单地视作最大输入长度 + 最大输出长度, 需根据具体需求和显存大小设置合适值。
* `deepspeed`: 设备资源不够可去掉。

> 可通过设置 `bits` = 8 或 4 进行量化。

* 要了解更多关于**参数配置**的信息，请参考 [src/utils/args](./src/args) 目录。


微调`LLaMA2-13B-Chat`模型的具体脚本可以在 [ft_scripts/fine_llama.bash](./ft_scripts/fine_llama.bash) 中找到。

微调`Baichuan2-13B-Chat`模型的具体脚本可以在 [ft_scripts/fine_baichuan.bash](./ft_scripts/fine_baichuan.bash) 中找到。



## 🔁4.领域内数据继续训练

尽管 `Baichuan2-IEPile` 和 `LLaMA2-IEPile` 模型已在多个通用数据集上接受了广泛的指令微调，并因此获得了一定的**通用信息抽取能力**，但它们在**特定领域**(如`法律`、`教育`、`科学`、`电信`)的数据处理上可能仍显示出一定的局限性。针对这一挑战，建议对这些模型在特定领域的数据集上进行**二次训练**。这将有助于模型更好地适应特定领域的语义和结构特征，从而显著增强其在**该领域内的信息抽取能力**。


#### 4.1训练数据转换

首先, 需要将**数据格式化**以包含`instruction`、`output`字段。为此，我们提供了一个脚本 [convert_func.py](./ie2instruction/convert_func.py)，它可以将数据批量转换成模型可以直接使用的格式。

> 在使用 [convert_func.py](./ie2instruction/convert_func.py) 脚本之前，请确保参考了 [data](./data) 目录。该目录详细说明了每种任务所需的数据格式要求。 `sample.json` 描述了转换前数据的格式，`schema.json` 展示了 schema 的组织结构， `train.json` 描述了转换后的数据格式。

> 此外，可直接使用包含12个主题（如人物、交通工具、艺术作品、自然科学、人造物品、天文对象等）的中英双语信息抽取数据集 [zjunlp/InstructIE](https://huggingface.co/datasets/zjunlp/InstructIE)。


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

* `language`: 支持`zh`, `en`两种语言, 不同语言使用的指令模版不同。
* `task`: 目前支持['`RE`', '`NER`', '`EE`', '`EET`', '`EEA`']五类任务。
* `split_num`: 定义单个指令中可包含的最大schema数目。默认值为4，设置为-1则不进行切分。推荐的任务切分数量依任务而异：**NER建议为6，RE、EE、EET、EEA均推荐为4**。
* `random_sort`: 是否对指令中的schema随机排序, 默认为False, 即按字母顺序排序。
* `split`: 指定数据集类型，可选`train`或`test`。

转换后的训练数据将包含 `task`, `source`, `instruction`, `output` 四个字段。



#### 4.2继续训练

```bash
output_dir='lora/llama2-13b-chat-v1-continue'
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node=8 --master_port=1287 src/test_finetune.py \
    --do_train --do_eval \
    --overwrite_output_dir \
    --model_name_or_path 'models/llama2-13B-Chat' \
    --checkpoint_dir 'zjunlp/llama2-13b-iepile-lora' \
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
    --lora_r 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --bf16 
```

* 若要基于微调后的LoRA权重继续训练，仅需将 `checkpoint_dir` 参数指向LoRA权重路径，例如设置为`'zjunlp/llama2-13b-iepile-lora'`。

> 可通过设置 `bits` = 8 或 4 进行量化。

> 请注意，在使用 **`LLaMA2-IEPile`** 或 **`Baichuan2-IEPile`** 时，保持lora_r和lora_alpha均为64，对于这些参数，我们不提供推荐设置。

* 若要基于微调后的模型权重继续训练，只需设定 `model_name_or_path` 参数为权重路径，如`'zjunlp/KnowLM-IE-v2'`，无需设置`checkpoint_dir`。


脚本可以在 [ft_scripts/fine_continue.bash](./ft_scripts/fine_continue.bash) 中找到。



## 👓5.预测

### 5.1测试数据转换

在准备测试数据转换之前，请访问 [data](./data) 目录以了解各任务所需的数据结构：1）输入数据格式参见 `sample.json`；2）schema格式请查看 `schema.json`；3）转换后数据格式可参照 `train.json`。**与训练数据不同, 测试数据的输入无需包含标注字段（`entity`, `relation`, `event`）**。


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

设置 `split` 为 **test** 时，请根据任务类型选择适当的schema数量：**NER推荐为6，而RE、EE、EET、EEA推荐为4**。转换后的测试数据将含有`id`, `task`, `source`, `instruction`, `label`五个字段。

`label` 字段将用于后续评估。若输入数据中缺少标注字段（`entity`, `relation`, `event`），则转换后的测试数据将不包含`label`字段，适用于那些无原始标注数据的场景。



### 5.2基础模型+Lora预测

**`LLaMA2-IEPile`** | **`Baichuan2-IEPile`** 模型下载链接：[zjunlp/llama2-13b-iepile-lora](https://huggingface.co/zjunlp/llama2-13b-iepile-lora/tree/main) | [zjunlp/baichuan2-13b-iepile-lora](https://huggingface.co/zjunlp/baichuan2-13b-iepile-lora)


```bash
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --stage sft \
    --model_name_or_path 'models/llama2-13B-Chat' \
    --checkpoint_dir 'zjunlp/llama2-13b-IEPile-lora' \
    --model_name 'llama' \
    --template 'llama2' \
    --do_predict \
    --input_file 'data/input.json' \
    --output_file 'results/llama2-13b-IEPile-lora_output.json' \
    --finetuning_type lora \
    --output_dir 'lora/test' \
    --predict_with_generate \
    --max_source_length 512 \
    --bf16 \
    --max_new_tokens 300
```

* 在进行推理时，`model_name`, `template`, 和 `bf16` 必须与训练时的设置相同。
* `model_name_or_path`: 指定所使用的基础模型路径，必须与相应的LoRA模型匹配。
* `checkpoint_dir`: LoRA的权重文件路径。
* `output_dir`: 此参数在推理时不起作用，可以随意指定一个路径。
* `input_file`, `output_file`: 分别指定输入的测试文件路径和预测结果的输出文件路径。
* `max_source_length`, `max_new_tokens`: 设置最大的输入长度和生成的新token数量，根据显存大小进行调整。

> 可通过设置 `bits` = 8 或 4 进行量化。

### 5.3IE专用模型预测

**`knowlm-ie-v2(based on Baichuan2)`** 模型下载链接：[zjunlp/knowlm-ie-v2](https://huggingface.co/zjunlp/knowlm-ie-v2)

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

`model_name_or_path`: IE专用模型权重路径

> 可通过设置 `bits` = 8 或 4 进行量化。

## 6.评估

我们提供了评估各个任务F1分数的脚本。

```bash
python ie2instruction/eval_func.py \
  --path1 data/NER/processed.json \
  --task NER 
```

* `task`: 目前支持['RE', 'NER', 'EE', 'EET', 'EEA']五类任务。
* 可以设置 `sort_by` 为 `source`, 分别计算每个数据集上的F1分数。



## 7.声明和许可

我们认为标注数据蕴含着人类的智慧宝库，它的存在是为了促进全人类的利益，并有助于提升我们的生活质量。我们强烈敦促所有的用户不要将我们的语料库用于任何可能对国家或公共安全造成伤害、违反法律法规的行为。
我们竭尽所能地保证所提供数据的质量与其合法性。但我们也意识到，尽管如此，可能还是存在一些不可预见的问题，诸如数据保护的担忧以及数据被滥用可能引起的风险和问题。对于这些潜在的问题，我们将不承担责任。
对于那些受限于比[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)协议更为严格的使用许可的原始数据，IEPile将恪守那些较为严格的条款。在其他所有情形下，我们的操作将基于[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)许可协议。



## 8.局限

从数据角度来看，我们的研究主要集中在基于schema的信息提取（IE）上，这限制了我们将研究成果推广至不遵循我们特定格式要求的人类指令的能力。此外，我们没有探索开放信息提取（Open IE）领域；然而，如果我们去除schema约束，我们的数据集将适用于开放信息提取场景。此外，我们的数据集目前仅包含英语和中文数据，在未来，我们希望能够包含更多语言的数据。
从模型的角度来看，由于计算资源的限制，我们的研究仅评估了两个模型：Baichuan和LLaMA，以及一些基线模型。我们的数据集可以应用于任何其他的大型语言模型（LLMs），如Qwen、ChatGLM。


## 9.引用
如果您使用IEPile或代码，请引用以下论文：



## 10. 致谢
我们非常感谢[MathPile](mathpile)和[KnowledgePile](https://huggingface.co/datasets/Query-of-CC/Knowledge_Pile)项目提供的宝贵灵感。我们对以下数据集构建者和维护者表示特别的谢意：[AnatEM](https://doi.org/10.1093/BIOINFORMATICS/BTT580)、[BC2GM](https://link.springer.com/chapter/10.1007/978-3-030-68763-2_48)、[BC4CHEMD](https://link.springer.com/chapter/10.1007/978-3-030-68763-2_48)、[NCBI-Disease](https://linkinghub.elsevier.com/retrieve/pii/S1532046413001974)、[BC5CDR](https://openreview.net/pdf?id=9EAQVEINuum)、[HarveyNER](https://aclanthology.org/2022.naacl-main.243/)、[CoNLL2003](https://aclanthology.org/W03-0419/)、[GENIA](https://pubmed.ncbi.nlm.nih.gov/12855455/)、[ACE2005](https://catalog.ldc.upenn.edu/LDC2006T06)、[MIT Restaurant](https://ieeexplore.ieee.org/document/6639301)、[MIT Movie](https://ieeexplore.ieee.org/document/6639301)、[FabNER](https://link.springer.com/article/10.1007/s10845-021-01807-x)、[MultiNERD](https://aclanthology.org/2022.findings-naacl.60/)、[Ontonotes](https://aclanthology.org/N09-4006/)、[FindVehicle](https://arxiv.org/abs/2304.10893)、[CrossNER](https://ojs.aaai.org/index.php/AAAI/article/view/17587)、[MSRA NER](https://aclanthology.org/W06-0115/)、[Resume NER](https://aclanthology.org/P18-1144/)、[CLUE NER](https://arxiv.org/abs/2001.04351)、[Weibo NER](https://aclanthology.org/D15-1064/)、[Boson](https://github.com/InsaneLife/ChineseNLPCorpus/tree/master/NER/boson)、[ADE Corpus](https://jbiomedsem.biomedcentral.com/articles/10.1186/2041-1480-3-15)、[GIDS](https://arxiv.org/abs/1804.06987)、[CoNLL2004](https://aclanthology.org/W04-2412/)、[SciERC](https://aclanthology.org/D18-1360/)、[Semeval-RE](https://aclanthology.org/S10-1006/)、[NYT11-HRL](https://ojs.aaai.org/index.php/AAAI/article/view/4688)、[KBP37](https://arxiv.org/abs/1508.01006)、[NYT](https://link.springer.com/chapter/10.1007/978-3-642-15939-8_10)、[Wiki-ZSL](https://aclanthology.org/2021.naacl-main.272/)、[FewRel](https://aclanthology.org/D18-1514/)、[CMeIE](https://link.springer.com/chapter/10.1007/978-3-030-60450-9_22)、[DuIE](https://link.springer.com/chapter/10.1007/978-3-030-32236-6_72)、[COAE2016](https://github.com/Sewens/COAE2016)、[IPRE](https://arxiv.org/abs/1907.12801)、[SKE2020](https://aistudio.baidu.com/datasetdetail/177191)、[CASIE](https://ojs.aaai.org/index.php/AAAI/article/view/6401)、[PHEE](https://aclanthology.org/2022.emnlp-main.376/)、[CrudeOilNews](https://aclanthology.org/2022.lrec-1.49/)、[RAMS](https://aclanthology.org/2020.acl-main.718/)、[WikiEvents](https://aclanthology.org/2021.naacl-main.69/)、[DuEE](https://link.springer.com/chapter/10.1007/978-3-030-60457-8_44)、[DuEE-Fin](https://link.springer.com/chapter/10.1007/978-3-031-17120-8_14)、[FewFC](https://ojs.aaai.org/index.php/AAAI/article/view/17720)、[CCF law](https://aistudio.baidu.com/projectdetail/4201483)等，这些数据集极大地促进了本研究的进展。我们也要对[InstructUIE](http://arxiv.org/abs/2304.08085)与[YAYI-UIE](http://arxiv.org/abs/2312.15548)为数据和模型在信息抽取领域做出的宝贵贡献表示感激。我们的研究成果同样得益于他们的创新和努力。此外，我们要对[hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)表示衷心的感谢，我们的微调代码实现在很大程度上参考了他们的工作。通过这些学术资源的辅助，我们得以完成本项研究，对此我们深表感激。
