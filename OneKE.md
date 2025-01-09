- [OneKE](#oneke)
- [模型介绍](#模型介绍)
- [模型使用](#模型使用)
  - [模型下载](#模型下载)
  - [环境安装](#环境安装)
  - [快速运行](#快速运行)
  - [vLLM 推理](#vllm-推理)
  - [gguf 格式转换](#gguf-格式转换)
  - [ollama 推理](#ollama-推理)
  - [在 Mac 上推理](#在-mac-上推理)
  - [多卡推理](#多卡推理)
- [数据准备](#数据准备)
  - [训练数据](#训练数据)
  - [测试数据](#测试数据)
- [模型训练](#模型训练)
  - [LoRA微调：](#lora微调)
  - [全量微调：](#全量微调)
- [模型评估](#模型评估)
- [常见问题](#常见问题)


## OneKE

## 模型介绍

OneKE是由蚂蚁集团和浙江大学联合研发的大模型知识抽取框架，具备中英文双语、多领域多任务的泛化知识抽取能力，并提供了完善的工具链支持。

基于非结构化文档的知识构建一直是知识图谱大规模落地的关键难题之一，因为真实世界的信息高度碎片化、非结构化，大语言模型在处理信息抽取任务时仍因抽取内容与自然语言表述之间的巨大差异导致效果不佳，自然语言文本信息表达中因隐式、长距离上下文关联存在较多的歧义、多义、隐喻等，给知识抽取任务带来较大的挑战。针对上述问题，蚂蚁集团与浙江大学依托多年积累的知识图谱与自然语言处理技术，联合构建和升级蚂蚁百灵大模型在知识抽取领域的能力，并发布中英双语大模型知识抽取框架OneKE，同时开源基于Chinese-Alpaca-2-13B全参数微调的版本。测评指标显示，OneKE在多个全监督及零样本实体/关系/事件抽取任务上取得了相对较好的效果。

统一知识抽取框架有比较广阔的应用场景，可大幅降低领域知识图谱的构建成本。通过从海量的数据中萃取结构化知识，构建高质量知识图谱并建立知识要素间的逻辑关联，可以实现可解释的推理决策，也可用于增强大模型缓解幻觉并提升稳定性，加速大模型垂直领域的落地应用。如应用在医疗领域通过知识抽取实现医生经验的知识化规则化管理，构建可控的辅助诊疗和医疗问答。应用在金融领域抽取金融指标、风险事件、因果逻辑及产业链等，实现自动的金融研报生成、风险预测、产业链分析等。应用在政务场景实现政务法规的知识化，提升政务服务的办事效率和准确决策。

<p align="center" width="100%">
<a href="" target="_blank"><img src="./assets/oneke.gif" alt="OneKE" style="width: 100%; min-width: 20px; display: block; margin: auto;"></a>
</p>

## 模型使用

### 模型下载

[HuggingFace](https://huggingface.co/zjunlp/OneKE), [ModelScope](https://modelscope.cn/models/ZJUNLP/OneKE), [WiseModel](https://wisemodel.cn/models/zjunlp/OneKE), 转换后的gguf格式的OneKE [OneKE-gguf](https://modelscope.cn/models/ZJUNLP/OneKE-gguf)

### 环境安装

```bash
conda create -n OneKE python=3.9
conda activate OneKE
pip install -r requirements.txt
```

### 快速运行

训练和推理建议至少具备**20GB的显存**

```python
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'zjunlp/OneKE' #选择你下载的模型存储在本地的位置
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


# 4bit量化OneKE
quantization_config=BitsAndBytesConfig(     
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    device_map="auto",  
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()


system_prompt = '<<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n'
sintruct = "{\"instruction\": \"You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.\", \"schema\": [\"person\", \"organization\", \"else\", \"location\"], \"input\": \"284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )\"}"
sintruct = '[INST] ' + system_prompt + sintruct + '[/INST]'

input_ids = tokenizer.encode(sintruct, return_tensors="pt").to(device)
input_length = input_ids.size(1)
generation_output = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_length=1024, max_new_tokens=512, return_dict_in_generate=True), pad_token_id=tokenizer.eos_token_id)
generation_output = generation_output.sequences[0]
generation_output = generation_output[input_length:]
output = tokenizer.decode(generation_output, skip_special_tokens=True)

print(output)
```

### vLLM 推理

vLLM的环境配置可见其官方安装配置文档 ([Installation](https://vllm.readthedocs.io/en/latest/getting_started/installation.html))

部署服务

```bash
python -m vllm.entrypoints.openai.api_server --model zjunlp/OneKE
```

终端使用Api推理

```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "/data2/lkw/OneKE", "prompt": "[INST] <<SYS>>You are a helpful assistant. 你是一个乐于助人的助手。<</SYS>>{\"instruction\": \"You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.\", \"schema\": [\"person\", \"organization\", \"else\", \"location\"], \"input\": \"284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )\"}[/INST]", "max_tokens": 1024, "temperature": 0}'
```

### gguf 格式转换

为了将模型权重从Hugging Face格式转换为GGUF格式，我们首先需要克隆llama.cpp的GitHub仓库，该仓库包含了必要的转换脚本。请按照以下步骤操作：

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```


接下来，使用提供的Python脚本convert_hf_to_gguf.py来执行格式转换。确保你已经安装了所需的Python环境和依赖项。下面是执行转换命令的具体方式：

```bash
python3 convert_hf_to_gguf.py \
    /disk/disk_20T/ghh/OneKE \
    --outfile /disk/disk_20T/ghh/OneKE.gguf \
    --outtype bf16 
```

请注意，--model_dir参数指定了原始模型文件的位置，而--outfile定义了转换后GGUF文件的保存位置。--outtype参数用来设置输出文件中数值的精度。

转换后的gguf格式的OneKE [OneKE-gguf](https://modelscope.cn/models/ZJUNLP/OneKE-gguf)

### ollama 推理

ollama 的环境配置可见其官方文档 https://github.com/ollama/ollama/tree/main

```bash
curl -fsSL https://ollama.com/install.sh | sh
```


创建 Modelfile 文件

```bash
FROM ./OneKE-13B-BF16.gguf
PARAMETER temperature 0
PARAMETER num_ctx 4096
TEMPLATE """[INST] <<SYS>>You are a helpful assistant. 你是一个乐于助人的助手。<</SYS>>{{ .Prompt }}[/INST]"""
```

启动 ollama

```bash
ollama serve
```

在另一个终端窗口输入命令

```bash
ollama create oneke -f Modelfile

ollama run oneke
```

输入和输出

```
>>> {\"instruction\": \"你是专门进行实体抽取的专家。请从input中抽取出符合schema定义的实体，不存在的实体类型
... 返回空列表。请按照JSON字符串的格式回答。\", \"schema\": [\"人物\", \"地理位置\", \"组织机构\"], \"input
... \": \"在这里恕弟不恭之罪，敢在尊前一诤：前人论书，每曰“字字有来历，笔笔有出处”，细读公字，何尝跳出前人
... 藩篱，自隶变而后，直至明季，兄有何新出？\"}
 {"人物": [], "地理位置": [], "组织机构": []}

>>> {\"instruction\": \"你是专门进行实体抽取的专家。请从input中抽取出符合schema定义的实体，不存在的实体类型
... 返回空列表。请按照JSON字符串的格式回答。\", \"schema\": [\"组织机构\", \"地理位置\", \"人物\"], \"input
... \": \"胡老说，当画画疲倦时就到院里去看看，给这盆花浇点水，给那棵花剪剪枝，回来再接着画，画累了再出去，
... 如此循环往复，脑体结合，有益健康，胜过吃药。\"}
 {"组织机构": [], "地理位置": [], "人物": ["胡"]}

>>> {\"instruction\": \"你是专门进行事件提取的专家。请从input中抽取出符合schema定义的事件，不存在的事件返回
... 空列表，不存在的论元返回NAN，如果论元存在多值请返回列表。请按照JSON字符串的格式回答。\", \"schema\": [{
... \"event_type\": \"产品行为-获奖\", \"trigger\": true, \"arguments\": [\"获奖人\", \"颁奖机构\", \"奖项\
... ", \"时间\"]}, {\"event_type\": \"组织行为-罢工\", \"trigger\": true, \"arguments\": [\"罢工人数\", \"
... 罢工人员\", \"所属组织\", \"时间\"]}, {\"event_type\": \"组织关系-裁员\", \"trigger\": true, \"argument
... s\": [\"裁员方\", \"时间\", \"裁员人数\"]}, {\"event_type\": \"组织关系-解散\", \"trigger\": true, \"ar
... guments\": [\"解散方\", \"时间\"]}], \"input\": \"消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”
... 了\"}
 {"产品行为-获奖": [], "组织行为-罢工": [], "组织关系-裁员": [{"trigger": "裁员", "arguments": {"裁员方
": "NAN", "时间": "5月份", "裁员人数": "900余人"}}], "组织关系-解散": []}
```


退出后删除

```bash
ollama stop oneke

ollama rm oneke
```


### 在 Mac 上推理

```python
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)

device = torch.device("mps")
model_path = 'zjunlp/OneKE' #选择你下载的模型存储在本地的位置
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    device_map="auto",  
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()
model = model.to(device)
```

`PYTORCH_ENABLE_MPS_FALLBACK=1 python test.py` 命令行启动。


### 多卡推理

```python
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, GenerationConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model

max_memory_each_gpu = '15GiB' 
gpu_device_ids = [0, 1] 
no_split_module_classes = ["LlamaDecoderLayer"]
model_path = '/disk/disk_20T/ghh/OneKE' #选择你下载的模型存储在本地的位置

max_memory = {
    device_id: max_memory_each_gpu for device_id in gpu_device_ids
}

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

with init_empty_weights():
    model = AutoModel.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)

device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes)

print("auto determined device_map", device_map)
device_map["llm.model.embed_tokens"] = 0
device_map["llm.model.layers.0"] = 0
device_map["llm.lm_head"] = 0
device_map["vpm"] = 0
device_map["resampler"] = 0
print("modified device_map", device_map)

load_checkpoint_in_model(model, model_path, device_map=device_map)

model = dispatch_model(model, device_map=device_map)
torch.set_grad_enabled(False)
model.eval()


system_prompt = '<<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n'
sintruct = "{\"instruction\": \"You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.\", \"schema\": [\"person\", \"organization\", \"else\", \"location\"], \"input\": \"284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )\"}"
sintruct = '[INST] ' + system_prompt + sintruct + '[/INST]'

input_ids = tokenizer.encode(sintruct, return_tensors="pt")
input_length = input_ids.size(1)
generation_output = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_length=1024, max_new_tokens=512, return_dict_in_generate=True), pad_token_id=tokenizer.eos_token_id)
generation_output = generation_output.sequences[0]
generation_output = generation_output[input_length:]
output = tokenizer.decode(generation_output, skip_special_tokens=True)

print(output)
```



## 数据准备

### 训练数据

训练数据需要包含task, source, instruction, output四个字段。

 `'instruction'` 的格式采用了类JSON字符串的结构，实质上是一种字典类型的字符串。它由以下三个字段构成： (1) `'instruction'`，即任务描述，以自然语言指定模型扮演的角色以及需要完成的任务； (2) `'schema'`，这是一份需提取的标签列表，明确指出了待抽取信息的关键字段，反应用户的需求，是动态可变的； (3) `'input'`，指的是用于信息抽取的源文本。

其中task有'`RE`', '`NER`', '`EE`', '`EET`', '`EEA`', `'KG'`等，不同task的具体训练数据格式均有所不同，请先确定自己的训练任务并且去[data](https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC/data)文件夹下确认该任务的训练数据格式。

以'`NER`'任务为例，以下是一条NER任务的训练数据示例：

```json
{"task": "NER", "source": "NER", "instruction": "{\"instruction\": \"你是专门进行实体抽取的专家。请从input中抽取出符合schema定义的实体，不存在的实体类型返回空列表。请按照JSON字符串的格式回答。\", \"schema\": [\"组织机构\", \"人物\", \"地理位置\"], \"input\": \"相比之下，青岛海牛队和广州松日队的雨中之战虽然也是0∶0，但乏善可陈。\"}", "output": "{\"组织机构\": [\"广州松日队\", \"青岛海牛队\"], \"人物\": [], \"地理位置\": []}"}
```

你需要先初步准备数据并写为以下格式：

```json
{"text": "相比之下，青岛海牛队和广州松日队的雨中之战虽然也是0∶0，但乏善可陈。", "entity": [{"entity": "广州松日队", "entity_type": "组织机构"}, {"entity": "青岛海牛队", "entity_type": "组织机构"}]}
```

再利用我们的格式转换脚本[convert_func.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/ie2instruction/convert_func.py)：

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

- `language`: 支持`zh`, `en`两种语言, 不同语言使用的指令模版不同。
- `task`: 目前支持['`RE`', '`NER`', '`EE`', '`EET`', '`EEA`', `'KG'`]任务。
- `split_num`: 定义单个指令中可包含的最大schema数目。默认值为4，设置为-1则不进行切分。推荐的任务切分数量依任务而异：**NER建议为6，RE、EE、EET、EEA均推荐为4、KG推荐为1**。
- `random_sort`: 是否对指令中的schema随机排序, 默认为False, 即按字母顺序排序。
- `split`: 指定数据集类型，可选`train`或`test`。

除此之外，鉴于特定领域内信息抽取的复杂性和对提示的依赖程度较高，我们支持在指令中融入Schema描述和示例Example来提升抽取任务的效果。（以下两个示例只列出了instruction部分，task, source, output 字段并未列出）
定制化schema解释指令：

```json
{
  "instruction": "你是专门进行实体抽取的专家。请从input中抽取出符合schema定义的实体，不存在的实体类型返回空列表。请按照JSON字符串的格式回答。", 
  "schema": {
    "职位": "实体类型描述个人或群体的职业或职务，包括特定角色名称如'制片方'，'报分员'，'苦行僧'，'油画家'。", 
    "景点": "景点实体类型包括建筑物、博物馆、纪念馆、美术馆、河流、山峰等。代表性实体有五角大楼、泰特当代美术馆、郑成功纪念馆、都喜天阙、巴里卡萨、罗博河、gunungbatur、愚公移山LIVE、徐悲鸿纪念馆、杜莎夫人蜡像馆等。", 
    "公司": "公司是一个实体类型，代表任何法人实体或商业组织。这个类型的实体可以是餐饮集团，制造商，零售商，酒店，银行，设计院等各种类型的公司。例如：'香格里拉酒店集团', 'JVC', '上海酷蕾专业电竞外设装备店', 'k2&bull;海棠湾', '武钢', 'louisvuitton', '苏格兰银行', '北京市建筑设计研究院', '7天', '万科集团'。", 
    "地址": "地址实体是指具有地理位置信息的实体，它可以代表一个国家、城市、区域、街道等具体的地方或者一个抽象的地理区域。例如：'曼哈顿下城区东南尖上的河边码头', '图阿普谢', '意大利威尼斯水乡', '湖州温泉高尔夫球场', '北卡罗来纳州', '京津区域', '开心网吧', '颐年护理院', '上塘镇浦东', '内蒙古自治区赤峰市'等。", 
    "组织机构": "组织机构实体是指集体性质的组织，比如公司、商铺、俱乐部、学校等。它们在社会和经济活动中扮演一定角色，并拥有一定的人格权。", 
    "电影": "电影实体包括中文或英文电影名称，有时也包括电影人物角色名。"
  }, 
  "input": "我很难想象在另外一个项目再做一个海渔广场，当时我们拿到这个项目的时候，我正好在三亚，"
}
```

定制化example示例指令：

```json
{
    "instruction": "你是专门进行实体抽取的专家。请从input中抽取出符合schema定义的实体，不存在的实体类型返回空列表。请按照JSON字符串的格式回答。你可以参考example进行抽取。",
    "schema": [
        "检查指标"
    ],
    "example": [
        {
            "input": "CKD诊断标准：1.以下任意一项指标持续超过3个月；且至少满足1项。(1)肾损伤标志：白蛋白尿[尿白蛋白排泄率(AER)≥30mg／24h；尿白蛋白肌酐比值(ACR)≥3mg／mmol]；尿沉渣异常；肾小管相关病变；组织学异常；影像学所见结构异常；肾移植病史。(2)肾小球滤过率下降：eGFR≤60ml·min-1·1.73m-2 ",
            "output": {
                "检查指标": [
                    "尿白蛋白排泄率(AER)",
                    "尿白蛋白肌酐比值(ACR)",
                    "肾小球滤过率",
                    "eGFR"
                ]
            }
        },
        {
            "input": "DPP-4抑制剂在特殊人群中的应用",
            "output": {
                "检查指标": []
            }
        }
    ],
    "input": "目前所有磺脲类药物说明书均将重度肝功能不全列为禁忌证。丙氨酸氨基转移酶(alanine transaminase，ALT)>3倍参考值上限可作为肝损害的敏感而特异的指标，若ALT>8~10倍参考值上限或者ALT>3倍参考值上限且血清总胆红素(total bilirubin，TBIL)>2倍参考值上限则是预测重度肝损害的特异指标，表明肝脏实质细胞受到损害，此时应禁用磺脲类药物。在临床使用中，伴有肝性脑病、腹水或凝血障碍的失代偿肝硬化患者应禁用该类药物以防发生低血糖。"
}
```

### 测试数据

测试数据的构建流程和训练数据的构建流程较为相似，在准备测试数据转换之前，请访问 [data](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/data) 目录以了解各任务所需的数据结构：1）输入数据格式参见 `sample.json`；2）schema格式请查看 `schema.json`；3）转换后数据格式可参照 `train.json`。**与训练数据不同, 测试数据的输入无需包含标注字段（`entity`, `relation`, `event`）**。

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

## 模型训练

我们对OneKE的继续训练提供了两种微调方法：即LoRA和全量微调。

### LoRA微调：

在[ft_scripts/fine_continue.bash](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/ft_scripts/fine_continue.bash)文件中进行参数的配置，然后运行。

```bash
output_dir='lora/oneke-continue'
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port=1287 src/finetune.py \
    --do_train --do_eval \
    --overwrite_output_dir \
    --model_name_or_path 'models/OneKE' \
    --stage 'sft' \
    --model_name 'llama' \
    --template 'llama2_zh' \
    --train_file 'data/train.json' \
    --valid_file 'data/dev.json' \
    --output_dir=${output_dir} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --preprocessing_num_workers 16 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --max_grad_norm 0.5 \
    --optim "adamw_torch" \
    --max_source_length 400 \
    --cutoff_len 700 \
    --max_target_length 300 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --lora_r 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --bf16 \
    --bits 4
```

- 若要基于微调后的LoRA权重继续训练，仅需将 `checkpoint_dir` 参数指向LoRA权重路径，例如设置为`'zjunlp/OneKE'`。
- 使用`deepspeed`, 可设置 `--deeepspeed configs/ds_config_bf16_stage2.json`
- 可通过设置 `bits`= 4 进行量化, RTX3090建议量化。
- 请注意，在使用 `llama2-13b-iepile-lora`、`baichuan2-13b-iepile-lora` 时，保持lora_r和lora_alpha均为64，对于这些参数，我们不提供推荐设置。
- 若要基于微调后的模型权重继续训练，只需设定 `model_name_or_path` 参数为权重路径，如`'zjunlp/OneKE'`，无需设置`checkpoint_dir`。

### 全量微调：

在[ft_scripts/fine_continue_full.bash](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/ft_scripts/fine_continue_full.bash)文件中进行参数的配置，然后运行。

```bash
output_dir='lora/oneke-continue'
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port=1287 src/finetune.py \
    --do_train --do_eval \
    --overwrite_output_dir \
    --model_name_or_path 'models/OneKE' \
    --stage 'sft' \
    --finetuning_type 'full' \
    --model_name 'llama' \
    --template 'llama2_zh' \
    --train_file 'data/train.json' \
    --valid_file 'data/dev.json' \
    --output_dir=${output_dir} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --preprocessing_num_workers 16 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --max_grad_norm 0.5 \
    --optim "adamw_torch" \
    --max_source_length 400 \
    --cutoff_len 700 \
    --max_target_length 300 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --lora_dropout 0.05 \
    --bf16 
```

## 模型评估

我们提供了评估各个任务F1分数的脚本[eval_func.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/ie2instruction/eval_func.py)，以NER任务为例：

```bash
python ie2instruction/eval_func.py \
  --path1 data/NER/processed.json \
  --task NER 
```

- `task`: 目前支持['`RE`', '`NER`', '`EE`', '`EET`', '`EEA`']五类任务。
- 可以设置 `sort_by` 为 `source`, 分别计算每个数据集上的F1分数。

## 常见问题

-  使用vllm进行推理时，请按照vllm官网的指引重新配置一个新的环境，并确保符合vllm所需的CUDA版本要求。

- 注意精度问题，目前OneKE训练的精度为bf16，若使用fp16可能会导致问题。
- 在ModelScope上直接部署模型可能会影响模型效果，建议谨慎操作。
- 请确保修改模型路径为你在本地下载的位置，以确保正常使用。

