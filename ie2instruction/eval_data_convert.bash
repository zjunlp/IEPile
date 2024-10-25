#!/bin/bash
# NER-zh
dir_path='/IEPile'
datasets=("MSRA" "clue" "resume_ner" "boson" "WEIBONER")
for dataset in "${datasets[@]}"; do
    mkdir -p ${dir_path}/IE-zh-converted/NER/${dataset}
    python ie2instruction/convert_func.py \
        --src_path ${dir_path}/IE-zh/NER/${dataset}/test.json \
        --tgt_path ${dir_path}/IE-zh-converted/NER/${dataset}/test.json \
        --schema_path ${dir_path}/IE-zh/NER/${dataset}/schema.json \
        --language zh \
        --task NER \
        --split_num 6 \
        --split test
done

# NER-en
datasets=("CrossNER_AI" "CrossNER_literature" "CrossNER_music" "CrossNER_politics" "CrossNER_science" "AnatEM" "bc2gm" "bc4chemd" "bc5cdr" "CoNLL2003" "FindVehicle" "FabNER" "GENIA_NER" "HarveyNER" "MultiNERD" "mit-movie" "mit-restaurant" "Ontonotes" "ncbi")
for dataset in "${datasets[@]}"; do
    mkdir -p ${dir_path}/IE-en-converted/NER/${dataset}
    python ie2instruction/convert_func.py \
        --src_path ${dir_path}/IE-en/NER/${dataset}/test.json \
        --tgt_path ${dir_path}/IE-en-converted/NER/${dataset}/test.json \
        --schema_path ${dir_path}/IE-en/NER/${dataset}/schema.json \
        --language en \
        --task NER \
        --split_num 6 \
        --split test
done

# RE-zh
datasets=("DuIE2.0" "CMeIE" "COAE2016" "IPRE" "SKE2020")
for dataset in "${datasets[@]}"; do
    mkdir -p ${dir_path}/IE-zh-converted/RE/${dataset}
    python ie2instruction/convert_func.py \
        --src_path ${dir_path}/IE-zh/RE/${dataset}/test.json \
        --tgt_path ${dir_path}/IE-zh-converted/RE/${dataset}/test.json \
        --schema_path ${dir_path}/IE-zh/RE/${dataset}/schema.json \
        --language zh \
        --task RE \
        --split_num 4 \
        --split test
done

# RE-en
datasets=("ADE_corpus" "New-York-Times-RE" "GIDS" "kbp37" "NYT11" "SciERC" "semval-RE" "conll04" "fewrel_0" "fewrel_1" "fewrel_2" "fewrel_3" "fewrel_4" "wiki_0" "wiki_1" "wiki_2" "wiki_3" "wiki_4")
for dataset in "${datasets[@]}"; do
    mkdir -p ${dir_path}/IE-en-converted/RE/${dataset}
    python ie2instruction/convert_func.py \
        --src_path ${dir_path}/IE-en/RE/${dataset}/test.json \
        --tgt_path ${dir_path}/IE-en-converted/RE/${dataset}/test.json \
        --schema_path ${dir_path}/IE-en/RE/${dataset}/schema.json \
        --language en \
        --task RE \
        --split_num 4 \
        --split test
done


# EE-zh
datasets=("DuEE-fin" "DuEE1.0" "ccf_law" "FewFC")
for dataset in "${datasets[@]}"; do
    mkdir -p ${dir_path}/IE-zh-converted/EE/${dataset}
    python ie2instruction/convert_func.py \
        --src_path ${dir_path}/IE-zh/EE/${dataset}/test.json \
        --tgt_path ${dir_path}/IE-zh-converted/EE/${dataset}/test.json \
        --schema_path ${dir_path}/IE-zh/EE/${dataset}/schema.json \
        --language zh \
        --task EE \
        --split_num 4 \
        --split test
done

# EE-en
datasets=("CASIE" "PHEE" "CrudeOilNews")
for dataset in "${datasets[@]}"; do
    mkdir -p ${dir_path}/IE-en-converted/EE/${dataset}
    python ie2instruction/convert_func.py \
        --src_path ${dir_path}/IE-en/EE/${dataset}/test.json \
        --tgt_path ${dir_path}/IE-en-converted/EE/${dataset}/test.json \
        --schema_path ${dir_path}/IE-en/EE/${dataset}/schema.json \
        --language en \
        --task EE \
        --split_num 4 \
        --split test
done

datasets=("life" "artifactexistence" "transaction" "conflict" "disaster" "personnel" "movement" "inspection" "justice" "manufacture" "government" "contact")
for dataset in "${datasets[@]}"; do
    mkdir -p ${dir_path}/IE-en-converted/EE/RAMS/${dataset}
    python ie2instruction/convert_func.py \
        --src_path ${dir_path}/IE-en/EE/RAMS/${dataset}/test.json \
        --tgt_path ${dir_path}/IE-en-converted/EE/RAMS/${dataset}/test.json \
        --schema_path ${dir_path}/IE-en/EE/RAMS/${dataset}/schema.json \
        --language en \
        --task EE \
        --split_num 4 \
        --split test
done

datasets=("Control" "Movement" "Transaction" "Life" "ArtifactExistence" "Cognitive" "Justice" "Conflict" "Medical" "GenericCrime" "Contact" "Personnel")
for dataset in "${datasets[@]}"; do
    mkdir -p ${dir_path}/IE-en-converted/EE/WikiEvents/${dataset}
    python ie2instruction/convert_func.py \
        --src_path ${dir_path}/IE-en/EE/WikiEvents/${dataset}/test.json \
        --tgt_path ${dir_path}/IE-en-converted/EE/WikiEvents/${dataset}/test.json \
        --schema_path ${dir_path}/IE-en/EE/WikiEvents/${dataset}/schema.json \
        --language en \
        --task EE \
        --split_num 4 \
        --split test
done

