#!/bin/bash

### 将本次作业计费到导师课题组，tutor_project改为导师创建的课题组名
#SBATCH --comment=joint_project

### 给你这个作业起个名字，方便识别不同的作业
#SBATCH --job-name="tokenize_llama-cn_pubmed_abstract"

### 指定该作业需要多少个节点
### 注意！没有使用多机并行（MPI/NCCL等），下面参数写1！不要多写，多写了也不会加速程序！
#SBATCH --nodes=1

### 指定该作业需要多少个CPU核心
### 注意！一般根据队列的CPU核心数填写，比如cpu队列64核，这里申请64核，并在你的程序中尽量使用多线程充分利用64核资源！
#SBATCH --ntasks=64

### 指定该作业在哪个队列上执行
#SBATCH --partition=cpu64c

### 以上参数用来申请所需资源
### 以下命令将在计算节点执行


### 激活一个 Anaconda 环境 your_env
conda activate llm

### 执行你的作业
### 本任务是tokenize你的zst数据文件, 
### input是zst数据的文件地址
### output-prefix写文件输出目录+输出文件的prefix, 最后生成的bin和idx文件会自动增加_text_document.bin or .idx的后缀
### vocab和merge-file都是tokenizer的文件
### workers多进程数目要开大才够快
### append-eod是说在把多个doc合并成一个sample的时候，doc和doc之间会加特殊字符eod作为分隔
python /home/share/gsai_joint_project/llama_train/gpt-neox-main/tools/preprocess_data.py \
            --input /fs/archive/share/jarvis/pile_original/PubMed_Abstracts.jsonl \
            --output-prefix /fs/fast/u2021000178/data/new_llama-cn_tokenized/pubmed_abstract/train \
            --vocab /home/share/jarvis/tokenizer/llama_chinese_tokenizer.model \
            --dataset-impl mmap \
            --tokenizer-type LlamaTokenizer \
            --append-eod \
            --workers=64 \


# python /home/share/gsai_joint_project/llama_train/gpt-neox-main/tools/preprocess_data.py \
#             --input /fs/fast/u2020000280/data/csl/csl-val.jsonl\
#             --output-prefix /fs/fast/u2021000178/data/new_llama-cn_tokenized/csl/val \
#             --vocab /home/share/jarvis/tokenizer/llama_chinese_tokenizer.model \
#             --dataset-impl mmap \
#             --tokenizer-type LlamaTokenizer \
#             --append-eod \
#             --workers=64 \