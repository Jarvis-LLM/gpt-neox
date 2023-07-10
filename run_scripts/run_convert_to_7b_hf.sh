#!/bin/bash

### 将本次作业计费到导师课题组，tutor_project改为导师创建的课题组名
#SBATCH --comment=joint_project

### 给你这个作业起个名字，方便识别不同的作业
#SBATCH --job-name=convert_to_hf

### 指定该作业需要多少个节点
### 注意！没有使用多机并行（MPI/NCCL等），下面参数写1！不要多写，多写了也不会加速程序！
#SBATCH --nodes=1

### 指定该作业需要多少个CPU核心
### 注意！一般根据队列的CPU核心数填写，比如cpu队列64核，这里申请64核，并在你的程序中尽量使用多线程充分利用64核资源！
#SBATCH --ntasks=64

### 指定该作业在哪个队列上执行
#SBATCH --partition=cpu64c

### output log
#SBATCH --output=sbatch_convert_llama_7B_v16.log

### 以上参数用来申请所需资源
### 以下命令将在计算节点执行

### 本例使用Anaconda中的Python，先将Python添加到环境变量配置好环境变量
source activate llm

### 执行你的作业

# python ./tools/convert_to_hf.py --input_dir /home/share/gsai_joint_project/gpt-neox-2.0/checkpoints_1-3B_v3-pile+chinese/global_step22000 --config_file /home/u2021000178/share/gsai_joint_project/gpt-neox-2.0/jarvis_configs/1-3B/v3-pile+chinese/hf_config.yml --output_dir /home/share/gsai_joint_project/1-3B_v3-pile+chinese_global_step22000_hf
# python ./tools/convert_sequential_to_hf.py --input_dir /home/u2021000178/share/gsai_joint_project/llama_train/gpt-neox-main/checkpoints_v7-llama7b-linly/global_step2000 --config_file /home/u2021000178/share/gsai_joint_project/llama_train/gpt-neox-main/jarvis_configs/1-3B/v7-llama7b+linly/hf_config.yml --output_dir /home/u2021000178/share/gsai_joint_project/v7-llama7b-linly_step2000_hf/
# python ./tools/convert_llama_to_hf.py --input_dir /home/share/gsai_joint_project/gpt-neox-2.0/llama_model/global_step0 --config_file /home/share/gsai_joint_project/llama_train/gpt-neox-main/jarvis_configs/1-3B/v7-llama7b+linly/hf_config.yml --output_dir /home/share/gsai_joint_project/llama_7b_test_convert/

input_dir="/fs/fast/share/jarvis/checkpoints/7B/v14-new_data_source/global_step7600"
config_file="/home/share/gsai_joint_project/llama_train/gpt-neox-main/jarvis_configs/7B/v14-new_data_source/hf_config.yml"
output_dir="/fs/archive/share/jarvis/checkpoints/llama-7B_hf_v14_step7600/"
tokenizer_file="/fs/fast/share/jarvis/tokenizer/llama/tokenizer.model"


mkdir $output_dir
python ./tools/convert_llama_to_hf.py --input_dir $input_dir --config_file $config_file --output_dir $output_dir
cp $tokenizer_file  $output_dir
chgrp -R 2400194 $output_dir
chmod 770 $output_dir
chmod 770 $output_dir/*
echo "-----> Task finished..."

