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
#SBATCH --ntasks=24

### 指定该作业在哪个队列上执行
#SBATCH --partition=cpu24c

### output log
#SBATCH --output=convert_to_hf_v32.log

### 以上参数用来申请所需资源
### 以下命令将在计算节点执行

### 本例使用Anaconda中的Python，先将Python添加到环境变量配置好环境变量
source activate llm

### 执行你的作业
model_id="v43-en_20B_cn_hq_10B_real"
step="global_step22000"
input_dir=/fs/fast/share/jarvis/checkpoints/1-3B/$model_id/$step
config_file=/home/share/gsai_joint_project/llama_train/gpt-neox-main/jarvis_configs/1-3B/$model_id/hf_config.yml
output_dir=/fs/archive/share/jarvis/checkpoints/$model_id"_"$step
tokenizer_file="/fs/archive/share/yulan/tokenizer/yulan_v1/*"

python ./tools/convert_to_hf.py --input_dir $input_dir --config_file $config_file  --output_dir $output_dir

cp $tokenizer_file $output_dir
chgrp -R 2400194 $output_dir

chmod -R 770 $output_dir
echo "-----> Task finished..."
