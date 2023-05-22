# Modification for LLama


1. deepspeed版本0.8.3, megatron/neox_arguments/arguments.py的1010行要写成>=2, 这是为了修gpt neox别的错误改成了1，但是1的话，llama train不了, 这个修了之后会报另一个deepspeed的错误，需要改源码/conda/envs/llm/lib/python3.8/site-packages/deepspeed/runtime/engine.py的3025行要加

```
for ckpt_name in zero_ckpt_names:
    if not os.path.exists(ckpt_name):
        return None
```

参考： https://github.com/EleutherAI/gpt-neox/issues/921
