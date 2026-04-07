# # 下载完模型可以注释掉
# import os
# os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
# from huggingface_hub import snapshot_download
# def load_from_hf(model_name: str):
#     # usage: model, params, tokenizer = load_from_hf(state-spaces/mamba-130m-hf)
#     base_path = snapshot_download(repo_id=model_name,cache_dir=None)
#     print(f"download done, {model_name}:\t{base_path}")
#     return load_from_cache(base_path)

import json
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from safetensors import safe_open
from model import Mamba, ModelArgs

from functools import partial


def load_from_cache(base_path: str):
    model,params = load_flax_mamba(base_path)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    return model,params,tokenizer

def load_flax_mamba(base_path: str):

    config_path = base_path + "/config.json"
    model_path = base_path + "/model.safetensors"

    # 1. 解析 config.json 获取模型结构
    with open(config_path, 'r') as f:
        config = json.load(f)

    args = ModelArgs(
        d_model=config.get('hidden_size', config.get('d_model', 768)),
        n_layer=config.get('num_hidden_layers', config.get('n_layer', 24)),
        vocab_size=config.get('vocab_size', 50277)
    )

    model = Mamba(args)

    # 2. 加载 safetensors 权重并转换为 Flax 参数字典
    params = {}

    with safe_open(model_path, framework="np", device="cpu") as f:
        # 提取 Embedding (注意 s) 和最后的 Norm
        params['embedding'] = {'embedding': jnp.array(f.get_tensor('backbone.embeddings.weight'))}
        params['norm_f'] = {'weight': jnp.array(f.get_tensor('backbone.norm_f.weight'))}

        # 遍历每一层进行权重映射
        for i in range(args.n_layer):
            layer_name = f'layers_{i}'
            pt_prefix = f'backbone.layers.{i}'

            # 卷积层权重维度转换: PyTorch(C_out, C_in, L) -> Flax(L, C_in, C_out)
            conv_weight = f.get_tensor(f'{pt_prefix}.mixer.conv1d.weight')
            conv_weight = jnp.transpose(conv_weight, (2, 1, 0))

            # 线性层需要转置 (.T)
            params[layer_name] = {
                'norm': {'weight': jnp.array(f.get_tensor(f'{pt_prefix}.norm.weight'))},
                'mixer': {
                    'A_log': jnp.array(f.get_tensor(f'{pt_prefix}.mixer.A_log')),
                    'D': jnp.array(f.get_tensor(f'{pt_prefix}.mixer.D')),
                    'in_proj': {'kernel': jnp.array(f.get_tensor(f'{pt_prefix}.mixer.in_proj.weight')).T},
                    'conv1d': {
                        'kernel': conv_weight,
                        'bias': jnp.array(f.get_tensor(f'{pt_prefix}.mixer.conv1d.bias'))
                    },
                    'x_proj': {'kernel': jnp.array(f.get_tensor(f'{pt_prefix}.mixer.x_proj.weight')).T},
                    'dt_proj': {
                        'kernel': jnp.array(f.get_tensor(f'{pt_prefix}.mixer.dt_proj.weight')).T,
                        'bias': jnp.array(f.get_tensor(f'{pt_prefix}.mixer.dt_proj.bias'))
                    },
                    'out_proj': {'kernel': jnp.array(f.get_tensor(f'{pt_prefix}.mixer.out_proj.weight')).T}
                }
            }

    return model, params

# 1. 预处理 Prompt，提取并返回最后一个 token 的 logits 以及模型的初始状态 (Prefill)
@partial(jax.jit, static_argnames=['model'])
def prefill(model, params, input_ids):
    logits, states = model.apply({'params': params}, input_ids)
    return logits[:, -1, :], states

# 2. 核心状态转移，形状锁定为 (batch,)，永远只编译 1 次 (Decoding)
@partial(jax.jit, static_argnames=['model'])
def step_fn(model, params, input_id, states):
    next_logits, new_states = model.apply({'params': params}, input_id, states, method=model.step)
    return next_logits, new_states

# generate函数不应该jit，它jit后运行速度会慢得多，但SPU似乎要求JIT
# @partial(jax.jit, static_argnames=['model','n_tokens_to_gen','sample','top_k'])
def generate(model, params, input_ids, n_tokens_to_gen: int = 50,
             sample: bool = True, top_k: int = 40, seed: int = 42):
    key = jax.random.PRNGKey(seed)
    next_token_logits, states = prefill(model, params, input_ids)

    generated = jnp.zeros((input_ids.shape[0], n_tokens_to_gen), dtype=input_ids.dtype)

    for i in range(n_tokens_to_gen):
        if top_k is not None:
            values, _ = jax.lax.top_k(next_token_logits, k=top_k)
            kth_values = values[:, -1:]
            next_token_logits = jnp.where(next_token_logits < kth_values, -1e9, next_token_logits)

        probs = jax.nn.softmax(next_token_logits, axis=-1)

        if sample:
            key, subkey = jax.random.split(key)
            next_id = jax.random.categorical(subkey, jnp.log(probs + 1e-9), axis=-1)
        else:
            next_id = jnp.argmax(probs, axis=-1)

        generated = generated.at[:, i].set(next_id)

        next_token_logits, states = step_fn(model, params, next_id, states)

    return generated

# generate函数不应该jit，它jit后运行速度会慢得多，但SPU似乎要求JIT
# @partial(jax.jit, static_argnames=['model','n_tokens_to_gen','sample','top_k'])
def generate_min_p(model, params, input_ids, n_tokens_to_gen: int = 50,
             min_p: float = 0.1, seed: int = 42):
    key = jax.random.PRNGKey(seed)
    next_token_logits, states = prefill(model, params, input_ids)

    generated = jnp.zeros((input_ids.shape[0], n_tokens_to_gen), dtype=input_ids.dtype)

    for i in range(n_tokens_to_gen):
        max_logit = jnp.max(next_token_logits, axis=-1, keepdims=True)
        threshold = jnp.log(min_p) + max_logit # math hack
        mask = next_token_logits >= threshold
        filtered_logits = jnp.where(mask, next_token_logits, -1e10)

        # 随机采样
        key, subkey = jax.random.split(key)
        next_id = jax.random.categorical(subkey, filtered_logits, axis=-1)

        generated = generated.at[:, i].set(next_id)

        next_token_logits, states = step_fn(model, params, next_id, states)

    return generated


if __name__ == '__main__':

    # model_name = 'state-spaces/mamba-130m-hf'
    base_path="/root/.cache/huggingface/hub/models--state-spaces--mamba-130m-hf/snapshots/1e76775f628fbf1350fbe4dbb3d971ba64af25a1"
    model, params, tokenizer = load_from_cache(base_path)

    print('\n------\nRun on CPU')
    prompt = 'Python is'
    input_ids = tokenizer.encode(prompt, return_tensors='jax')
    output_ids = generate_min_p(model, params, input_ids, 10, seed=42)
    print(prompt, tokenizer.decode(output_ids[0]), sep='')

    # import time
    #
    # print("\nbenchmark")
    # rounds = 20
    # gen_len = 50
    # prompt = 'Python is'
    #
    # input_ids = tokenizer.encode(prompt, return_tensors='jax')
    # time0 = time.time()
    # time1 = time.time()
    # for s in range(1, rounds + 1):
    #     output_ids = generate(model, params, input_ids, gen_len, seed=s)
    #     # print(prompt, tokenizer.decode(output_ids[0]), sep='')
    #     print(f"round:{s},\trun: {time.time() - time1:.3f},\ttotal: {time.time() - time0:.3f}")
    #     time1 = time.time()
    # elapsed = time.time() - time0
    # print(f"avg runtime: {elapsed / rounds}, tokens/second: {rounds * gen_len / elapsed}")
    # # autodl cpu: avg runtime: 1.0532284736633302, tokens/second: 47.473080390706144
