# # 下载完模型可以注释掉，别忘了demo_ex顶上也要
# import os
#
# os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
# os.environ['HF_HOME'] = "/root/autodl-shared/hf_cache"  # autodl
# # os.environ['HF_HOME'] = "/root/shared-nvme/hf_cache"  # paratera
#
# from huggingface_hub import snapshot_download
#
#
# def setup_hf_cache(model_name: str):
#     # usage: base_path = setup_hf_cache("state-spaces/mamba-130m-hf")
#     base_path = snapshot_download(repo_id=model_name, cache_dir=None)
#     print(f'base_path = "{base_path}"')
#     return base_path


import json
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from safetensors import safe_open
from .model import Mamba, ModelArgs  # 删掉.来运行demo

from functools import partial


def load_from_cache(base_path: str):
    model, params = load_flax_mamba(base_path)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    return model, params, tokenizer


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


# 基线，只测时间
def sampler_baseline(logits, key=0):
    return [0]


# 贪心采样
def sampler_greedy(logits, key=0):
    return jnp.argmax(logits, axis=-1)


# top k
def sampler_top_k(logits, key=jax.random.PRNGKey(42), top_k=40):
    # 1. 找到第 k 大的 logit 值（阈值）
    values, _ = jax.lax.top_k(logits, k=top_k)
    kth = values[:, -1:]  # 形状 (batch, 1)

    # 2. 掩码：小于阈值的设为 -1e9（或 -inf）
    masked_logits = jnp.where(logits < kth, -1e9, logits)

    # 3. 直接采样（categorical 内部会做 softmax）
    return jax.random.categorical(key, masked_logits, axis=-1)


# min p，变形优化版本
def sampler_min_p(logits, key=jax.random.PRNGKey(42), min_p=0.1):
    max_logit = jnp.max(logits, axis=-1, keepdims=True)
    threshold = jnp.log(min_p) + max_logit  # math hack
    mask = logits >= threshold
    filtered_logits = jnp.where(mask, logits, -1e10)
    return jax.random.categorical(key, filtered_logits, axis=-1)


# 老代码
def sampler_legacy(logits, key=jax.random.PRNGKey(42), sample: bool = True, top_k: int = 40):
    if top_k is not None:
        values, _ = jax.lax.top_k(logits, k=top_k)
        kth_values = values[:, -1:]
        logits = jnp.where(logits < kth_values, -1e9, logits)

    probs = jax.nn.softmax(logits, axis=-1)

    if sample:
        key, subkey = jax.random.split(key)
        next_id = jax.random.categorical(subkey, jnp.log(probs + 1e-9), axis=-1)
    else:
        next_id = jnp.argmax(probs, axis=-1)

    return next_id


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
def generate(model, params, input_ids, n_tokens_to_gen: int = 50, sampler=sampler_min_p, seed: int = 42):
    key = jax.random.PRNGKey(seed)
    next_token_logits, states = prefill(model, params, input_ids)
    generated = jnp.zeros((input_ids.shape[0], n_tokens_to_gen), dtype=input_ids.dtype)

    for i in range(n_tokens_to_gen):
        key, subkey = jax.random.split(key)
        next_id = sampler(logits=next_token_logits, key=subkey)
        generated = generated.at[:, i].set(next_id)
        next_token_logits, states = step_fn(model, params, next_id, states)

    return generated

def generate_static(model, params, input_ids, n_tokens_to_gen: int = 50, sampler=sampler_greedy, seed: int = 0):
    next_token_logits, states = prefill(model, params, input_ids)
    generated = jnp.zeros((input_ids.shape[0], n_tokens_to_gen), dtype=input_ids.dtype)

    for i in range(n_tokens_to_gen):
        next_id = sampler(logits=next_token_logits)
        generated = generated.at[:, i].set(next_id)
        next_token_logits, states = step_fn(model, params, next_id, states)

    return generated


def generate_demo(model, params, tokenizer, prompt: str, n_tokens_to_gen: int = 50, sampler=sampler_min_p, seed=42):
    print(prompt, end="")
    key = jax.random.PRNGKey(seed)
    input_ids = tokenizer.encode(prompt, return_tensors='jax')
    next_token_logits, states = prefill(model, params, input_ids)

    for i in range(n_tokens_to_gen):
        key, subkey = jax.random.split(key)
        next_id = sampler(logits=next_token_logits, key=subkey)
        print(tokenizer.decode([next_id.item()]), end="")
        next_token_logits, states = step_fn(model, params, next_id, states)

    return None


# 仅用于检查模型rnn部分(step)部分正确性，慢到爆炸
def generate_demo_no_rnn(model, params, tokenizer, prompt: str, n_tokens_to_gen: int = 50, sampler=sampler_min_p, seed=42):
    print(prompt, end="")
    key = jax.random.PRNGKey(seed)
    input_ids = tokenizer.encode(prompt, return_tensors='jax')

    for i in range(n_tokens_to_gen):
        next_token_logits, _ = prefill(model, params, input_ids)
        key, subkey = jax.random.split(key)
        next_id = sampler(logits=next_token_logits, key=subkey)
        input_ids = jnp.concatenate([input_ids, next_id[None,:]], axis=1)
        print(tokenizer.decode([next_id.item()]), end="")
        # print(tokenizer.decode(input_ids[0].tolist()))

    return None

# 贪心搜索，理论最快，但效果最差
def generate_greedy(model, params, input_ids, n_tokens_to_gen: int = 50, seed: int = 0):
    return generate_static(model, params, input_ids, n_tokens_to_gen, sampler_greedy)


# 优化掉第一个softmax的版本
def generate_topk(model, params, input_ids, n_tokens_to_gen: int = 50, top_k: int = 40, seed: int = 42):
    sampler = lambda logits, key: sampler_top_k(logits, key, top_k)
    return generate(model, params, input_ids, n_tokens_to_gen, sampler, seed)


# min p 采样，理论上计算开销更低
def generate_minp(model, params, input_ids, n_tokens_to_gen: int = 50, min_p: float = 0.1, seed: int = 42):
    sampler = lambda logits, key: sampler_min_p(logits, key, min_p)
    return generate(model, params, input_ids, n_tokens_to_gen, sampler, seed)


# 独立版本的min p采样，
# @partial(jax.jit, static_argnames=['model','n_tokens_to_gen','sample','top_k'])
def generate_minp_s(model, params, input_ids, n_tokens_to_gen: int = 50, min_p: float = 0.1, seed: int = 42):
    key = jax.random.PRNGKey(seed)
    next_token_logits, states = prefill(model, params, input_ids)
    generated = jnp.zeros((input_ids.shape[0], n_tokens_to_gen), dtype=input_ids.dtype)

    for i in range(n_tokens_to_gen):
        max_logit = jnp.max(next_token_logits, axis=-1, keepdims=True)
        threshold = jnp.log(min_p) + max_logit  # math hack
        mask = next_token_logits >= threshold
        filtered_logits = jnp.where(mask, next_token_logits, -1e10)

        key, subkey = jax.random.split(key)
        next_id = jax.random.categorical(subkey, filtered_logits, axis=-1)

        generated = generated.at[:, i].set(next_id)

        next_token_logits, states = step_fn(model, params, next_id, states)

    return generated


if __name__ == '__main__':
    # base_path = setup_hf_cache("state-spaces/mamba-130m-hf")
    base_path="/root/autodl-shared/hf_cache/hub/models--state-spaces--mamba-130m-hf/snapshots/1e76775f628fbf1350fbe4dbb3d971ba64af25a1"  # autodl
    # base_path = "/root/shared-nvme/hf_cache/hub/models--state-spaces--mamba-130m-hf/snapshots/1e76775f628fbf1350fbe4dbb3d971ba64af25a1"  # paratera
    model, params, tokenizer = load_from_cache(base_path)

    # model.args.use_zoh=True # defaults False
    #
    # sampler = lambda logits,key: sampler_min_p(logits,key,min_p=0.05)
    # def generate_demo_ex(prompt: str, n_tokens_to_gen: int = 50, seed: int = 42):
    #     generate_demo(model, params, tokenizer, prompt, n_tokens_to_gen, sampler=sampler, seed=seed)
    #     # generate_demo_no_rnn(model, params, tokenizer, prompt, n_tokens_to_gen, sampler=sampler, seed=seed)
    #
    # prompt = input("prompt: ")
    # generate_demo_ex(prompt)


    # prefill benchmark
    import time

    input_len = 4000
    input_file = "../LICENSE" # AGPL3, 6937 tokens
    laps = 10

    with open(input_file, 'r') as file:
        file_content = file.read()

    licence_ids = tokenizer.encode(file_content, return_tensors='jax')
    licence_len = len(licence_ids[0])
    warmup_ids = licence_ids[:,:input_len]

    print(f"Prefill {input_len} tokens")

    print("="*20 + "Serial" + "="*20)
    model.args.use_parallel_scan=False
    time0 = time.time()
    _ = prefill(model, params, warmup_ids)
    print(f"warmup: {time.time()-time0}s")
    time_sum = .0
    for i in range(laps):
        benchmark_start = hash(time.time_ns() + i) % (licence_len-input_len) # 超简易随机数生成
        benchmark_ids = licence_ids[:,benchmark_start:benchmark_start+input_len]
        time0 = time.time()
        _ = prefill(model, params, benchmark_ids)
        time1 = time.time() - time0
        print(f"scan {i:2d}:{time1}s")
        time_sum += time1
    print(f"average: {time_sum/laps}s")

    # parallel scan takes more time to compile, but it can be more performant later
    print("="*20 + "Parallel" + "="*20)
    model.args.use_parallel_scan=True
    _ = prefill(model, params, warmup_ids)
    print(f"warmup: {time.time()-time0}s")
    time_sum = .0
    for i in range(laps):
        benchmark_start = hash(time.time_ns() + i) % (licence_len-input_len) # 超简易随机数生成
        benchmark_ids = licence_ids[:,benchmark_start:benchmark_start+input_len]
        time0 = time.time()
        _ = prefill(model, params, benchmark_ids)
        time1 = time.time() - time0
        print(f"scan {i:2d}:{time1}s")
        time_sum += time1
    print(f"average: {time_sum/laps}s")


    # from time import time
    #
    # print('\n------\nRun on CPU')
    # prompt = 'Python is'
    # input_ids = tokenizer.encode(prompt, return_tensors='jax')
    #
    # print("warmup:")
    # time0 = time()
    # output_ids = generate_minp(model, params, input_ids, 10, seed=42)
    # print(prompt, tokenizer.decode(output_ids[0]), sep='')
    # print(f"elapsed: {time()-time0}")
    #
    # print("split")
    # time0 = time()
    # output_ids = generate_minp(model, params, input_ids, 10, seed=42)
    # print(prompt, tokenizer.decode(output_ids[0]), sep='')
    # print(f"elapsed: {time()-time0}")
    #
    # print("standalone")
    # time0 = time()
    # output_ids = generate_minp_s(model, params, input_ids, 10, seed=42)
    # print(prompt, tokenizer.decode(output_ids[0]), sep='')
    # print(f"elapsed: {time()-time0}")
