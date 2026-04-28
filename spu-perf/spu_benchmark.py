import os
import json
import jax
import jax.numpy as jnp
from time import time

# 调整为您环境的 import
import spu.utils.distributed as ppd
import spu.libspu as libspu

from flax_rnn.helper import setup_hf_cache, load_from_cache, prefill, sampler_min_p
from flax_rnn.model import ResidualBlock, RMSNorm, ModelArgs

import logging
logging.disable()

# ==========================================
# 1. 定义密态调度器 (无自动解密)
# ==========================================
def run_sealed(func, static_argnums=(), copts=libspu.CompilerOptions()):
    """将函数抛给 SPU 执行，返回密态对象，不进行 ppd.get() 解密"""
    def wrapper(*args, **kwargs):
        return ppd.device("SPU")(func, static_argnums, copts)(*args, **kwargs)
    return wrapper

# ==========================================
# 2. 定义每一层的纯函数 (供 SPU JIT 编译)
# ==========================================
def spu_embed_step(input_id, emb_weight):
    # 根据输入的 token ID 提取 embedding
    return emb_weight[input_id]

def spu_layer_step(x, conv_state, ssm_state, layer_params, args: ModelArgs):
    # 显式实例化 ResidualBlock 处理单层 step
    model = ResidualBlock(args)
    # 调用 flax 的 apply，传入当前层的参数和状态
    return model.apply({'params': layer_params}, x, conv_state, ssm_state, method=model.step)

def spu_head_step(x, norm_f_params, emb_weight, args: ModelArgs):
    # 最后的 LayerNorm 和 线性映射
    x = RMSNorm(args.d_model).apply({'params': norm_f_params}, x)
    logits = jnp.dot(x, emb_weight.T)
    return logits

# ==========================================
# 3. 分层调度的 SPU Generate 函数
# ==========================================
def generate_spu_profile(model, params, input_ids, n_tokens_to_gen: int = 10, min_p: float = 0.1, seed: int = 42):
    # 提取模型配置
    args = model.args
    key = jax.random.PRNGKey(seed)

    # 1. 在 CPU (Driver端) 进行明文 Prefill，获取初始的 logits 和 states
    print("[CPU] Running Prefill...")
    next_token_logits, states = prefill(model, params, input_ids)

    # 我们假设 P1 负责做明密文转换的中转（单机多进程模拟时这很常见）
    P1 = ppd.device("P1")

    # 将模型参数传输/转为密态 (只取用到的部分)
    print("[SPU] Sealing parameters...")
    emb_weight_s = P1(lambda x: x)(params['embedding']['embedding'])
    norm_f_params_s = P1(lambda x: x)(params['norm_f'])

    layers_params_s = []
    for i in range(args.n_layer):
        layers_params_s.append(P1(lambda x: x)(params[f'layers_{i}']))

    # 包装 SPU 密态执行函数
    run_embed = run_sealed(spu_embed_step)
    run_layer = run_sealed(spu_layer_step, static_argnums=(4,))
    run_head = run_sealed(spu_head_step, static_argnums=(3,))

    output_ids = []

    print("Start SPU generation loop...")
    for step_idx in range(n_tokens_to_gen):
        step_time0 = time()

        # 2. 对从上一轮（或 Prefill）拿到的 Logits 进行采样 (CPU 端)
        key, subkey = jax.random.split(key)
        next_id_plain = sampler_min_p(logits=next_token_logits, key=subkey, min_p=min_p)
        output_ids.append(next_id_plain.item())

        # 3. 密封新采样的 token 和当前状态发送给 SPU
        input_id_s = P1(lambda x: x)(next_id_plain)

        states_s = []
        for conv_state, ssm_state in states:
            states_s.append((P1(lambda x: x)(conv_state), P1(lambda x: x)(ssm_state)))

        # [SPU Profile: Embedding]
        x_s = run_embed(input_id_s, emb_weight_s)

        # [SPU Profile: Layer-by-Layer]
        new_states_s = []
        for i in range(args.n_layer):
            conv_s, ssm_s = states_s[i]
            x_s, new_conv_s, new_ssm_s = run_layer(x_s, conv_s, ssm_s, layers_params_s[i], args)
            new_states_s.append((new_conv_s, new_ssm_s))

        # [SPU Profile: LM Head]
        logits_s = run_head(x_s, norm_f_params_s, emb_weight_s, args)

        # 4. 把状态和 Logits 拿回 CPU (准备下一次循环)
        # 注意：真实生产环境中 states 应该一直留在 SPU，只有 logits 被 get() 拿回客户端
        # 为了尽量复用您的 CPU 循环逻辑和验证结果正确性，这里临时 fetch 回来。
        next_token_logits = ppd.get(logits_s)
        states = ppd.get(new_states_s)

        print(f"-> Step {step_idx + 1} elapsed: {time() - step_time0:.2f}s | Sampled token ID: {next_id_plain.item()}")

    return output_ids

# ==========================================
# 4. 运行入口
# ==========================================
if __name__ == '__main__':
    # 导入 Emulator（根据您环境的具体路径调整，可能是 sml.utils.emulation 或 spu_emulator）
    import sml.utils.emulation as emulation

    # === 1. SPU 集群初始化 ===
    print("Starting SPU Emulator cluster...")
    # 启动本地多进程模拟集群 (会自动拉起后台进程并调用 ppd.init)
    emulator = emulation.Emulator("3pc_no_profile.json", emulation.Mode.MULTIPROCESS)
    emulator.up()

    try:
        # === 2. 模型加载 ===
        # 替换为您环境的实际缓存路径
        base_path = "/root/autodl-shared/hf_cache/hub/models--state-spaces--mamba-130m-hf/snapshots/1e76775f628fbf1350fbe4dbb3d971ba64af25a1"
        model, params, tokenizer = load_from_cache(base_path)

        print('\n------\nRun on SPU (Layer-by-Layer Profiling)')
        prompt = 'Python is'
        input_ids = tokenizer.encode(prompt, return_tensors='jax')

        # === 3. Warmup 环节 ===
        print("SPU warmup (JIT compilation will happen here):")
        time0 = time()
        output_ids = generate_spu_profile(model, params, input_ids, n_tokens_to_gen=2, seed=42)
        print(prompt, tokenizer.decode(output_ids), sep='')
        print(f"Warmup elapsed: {time()-time0:.2f}s\n")

        # === 4. 实际 Benchmark ===
        print("SPU run!")
        time0 = time()
        output_ids = generate_spu_profile(model, params, input_ids, n_tokens_to_gen=10, seed=42)
        print(f"\nFinal Output: {prompt}", tokenizer.decode(output_ids), sep='')
        print(f"Total elapsed: {time()-time0:.2f}s")

    finally:
        # === 5. 关闭集群 ===
        # 非常重要：测试结束后必须调用 down() 杀死后台进程，否则端口会被一直占用
        print("Shutting down SPU Emulator cluster...")
        emulator.down()