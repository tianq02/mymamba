import jax
import jax.numpy as jnp
from time import time
import spu.utils.distributed as ppd
import spu.libspu as libspu
from flax_rnn.model import ResidualBlock, RMSNorm, ModelArgs
from flax_rnn.helper import sampler_min_p, sampler_greedy, load_from_cache
import gc

import logging
logging.disable()

# ==========================================
# 1. 密态调度器
# ==========================================
def run_sealed(func, static_argnums=(), copts=libspu.CompilerOptions()):
    def wrapper(*args, **kwargs):
        return ppd.device("SPU")(func, static_argnums, copts)(*args, **kwargs)
    return wrapper

# ==========================================
# 2. 定义 SPU 纯函数 (分为 Prefill 和 Decode 两组)
# ==========================================
# --- Prefill 组 (处理形状: [batch, seq_len, d_model]) ---
def spu_embed_prefill(input_ids, emb_weight):
    return emb_weight[input_ids]

def spu_layer_prefill(x, layer_params, args: ModelArgs):
    # 处理整个 Prompt 序列，返回输出和最终的 (conv_state, ssm_state)
    model = ResidualBlock(args)
    return model.apply({'params': layer_params}, x)

def spu_head_prefill(x, norm_f_params, emb_weight, args: ModelArgs):
    # 取序列最后一个位置的特征计算 Logits
    x_last = x[:, -1, :]
    x_last = RMSNorm(args.d_model).apply({'params': norm_f_params}, x_last)
    return jnp.dot(x_last, emb_weight.T)

# --- Decode 组 (处理形状: [batch, 1, d_model] 并在内部被降维) ---
def spu_embed_step(input_id, emb_weight):
    return emb_weight[input_id]

def spu_layer_step(x, conv_state, ssm_state, layer_params, args: ModelArgs):
    model = ResidualBlock(args)
    return model.apply({'params': layer_params}, x, conv_state, ssm_state, method=model.step)

def spu_head_step(x, norm_f_params, emb_weight, args: ModelArgs):
    x = RMSNorm(args.d_model).apply({'params': norm_f_params}, x)
    return jnp.dot(x, emb_weight.T)

# ==========================================
# 3. 终极版：全密态流水线 Generate
# ==========================================
def generate_spu_ultimate(model, params, input_ids_plain, n_tokens_to_gen: int = 10, min_p: float = 0.1, seed: int = 42):
    args = model.args
    key = jax.random.PRNGKey(seed)

    # 假设 P1 为 Client(提供输入)，P2 为 Server(提供权重)
    # 演示中我们均使用 P1 进行加密模拟
    P1 = ppd.device("P1")

    print("[Secure Pipeline] Sealing inputs and weights...")
    input_ids_s = P1(lambda x: x)(input_ids_plain)

    emb_weight_s = P1(lambda x: x)(params['embedding']['embedding'])
    norm_f_params_s = P1(lambda x: x)(params['norm_f'])
    layers_params_s = [P1(lambda x: x)(params[f'layers_{i}']) for i in range(args.n_layer)]

    # 包装函数
    run_em_pre = run_sealed(spu_embed_prefill)
    run_ly_pre = run_sealed(spu_layer_prefill, static_argnums=(2,))
    run_hd_pre = run_sealed(spu_head_prefill, static_argnums=(3,))

    run_em_dec = run_sealed(spu_embed_step)
    run_ly_dec = run_sealed(spu_layer_step, static_argnums=(4,))
    run_hd_dec = run_sealed(spu_head_step, static_argnums=(3,))

    run_sampler = run_sealed(sampler_min_p, static_argnums=(2,))

    output_ids_plain = []

    # ---------------------------------------------------------
    # 第一阶段：密态 Prefill (按层执行，防止 JIT 内存溢出)
    # ---------------------------------------------------------
    print("[SPU] Running Layer-by-Layer Prefill...")
    prefill_time0 = time()

    x_s = run_em_pre(input_ids_s, emb_weight_s)
    states_s = []

    for i in range(args.n_layer):
        x_s, layer_state_s = run_ly_pre(x_s, layers_params_s[i], args)
        states_s.append(layer_state_s)  # 状态直接在 SPU 中驻留！

    logits_s = run_hd_pre(x_s, norm_f_params_s, emb_weight_s, args)

    # 密态采样第一个 Token
    key, subkey = jax.random.split(key)
    subkey_s = P1(lambda x: x)(subkey)  # 客户端向 SPU 提供加密的随机种子
    next_id_s = run_sampler(logits_s, subkey_s, min_p)

    # 仅解密 Token 给客户端
    next_id_plain = ppd.get(next_id_s)
    output_ids_plain.append(next_id_plain.item())

    print(f"-> Prefill elapsed: {time() - prefill_time0:.2f}s | Token: {next_id_plain.item()}")

    # ---------------------------------------------------------
    # 第二阶段：密态 Decode 循环 (全程无中间状态泄露)
    # ---------------------------------------------------------
    print("[SPU] Running Decode Loop...")
    for step_idx in range(n_tokens_to_gen - 1):
        step_time0 = time()

        # 注意：直接使用上一轮还在 SPU 中的 next_id_s，客户端甚至连新 Token 都不需要传回！
        x_s = run_em_dec(next_id_s, emb_weight_s)

        new_states_s = []
        for i in range(args.n_layer):
            conv_s, ssm_s = states_s[i]
            x_s, new_conv_s, new_ssm_s = run_ly_dec(x_s, conv_s, ssm_s, layers_params_s[i], args)
            new_states_s.append((new_conv_s, new_ssm_s))

        del states_s
        del next_id_s

        states_s = new_states_s  # 状态继续在 SPU 轮转

        logits_s = run_hd_dec(x_s, norm_f_params_s, emb_weight_s, args)

        # 客户端分发新的随机种子，在密态完成采样
        key, subkey = jax.random.split(key)
        subkey_s = P1(lambda x: x)(subkey)
        next_id_s = run_sampler(logits_s, subkey_s, min_p)

        # 解密 Token 给客户端呈现
        next_id_plain = ppd.get(next_id_s)
        output_ids_plain.append(next_id_plain.item())

        print(f"-> Decode Step {step_idx + 1} elapsed: {time() - step_time0:.2f}s | Token: {next_id_plain.item()}")
        gc.collect()

    return output_ids_plain

# ==========================================
# 4. 运行入口
# ==========================================
if __name__ == '__main__':
    # 导入 Emulator（根据您环境的具体路径调整，可能是 sml.utils.emulation 或 spu_emulator）
    import sml.utils.emulation as emulation

    # === 1. SPU 集群初始化 ===
    print("Starting SPU Emulator cluster...")
    # 启动本地多进程模拟集群 (会自动拉起后台进程并调用 ppd.init)
    emulator = emulation.Emulator("3pc.json", emulation.Mode.MULTIPROCESS)
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
        output_ids = generate_spu_ultimate(model, params, input_ids[:1], n_tokens_to_gen=2, seed=42)
        print(prompt, tokenizer.decode(output_ids), sep='')
        print(f"Warmup elapsed: {time()-time0:.2f}s\n")

        # 原因未知，但分两段跑总会爆内存
        # === 4. 实际 Benchmark ===
        print("SPU run!")
        time0 = time()
        output_ids = generate_spu_ultimate(model, params, input_ids, n_tokens_to_gen=10, seed=42)
        print(f"\nFinal Output: {prompt}", tokenizer.decode(output_ids), sep='')
        print(f"Total elapsed: {time()-time0:.2f}s")

    finally:
        # === 5. 关闭集群 ===
        # 非常重要：测试结束后必须调用 down() 杀死后台进程，否则端口会被一直占用
        print("Shutting down SPU Emulator cluster...")
        emulator.down()