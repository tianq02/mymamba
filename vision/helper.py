import jax
import jax.numpy as jnp
from safetensors import safe_open
from vision.model import MambaVision
from PIL import Image
import json

def load_from_hf(model_repo_id: str = "nvidia/MambaVision-T-1K"):
    import os
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
    os.environ['HF_HOME'] = "/root/autodl-shared/hf_cache"

    from huggingface_hub import hf_hub_download

    # MambaVision-T-1K, MambaVision-T2-1K, MambaVision-S-1K.
    model_filename = "model.safetensors"
    model_metadata = "config.json"

    # 也可以通过snapshot_download下载，但我们的代码用不上pth.tar，浪费空间。
    param_path = hf_hub_download(model_repo_id, model_filename)
    model_path = hf_hub_download(model_repo_id, model_metadata)

    # 将输出粘贴到cell中，然后注释上面的代码，后续运行无需网络。
    print(f'param_path = "{param_path}"')
    print(f'model_path = "{model_path}"')

    return param_path, model_path


def load_label(label_path: str, key: str|None = None):
    """
    加载id字典
    可以使用模型config.json的id2label
    labels = load_label(model_path, key="id2label")
    """
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            label_dict = json.load(f)
        if key is not None:
            label_dict = label_dict.get(key)
        labels = list(label_dict.values())
    except Exception as e:
        labels = [f"Class_{i}" for i in range(1000)]
    return labels


def load_model(model_path: str):
    """
    加载模型
    """
    with open(model_path, "r", encoding="utf-8") as f:
        config: dict = json.load(f)
        model = MambaVision(
            dim = config.get("dim", 80),
            in_dim = config.get("in_dim", 32),
            depths = config.get("depths", (1, 3, 8, 4)),
            window_size = config.get("window_size", (8, 8, 14, 7)),
            mlp_ratio = config.get("mlp_ratio", 4.0),
            num_heads = config.get("num_heads", (2, 4, 8, 16)),
            num_classes = config.get("num_classes", 1000),
            layer_scale = None,
            mean = jnp.array(config.get("mean", [0.485, 0.456, 0.406])),
            std = jnp.array(config.get("std", [0.229, 0.224, 0.225])),
        )

    return model


def load_param(param_path: str):
    """
    加载模型参数
    这里写得又臭又长，想要彻底解决需要改model
    """

    def map_pt_key_to_flax_path(pt_key):
        """
        精确将 PyTorch 权重名称映射为我们的 Flax 骨架路径
        """
        if "num_batches_tracked" in pt_key:
            return None

        # 移除最前面的 model.
        k = pt_key.replace("model.", "")

        # 1. PatchEmbed
        if k.startswith("patch_embed.conv_down.0"):
            k = k.replace("patch_embed.conv_down.0", "PatchEmbed_0/Conv_0")
        elif k.startswith("patch_embed.conv_down.1"):
            k = k.replace("patch_embed.conv_down.1", "PatchEmbed_0/BatchNorm_0")
        elif k.startswith("patch_embed.conv_down.3"):
            k = k.replace("patch_embed.conv_down.3", "PatchEmbed_0/Conv_1")
        elif k.startswith("patch_embed.conv_down.4"):
            k = k.replace("patch_embed.conv_down.4", "PatchEmbed_0/BatchNorm_1")

        # 2. 全局 Norm & Head
        elif k.startswith("norm."):
            k = k.replace("norm", "BatchNorm_0")
        elif k.startswith("head."):
            k = k.replace("head", "Dense_0")

        # 3. 核心 Levels
        elif k.startswith("levels."):
            parts = k.split('.')
            lvl = parts[1]

            if "downsample" in k:
                k = k.replace(f"levels.{lvl}.downsample.reduction.0", f"MambaVisionLayer_{lvl}/Downsample_0/Conv_0")
            else:
                blk = parts[3]
                base = f"MambaVisionLayer_{lvl}"

                if int(lvl) in [0, 1]:  # 前两个 stage 是纯 ConvBlock
                    k = k.replace(f"levels.{lvl}.blocks.{blk}.conv1", f"{base}/ConvBlock_{blk}/Conv_0")
                    k = k.replace(f"levels.{lvl}.blocks.{blk}.conv2", f"{base}/ConvBlock_{blk}/Conv_1")
                    k = k.replace(f"levels.{lvl}.blocks.{blk}.norm1", f"{base}/ConvBlock_{blk}/BatchNorm_0")
                    k = k.replace(f"levels.{lvl}.blocks.{blk}.norm2", f"{base}/ConvBlock_{blk}/BatchNorm_1")
                else:  # 后两个 stage 是 MambaBlock / AttentionBlock
                    k = k.replace(f"levels.{lvl}.blocks.{blk}.norm1", f"{base}/Block_{blk}/LayerNorm_0")
                    k = k.replace(f"levels.{lvl}.blocks.{blk}.norm2", f"{base}/Block_{blk}/LayerNorm_1")
                    k = k.replace(f"levels.{lvl}.blocks.{blk}.mlp.fc1", f"{base}/Block_{blk}/Mlp_0/Dense_0")
                    k = k.replace(f"levels.{lvl}.blocks.{blk}.mlp.fc2", f"{base}/Block_{blk}/Mlp_0/Dense_1")

                    mixer_prefix = f"levels.{lvl}.blocks.{blk}.mixer"
                    # 区分 Attention 还是 Mamba
                    if "qkv" in k or ("proj" in k and not any(x in k for x in ["in_proj", "out_proj", "dt_proj", "x_proj"])):
                        k = k.replace(f"{mixer_prefix}.qkv", f"{base}/Block_{blk}/Attention_0/Dense_0")
                        k = k.replace(f"{mixer_prefix}.proj", f"{base}/Block_{blk}/Attention_0/Dense_1")
                    else:
                        k = k.replace(f"{mixer_prefix}", f"{base}/Block_{blk}/MambaVisionMixer_0")

        # 关键修复：把所有剩余的 '.' 变成 '/' 以便生成嵌套字典
        k = k.replace(".", "/")

        # 4. 参数名转换
        k = k.replace("/weight", "/kernel")
        k = k.replace("/running_mean", "/mean")
        k = k.replace("/running_var", "/var")
        if "BatchNorm" in k or "LayerNorm" in k:
            k = k.replace("/kernel", "/scale")

        return k

    def convert_state_dict_pt_to_flax(pt_state_dict):
        flax_dict = {}

        for pt_key, pt_tensor in pt_state_dict.items():
            flax_path = map_pt_key_to_flax_path(pt_key)
            if flax_path is None:
                continue

            np_tensor = pt_tensor.numpy() if hasattr(pt_tensor, 'numpy') else pt_tensor

            # 维度转换策略
            if "conv1d" in flax_path:
                # 1D卷积: PyTorch (Out, In, L) -> Flax (L, In, Out)
                flax_tensor = jnp.transpose(np_tensor, (2, 1, 0))
            elif len(np_tensor.shape) == 4:
                # 2D卷积: PyTorch (Out, In, H, W) -> Flax (H, W, In, Out)
                flax_tensor = jnp.transpose(np_tensor, (2, 3, 1, 0))
            elif len(np_tensor.shape) == 2 and "A_log" not in flax_path:
                # 全连接: PyTorch (Out, In) -> Flax (In, Out)
                flax_tensor = jnp.transpose(np_tensor, (1, 0))
            else:
                # Bias/Scale/A_log/D: 保持不变
                flax_tensor = jnp.array(np_tensor)

            # 构建标准的 Flax 嵌套字典
            keys = flax_path.split('/')
            current_dict = flax_dict
            for k in keys[:-1]:
                if k not in current_dict:
                    current_dict[k] = {}
                current_dict = current_dict[k]
            current_dict[keys[-1]] = flax_tensor

        return flax_dict

    def load_pretrained_mambavision(param_path):
        pt_state_dict = {}
        with safe_open(param_path, framework="np", device="cpu") as f:
            for k in f.keys():
                pt_state_dict[k] = f.get_tensor(k)

        print("Converting PyTorch weights to Flax format...")
        flax_params = convert_state_dict_pt_to_flax(pt_state_dict)

        params_dict = {'params': {}, 'batch_stats': {}}

        def separate_stats(d, target_params, target_stats):
            for k, v in d.items():
                if isinstance(v, dict):
                    if 'mean' in v or 'var' in v:
                        target_stats[k] = {}
                        target_params[k] = {}
                        for sub_k, sub_v in v.items():
                            if sub_k in ['mean', 'var']:
                                target_stats[k][sub_k] = sub_v
                            else:
                                target_params[k][sub_k] = sub_v
                    else:
                        target_params[k] = {}
                        target_stats[k] = {}
                        separate_stats(v, target_params[k], target_stats[k])
                        if not target_stats[k]: del target_stats[k]
                        if not target_params[k]: del target_params[k]
                else:
                    target_params[k] = v

        separate_stats(flax_params, params_dict['params'], params_dict['batch_stats'])
        return params_dict

    return load_pretrained_mambavision(param_path)


def print_dict(d, indent=0):
    """递归打印嵌套字典结构，叶节点显示摘要信息"""
    prefix = "  " * indent
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                print_dict(v, indent + 1)
            else:
                if hasattr(v, 'shape') and hasattr(v, 'dtype'):
                    print(f"{prefix}{k}: array(shape={v.shape}, dtype={v.dtype})")
                else:
                    print(f"{prefix}{k}: {v}")
    else:
        print(f"{prefix}{d}")


def preprocess_image(
        img,
        input_size:tuple[int,int]|None = (224,224),
        mean: jax.Array|None = None,
        std: jax.Array|None = None
):
    # Resize: 原版 MambaVision-T 的 crop_pct=1.0，直接 resize 到 224x224
    if input_size is not None:
        img = img.resize(input_size, Image.Resampling.BICUBIC)

    # 转为 jnp array 并归一化到 [0, 1]
    img_arr = jnp.array(img, dtype=jnp.float32) / 255.0

    img_mean = img_arr.mean(axis=(0,1)) if mean is None else mean
    img_std = img_arr.std(axis=(0,1)) if std is None else std

    # 减均值除以方差
    img_normalized = (img_arr - img_mean) / img_std

    # 增加 Batch 维度 (B, H, W, C) -> Flax 默认支持 channels-last
    img_batched = jnp.expand_dims(img_normalized, axis=0)
    return jnp.array(img_batched)


if __name__ == "__main__":

    # param_path, model_path = load_from_hf("nvidia/MambaVision-T-1K")
    param_path = "/root/autodl-shared/hf_cache/hub/models--nvidia--MambaVision-T-1K/snapshots/b1de77e17599566d98efb701c0231b1095dc3a67/model.safetensors"
    model_path = "/root/autodl-shared/hf_cache/hub/models--nvidia--MambaVision-T-1K/snapshots/b1de77e17599566d98efb701c0231b1095dc3a67/config.json"

    labels = load_label(model_path, key="id2label")
    params = load_param(param_path)
    model = load_model(model_path)

    # 预处理图像
    image = Image.open("000000020247.jpg").convert('RGB')
    inputs = preprocess_image(image, (224, 224), model.mean, model.std)

    logits = model.apply(params, inputs, train=False)

    # 4. 获取预测结果
    predicted_class_idx = jnp.argmax(logits, axis=-1).item()
    predicted_label = labels[predicted_class_idx]

    print("="*40)
    print(f"Predicted class index: {predicted_class_idx}")
    print(f"Predicted class label: {predicted_label}")
    print("="*40)



