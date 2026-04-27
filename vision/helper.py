import jax
import jax.numpy as jnp
from safetensors import safe_open
from vision.model import MambaVision
from PIL import Image
import json

def load_from_hf(model_repo_id: str = "nvidia/MambaVision-T-1K"):
    import os
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
    # os.environ['HF_HOME'] = "/root/autodl-shared/hf_cache"  # autodl
    os.environ['HF_HOME'] = "/root/shared-nvme/hf_cache"  # paratera

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


def map_pt_key_to_flax_path(pt_key: str):
    """映射 PyTorch 权重名称为 Flax 骨架路径 (保留了必要的替换逻辑)"""
    if "num_batches_tracked" in pt_key:
        return None
    k = pt_key.replace("model.", "")

    if k.startswith("patch_embed"):
        k = k.replace("patch_embed.conv_down.0", "PatchEmbed_0/Conv_0") \
            .replace("patch_embed.conv_down.1", "PatchEmbed_0/BatchNorm_0") \
            .replace("patch_embed.conv_down.3", "PatchEmbed_0/Conv_1") \
            .replace("patch_embed.conv_down.4", "PatchEmbed_0/BatchNorm_1")
    elif k.startswith("norm."): k = k.replace("norm", "BatchNorm_0")
    elif k.startswith("head."): k = k.replace("head", "Dense_0")
    elif k.startswith("levels."):
        parts = k.split('.')
        lvl = parts[1]
        if "downsample" in k:
            k = k.replace(f"levels.{lvl}.downsample.reduction.0", f"MambaVisionLayer_{lvl}/Downsample_0/Conv_0")
        else:
            blk, base = parts[3], f"MambaVisionLayer_{lvl}"
            if int(lvl) in [0, 1]:
                k = k.replace(f"levels.{lvl}.blocks.{blk}.conv1", f"{base}/ConvBlock_{blk}/Conv_0") \
                    .replace(f"levels.{lvl}.blocks.{blk}.conv2", f"{base}/ConvBlock_{blk}/Conv_1") \
                    .replace(f"levels.{lvl}.blocks.{blk}.norm1", f"{base}/ConvBlock_{blk}/BatchNorm_0") \
                    .replace(f"levels.{lvl}.blocks.{blk}.norm2", f"{base}/ConvBlock_{blk}/BatchNorm_1")
            else:
                k = k.replace(f"levels.{lvl}.blocks.{blk}.norm1", f"{base}/Block_{blk}/LayerNorm_0") \
                    .replace(f"levels.{lvl}.blocks.{blk}.norm2", f"{base}/Block_{blk}/LayerNorm_1") \
                    .replace(f"levels.{lvl}.blocks.{blk}.mlp.fc1", f"{base}/Block_{blk}/Mlp_0/Dense_0") \
                    .replace(f"levels.{lvl}.blocks.{blk}.mlp.fc2", f"{base}/Block_{blk}/Mlp_0/Dense_1")

                mixer_prefix = f"levels.{lvl}.blocks.{blk}.mixer"
                if "qkv" in k or ("proj" in k and not any(x in k for x in ["in_proj", "out_proj", "dt_proj", "x_proj"])):
                    k = k.replace(f"{mixer_prefix}.qkv", f"{base}/Block_{blk}/Attention_0/Dense_0") \
                        .replace(f"{mixer_prefix}.proj", f"{base}/Block_{blk}/Attention_0/Dense_1")
                else:
                    k = k.replace(f"{mixer_prefix}", f"{base}/Block_{blk}/MambaVisionMixer_0")

    # 替换参数名并返回分割数组
    k = k.replace(".", "/").replace("/weight", "/kernel") \
        .replace("/running_mean", "/mean").replace("/running_var", "/var")
    if "BatchNorm" in k or "LayerNorm" in k:
        k = k.replace("/kernel", "/scale")
    return k.split('/')

def insert_in_dict(target_dict: dict, keys: list[str], value):
    """辅助函数：根据 keys 列表在字典中创建嵌套路径并赋值"""
    d = target_dict
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value

def load_param(param_path: str):

    params = {}
    batch_stats = {}

    with safe_open(param_path, framework="np", device="cpu") as f:
        for pt_key in f.keys():
            keys = map_pt_key_to_flax_path(pt_key)
            if not keys: continue

            np_tensor = f.get_tensor(pt_key)

            # 维度转换策略
            if "conv1d" in pt_key:
                flax_tensor = jnp.transpose(np_tensor, (2, 1, 0)) # 1D Conv
            elif len(np_tensor.shape) == 4:
                flax_tensor = jnp.transpose(np_tensor, (2, 3, 1, 0)) # 2D Conv
            elif len(np_tensor.shape) == 2 and "A_log" not in keys[-1]:
                flax_tensor = jnp.transpose(np_tensor, (1, 0)) # Dense
            else:
                flax_tensor = jnp.array(np_tensor) # 1D params

            # 动态路由：如果是 mean 或 var 则放入 batch_stats，否则放入 params
            target_dict = batch_stats if keys[-1] in ['mean', 'var'] else params
            insert_in_dict(target_dict, keys, flax_tensor)

    return {'params': params, 'batch_stats': batch_stats}


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

    print_dict(params)

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



