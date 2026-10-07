"""PEFT -> MLX adapter conversion, on a tiny Qwen3 with fake tensors (no real weights)."""

import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import Model, ModelArgs
from mlx_lm.tuner.utils import load_adapters

spec = importlib.util.spec_from_file_location(
    "convert_adapter", Path(__file__).parents[1] / "scripts" / "convert_adapter.py"
)
convert_adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert_adapter)

CFG = {
    "model_type": "qwen3",
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "rms_norm_eps": 1e-6,
    "vocab_size": 50,
    "max_position_embeddings": 128,
    "tie_word_embeddings": False,
    "rope_theta": 10000.0,
}
R = 4
OUT_DIMS = {"q_proj": 32, "k_proj": 16, "v_proj": 16, "o_proj": 32}
IN_DIMS = {"q_proj": 32, "k_proj": 32, "v_proj": 32, "o_proj": 32}


@pytest.fixture
def peft_dir(tmp_path):
    tensors = {}
    for layer in range(2):
        for mod, out in OUT_DIMS.items():
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{mod}"
            tensors[f"{prefix}.lora_A.weight"] = mx.random.normal((R, IN_DIMS[mod]))
            tensors[f"{prefix}.lora_B.weight"] = mx.random.normal((out, R))
    d = tmp_path / "peft"
    d.mkdir()
    mx.save_safetensors(str(d / "adapter_model.safetensors"), tensors)
    (d / "adapter_config.json").write_text(
        json.dumps({"r": R, "lora_alpha": 8, "target_modules": list(OUT_DIMS)})
    )
    return d, tensors


def test_convert_writes_expected_files_and_config(peft_dir, tmp_path):
    d, _ = peft_dir
    config = convert_adapter.convert(d, tmp_path / "mlx", CFG)
    assert config["num_layers"] == 2
    assert config["lora_parameters"]["scale"] == 2.0  # alpha / r
    assert config["lora_parameters"]["rank"] == R
    assert (tmp_path / "mlx" / "adapters.safetensors").exists()
    assert len(mx.load(str(tmp_path / "mlx" / "adapters.safetensors"))) == 2 * 4 * 2


def test_transpose_matches_peft_math(peft_dir, tmp_path):
    d, peft = peft_dir
    convert_adapter.convert(d, tmp_path / "mlx", CFG)
    model = Model(ModelArgs(**CFG))
    mx.eval(model.parameters())
    base_q = model.model.layers[0].self_attn.q_proj
    x = mx.random.normal((3, 32))
    base_out = base_q(x)

    load_adapters(model, str(tmp_path / "mlx"))
    lora_q = model.model.layers[0].self_attn.q_proj
    assert lora_q.scale == 2.0

    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    a, b = peft[f"{prefix}.lora_A.weight"], peft[f"{prefix}.lora_B.weight"]
    expected = base_out + 2.0 * (x @ a.T) @ b.T  # what PEFT computes
    assert mx.allclose(lora_q(x), expected, atol=1e-4)

    lora_q.scale = 0.0
    assert mx.allclose(lora_q(x), base_out)


def test_converted_keys_match_mlx_lora_model(peft_dir, tmp_path):
    d, _ = peft_dir
    convert_adapter.convert(d, tmp_path / "mlx", CFG)
    converted = set(mx.load(str(tmp_path / "mlx" / "adapters.safetensors")))
    model = Model(ModelArgs(**CFG))
    from mlx_lm.tuner.utils import linear_to_lora_layers

    linear_to_lora_layers(
        model,
        2,
        {"rank": R, "scale": 2.0, "dropout": 0.0, "keys": [f"self_attn.{m}" for m in OUT_DIMS]},
    )
    model_keys = {
        k
        for k, _ in tree_flatten(model.parameters())
        if k.rsplit(".", 1)[-1] in ("lora_a", "lora_b")
    }
    assert converted == model_keys


def test_validate_rejects_wrong_shape(peft_dir, tmp_path):
    _, peft = peft_dir
    tensors = convert_adapter.convert_tensors(peft)
    tensors["model.layers.0.self_attn.k_proj.lora_b"] = mx.zeros((R, 99))
    with pytest.raises(ValueError, match="k_proj"):
        convert_adapter.validate(tensors, R, list(OUT_DIMS), 2, CFG)


def test_validate_rejects_missing_tensor(peft_dir):
    _, peft = peft_dir
    tensors = convert_adapter.convert_tensors(peft)
    del tensors["model.layers.1.self_attn.o_proj.lora_a"]
    with pytest.raises(ValueError, match="expected 16 tensors"):
        convert_adapter.validate(tensors, R, list(OUT_DIMS), 2, CFG)


def test_unexpected_key_and_unsupported_config():
    with pytest.raises(ValueError, match="unexpected tensor"):
        convert_adapter.convert_tensors({"foo.bar": mx.zeros((1, 1))})
    with pytest.raises(ValueError, match="use_dora"):
        convert_adapter.mlx_adapter_config(
            {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj"], "use_dora": True}, 1
        )


def test_layers_to_transform_rejected():
    with pytest.raises(ValueError, match="layers_to_transform"):
        convert_adapter.mlx_adapter_config(
            {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj"], "layers_to_transform": [3]}, 4
        )
