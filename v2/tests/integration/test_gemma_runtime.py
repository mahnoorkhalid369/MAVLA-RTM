"""CPU checks of the installed runtime, without loading the production model weights."""
from types import SimpleNamespace
import pytest


def test_system_prefix_reuse_matches_gemma_with_sliding_window_on_cpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Gemma4Config, Gemma4TextConfig, Gemma4ForConditionalGeneration
    from agent.prefix_cache import SystemPrefixCache
    config = Gemma4TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=8,
                             vocab_size_per_layer_input=64, hidden_size_per_layer_input=8,
                             sliding_window=16, max_position_embeddings=128,
                             layer_types=["sliding_attention", "full_attention"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = Gemma4ForConditionalGeneration(Gemma4Config(text_config=config)).eval()
    class Inputs(dict):
        def to(self, device):
            return self
    ids = torch.tensor([[2]+[5,6,7,8]*10+[11,12,13]])
    prefix = ids[:,:38]
    processor = SimpleNamespace(apply_chat_template=lambda *a, **k: Inputs(input_ids=prefix, attention_mask=torch.ones_like(prefix)))
    full = {"input_ids":ids, "attention_mask":torch.ones_like(ids)}
    with torch.inference_mode():
        expected = model.generate(**full, max_new_tokens=4, do_sample=False)
        cache = SystemPrefixCache()
        kwargs, _ = cache.prepare(processor, model, full, "system", True)
        actual = model.generate(**full, **kwargs, max_new_tokens=4, do_sample=False)
        assert torch.equal(actual, expected)
        kwargs, info = cache.prepare(processor, model, full, "system", True)
        again = model.generate(**full, **kwargs, max_new_tokens=4, do_sample=False)
        assert info["hit"] and torch.equal(again, expected)
