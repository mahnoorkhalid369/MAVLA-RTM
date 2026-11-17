"""Opt-in system-only KV reuse. Never cache user text, images, or generated continuations.

Must pass the recorded-input equivalence benchmark before use in physical runs.
The immutable stored cache is cloned because generate() mutates its input cache.
"""

from copy import deepcopy


class SystemPrefixCache:
    def __init__(self):
        self.key, self.cache, self.length = None, None, 0

    def prepare(self, processor, model, inputs, system, thinking):
        if not system:
            return {}, {"hit": False, "tokens": 0}
        prefix = processor.apply_chat_template([{"role": "system", "content": system}], tokenize=True,
                                               return_dict=True, processor_kwargs={"return_tensors": "pt"}, add_generation_prompt=False,
                                               enable_thinking=thinking).to(model.device)
        ids, full = prefix["input_ids"], inputs["input_ids"]
        length = ids.shape[-1]
        # Template/BPE boundary changes must fall back to an ordinary full prefill.
        if length >= full.shape[-1] or ids.tolist()[0] != full[:, :length].tolist()[0]:
            return {}, {"hit": False, "tokens": 0, "reason": "system tokens are not an exact prefix"}
        key = tuple(ids.tolist()[0])
        hit = key == self.key and self.cache is not None
        if not hit:
            output = model(**prefix, use_cache=True, return_dict=True, logits_to_keep=1)
            self.cache, self.key, self.length = output.past_key_values, key, length
        return {"past_key_values": deepcopy(self.cache)}, {"hit": hit, "tokens": self.length}
