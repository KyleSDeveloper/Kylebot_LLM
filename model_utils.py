"""
model_utils.py
Utility loader that supports any HF causal-LM and optional 4-bit quant (bitsandbytes).
"""

import logging, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

log = logging.getLogger(__name__)

def load_model(model_name="gpt2", quant_4bit=False, device_map="auto"):
    """Returns (tokenizer, model) ready for inference."""
    log.info(f"Loading {model_name}  |  4-bit={quant_4bit}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    use_cuda = torch.cuda.is_available()

    if quant_4bit and not use_cuda:
        log.warning("--quant ignored (no CUDA)"); quant_4bit = False

    if quant_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map,
            quantization_config=bnb_cfg, trust_remote_code=True
        )
    else:
        kwargs = dict(device_map=device_map, trust_remote_code=True)
        if use_cuda: kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    model.eval()
    log.info(f"{sum(p.numel() for p in model.parameters())/1e6:.1f} M params loaded")
    return tok, model
