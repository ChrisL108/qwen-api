# Notes:

- https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLConfig
- Can test the large model here: https://huggingface.co/spaces/Qwen/Qwen2.5-VL-72B-Instruct
- For GPU deployment, update `DEVICE_MAP` and `TORCH_DTYPE` accordingly (see `app/config.py`).

## Possible TODOs:
- [ ] We can send multiple images (frames) to conversation prompt. See if batching is faster. https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/qwen2_5_vl#multiple-image-inputs

- [ ] Add Flash-Attention 2 to the model. https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/qwen2_5_vl#flash-attention-2-to-speed-up-generation
  - [ ] Requirements:
	- can only be used when a model is loaded in torch.float16 or torch.bfloat16
	- CUDA toolkit or ROCm toolkit
	- PyTorch 2.2 and above.
	- `packaging` Python package (`pip install packaging`)
	- `ninja` Python package (`pip install ninja`) *
	- Linux.