from vllm_omni.diffusion.registry import DiffusionModelRegistry, _DIFFUSION_MODELS

print("HiDreamO1ImagePipeline" in _DIFFUSION_MODELS)
print([k for k in _DIFFUSION_MODELS if "HiDream" in k])

cls = DiffusionModelRegistry._try_load_model_cls("HiDreamO1ImagePipeline")
print("Resolved class:", cls)
