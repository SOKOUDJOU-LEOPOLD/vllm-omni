# Reproduction Process — Issue #3733 ([New Model]: HiDream-O1-Image)

## Environment Setup

- Repo: `vllm-project/vllm-omni`, branch `repro/hidream-o1-image-3733` (off `main`,
  commit `7b837944`).
- Python 3.13.1 (`pyenv-win`), no GPU required for this reproduction — it only exercises
  the registry/config layer, not actual model execution.
- **Challenge 1**: `import vllm_omni` fails immediately with
  `ModuleNotFoundError: No module named 'aenum'` (`vllm_omni/patch.py`).
  **Fix**: `pip install aenum` (small, pure-Python, no GPU/CUDA implications).
- **Challenge 2**: after fixing `aenum`, `import vllm_omni` fails again with
  `ModuleNotFoundError: No module named 'vllm'` — `vllm_omni/__init__.py` eagerly imports
  `vllm_omni.transformers_utils.configs`, which pulls in
  `vllm_omni.model_executor.models.registry`, which does `from vllm.model_executor.models.registry
  import ...`. So **`vllm_omni` cannot be imported at all without a full `vllm` install**
  (a multi-GB, CUDA/torch-version-pinned dependency).
  **Not fixed** — installing full `vllm` is out of scope for this lightweight reproduction
  (no GPU available in this environment, and the download/build is heavyweight). This is itself
  a useful data point: a "does the registry even know about this model" check can't be done as a
  standalone unit-level script — it requires the full `pip install -e .` dev environment from the
  project's CONTRIBUTING/dev-setup docs.
- **Workaround used for this reproduction**: instead of importing `vllm_omni` end-to-end, I
  (a) statically grepped `vllm_omni/diffusion/registry.py` for `HiDream` entries, and
  (b) used `huggingface_hub` (already installed, no `vllm` dependency) to inspect the actual
  `HiDream-ai/HiDream-O1-Image` repo layout/config directly from the Hub. Together these two
  checks are sufficient to confirm the gap described in the issue without needing a GPU.

## Steps to Reproduce

1. In `vllm_omni/diffusion/registry.py`, search the `_DIFFUSION_MODELS` registry dict for any
   HiDream-O1-Image entry:
   ```bash
   grep -n "HiDream" -i vllm_omni/diffusion/registry.py
   ```
2. Inspect the actual `HiDream-ai/HiDream-O1-Image` Hub repo layout and `config.json`:
   ```bash
   python reproduction/issue_3733_hidream_o1_image/check_hf_repo_layout.py
   python reproduction/issue_3733_hidream_o1_image/check_hf_config.py
   ```
3. **Observed result**:
   - Step 1 shows only **one** HiDream entry in the registry: `"HiDreamImagePipeline"`
     (registered for the *different*, older `HiDream-I1-Image` model, mapped to
     `vllm_omni/diffusion/models/hidream_image/pipeline_hidream_image.py`). There is **no**
     `"HiDreamO1ImagePipeline"` (or any other O1-specific) entry anywhere in `_DIFFUSION_MODELS`.
   - Step 2 shows the `HiDream-ai/HiDream-O1-Image` repo:
     - Has **no `model_index.json`** — the file vLLM-Omni's diffusion loader normally reads to
       get `_class_name` and look it up in `_DIFFUSION_MODELS`.
     - Ships a flat `model.safetensors.index.json` + 8 sharded `model-0000X-of-00008.safetensors`
       files at the repo **root** (not under a `transformer/` subfolder like diffusers pipelines).
     - `config.json` has `architectures: ["Qwen3VLForConditionalGeneration"]` and
       `model_type: "qwen3_vl"`, with **no `_class_name`** field at all.

   Net effect: `Omni(model="HiDream-ai/HiDream-O1-Image")` has no `model_index.json` to read, and
   even if vLLM-Omni's diffusion config-resolution fell back to `config.json`'s `architectures`,
   `"Qwen3VLForConditionalGeneration"` is not a key in `_DIFFUSION_MODELS` either — so model
   resolution would fail with the registry's
   `ValueError(f"Model class {model_class_name} not found in diffusion model registry.")`
   (`vllm_omni/diffusion/registry.py:382`), confirming the issue: **the model architecture is
   simply not implemented/registered yet**, this is a "New Model" gap, not a bug in existing code.

## Reproduction Evidence

- **Commit showing reproduction**: `<to be filled in after pushing repro/hidream-o1-image-3733
  to fork>` — contains this `REPRODUCTION.md` plus the two standalone check scripts
  (`check_registry.py`, `check_hf_repo_layout.py`, `check_hf_config.py`) under
  `reproduction/issue_3733_hidream_o1_image/`.
- **Logs**: raw output of all three scripts captured below.

  `check_registry.py` (run after `pip install aenum`, still fails — documents Challenge 2):
  ```
  ModuleNotFoundError: No module named 'vllm'
  ```

  `check_hf_repo_layout.py`:
  ```
  Files in HiDream-ai/HiDream-O1-Image:
   - .gitattributes
   - .mdl
   - .msc
   - .mv
   - README.md
   - assets/...
   - chat_template.json
   - config.json
   - configuration.json
   - generation_config.json
   - merges.txt
   - model-00001-of-00008.safetensors
   - ... (8 shards total)
   - model.safetensors.index.json
   - preprocessor_config.json
   - tokenizer.json
   - tokenizer_config.json
   - video_preprocessor_config.json
   - vocab.json

  Has model_index.json: False
  Has model.safetensors.index.json: True
  ```

  `check_hf_config.py`:
  ```
  Top-level config.json keys: ['architectures', 'image_token_id', 'model_type', 'text_config',
  'tie_word_embeddings', 'transformers_version', 'video_token_id', 'vision_config',
  'vision_end_token_id', 'vision_start_token_id']
  architectures: ['Qwen3VLForConditionalGeneration']
  model_type: qwen3_vl
  _class_name: None
  ```

- **My findings**:
  1. `vllm_omni`'s registry already supports a *different* HiDream model
     (`HiDreamImagePipeline` for `HiDream-I1-Image`), but has zero awareness of
     `HiDream-O1-Image` — confirming the issue is asking for a brand-new pipeline, not a fix to
     an existing one.
  2. `HiDream-ai/HiDream-O1-Image` on the Hub is laid out exactly like the issue/architecture
     analysis predicted: no `model_index.json`, flat root-level sharded safetensors +
     `model.safetensors.index.json`, and a `config.json` that identifies it as a
     `Qwen3VLForConditionalGeneration` (`model_type: qwen3_vl`) — i.e. a VLM checkpoint being
     repurposed as an image generator, with none of the diffusion-specific metadata vLLM-Omni's
     loader expects.
  3. This validates the "Match" findings from the implementation plan: weight loading should
     follow the **BAGEL "Pattern 2: custom safetensors at root"** approach
     (`weights_sources` with `subfolder=None` + custom `load_weights()` name remapping against
     `model.safetensors.index.json`), and the transformer should be built on top of the
     `Qwen3VL` backbone (closest in-repo precedent: **SenseNova-U1**, an LLM-as-denoiser model).
  4. Separately, simply importing `vllm_omni` requires the full `vllm` package — so any future
     contributor doing a from-scratch dev setup on a non-GPU machine should expect to need the
     full `pip install -e .` (with `vllm` + matching torch/CUDA) before they can run even basic
     unit-level checks against the registry.

## Branch Link

`repro/hidream-o1-image-3733` (this branch) — to be pushed to fork:
`<https://github.com/<your-username>/vllm-omni/tree/repro/hidream-o1-image-3733>`
