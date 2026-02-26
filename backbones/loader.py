"""
Load vision-language backbone: CLIP or Qwen2.5-VL-7B-Instruct.
Config: backbone = "ViT-B/16" (CLIP) or "Qwen2.5-VL-7B-Instruct" (Qwen).
Returns (model, preprocess) with unified interface: model.encode_image(x), model has .dtype.
Classifier weights via get_classifier_weights(classnames, template, model) in utils.
"""
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

QWEN_VL_NAMES = ("Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct")


def _is_qwen_backbone(backbone):
    return any(b in backbone for b in ("Qwen2.5-VL", "qwen2.5-vl", "Qwen2.5_VL"))


def load_backbone(backbone_name):
    if _is_qwen_backbone(backbone_name):
        return _load_qwen_vl(backbone_name)
    import clip
    model, preprocess = clip.load(backbone_name)
    model.eval()
    return model, preprocess


def _load_qwen_vl(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    except ImportError:
        raise ImportError("Install transformers and accelerate: pip install transformers accelerate")
    if "Qwen2.5-VL" in model_name and not model_name.startswith("Qwen/"):
        model_name = "Qwen/" + model_name
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=224 * 224, max_pixels=224 * 224)
    wrapper = QwenVLWrapper(model, processor)
    # Preprocess: resize to 224 and to tensor; dataloader gives tensor, wrapper.encode_image converts to PIL for processor
    preprocess = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])
    return wrapper, preprocess


class QwenVLWrapper(torch.nn.Module):
    """Wrapper so Qwen2.5-VL can be used like CLIP: encode_image(batch_tensor) -> (B, D) normalized."""

    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        self._hidden_size = getattr(model.config, "hidden_size", 3584)

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def _tensor_batch_to_pil(self, images):
        # images: (B, 3, H, W) in [0,1]
        out = []
        for i in range(images.shape[0]):
            x = images[i].cpu().float()
            if x.max() <= 1.0:
                x = (x * 255).clamp(0, 255)
            x = x.permute(1, 2, 0).numpy().astype(np.uint8)
            out.append(Image.fromarray(x))
        return out

    def encode_image(self, images):
        # images: (B, 3, H, W) tensor from dataloader (after our preprocess: 224x224, [0,1])
        pil_list = self._tensor_batch_to_pil(images)
        device = next(self.model.parameters()).device
        all_hidden = []
        for pil_img in pil_list:
            msg = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": "."}]}]
            inp = self.processor.apply_chat_template(
                msg,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            with torch.no_grad():
                out = self.model(**inp, output_hidden_states=True)
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                h = out.last_hidden_state.float()
            else:
                h = out.hidden_states[-1].float()
            # (1, seq_len, hidden_size); pool over sequence
            feat = h.mean(dim=1)
            all_hidden.append(feat)
        features = torch.cat(all_hidden, dim=0)
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        return features.to(device=device, dtype=images.dtype if images.is_cuda else features.dtype)

    def encode_text(self, texts):
        """Encode text prompts (list of strings) to normalized (D, len(texts)) for classifier."""
        device = next(self.model.parameters()).device
        all_hidden = []
        for text in texts:
            msg = [{"role": "user", "content": [{"type": "text", "text": text}]}]
            inp = self.processor.apply_chat_template(
                msg,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
                inp.pop(key, None)
            with torch.no_grad():
                out = self.model.model(**inp, output_hidden_states=True)
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                h = out.last_hidden_state.float()
            else:
                h = out.hidden_states[-1].float()
            feat = h[:, -1, :]
            all_hidden.append(feat)
        features = torch.cat(all_hidden, dim=0)
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        return features.T
