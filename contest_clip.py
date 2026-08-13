"""NLPrompt model components for the contest dataset.

This is a self-contained, Dassl-free re-implementation of the building blocks used
by ``trainers/nlprompt.py`` (PromptLearner, TextEncoder, CustomCLIP, the
GeneralizedCrossEntropy loss and the CLIP loading helper) so that the new
train/test scripts for the contest dataset do not depend on the Dassl framework.

The components are copied verbatim from the project's NLPrompt implementation and
only the configuration is passed as plain arguments instead of a ``yacs`` CfgNode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip as clipmod


def load_clip_to_cpu(backbone_path, device="cpu"):
    """Load a CLIP backbone (state-dict or TorchScript archive) and wrap it with
    the NLPrompt design details (no per-layer prompts)."""
    try:
        model = torch.jit.load(backbone_path, map_location=device).eval()
        state_dict = model.state_dict()
    except RuntimeError:
        state_dict = torch.load(backbone_path, map_location=device, weights_only=False)
    design_details = {
        "trainer": "NLPrompt",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
    }
    model = clipmod.build_model(state_dict, design_details)
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=16, ctx_init="", csc=False,
                 class_token_position="end"):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        # token_embedding lives on whatever device clip_model was placed on; the
        # tokenized index tensors from clipmod.tokenize() are CPU by default, so we
        # must move them to the same device or F.embedding raises a device mismatch.
        device = next(clip_model.token_embedding.parameters()).device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution  # we use CLIP's native size
        cfg_imsize = clip_imsize
        assert cfg_imsize == clip_imsize, \
            f"cfg_imsize ({cfg_imsize}) must equal clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clipmod.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :].float()  # fp32 params
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        if csc:
            ctx_vectors = ctx_vectors.repeat(n_cls, 1, 1)
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(clipmod.tokenize(e)) for e in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clipmod.tokenize(p).to(device) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.csc = csc
        self.class_token_position = class_token_position
        self.name_lens = name_lens
        self.dtype = dtype

    def forward(self):
        ctx = self.ctx.to(self.dtype)
        if not self.csc:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        else:
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1, ...]
                class_i = suffix[i:i + 1, :name_len, ...]
                suffix_i = suffix[i:i + 1, name_len:, ...]
                ctx_i_half1 = ctx[i:i + 1, :half_n_ctx, ...]
                ctx_i_half2 = ctx[i:i + 1, half_n_ctx:, ...]
                prompt = torch.cat(
                    [prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        return prompts


class GeneralizedCrossEntropy(nn.Module):
    """Generalized Cross Entropy (Zhang & Sabuncu, 2018) - robust to label noise."""

    def __init__(self, q=0.7, ignore_index=-100, reduction="none"):
        super().__init__()
        self.q = q
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        probs_y = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = (1.0 - torch.pow(probs_y, self.q)) / self.q
        if self.ignore_index >= 0:
            ignore_mask = (targets == self.ignore_index)
            loss = loss * (~ignore_mask).float()
            if self.reduction == "mean":
                return loss.sum() / (1 - ignore_mask).float().sum().clamp(min=1)
            elif self.reduction == "sum":
                return loss.sum()
            return loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=16, ctx_init="", csc=False,
                 class_token_position="end"):
        super().__init__()
        self.prompt_learner = PromptLearner(
            classnames, clip_model, n_ctx, ctx_init, csc, class_token_position
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, images):
        image_features = self.image_encoder(images.type(self.dtype))
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits
