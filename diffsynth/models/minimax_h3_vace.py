import torch
import torch.nn as nn
from .minimax_h3_dit import MiniMaxH3DiT, MiniMaxH3DiTBlock, MiniMaxH3TimeEmbedder
from ..core.gradient import gradient_checkpoint_forward


class MiniMaxH3VaceBlock(MiniMaxH3DiTBlock):
    def __init__(self, hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size, time_embed_dim, adaln_out_features, norm_eps, qk_norm_eps):
        super().__init__(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size, time_embed_dim, adaln_out_features, norm_eps, qk_norm_eps)
        self.after_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, *, t_emb, combined_indices, rope_freqs, cu_seqlens, max_seqlen):
        c = super().forward(c, t_emb=t_emb, combined_indices=combined_indices, rope_freqs=rope_freqs, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.after_proj(c), c


class MiniMaxH3VaceModel(nn.Module):
    """The VACE bypass, which encodes the control video into per-layer hints.

    The bypass only reads the control latents, never the noisy video, and is always
    modulated at `cond_noise_aug` (the clean-conditioning timestep). Its hints are
    therefore constant over the sampling loop and only need to be computed once.
    Every module mirrors its counterpart in
    the backbone so that `init_from_dit` can warm-start all of them: `video_patch_proj`
    embeds the control latents, `time_embedder` modulates them and `vace_blocks`
    produce the hints."""

    def __init__(
        self,
        vace_layers=(0, 1, 2, 3, 4, 5, 6, 7),
        vace_in_dim=96,
        hidden_size=5376,
        num_attention_heads=56,
        attention_head_dim=128,
        ffn_hidden_size=14336,
        timestep_input_dim=256,
        time_embed_hidden_size=5376,
        time_embed_dim=2688,
        adaln_out_features=96768,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
        cond_noise_aug=0.999,
    ):
        super().__init__()
        self.vace_layers = sorted(vace_layers)
        self.vace_in_dim = vace_in_dim
        # Timestep the control latents are modulated at; see `forward`. Not a
        # parameter, so it does not change the checkpoint layout.
        self.cond_noise_aug = cond_noise_aug
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # These two mirror the backbone's modules of the same name, so that
        # `init_from_dit` can copy their weights.
        self.video_patch_proj = nn.Linear(vace_in_dim, hidden_size, bias=True)
        self.time_embedder = MiniMaxH3TimeEmbedder(timestep_input_dim, time_embed_hidden_size, time_embed_dim)

        # vace blocks
        self.vace_blocks = nn.ModuleList([
            MiniMaxH3VaceBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size, time_embed_dim, adaln_out_features, norm_eps, qk_norm_eps)
            for _ in self.vace_layers
        ])

    def init_from_dit(self, dit: MiniMaxH3DiT):
        """Warm-start every module from the backbone. `after_proj` stays
        zero-initialized, so the control signal contributes nothing at first."""
        self.video_patch_proj.load_state_dict(dit.video_patch_proj.state_dict())
        self.time_embedder.load_state_dict(dit.time_embedder.state_dict())
        for layer_id, block in zip(self.vace_layers, self.vace_blocks):
            block.load_state_dict(dit.blocks[layer_id].state_dict(), strict=False)

    def forward(
        self,
        vace_context, rope_freqs, img_pos,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        # The control tokens share the target video's positions in the packed
        # sequence, so RoPE frequencies are gathered from there.
        c = self.video_patch_proj(vace_context)
        rope_freqs = rope_freqs[img_pos[:c.shape[0]]]
        cu_seqlens = torch.tensor([0, c.shape[0]], dtype=torch.int32, device=c.device)
        max_seqlen = c.shape[0]

        # The control video is a clean signal, not a noisy latent, so the bypass is
        # modulated at the level the backbone uses for clean conditioning rows.
        # MiniMax-H3's flow-matching convention is `x_t = t*x0 + (1-t)*noise`, so
        # t=1 is clean and t=0 is pure noise: passing 0 here told every control
        # block the control latents were pure noise, i.e. the exact opposite of
        # what they are, and selected the AdaLN modulation the backbone learned for
        # "ignore this signal". 0.999 matches `pipe.imgvid_cond_noise_aug`, the
        # level the backbone was trained to read clean conditioning latents at.
        # `t_emb` holds a single row, and the AdaLN indices select its video
        # modality (tag 0) for every control token.
        t_emb = self.time_embedder(
            torch.full((1,), self.cond_noise_aug, dtype=torch.float32, device=c.device), dtype=c.dtype
        )
        combined_indices = torch.zeros(c.shape[0], dtype=torch.long, device=c.device)

        hints = []
        for block in self.vace_blocks:
            hint, c = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                c,
                t_emb=t_emb,
                combined_indices=combined_indices,
                rope_freqs=rope_freqs,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            hints.append(hint)
        return hints
