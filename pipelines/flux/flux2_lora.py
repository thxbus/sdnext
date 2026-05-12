"""Flux2/Klein native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``f2`` in ``allow_native``). Reads the safetensors directly and writes into
sdnext's existing ``network_layer_mapping``, returning a ``Network`` populated
with ``NetworkModule*`` entries that ``network_activate`` will apply.

Recognized key prefixes for every family: ``diffusion_model.``,
``transformer.``, ``lora_unet_``, ``lycoris_``, ``base_model.model.``
(PEFT save wrapper), bare BFL paths (e.g. ``double_blocks.``), and bare
diffusers paths (``transformer_blocks.`` / ``single_transformer_blocks.``,
produced by ``Flux2Transformer2DModel.save_lora_adapter()``). Diffusers-PEFT
``lora_A``/``lora_B`` are normalized to ``lora_down``/``lora_up`` and a
``.<adapter_name>.`` infix between the suffix and ``.weight`` (e.g.
``.lora_A.default.weight``) is stripped to match the standard suffix table.

BFL/kohya keys are mapped to diffusers paths via ``F2_SINGLE_MAP`` /
``F2_DOUBLE_MAP`` / ``F2_QKV_MAP``. Fused QKV in double_blocks is split into
three Q/K/V targets at lookup time. PEFT keys are diffusers paths already and
are returned verbatim with no chunking.

Per-family fused-QKV handling:

- LoRA: load-time chunk of ``lora_up`` along dim 0 (the down-side is shared).
- LoKR: apply-time slice via :class:`NetworkModuleLokrChunk`, which builds
  ``kron(w1, w2)`` once and returns the designated row range.
- LoHA: apply-time slice via :class:`NetworkModuleHadaChunk`, which slices
  ``w1a``/``w2a`` and computes the partial Hadamard product. Tucker
  (CP-decomposed) LoHAs are not chunked and are skipped on fused targets.
- OFT, IA3, GLoRA, Full: no chunk class exists and the math is not row-sliceable
  without re-deriving per-projection structure. Fused groups are skipped with a
  warning.
- Norm: targets 1-D LayerNorm/RMSNorm parameters; never fused.

LyCORIS algorithm coverage relative to upstream
``KohakuBlueleaf/LyCORIS/lycoris/modules/``:

- Native: LoRA, LoKR, LoHA, OFT, BOFT, IA3, GLoRA, Norm, Full.
- Saved as standard LoRA: LoCon and DyLoRA. Both ``custom_state_dict``
  outputs collapse to ``lora_up.weight``/``lora_down.weight``/``alpha``
  (LoCon bakes its ``scalar`` into ``lora_up``; DyLoRA concats its
  per-block slabs into a max-rank matrix), so ``try_load_lora`` loads
  them losslessly relative to upstream's own export.
- Deferred: TLoRA. The file saves only ``q_layer.weight`` /
  ``p_layer.weight`` / ``lambda_layer`` / ``alpha``; the base SVD
  reference (``base_q`` / ``base_p`` / ``base_lambda``) that the
  delta math subtracts is unsaved by upstream design and the
  ``sig_type`` selection mode is unrecoverable from the file, so
  any loader has a silent-correctness gap for ``sig_type != 'principal'``.
  Files fail cleanly with "not loaded".

Diffusers-PEFT fallback (used when ``lora_force_diffusers`` is on) is preserved
via :func:`apply_patch`, which monkey-patches ``Flux2LoraLoaderMixin.lora_state_dict``
to inject the ``diffusion_model.`` prefix for bare-BFL keys and bake kohya
``.alpha`` scaling into ``lora_down`` weights.
"""

import os
import time
import torch
from modules import shared, sd_models
from modules.logger import log
from modules.lora import (
    network, network_lora, network_lokr, network_hada, network_oft, network_boft,
    network_ia3, network_glora, network_norm, network_full, lora_convert,
)
from modules.lora import lora_common as l


# === Format detection ===

# Prefixes we recognize as the "true" format-identifying prefix on a state-dict
# key. The PEFT save wrapper ``base_model.model.`` is handled separately as a
# pre-strip step (see :func:`_unwrap_peft_wrapper`) because it can wrap any of
# the prefixes below — peft.save_pretrained prepends it indiscriminately.
#
# - ``diffusion_model.`` — AI-toolkit / BFL native (e.g. ostris/ai-toolkit)
# - ``transformer.``      — diffusers PEFT in-memory (e.g. HF DreamBooth scripts)
# - ``lora_unet_``        — kohya-ss/sd-scripts standard
# - ``lycoris_``          — LyCORIS-standalone save (e.g. SimpleTuner LoKR);
#                           the path under this prefix is an underscore-rendered
#                           diffusers path, not a BFL path
KNOWN_PREFIXES = ("diffusion_model.", "transformer.", "lora_unet_", "lycoris_")

BARE_FLUX_PREFIXES = (
    "single_blocks.", "double_blocks.", "img_in.", "txt_in.",
    "final_layer.", "time_in.", "single_stream_modulation.",
    "double_stream_modulation_",
)

# Bare diffusers paths (no wrapping prefix) — produced by
# ``Flux2Transformer2DModel.save_lora_adapter()`` after attaching a PEFT adapter.
# These are already-diffusers paths and pass through ``resolve_targets`` verbatim.
BARE_DIFFUSERS_PREFIXES = ("single_transformer_blocks.", "transformer_blocks.")
BARE_DIFFUSERS_PREFIX_USED = "bare_diffusers"  # sentinel value for ``parse_key`` return

SUFFIX_NORMALIZE = {
    "lora_A.weight": "lora_down.weight",
    "lora_B.weight": "lora_up.weight",
}


# === Family suffix tables (alpha / scale / bias / dora_scale flow into weights.w via base NetworkModule.__init__) ===

LORA_SUFFIXES = (
    ".lora_down.weight", ".lora_up.weight", ".lora_mid.weight",
    ".lora_A.weight",    ".lora_B.weight",
    ".alpha", ".dora_scale", ".bias", ".scale",
)
LOKR_SUFFIXES = (
    ".lokr_w1", ".lokr_w2",
    ".lokr_w1_a", ".lokr_w1_b",
    ".lokr_w2_a", ".lokr_w2_b",
    ".lokr_t2",
    ".alpha", ".dora_scale", ".bias", ".scale",
)
LOHA_SUFFIXES = (
    ".hada_w1_a", ".hada_w1_b",
    ".hada_w2_a", ".hada_w2_b",
    ".hada_t1",   ".hada_t2",
    ".alpha", ".dora_scale", ".bias", ".scale",
)
OFT_SUFFIXES = (
    ".oft_blocks", ".oft_diag",
    ".alpha", ".dora_scale", ".bias", ".scale",
)
IA3_SUFFIXES = (
    ".weight", ".on_input",
    ".alpha", ".scale",
)
GLORA_SUFFIXES = (
    ".a1.weight", ".a2.weight",
    ".b1.weight", ".b2.weight",
    ".alpha", ".dora_scale", ".scale",
)
NORM_SUFFIXES = (
    ".w_norm", ".b_norm",
    ".alpha", ".scale",
)
FULL_SUFFIXES = (
    ".diff", ".diff_b",
    ".alpha", ".scale",
)

LORA_MARKERS = (
    ".lora_down.weight", ".lora_up.weight",
    ".lora_A.weight", ".lora_B.weight",
    # PEFT named-adapter saves embed the slot name as ``.lora_A.<name>.weight``;
    # the trailing-dot forms catch every variant.
    ".lora_A.", ".lora_B.",
)
LOKR_MARKERS = (".lokr_w1", ".lokr_w2")
LOHA_MARKERS = (".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b")
OFT_MARKERS = (".oft_blocks", ".oft_diag")
IA3_MARKERS = (".on_input",)  # NOT .weight — too generic, overlaps every other family
GLORA_MARKERS = (".a1.weight", ".a2.weight", ".b1.weight", ".b2.weight")
NORM_MARKERS = (".w_norm",)
FULL_MARKERS = (".diff",)


# === BFL → diffusers mapping ===

# Single-block (single_transformer_blocks.{i}.<target>) — both projections are single fused diffusers modules,
# so no chunking is needed for any adapter family.
F2_SINGLE_MAP = {
    'linear1': 'attn.to_qkv_mlp_proj',
    'linear2': 'attn.to_out',
}

# Double-block non-QKV targets (transformer_blocks.{i}.<target>).
F2_DOUBLE_MAP = {
    'img_attn.proj': 'attn.to_out.0',
    'txt_attn.proj': 'attn.to_add_out',
    'img_mlp.0': 'ff.linear_in',
    'img_mlp.2': 'ff.linear_out',
    'txt_mlp.0': 'ff_context.linear_in',
    'txt_mlp.2': 'ff_context.linear_out',
}

# Double-block fused QKV targets — diffusers exposes Q/K/V as separate modules,
# so resolve_targets emits three (path, chunk_index, num_chunks=3) entries.
F2_QKV_MAP = {
    'img_attn.qkv': ('attn', ['to_q', 'to_k', 'to_v']),
    'txt_attn.qkv': ('attn', ['add_q_proj', 'add_k_proj', 'add_v_proj']),
}

# Kohya underscore suffix → BFL dot suffix (last underscore becomes dot).
# Used to convert kohya key fragments to look up F2_DOUBLE_MAP / F2_QKV_MAP.
KOHYA_SUFFIX_MAP = {
    'img_attn_proj': 'img_attn.proj',
    'txt_attn_proj': 'txt_attn.proj',
    'img_attn_qkv': 'img_attn.qkv',
    'txt_attn_qkv': 'txt_attn.qkv',
    'img_mlp_0': 'img_mlp.0',
    'img_mlp_2': 'img_mlp.2',
    'txt_mlp_0': 'txt_mlp.0',
    'txt_mlp_2': 'txt_mlp.2',
}


# === Shared scaffolding ===


def has_marker(state_dict, markers):
    return any(any(m in k for m in markers) for k in state_dict)


def resolve_mapping():
    sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)
    lora_convert.assign_network_names_to_compvis_modules(sd_model)
    return getattr(shared.sd_model, 'network_layer_mapping', {}) or {}


def new_network(name, network_on_disk):
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)
    return net


def finalize_network(net, name, family, lora_scale, t0, unmapped=0, mismatch=0, skipped=0):
    if len(net.modules) == 0:
        if unmapped or mismatch or skipped:
            log.debug(
                f'Network load: type={family} name="{name}" native no-match'
                f' unmapped={unmapped} mismatch={mismatch} skipped={skipped}'
            )
        return None
    log.debug(
        f'Network load: type={family} name="{name}" native modules={len(net.modules)}'
        f' unmapped={unmapped} mismatch={mismatch} skipped={skipped} scale={lora_scale}'
    )
    l.timer.activate += time.time() - t0
    return net


def shapes_match(sd_module, down_w: torch.Tensor, up_w: torch.Tensor) -> bool:
    if not hasattr(sd_module, 'weight'):
        return False
    if hasattr(sd_module, 'sdnq_dequantizer'):
        mod_shape = sd_module.sdnq_dequantizer.original_shape
    else:
        mod_shape = sd_module.weight.shape
    if len(mod_shape) < 2 or len(down_w.shape) < 2 or len(up_w.shape) < 2:
        return False
    return down_w.shape[1] == mod_shape[1] and up_w.shape[0] == mod_shape[0]


def _unwrap_peft_wrapper(key):
    """Strip the ``base_model.model.`` prefix added by ``peft.save_pretrained``.

    PeftModel.save_pretrained prepends this wrapper to every adapter key. The
    content underneath can be any of the formats KNOWN_PREFIXES already handle:

    - BFL keys (e.g. fal/flux-2-klein-4B-outpaint-lora:
      ``base_model.model.double_blocks.0.img_attn.proj.lora_A.weight``)
    - Diffusers paths under ``transformer.`` (HF DreamBooth scripts that
      target diffusers modules and let peft wrap them)
    - Bare-BFL keys (rare but possible)

    Stripping the wrapper once is enough; the rest of :func:`parse_key` then
    matches the unwrapped key against KNOWN_PREFIXES or the bare-BFL fallback
    normally. Mirrors the diffusers ``Flux2LoraLoaderMixin.lora_state_dict``
    behavior at lora_pipeline.py:5684-5686, which renames the prefix to
    ``diffusion_model.`` before feeding the key to the AI-toolkit converter.
    """
    if key.startswith("base_model.model."):
        return key[len("base_model.model."):]
    return key


def _strip_peft_adapter_name(key):
    """Normalize ``.lora_[AB].<adapter_name>.weight`` to ``.lora_[AB].weight``.

    ``peft.PeftModel`` and the diffusers ``save_lora_adapter`` exporter embed the
    adapter slot name into the saved key (``"default"`` when not explicitly
    set). Strip a single non-dotted name segment so the suffix table matches
    without listing every plausible adapter name.
    """
    for inner in (".lora_A.", ".lora_B."):
        idx = key.find(inner)
        if idx == -1:
            continue
        rest = key[idx + len(inner):]
        if rest == "weight" or not rest.endswith(".weight"):
            continue
        adapter_name = rest[:-len(".weight")]
        if adapter_name and "." not in adapter_name:
            return key[:idx] + inner + "weight"
    return key


def parse_key(key, suffixes):
    """Return ``(prefix_used, base, suffix_normalized)`` or ``None``.

    ``prefix_used`` is the matched ``KNOWN_PREFIXES`` element, or ``None`` for
    bare BFL keys. ``base`` is the format-native module path (kohya / lycoris
    underscore-style or BFL / diffusers dot-style depending on prefix).
    """
    key = _unwrap_peft_wrapper(key)
    key = _strip_peft_adapter_name(key)
    prefix_used = None
    stripped = key
    for p in KNOWN_PREFIXES:
        if key.startswith(p):
            prefix_used = p
            stripped = key[len(p):]
            break
    if prefix_used is None:
        if any(key.startswith(p) for p in BARE_DIFFUSERS_PREFIXES):
            prefix_used = BARE_DIFFUSERS_PREFIX_USED
        elif not any(key.startswith(p) for p in BARE_FLUX_PREFIXES):
            return None

    matched_suffix = None
    split_at = -1
    for marker in suffixes:
        if stripped.endswith(marker):
            split_at = len(stripped) - len(marker)
            matched_suffix = marker.lstrip('.')
            break
    if split_at < 0:
        return None

    base = stripped[:split_at]
    if not base:
        return None

    suffix = SUFFIX_NORMALIZE.get(matched_suffix, matched_suffix)
    return prefix_used, base, suffix


def group_by_suffixes(state_dict, suffixes):
    """Group state_dict entries by ``(prefix_used, base)``.

    Returns ``{(prefix_used, base): {suffix: tensor, ...}}`` where
    ``prefix_used`` is a ``KNOWN_PREFIXES`` element or ``None`` for bare-BFL.
    Per-family loaders apply their own key-presence gates on each group.
    """
    groups: dict[tuple, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        parsed = parse_key(key, suffixes)
        if parsed is None:
            continue
        prefix_used, base, suffix = parsed
        slot = groups.get((prefix_used, base))
        if slot is None:
            slot = {}
            groups[(prefix_used, base)] = slot
        slot[suffix] = value
    return groups


def resolve_targets(prefix_used, base):
    """Return ``[(diffusers_path, chunk_index, num_chunks), ...]`` for a parsed group key.

    For kohya prefix, applies ``KOHYA_SUFFIX_MAP`` then ``F2_*_MAP``. For
    BFL/bare-BFL, applies ``F2_*_MAP`` directly. For PEFT (``transformer.``),
    returns the base verbatim with no chunking — it is already a diffusers path.
    """
    if prefix_used == 'lora_unet_':
        return _kohya_to_diffusers_targets(base)
    if prefix_used in (None, 'diffusion_model.'):
        return _bfl_to_diffusers_targets(base)
    if prefix_used == 'transformer.':
        return [(base, None, None)]
    if prefix_used == BARE_DIFFUSERS_PREFIX_USED:
        # Already-diffusers path with no wrapping prefix (e.g. produced by
        # Flux2Transformer2DModel.save_lora_adapter()). Pass through verbatim.
        return [(base, None, None)]
    if prefix_used == 'lycoris_':
        # base is an already-underscored diffusers path (e.g.
        # 'transformer_blocks_0_attn_add_k_proj'). The caller's network_key
        # construction does base.replace('.', '_'); for already-underscored
        # paths that's a no-op, so the network_key matches the entry stamped
        # by lora_convert.assign_network_names_to_compvis_modules
        # (e.g. 'lora_transformer_transformer_blocks_0_attn_add_k_proj').
        return [(base, None, None)]
    return []


def _kohya_to_diffusers_targets(stripped):
    """For kohya keys like ``double_blocks_0_img_attn_proj`` or ``single_blocks_5_linear1``."""
    targets: list[tuple[str, int | None, int | None]] = []
    if stripped.startswith('single_blocks_'):
        rest = stripped[len('single_blocks_'):]
        idx, _, suffix = rest.partition('_')
        if suffix in F2_SINGLE_MAP:
            targets.append((f'single_transformer_blocks.{idx}.{F2_SINGLE_MAP[suffix]}', None, None))
    elif stripped.startswith('double_blocks_'):
        rest = stripped[len('double_blocks_'):]
        idx, _, kohya_suffix = rest.partition('_')
        bfl_suffix = KOHYA_SUFFIX_MAP.get(kohya_suffix)
        if bfl_suffix is None:
            return targets
        if bfl_suffix in F2_DOUBLE_MAP:
            targets.append((f'transformer_blocks.{idx}.{F2_DOUBLE_MAP[bfl_suffix]}', None, None))
        elif bfl_suffix in F2_QKV_MAP:
            attn_prefix, proj_keys = F2_QKV_MAP[bfl_suffix]
            for i, proj_key in enumerate(proj_keys):
                targets.append((f'transformer_blocks.{idx}.{attn_prefix}.{proj_key}', i, len(proj_keys)))
    return targets


def _bfl_to_diffusers_targets(base):
    """For BFL keys like ``double_blocks.0.img_attn.proj`` or ``single_blocks.5.linear1``."""
    targets: list[tuple[str, int | None, int | None]] = []
    parts = base.split('.')
    if len(parts) < 3:
        return targets
    block_type, block_idx, module_suffix = parts[0], parts[1], '.'.join(parts[2:])
    if block_type == 'single_blocks' and module_suffix in F2_SINGLE_MAP:
        targets.append((f'single_transformer_blocks.{block_idx}.{F2_SINGLE_MAP[module_suffix]}', None, None))
    elif block_type == 'double_blocks':
        if module_suffix in F2_DOUBLE_MAP:
            targets.append((f'transformer_blocks.{block_idx}.{F2_DOUBLE_MAP[module_suffix]}', None, None))
        elif module_suffix in F2_QKV_MAP:
            attn_prefix, proj_keys = F2_QKV_MAP[module_suffix]
            for i, proj_key in enumerate(proj_keys):
                targets.append((f'transformer_blocks.{block_idx}.{attn_prefix}.{proj_key}', i, len(proj_keys)))
    return targets


# === Native loaders ===


def try_load(name, network_on_disk, lora_scale):
    """Run every Flux2 family loader in dispatch order, merge any that match.

    Per-family ``try_load_*`` entry points stay public; this is the single
    umbrella the dispatcher in ``modules.lora.lora_load.load_safetensors``
    calls. Order matters only for marker-cost: LoRA / LoKR are most common
    so their fast bail-out runs first; the rare families come last.

    Returns a ``Network`` with the union of modules from every matching
    family loader, or ``None`` if no loader recognized the file.
    """
    net = None
    for try_fn in (
        try_load_lora, try_load_lokr, try_load_loha, try_load_oft,
        try_load_ia3, try_load_glora, try_load_norm, try_load_full,
    ):
        sub = try_fn(name, network_on_disk, lora_scale)
        if sub is None:
            continue
        if net is None:
            net = sub
        else:
            net.modules.update(sub.modules)
    return net


def try_load_lora(name, network_on_disk, lora_scale):
    """Load a Flux2/Klein LoRA (plus DoRA via the universal ``finalize_updown`` hook) as native modules.

    Handles kohya, AI-toolkit/BFL, diffusers PEFT, and bare-BFL key formats.
    Fused QKV in double_blocks is split at load time by chunking the up-weight
    along dim 0; the down-weight is shared across Q/K/V.
    """
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, LORA_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    groups = group_by_suffixes(state_dict, LORA_SUFFIXES)

    unmapped = 0
    mismatch = 0
    for (prefix, base), w in groups.items():
        if 'lora_down.weight' not in w or 'lora_up.weight' not in w:
            continue
        for diffusers_path, chunk_idx, num_chunks in resolve_targets(prefix, base):
            network_key = "lora_transformer_" + diffusers_path.replace(".", "_")
            sd_module = mapping.get(network_key)
            if sd_module is None:
                unmapped += 1
                continue

            if chunk_idx is not None:
                chunks = torch.chunk(w['lora_up.weight'], num_chunks, dim=0)
                target_w = dict(w)
                target_w['lora_up.weight'] = chunks[chunk_idx].contiguous()
            else:
                target_w = w

            if not shapes_match(sd_module, target_w['lora_down.weight'], target_w['lora_up.weight']):
                log.warning(
                    f'Network load: type=LoRA name="{name}" key={network_key}'
                    f' lora={target_w["lora_down.weight"].shape[1]}x{target_w["lora_up.weight"].shape[0]}'
                    f' module={getattr(sd_module, "weight", None).shape if hasattr(sd_module, "weight") else "?"}'
                    f' shape mismatch'
                )
                mismatch += 1
                continue

            nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=target_w, sd_module=sd_module)
            net.modules[network_key] = network_lora.NetworkModuleLora(net, nw)

    return finalize_network(net, name, 'LoRA', lora_scale, t0, unmapped=unmapped, mismatch=mismatch)


def try_load_lokr(name, network_on_disk, lora_scale):
    """Load a Flux2/Klein LoKR as native modules.

    Stores only the compact LoKR factors (``w1``/``w2``) and computes
    ``kron(w1, w2)`` on-the-fly during weight application. For fused QKV
    targets in double_blocks, :class:`NetworkModuleLokrChunk` materializes the
    full Kronecker product and returns the designated Q/K/V slice.
    """
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, LOKR_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    groups = group_by_suffixes(state_dict, LOKR_SUFFIXES)

    unmapped = 0
    for (prefix, base), w in groups.items():
        has_1 = "lokr_w1" in w or ("lokr_w1_a" in w and "lokr_w1_b" in w)
        has_2 = "lokr_w2" in w or ("lokr_w2_a" in w and "lokr_w2_b" in w)
        if not (has_1 and has_2):
            continue
        for diffusers_path, chunk_idx, num_chunks in resolve_targets(prefix, base):
            network_key = "lora_transformer_" + diffusers_path.replace(".", "_")
            sd_module = mapping.get(network_key)
            if sd_module is None:
                unmapped += 1
                continue
            nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
            if chunk_idx is not None:
                net.modules[network_key] = network_lokr.NetworkModuleLokrChunk(net, nw, chunk_idx, num_chunks)
            else:
                net.modules[network_key] = network_lokr.NetworkModuleLokr(net, nw)

    return finalize_network(net, name, 'LoKR', lora_scale, t0, unmapped=unmapped)


def try_load_loha(name, network_on_disk, lora_scale):
    """Load a Flux2/Klein LoHA (Hadamard product) adapter as native modules.

    Standard non-Tucker LoHA on fused QKV in double_blocks is supported via
    :class:`NetworkModuleHadaChunk`, which slices ``w1a``/``w2a`` at the
    chunk's row range and computes the partial Hadamard. Tucker
    (CP-decomposed) LoHAs are skipped on fused targets because the chunk
    class does not implement the CP path; non-fused Tucker LoHAs go through
    the standard :class:`NetworkModuleHada`.
    """
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, LOHA_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    groups = group_by_suffixes(state_dict, LOHA_SUFFIXES)

    unmapped = 0
    skipped = 0
    for (prefix, base), w in groups.items():
        if not all(k in w for k in ('hada_w1_a', 'hada_w1_b', 'hada_w2_a', 'hada_w2_b')):
            continue
        is_tucker = 'hada_t1' in w or 'hada_t2' in w
        targets = resolve_targets(prefix, base)
        is_fused = any(t[1] is not None for t in targets)
        if is_fused and is_tucker:
            log.warning(f'Network load: type=LoHA name="{name}" key={base} Tucker fused QKV skipped (unsupported)')
            skipped += 1
            continue
        for diffusers_path, chunk_idx, num_chunks in targets:
            network_key = "lora_transformer_" + diffusers_path.replace(".", "_")
            sd_module = mapping.get(network_key)
            if sd_module is None:
                unmapped += 1
                continue
            nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
            if chunk_idx is not None:
                net.modules[network_key] = network_hada.NetworkModuleHadaChunk(net, nw, chunk_idx, num_chunks)
            else:
                net.modules[network_key] = network_hada.NetworkModuleHada(net, nw)

    return finalize_network(net, name, 'LoHA', lora_scale, t0, unmapped=unmapped, skipped=skipped)


def try_load_oft(name, network_on_disk, lora_scale):
    """Load a Flux2/Klein OFT or BOFT adapter as native modules.

    Both algorithms share the ``oft_blocks`` save key and are discriminated
    by tensor dimensionality, mirroring LyCORIS's own ``algo_check``:

    - **OFT** — 3-D ``(num_blocks, block_size, block_size)``. Both kohya
      (``oft_blocks`` + alpha-as-constraint) and LyCORIS (``oft_diag``)
      layouts route through :class:`NetworkModuleOFT`.
    - **BOFT** — 4-D ``(boft_m, block_num, block_size, block_size)``,
      a cascade of butterfly factors. Routes through
      :class:`NetworkModuleBOFT` which ports the butterfly-cascade
      ``make_weight`` from LyCORIS boft.py.

    Fused QKV in double_blocks is skipped with a warning for both: an OFT
    block structure (and BOFT's per-stage block partition) is tied to the
    target module's ``out_features``, so a per-Q/K/V split would require
    re-deriving the rotations per chunk. Single-block ``linear1`` (a single
    fused diffusers module) and all non-QKV double-block targets work fully.
    """
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, OFT_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    groups = group_by_suffixes(state_dict, OFT_SUFFIXES)

    unmapped = 0
    skipped = 0
    for (prefix, base), w in groups.items():
        if not ('oft_blocks' in w or 'oft_diag' in w):
            continue
        is_boft = 'oft_blocks' in w and w['oft_blocks'].ndim == 4
        targets = resolve_targets(prefix, base)
        if any(t[1] is not None for t in targets):
            log.warning(f'Network load: type={"BOFT" if is_boft else "OFT"} name="{name}" key={base} fused QKV skipped (unsupported)')
            skipped += 1
            continue
        for diffusers_path, _, _ in targets:
            network_key = "lora_transformer_" + diffusers_path.replace(".", "_")
            sd_module = mapping.get(network_key)
            if sd_module is None:
                unmapped += 1
                continue
            nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
            if is_boft:
                net.modules[network_key] = network_boft.NetworkModuleBOFT(net, nw)
            else:
                net.modules[network_key] = network_oft.NetworkModuleOFT(net, nw)

    return finalize_network(net, name, 'OFT', lora_scale, t0, unmapped=unmapped, skipped=skipped)


def try_load_ia3(name, network_on_disk, lora_scale):
    """Load a Flux2/Klein IA3 adapter as native modules.

    IA3 stores a per-row or per-column scale vector keyed under ``.weight``
    plus an ``.on_input`` flag selecting which axis. The ``.on_input`` marker
    is the format disambiguator — ``.weight`` alone is too generic and
    overlaps every other family's ``.lora_down.weight`` / ``.hada_w*`` keys,
    so the SUFFIXES table includes it but the MARKERS gate insists on
    ``.on_input``.

    Fused QKV in double_blocks is skipped: ``on_input=True`` IA3 vectors
    would replicate cleanly to Q/K/V (same ``in_features``) but
    ``on_input=False`` requires slicing the output-axis vector across the
    three projections, and there is zero real-world IA3-on-DiT prevalence to
    justify the asymmetry.
    """
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, IA3_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    groups = group_by_suffixes(state_dict, IA3_SUFFIXES)

    unmapped = 0
    skipped = 0
    for (prefix, base), w in groups.items():
        if not ('weight' in w and 'on_input' in w):
            continue
        targets = resolve_targets(prefix, base)
        if any(t[1] is not None for t in targets):
            log.warning(f'Network load: type=IA3 name="{name}" key={base} fused QKV skipped (unsupported)')
            skipped += 1
            continue
        for diffusers_path, _, _ in targets:
            network_key = "lora_transformer_" + diffusers_path.replace(".", "_")
            sd_module = mapping.get(network_key)
            if sd_module is None:
                unmapped += 1
                continue
            nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
            net.modules[network_key] = network_ia3.NetworkModuleIa3(net, nw)

    return finalize_network(net, name, 'IA3', lora_scale, t0, unmapped=unmapped, skipped=skipped)


def try_load_glora(name, network_on_disk, lora_scale):
    """Load a Flux2/Klein GLoRA adapter as native modules.

    GLoRA stores four low-rank components (``a1``/``a2``/``b1``/``b2``) and
    computes ``ΔW = w2b @ w1b + (target @ w2a) @ w1a`` — the second term is
    target-dependent. Fused QKV in double_blocks is skipped with a warning
    because the target-dependent term doesn't slice cleanly without
    redirecting calc_updown to a fused proxy weight, and zero real-world
    GLoRA-on-DiT files exist.

    Depends on the ``self.dim`` initialization fix in network_glora.py so
    that alpha-based ``calc_scale`` is honored.
    """
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, GLORA_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    groups = group_by_suffixes(state_dict, GLORA_SUFFIXES)

    unmapped = 0
    skipped = 0
    for (prefix, base), w in groups.items():
        if not all(k in w for k in ('a1.weight', 'a2.weight', 'b1.weight', 'b2.weight')):
            continue
        targets = resolve_targets(prefix, base)
        if any(t[1] is not None for t in targets):
            log.warning(f'Network load: type=GLoRA name="{name}" key={base} fused QKV skipped (unsupported)')
            skipped += 1
            continue
        for diffusers_path, _, _ in targets:
            network_key = "lora_transformer_" + diffusers_path.replace(".", "_")
            sd_module = mapping.get(network_key)
            if sd_module is None:
                unmapped += 1
                continue
            nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
            net.modules[network_key] = network_glora.NetworkModuleGLora(net, nw)

    return finalize_network(net, name, 'GLoRA', lora_scale, t0, unmapped=unmapped, skipped=skipped)


def try_load_norm(name, network_on_disk, lora_scale):
    """Load a Flux2/Klein Norm adapter (LayerNorm/RMSNorm weight + bias deltas) as native modules.

    Norm adapters target the RMSNorm modules inside Flux2 attention
    (``attn.norm_q``, ``attn.norm_k``, ``attn.norm_added_q``,
    ``attn.norm_added_k``) — the only norm modules in Flux2 with trainable
    weights. The block-level ``norm1``/``norm2`` LayerNorms have
    ``elementwise_affine=False`` and are not adaptable.

    Loader-local stamping: ``modules/lora/lora_convert.py:assign_network_names_to_compvis_modules``
    deliberately skips setting ``module.network_layer_name`` for transformer
    norm modules (except SD3) because of legacy CompVis UNet collisions. This
    loader bypasses the guard locally — for each target it actually binds, it
    sets ``network_layer_name`` directly on the host module so
    ``network_activate`` will apply the delta. No edit to the shared
    ``lora_convert`` carve-out is required, and no norm module is touched
    unless a Norm adapter explicitly targets it.

    BFL/kohya prefix support is deferred — there is no public Flux2 BFL norm
    mapping table to verify against. PEFT prefix (the format produced by
    ``peft`` training) works directly because the base path is already a
    diffusers path.
    """
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, NORM_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    groups = group_by_suffixes(state_dict, NORM_SUFFIXES)

    unmapped = 0
    for (prefix, base), w in groups.items():
        if 'w_norm' not in w:
            continue
        targets = resolve_targets(prefix, base)
        if not targets:
            unmapped += 1
            continue
        for diffusers_path, chunk_idx, _ in targets:
            if chunk_idx is not None:
                continue  # norm targets are not fused
            network_key = "lora_transformer_" + diffusers_path.replace(".", "_")
            sd_module = mapping.get(network_key)
            if sd_module is None:
                unmapped += 1
                continue
            # Bypass the lora_convert.py:502 transformer-norm guard locally.
            # Stamping is idempotent and only touches modules a Norm adapter targets.
            if not getattr(sd_module, 'network_layer_name', None):
                sd_module.network_layer_name = network_key
            nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
            net.modules[network_key] = network_norm.NetworkModuleNorm(net, nw)

    return finalize_network(net, name, 'Norm', lora_scale, t0, unmapped=unmapped)


def try_load_full(name, network_on_disk, lora_scale):
    """Load a Flux2/Klein Full (full-rank) adapter as native modules.

    Full adapters carry a complete weight delta (``diff``, same shape as the
    host weight) and an optional bias delta (``diff_b``) via
    :class:`NetworkModuleFull`. Most realistic use: small per-block bias-only
    adjustments in distillation LoRAs.

    Fused QKV in double_blocks is skipped with a warning. Full's ``diff`` has
    the host weight's full shape; row-slicing across three projections is
    well-defined arithmetically but no chunk class exists and zero
    real-world Full-on-fused-DiT files exist. Single-block linear1 (a single
    fused diffusers module) and non-QKV double-block targets work fully.
    """
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, FULL_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    groups = group_by_suffixes(state_dict, FULL_SUFFIXES)

    unmapped = 0
    skipped = 0
    for (prefix, base), w in groups.items():
        if 'diff' not in w:
            continue
        targets = resolve_targets(prefix, base)
        if any(t[1] is not None for t in targets):
            log.warning(f'Network load: type=Full name="{name}" key={base} fused QKV skipped (unsupported)')
            skipped += 1
            continue
        for diffusers_path, _, _ in targets:
            network_key = "lora_transformer_" + diffusers_path.replace(".", "_")
            sd_module = mapping.get(network_key)
            if sd_module is None:
                unmapped += 1
                continue
            nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
            net.modules[network_key] = network_full.NetworkModuleFull(net, nw)

    return finalize_network(net, name, 'Full', lora_scale, t0, unmapped=unmapped, skipped=skipped)


# === Diffusers-PEFT path helpers (used when lora_force_diffusers is on) ===


def apply_lora_alphas(state_dict):
    """Bake kohya-format ``.alpha`` scaling into ``lora_down`` weights and remove alpha keys.

    Diffusers' Flux2 converter only handles ``lora_A``/``lora_B`` (or
    ``lora_down``/``lora_up``) keys. Kohya-format LoRAs store per-layer alpha
    values as separate ``.alpha`` keys that the converter does not consume,
    causing a ``ValueError`` on leftover keys. This matches the approach used
    by ``_convert_kohya_flux_lora_to_diffusers`` for Flux 1.
    """
    alpha_keys = [k for k in state_dict if k.endswith('.alpha')]
    if not alpha_keys:
        return state_dict
    for alpha_key in alpha_keys:
        base = alpha_key[:-len('.alpha')]
        down_key = f'{base}.lora_down.weight'
        if down_key not in state_dict:
            continue
        down_weight = state_dict[down_key]
        rank = down_weight.shape[0]
        alpha = state_dict.pop(alpha_key).item()
        scale = alpha / rank
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2
        state_dict[down_key] = down_weight * scale_down
        up_key = f'{base}.lora_up.weight'
        if up_key in state_dict:
            state_dict[up_key] = state_dict[up_key] * scale_up
    remaining = [k for k in state_dict if k.endswith('.alpha')]
    if remaining:
        log.debug(f'Network load: type=LoRA stripped {len(remaining)} orphaned alpha keys')
        for k in remaining:
            del state_dict[k]
    return state_dict


def preprocess_f2_keys(state_dict):
    """Add ``diffusion_model.`` prefix to bare BFL-format keys so
    ``Flux2LoraLoaderMixin``'s format detection routes them to the converter."""
    if any(k.startswith("diffusion_model.") or k.startswith("base_model.model.") for k in state_dict):
        return state_dict
    if any(k.startswith(p) for k in state_dict for p in BARE_FLUX_PREFIXES):
        log.debug('Network load: type=LoRA adding diffusion_model prefix for bare BFL-format keys')
        state_dict = {f"diffusion_model.{k}": v for k, v in state_dict.items()}
    return state_dict


patched = False


def apply_patch():
    """Patch ``Flux2LoraLoaderMixin.lora_state_dict`` to handle bare BFL-format keys.

    When a LoRA file has bare BFL keys (no ``diffusion_model.`` prefix), the
    original ``lora_state_dict`` won't detect them as AI toolkit format. This
    patch checks for bare keys after the original returns and adds the prefix +
    re-runs conversion. Used only on the diffusers-PEFT fallback path.
    """
    global patched # pylint: disable=global-statement
    if patched:
        return
    patched = True

    from diffusers.loaders.lora_pipeline import Flux2LoraLoaderMixin
    original_lora_state_dict = Flux2LoraLoaderMixin.lora_state_dict.__func__

    @classmethod # pylint: disable=no-self-argument
    def patched_lora_state_dict(cls, pretrained_model_name_or_path_or_dict, **kwargs):
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = preprocess_f2_keys(pretrained_model_name_or_path_or_dict)
            pretrained_model_name_or_path_or_dict = apply_lora_alphas(pretrained_model_name_or_path_or_dict)
        elif isinstance(pretrained_model_name_or_path_or_dict, (str, os.PathLike)):
            path = str(pretrained_model_name_or_path_or_dict)
            if path.endswith('.safetensors'):
                try:
                    from safetensors import safe_open
                    with safe_open(path, framework="pt") as f:
                        keys = list(f.keys())
                    needs_load = (
                        any(k.endswith('.alpha') for k in keys)
                        or (not any(k.startswith("diffusion_model.") or k.startswith("base_model.model.") for k in keys)
                            and any(k.startswith(p) for k in keys for p in BARE_FLUX_PREFIXES))
                    )
                    if needs_load:
                        from safetensors.torch import load_file
                        sd = load_file(path)
                        sd = preprocess_f2_keys(sd)
                        pretrained_model_name_or_path_or_dict = apply_lora_alphas(sd)
                except Exception:
                    pass
        return original_lora_state_dict(cls, pretrained_model_name_or_path_or_dict, **kwargs)

    Flux2LoraLoaderMixin.lora_state_dict = patched_lora_state_dict
