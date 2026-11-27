from ..core.loader import load_model, hash_model_file, load_keys_dict, load_metadata_from_safetensors
from ..core.vram import AutoWrappedModule
from ..core.quant import QuantizeConfig, MixedQuantizeConfig
from ..configs import MODEL_CONFIGS, VRAM_MANAGEMENT_MODULE_MAPS, VERSION_CHECKER_MAPS
import importlib, json, torch


# Default hyperparams shared by the published Fun-Controlnet-Union checkpoint and the
# DiffSynth training export (blocks.* / control_patch_proj.*). control_layers is filled
# in per file from metadata or by inferring from the number of blocks.
_MINIMAX_H3_CONTROLNET_BASE_KWARGS = {
    "control_in_dim": 49,
    "control_apply_audio": False,
    "hidden_size": 5376,
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "ffn_hidden_size": 14336,
    "time_embed_dim": 2688,
    "adaln_out_features": 96768,
    "patch_size": (1, 2, 2),
    "norm_eps": 1e-05,
    "qk_norm_eps": 1e-05,
}


class ModelPool:
    def __init__(self):
        self.model = []
        self.model_name = []
        self.model_path = []
        
    def import_model_class(self, model_class):
        split = model_class.rfind(".")
        model_resource, model_class = model_class[:split], model_class[split+1:]
        model_class = importlib.import_module(model_resource).__getattribute__(model_class)
        return model_class
    
    def need_to_enable_vram_management(self, vram_config):
        return vram_config["offload_dtype"] is not None and vram_config["offload_device"] is not None
    
    def fetch_module_map(self, model_class, vram_config):
        if self.need_to_enable_vram_management(vram_config):
            if model_class in VRAM_MANAGEMENT_MODULE_MAPS:
                vram_module_map = VRAM_MANAGEMENT_MODULE_MAPS[model_class] if model_class not in VERSION_CHECKER_MAPS else VERSION_CHECKER_MAPS[model_class]()
                module_map = {self.import_model_class(source): self.import_model_class(target) for source, target in vram_module_map.items()}
            else:
                module_map = {self.import_model_class(model_class): AutoWrappedModule}
        else:
            module_map = None
        return module_map
    
    def load_model_file(self, config, path, vram_config, vram_limit=None, state_dict=None, quantize=None):
        model_class = self.import_model_class(config["model_class"])
        # Resolve the quant config only after the model class import: a model file may
        # register its own method/backend at import time (e.g. `ideogram4_fp8`), and
        # `QuantizeConfig` looks the method up at construction.
        quantize = self.resolve_quant_config(config, quantize)
        model_config = config.get("extra_kwargs", {})
        if "state_dict_converter" in config:
            state_dict_converter = self.import_model_class(config["state_dict_converter"])
        else:
            state_dict_converter = None
        module_map = self.fetch_module_map(config["model_class"], vram_config)
        model = load_model(
            model_class, path, model_config,
            vram_config["computation_dtype"],
            vram_config["computation_device"],
            state_dict_converter,
            use_disk_map=True,
            vram_config=vram_config, module_map=module_map, vram_limit=vram_limit,
            state_dict=state_dict,
            quantize=quantize,
        )
        return model
    
    def default_vram_config(self):
        vram_config = {
            "offload_dtype": None,
            "offload_device": None,
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cpu",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cpu",
        }
        return vram_config
    
    def resolve_quant_config(self, config, quantize):
        registered = config.get("quant_config")
        if registered is not None:
            if quantize is not None:
                print(f"Warning: `{config.get('model_name')}` is already a pre-quantized checkpoint; ignoring the passed quantize option.")
            registered = dict(registered)
            child_configs = registered.pop("configs", None)
            if child_configs is not None:
                return MixedQuantizeConfig(configs=[QuantizeConfig(**child) for child in child_configs], **registered)
            return QuantizeConfig(**registered)
        return quantize

    def _minimax_h3_controlnet_block_ids(self, keys_dict):
        block_ids = set()
        for key in keys_dict:
            # DiffSynth training export: blocks.{i}.*
            # Official Fun-Controlnet-Union (pre-converter): control_blocks.{i}.*
            for prefix in ("blocks.", "control_blocks."):
                if key.startswith(prefix):
                    rest = key[len(prefix):]
                    head = rest.split(".", 1)[0]
                    if head.isdigit():
                        block_ids.add(int(head))
        return sorted(block_ids)

    def _looks_like_minimax_h3_controlnet(self, keys_dict):
        has_proj = any(k.startswith("control_patch_proj.") or k.startswith("control_proj_in.") for k in keys_dict)
        return has_proj and bool(self._minimax_h3_controlnet_block_ids(keys_dict))

    def _resolve_minimax_h3_control_layers(self, path, num_blocks):
        from .minimax_h3_controlnet import DEFAULT_CONTROL_LAYERS, default_control_layers_for_num_blocks, parse_control_layers

        metadata = {}
        try:
            if isinstance(path, str) and path.endswith(".safetensors"):
                metadata = load_metadata_from_safetensors(path)
            elif isinstance(path, list) and path and str(path[0]).endswith(".safetensors"):
                metadata = load_metadata_from_safetensors(path)
        except Exception:
            metadata = {}
        raw = metadata.get("minimax_h3_control_layers")
        if raw:
            layers = parse_control_layers(raw)
            if len(layers) != num_blocks:
                raise ValueError(
                    f"Checkpoint metadata control_layers={layers} has {len(layers)} entries "
                    f"but the file contains {num_blocks} ControlNet blocks"
                )
            return layers
        if num_blocks == len(DEFAULT_CONTROL_LAYERS):
            return DEFAULT_CONTROL_LAYERS
        return default_control_layers_for_num_blocks(num_blocks)

    def _apply_minimax_h3_control_layers_from_metadata(self, model, path):
        """Same-hash 5-block exports can still retarget injection points via metadata."""
        if getattr(model, "control_layers", None) is None or not hasattr(model, "set_control_layers"):
            return model
        try:
            layers = self._resolve_minimax_h3_control_layers(path, len(model.blocks))
        except Exception:
            return model
        if tuple(layers) != tuple(model.control_layers):
            model.set_control_layers(layers)
            print(f"Applied ControlNet control_layers from checkpoint metadata: {layers}")
        return model

    def _build_minimax_h3_controlnet_config(self, path):
        keys_dict = load_keys_dict(path)
        if not self._looks_like_minimax_h3_controlnet(keys_dict):
            return None
        block_ids = self._minimax_h3_controlnet_block_ids(keys_dict)
        num_blocks = max(block_ids) + 1 if block_ids else 0
        if num_blocks <= 0 or block_ids != list(range(num_blocks)):
            return None
        control_layers = self._resolve_minimax_h3_control_layers(path, num_blocks)
        extra_kwargs = dict(_MINIMAX_H3_CONTROLNET_BASE_KWARGS)
        extra_kwargs["control_layers"] = control_layers
        # Official weights still need the key converter; DiffSynth exports do not.
        needs_converter = any(k.startswith("control_blocks.") or k.startswith("control_proj_in.") for k in keys_dict)
        config = {
            "model_name": "minimax_h3_controlnet",
            "model_class": "diffsynth.models.minimax_h3_controlnet.MiniMaxH3ControlNet",
            "extra_kwargs": extra_kwargs,
        }
        if needs_converter:
            config["state_dict_converter"] = "diffsynth.utils.state_dict_converters.minimax_h3_controlnet.MiniMaxH3ControlNetStateDictConverter"
        return config

    def auto_load_model(self, path, vram_config=None, vram_limit=None, clear_parameters=False, state_dict=None, quantize=None):
        print(f"Loading models from: {json.dumps(path, indent=4)}")
        if vram_config is None:
            vram_config = self.default_vram_config()
        model_hash = hash_model_file(path)
        loaded = False
        for config in MODEL_CONFIGS:
            if config["model_hash"] == model_hash:
                model = self.load_model_file(config, path, vram_config, vram_limit=vram_limit, state_dict=state_dict, quantize=quantize)
                if config.get("model_name") == "minimax_h3_controlnet":
                    model = self._apply_minimax_h3_control_layers_from_metadata(model, path)
                if clear_parameters: self.clear_parameters(model)
                self.model.append(model)
                model_name = config["model_name"]
                self.model_name.append(model_name)
                self.model_path.append(path)
                model_info = {"model_name": model_name, "model_class": config["model_class"], "extra_kwargs": config.get("extra_kwargs")}
                if hasattr(model, "control_layers"):
                    model_info["control_layers"] = list(model.control_layers)
                print(f"Loaded model: {json.dumps(model_info, indent=4)}")
                loaded = True
        if not loaded:
            # Variable-depth MiniMax-H3 ControlNet exports change the state-dict hash, so
            # fall back to structural detection + optional control_layers metadata.
            fallback = self._build_minimax_h3_controlnet_config(path)
            if fallback is not None:
                model = self.load_model_file(fallback, path, vram_config, vram_limit=vram_limit, state_dict=state_dict, quantize=quantize)
                if clear_parameters: self.clear_parameters(model)
                self.model.append(model)
                self.model_name.append(fallback["model_name"])
                self.model_path.append(path)
                model_info = {
                    "model_name": fallback["model_name"],
                    "model_class": fallback["model_class"],
                    "extra_kwargs": fallback.get("extra_kwargs"),
                    "control_layers": list(model.control_layers),
                    "detected_by": "minimax_h3_controlnet_structure",
                }
                print(f"Loaded model: {json.dumps(model_info, indent=4)}")
                loaded = True
        if not loaded:
            raise ValueError(f"Cannot detect the model type. File: {path}. Model hash: {model_hash}")
    
    def fetch_model(self, model_name, index=None):
        fetched_models = []
        fetched_model_paths = []
        for model, model_path, model_name_ in zip(self.model, self.model_path, self.model_name):
            if model_name == model_name_:
                fetched_models.append(model)
                fetched_model_paths.append(model_path)
        if len(fetched_models) == 0:
            print(f"No {model_name} models available. This is not an error.")
            model = None
        elif len(fetched_models) == 1:
            print(f"Using {model_name} from {json.dumps(fetched_model_paths[0], indent=4)}.")
            model = fetched_models[0]
        else:
            if index is None:
                model = fetched_models[0]
                print(f"More than one {model_name} models are loaded: {fetched_model_paths}. Using {model_name} from {json.dumps(fetched_model_paths[0], indent=4)}.")
            elif isinstance(index, int):
                model = fetched_models[:index]
                print(f"More than one {model_name} models are loaded: {fetched_model_paths}. Using {model_name} from {json.dumps(fetched_model_paths[:index], indent=4)}.")
            else:
                model = fetched_models
                print(f"More than one {model_name} models are loaded: {fetched_model_paths}. Using {model_name} from {json.dumps(fetched_model_paths, indent=4)}.")
        return model

    def clear_parameters(self, model: torch.nn.Module):
        for name, module in model.named_children():
            self.clear_parameters(module)
        for name, param in model.named_parameters(recurse=False):
            setattr(model, name, None)
