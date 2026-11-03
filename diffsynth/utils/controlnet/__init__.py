from .controlnet_input import ControlNetInput
from .annotator import Annotator

# Lazy: each of these is also a __main__ entry point (`python -m
# diffsynth.utils.controlnet.video_blur`), and importing it here would make runpy
# re-execute an already imported module. They also pull in heavy optional deps
# (controlnet_aux, transformers) that only the preprocessing steps need.
_LAZY = {"video_to_lineart": "video_lineart", "video_to_blur": "video_blur",
         "caption_windows": "video_caption"}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(name)


__all__ = ["ControlNetInput", "Annotator", *_LAZY]
