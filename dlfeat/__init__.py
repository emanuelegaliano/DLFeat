# dlfeat/__init__.py

from .dlfeat_lib import (
    DLFeatExtractor,
    list_available_models,
    run_self_tests,
    MODEL_CONFIGS,
    __version__,
    register_image_model,
    register_video_model,
    register_audio_model,
    register_text_model,
    register_multimodal_image_text_model,
    register_multimodal_video_text_model,
)

__all__ = [
    "DLFeatExtractor",
    "list_available_models",
    "run_self_tests",
    "MODEL_CONFIGS",
    "__version__",
    "register_image_model",
    "register_video_model",
    "register_audio_model",
    "register_text_model",
    "register_multimodal_image_text_model",
    "register_multimodal_video_text_model",
]