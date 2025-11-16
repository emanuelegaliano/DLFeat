# DLFeat: Deep Learning Feature Extraction Library

<p align="center">
  <img src="dlfeat.jpg" alt="DLFeat Logo" width="200"/>
</p>

<p align="center">
  <a href="https://antoninofurnari.github.io/DLFeat/">
    <img src="https://img.shields.io/badge/docs-gh--pages-blue.svg" alt="Documentation Status">
  </a>
  <!--<a href="https://pypi.org/project/dlfeat/"> <img src="https://img.shields.io/pypi/v/dlfeat.svg" alt="PyPI version">
  </a>-->
  <a > <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"> </a>
  <!-- <a href="YOUR_ACTIONS_LINK_HERE">
    <img src="https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME/actions/workflows/YOUR_CI_WORKFLOW.yml/badge.svg" alt="Build Status">
  </a> -->
</p>

**DLFeat** is a Python library designed for easy and modular feature extraction 
from various data modalities including images, videos, audio, and text. 
It leverages powerful pre-trained models from libraries like PyTorch, torchvision, 
Transformers, Sentence-Transformers, and TIMM. 
The goal is to provide a "black box" tool suitable for educational and research purposes, 
allowing users to quickly extract meaningful features for data analysis tasks 
without needing to delve into the complexities of each model's architecture or training.

## Core Features

* **Unified API**: Consistent `DLFeatExtractor` class for all modalities.
* **Scikit-learn Compatible**: Implements `BaseEstimator` and `TransformerMixin` for easy pipeline integration.
* **Multi-Modal Support**: Extract features from images, videos, audio, and text.
* **Extensive Model Zoo**: Access to a wide range of pre-trained models. See the [full documentation](https://antoninofurnari.github.io/DLFeat/) for details.
* **Automatic Handling**: Manages model loading, preprocessing, and device placement (CPU/GPU).
* **Single-File Library**: Easy to distribute and integrate (dependencies must be installed separately).
* **Self-Testing**: Built-in function `run_self_tests()` to verify model availability and basic functionality.

## Model Zoo
DLFeat provides access to a variety of pre-trained models. For a comprehensive list including performance metrics, feature dimensions, and source libraries, please see the Model Zoo page in [our documentation](https://antoninofurnari.github.io/DLFeat/model_zoo.html).

## Documentation
For complete API reference, tutorials, and the full Model Zoo, please visit our [GitHub Pages Documentation Site](https://antoninofurnari.github.io/DLFeat/).

## Running Self-Tests
To verify the installation and basic functionality of the models in your environment, you can run the built-in self-tests:

```python
from DLFeat import run_self_tests

# Test a default set of representative models
results = run_self_tests()

# Or, to test all configured models (can be time-consuming):
# results_all = run_self_tests(models_to_test='all', verbose=False) 
# verbose=False will give a cleaner summary table without intermediate logs
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
This library is inspired by the ease of use of VLFeat and generated with Gemini 2.5 Pro.

It leverages excellent open-source libraries such as PyTorch, Hugging Face Transformers, TIMM, and others.
