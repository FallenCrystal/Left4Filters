# Model Weights & Third-Party Notices

## Code License
All source code, training scripts, feature extraction utilities, fusion head architecture, and deployment wrappers in this repository are licensed under the **MIT License** (see [`LICENSE`](LICENSE)).

## Pretrained Visual Backbone Attribution (`deploy/vision_encoder.onnx`)
The visual encoder weights shipped in `deploy/vision_encoder.onnx` are fine-tuned and reparameterized from **Apple MobileCLIP2-S0** (`apple/MobileCLIP2-S0`, pretrained on DFNDR-2B):

- **Upstream Repository**: https://huggingface.co/apple/MobileCLIP2-S0
- **Upstream Codebase**: https://github.com/apple/ml-mobileclip
- **Upstream Model License**: **Apple Machine Learning Research Model License (`apple-amlr`)**
- **Reference**: Faghri et al., *"MobileCLIP2: Improving Multi-Modal Reinforced Training"*, TMLR 2025 (arXiv:2508.20691).

Any redistribution or commercial/research use of `deploy/vision_encoder.onnx` must comply with the terms of the **Apple Machine Learning Research Model License (`apple-amlr`)**.

## Data Source Attribution
Training and evaluation samples were obtained via the public REST catalog cache API of **[Left4Addons](https://github.com/FallenCrystal/Left4Addons)** (`l4a-server`), indexing publicly listed Steam Workshop items for *Left 4 Dead 2* (AppID 550). No raw Steam Workshop media assets, creator identifiers, or private backend endpoints are redistributed in this repository.
