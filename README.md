# Detectron2 GLIP

This is a Detectron2 implementation of "GLIP: Grounded Language-Image Pre-training" (see the [original repo](https://github.com/microsoft/GLIP)), since the original is implemented in the now deprecated `maskrcnn_benchmark` and only works with old PyTorch versions.

## Issues
 - This is still a quite **bare-bones** implementation, I focused on getting training and inference to work with the pretrained GLIP-L model, but none of the other options are supported yet.
 - Some settings are not used anymore, others from detectron2 might not be used but hardcoded instead.
 - Performance is not 100% the same as the original implementation, but it's quite close in the tests I did.
 - DropBlock is not implemented in the FPN yet.

## Usage
Download the pretrained GLIP model from the [original repo](https://github.com/microsoft/GLIP).

When in the same directory as the `setup.py` file, install the package with `pip install -e .`.

Load the model with the following code (if using a model trained with the original GLIP code):
```python
from glip_detectron2.config import cfg
from glip_detectron2.utils.checkpoint import init_glip_model

cfg = cfg.merge_from_file('config/pretrained/glip_L_finetune.yaml')
model = init_glip_model(cfg)
```
