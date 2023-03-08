# Detectron2 GLIP

This is a Detectron2 implementation of "GLIP: Grounded Language-Image Pre-training" (see the [original repo](https://github.com/microsoft/GLIP)).

## Issues
 - This is still a quite bare-bones implementation, I focused on getting training and inference to work with the pretrained GLIP-L model, but none of the other options are supported yet.
 - Some settings are not used anymore, others from detectron2 might not be used but hardcoded instead.
 - Performance is not 100% the same as the original implementation, but it's quite close in the tests I did.
 - DropBlock is not implemented in the FPN yet.
