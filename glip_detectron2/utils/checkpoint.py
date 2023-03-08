import torch


def load_old_glip_checkpoint(weights_path, map_location='cpu'):
    state_dict = torch.load(weights_path, map_location=map_location)['model']
    state_dict = translate_glip_dict(state_dict)
    return state_dict


def translate_glip_dict(state_dict):
    # old state dict includes cell anchors, which are not needed
    return {_state_translate(k): v for k, v in state_dict.items() if not k.startswith('module.rpn.anchor_generator.cell_anchors.')}


def _state_translate(k):
    if k.startswith('module.'):
        k = k[len('module.'):]

    if k.startswith('backbone.fpn.'):
        for i in range(2, 5):
            k = k.replace('top_blocks', 'top_block')
            k = k.replace(f'_inner{i}', f'_lateral{ i +1}')
            k = k.replace(f'_layer{i}', f'_output{ i +1}')
    elif k.startswith('backbone.body.'):
        k = k.replace('backbone.body.', 'backbone.fpn.bottom_up.')

    return k
