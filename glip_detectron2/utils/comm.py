from detectron2.utils.comm import get_world_size, dist


def reduce_sum(tensor):
    if get_world_size() <= 1:
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
