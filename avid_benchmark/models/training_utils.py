"""Small testable training operations shared with the model adapters."""


def offset_mixco_permutation(local_permutation,previous_batches):
    return local_permutation + sum(batch.shape[0] for batch in previous_batches)


def gather_contrastive_batch(predictions, targets, permutation=None, betas=None, selected=None):
    """Gather rank-major candidates, retaining prediction gradients and MixCo indices."""
    import torch
    import torch.distributed as dist
    from torch.distributed.nn.functional import all_gather

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return predictions, targets, permutation, betas, selected
    sizes = [torch.zeros((), dtype=torch.long, device=predictions.device) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, torch.tensor(len(predictions), device=predictions.device))
    if len({int(size) for size in sizes}) != 1:
        raise ValueError('Distributed contrastive batches must have equal sizes on every rank.')

    def gather_constant(value):
        if value is None:
            return None
        parts = [torch.empty_like(value) for _ in sizes]
        dist.all_gather(parts, value.contiguous())
        return torch.cat(parts)

    if permutation is not None:
        permutation = permutation + dist.get_rank() * len(predictions)
    return (
        torch.cat(all_gather(predictions.contiguous())),
        gather_constant(targets),
        gather_constant(permutation),
        gather_constant(betas),
        gather_constant(selected),
    )
