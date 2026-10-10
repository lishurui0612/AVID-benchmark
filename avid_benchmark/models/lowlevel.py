"""Restore AVID low-level architecture exactly from its checkpoint tensors."""


def validate_response_dimension(responses, expected=None):
    if responses.ndim not in (2, 3):
        raise ValueError(f'Expected [B,V] or [B,R,V], got {tuple(responses.shape)}')
    actual = int(responses.shape[-1])
    if expected is not None and actual != int(expected):
        raise ValueError(f'Low-level input expects {expected} vertices, data has {actual}')
    return actual


def load_legacy_attention_state(model, state_dict):
    """Rename old Diffusers projections only when their modern keys are expected."""
    expected = set(model.state_dict())
    restored = dict(state_dict)
    migrations = {}
    for key in state_dict:
        if '.attentions.' not in key or key in expected:
            continue
        for before, after in (('.query.', '.to_q.'), ('.key.', '.to_k.'),
                              ('.value.', '.to_v.'), ('.proj_attn.', '.to_out.0.')):
            target = key.replace(before, after)
            if target != key and target in expected:
                if target in restored:
                    raise ValueError(f'Conflicting legacy/current low-level keys: {key}, {target}')
                restored[target] = restored.pop(key)
                migrations[key] = target
                break
    model.load_state_dict(restored, strict=True)
    return migrations


def load_lowlevel_model(model_class, state_dict, input_dim=None):
    hidden_dim, checkpoint_input_dim = state_dict['lin0.0.weight'].shape
    if input_dim is not None and input_dim != checkpoint_input_dim:
        raise ValueError(f'Low-level checkpoint expects {checkpoint_input_dim} vertices, '
                         f'but target data has {input_dim}')
    channels = state_dict['norm.weight'].shape[0]
    modes = {64: '4x', 256: '8x', 512: '16x'}
    if channels not in modes:
        raise ValueError(f'Unsupported low-level decoder channels: {channels}')
    blocks = {int(key.split('.')[1]) for key in state_dict if key.startswith('mlp.')}
    if blocks != set(range(len(blocks))):
        raise ValueError('Low-level checkpoint has incomplete residual blocks')
    model = model_class(in_dim=checkpoint_input_dim, h=hidden_dim,
                        n_blocks=len(blocks), ups_mode=modes[channels],
                        use_cont=any(key.startswith('maps_projector.') for key in state_dict))
    model._avid_lowlevel_key_migrations = load_legacy_attention_state(model, state_dict)
    return model
