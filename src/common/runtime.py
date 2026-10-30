"""Read-only numerical-runtime evidence; never changes backend settings."""
import os
import platform


def tensor_sha256(tensor, torch):
    import hashlib
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def runtime_snapshot(torch, model=None):
    snapshot = dict(python=platform.python_version(), platform=platform.platform(),
                    torch=str(torch.__version__), cuda_runtime=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(), threads=torch.get_num_threads(),
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    environment={key: os.environ.get(key) for key in (
                        'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE', 'CUDA_VISIBLE_DEVICES',
                        'CUBLASLT_WORKSPACE_SIZE', 'PYTORCH_CUDA_ALLOC_CONF')})
    snapshot['backend_flags'] = {}
    for group, names in [(torch.backends.cuda.matmul, (
            'allow_tf32', 'allow_fp16_reduced_precision_reduction', 'allow_bf16_reduced_precision_reduction')),
            (torch.backends.cudnn, ('benchmark', 'deterministic', 'allow_tf32'))]:
        prefix = 'matmul' if group is torch.backends.cuda.matmul else 'cudnn'
        for name in names:
            snapshot['backend_flags'][prefix + '.' + name] = getattr(group, name, None)
    for name in ('flash_sdp_enabled', 'mem_efficient_sdp_enabled', 'math_sdp_enabled',
                 'cudnn_sdp_enabled'):
        value = getattr(torch.backends.cuda, name, None)
        snapshot['backend_flags'][name] = value() if callable(value) else None
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        snapshot['gpu'] = dict(name=prop.name, capability=list(torch.cuda.get_device_capability()),
                               multiprocessors=prop.multi_processor_count, total_memory=prop.total_memory)
        free, total = torch.cuda.mem_get_info()
        snapshot['gpu_memory'] = dict(free_bytes=free, total_bytes=total,
            process_allocated_bytes=torch.cuda.memory_allocated(),
            process_reserved_bytes=torch.cuda.memory_reserved())
    if model is not None:
        snapshot['attention_implementation'] = getattr(model.config, '_attn_implementation', None)
        snapshot['parameter_dtypes'] = sorted({str(p.dtype) for p in model.parameters()})
        snapshot['buffer_fingerprints'] = {
            name: dict(dtype=str(value.dtype), shape=list(value.shape), sha256=tensor_sha256(value, torch))
            for name, value in model.named_buffers()}
    return snapshot
