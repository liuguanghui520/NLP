import importlib.metadata
import torch

print('LLaMA-Factory:', importlib.metadata.version('llamafactory'))
print('PyTorch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA is unavailable'
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(128, 128, device='cuda')
y = x @ x.T
torch.cuda.synchronize()
assert torch.isfinite(y).all().item()
print('CUDA matrix multiplication: PASS')
import bitsandbytes
print('bitsandbytes:', bitsandbytes.__version__)
from llamafactory.webui.interface import create_ui
demo = create_ui()
demo.launch(server_name='127.0.0.1', server_port=7860, prevent_thread_lock=True, inbrowser=False)
import urllib.request
with urllib.request.urlopen('http://127.0.0.1:7860/', timeout=30) as response:
    assert response.status == 200
    print('WebUI HTTP:', response.status)
demo.close()
