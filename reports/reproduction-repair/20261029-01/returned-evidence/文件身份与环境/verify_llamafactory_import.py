import sys
sys.path.insert(0,r'/home/yilin/LLaMA-Factory/src')
import llamafactory
print('llamafactory',llamafactory.__version__,llamafactory.__file__)
try:
 from llamafactory.chat import ChatModel
 print('chat_model_import=passed')
except Exception as exc:
 print('chat_model_import=failed',repr(exc))
 raise
