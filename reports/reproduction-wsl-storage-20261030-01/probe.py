import json,sys
from pathlib import Path
from unittest.mock import patch
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
from src.model_preflight import storage_check,windows_mount_path
result={"native_environment":storage_check(root,{"factory_root":"/home/lgh/LLaMA-Factory"})}
with patch("sys.prefix",str(root)):
 result["mounted_environment"]=storage_check(root,{"factory_root":"/home/lgh/LLaMA-Factory"})
result["actual_windows_mount_detected"]=windows_mount_path(root)
assert result["actual_windows_mount_detected"]
assert not result["native_environment"]["errors"]
assert result["mounted_environment"]["errors"]
result.update(status="passed",model_loading=False,inference=False,training=False)
(root/"reports/reproduction-wsl-storage-20261030-01/storage-probe.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,ensure_ascii=False))
