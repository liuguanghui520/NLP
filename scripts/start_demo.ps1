param(
    [int]$Port = 8017,
    [string]$Distro = 'Ubuntu',
    [string]$WslUser = 'lgh',
    [string]$Python = '/home/lgh/.venvs/llamafactory/bin/python',
    [string]$Preload = 'nlp-qwen-lora-t01-seed42'
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskRunId = 'demo-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + ([guid]::NewGuid().ToString('N').Substring(0,6))
$taskOutput = Join-Path $taskRoot ('results/demo/' + $taskRunId)
$taskWslRoot = (& wsl -d $Distro -u $WslUser -- wslpath -a $taskRoot).Trim()
if ($LASTEXITCODE -ne 0) { throw '无法转换项目路径。' }
$taskWslOutput = $taskWslRoot + '/results/demo/' + $taskRunId
New-Item -ItemType Directory -Path $taskOutput | Out-Null
Write-Host '本地演示服务正在启动；不会重新训练。'
Write-Host "页面 http://127.0.0.1:$Port；Cherry API地址 http://127.0.0.1:$Port/v1"
Write-Host "日志 $taskOutput；Ctrl+C停止。其他成员请指定自己的WSL用户和Python环境地址。"
# The server creates its own fresh child, while stdout/stderr remain in this run.
& wsl -d $Distro -u $WslUser --cd $taskWslRoot -- $Python -u -m src.demo.server --paths configs/paths.local.json --output ($taskWslOutput + '/server') --port $Port --preload $Preload 1> (Join-Path $taskOutput 'stdout.log') 2> (Join-Path $taskOutput 'stderr.log')
if ($LASTEXITCODE -ne 0) { throw "服务退出，查看 $taskOutput 下的日志。" }
