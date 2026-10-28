param([Parameter(Mandatory=$true)][string]$Output)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$targetPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $Output))
if (-not $targetPath.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Output must be inside the project.'
}
if (Test-Path -LiteralPath $targetPath) { throw 'Use a new environment snapshot directory.' }
New-Item -ItemType Directory -Path $targetPath | Out-Null
$collectedAt = [DateTimeOffset]::Now.ToString('o')
$cpu = @(Get-CimInstance Win32_Processor | Select-Object Name, Manufacturer, NumberOfCores, NumberOfLogicalProcessors)
$system = Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer, Model, TotalPhysicalMemory
$modules = @(Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity)
$os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber
$gpu = @(& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)
if ($LASTEXITCODE -ne 0) { throw 'GPU environment query failed.' }
$driverDetails = @(& nvidia-smi --version)
if ($LASTEXITCODE -ne 0) { throw 'GPU version query failed.' }
$wslPython = @'
import importlib.metadata as metadata
import json, platform, shutil, subprocess
from pathlib import Path
import torch
packages = {}
for name in ['torch','transformers','peft','scikit-learn','numpy','scipy','matplotlib','llamafactory']:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
nvcc = shutil.which('nvcc')
toolkit = subprocess.run([nvcc, '--version'], capture_output=True, text=True) if nvcc else None
print(json.dumps(dict(platform=platform.platform(), python=platform.python_version(),
    distribution=Path('/etc/os-release').read_text(), torch_cuda_build=torch.version.cuda,
    cuda_available=torch.cuda.is_available(), packages=packages,
    wsl_meminfo=Path('/proc/meminfo').read_text().splitlines()[:3],
    nvcc_path=nvcc, nvcc_version=toolkit.stdout if toolkit and toolkit.returncode==0 else None,
    toolkit_note='nvcc absent from PATH does not establish that no toolkit is installed'), ensure_ascii=True))
'@
$wslOut = @(& wsl -d Ubuntu -u lgh -- /home/lgh/.venvs/llamafactory/bin/python -c $wslPython 2> (Join-Path $targetPath 'wsl-stderr.log'))
if ($LASTEXITCODE -ne 0) { throw 'WSL software environment query failed; raw stderr retained.' }
$wslOut | Set-Content -LiteralPath (Join-Path $targetPath 'wsl-stdout.log') -Encoding utf8
$gpu | Set-Content -LiteralPath (Join-Path $targetPath 'gpu-query.log') -Encoding utf8
$driverDetails | Set-Content -LiteralPath (Join-Path $targetPath 'nvidia-version.log') -Encoding utf8
$wslJson = ($wslOut | Where-Object { $_.TrimStart().StartsWith('{') } | Select-Object -Last 1) | ConvertFrom-Json
if (-not $wslJson) { throw 'No structured WSL environment output.' }
$snapshot = [ordered]@{
    schema_version = 1
    collected_at = $collectedAt
    scope = 'Post-experiment supplementary collection; not a reconstructed benchmark-time snapshot'
    cpu = $cpu
    host = $system
    host_memory_modules = $modules
    os = $os
    gpu_csv = $gpu
    driver_version_output = $driverDetails
    wsl = $wslJson
    historical_environment_source = 'reports/audits/environment.json'
    collector = 'scripts/collect_benchmark_environment.ps1'
    training = $false
    model_loading = $false
    inference = $false
}
$snapshot | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath (Join-Path $targetPath 'environment.json') -Encoding utf8
$hashes = [ordered]@{}
Get-ChildItem -LiteralPath $targetPath -File | ForEach-Object { $hashes[$_.Name] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
$manifest = [ordered]@{
    collected_at = $collectedAt
    collector_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    historical_environment_sha256 = (Get-FileHash -LiteralPath (Join-Path $projectRoot 'reports/audits/environment.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    files_sha256 = $hashes
    note = 'CPU/GPU/WSL read-only software and hardware checks; no timing benchmark rerun'
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $targetPath 'manifest.json') -Encoding utf8
Write-Output (Join-Path $targetPath 'environment.json')
