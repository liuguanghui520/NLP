# Guard for run_bert_local.bat
# Exit 0 -> safe to start a training run
# Exit 1 -> another _bert_run.py is already running, do not start a second one
$ErrorActionPreference = 'SilentlyContinue'

$running = @(Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
    Where-Object { $_.CommandLine -like '*_bert_run.py*' })

if ($running.Count -gt 0) {
    foreach ($p in $running) {
        Write-Host ("  PID {0} is already running _bert_run.py" -f $p.ProcessId)
    }
    exit 1
}

Write-Host "  no other training process found"
exit 0
