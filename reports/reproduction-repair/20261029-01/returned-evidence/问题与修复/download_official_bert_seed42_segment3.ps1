param(
    [string]$Attempt = "001"
)

$ErrorActionPreference = "Stop"
$logDir = "D:\NLPP\复现证据\命令与原始日志\04_第二层模型重新预测"
$target = "D:\NLPP\resource-cache\trained-artifacts-20261029\bert-len512-seed42.zip.resume-004-official.segments"
$url = "https://github.com/liuguanghui520/NLP/releases/download/trained-artifacts-20261029-01/bert-len512-seed42.zip"
$segmentStart = [int64]95263140
$segmentLength = [int64]31754380
$partCount = 8

if (Test-Path -LiteralPath $target) {
    throw "Target already exists: $target"
}
New-Item -ItemType Directory -Path $target | Out-Null

$processes = [System.Collections.Generic.List[object]]::new()
$ranges = [System.Collections.Generic.List[object]]::new()
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

for ($index = 0; $index -lt $partCount; $index++) {
    $start = $segmentStart + [int64][math]::Floor($segmentLength * $index / $partCount)
    $end = $segmentStart + [int64][math]::Floor($segmentLength * ($index + 1) / $partCount) - 1
    $prefix = "{0:D3}" -f $index
    $output = Join-Path $target "$prefix.part"
    $range = "$start-$end"
    $arguments = @(
        "-fL", "--retry", "5", "--retry-all-errors", "--retry-delay", "5",
        "--connect-timeout", "20", "--max-time", "1200",
        "--range", $range, "-o", $output, $url
    )
    $processes.Add((Start-Process -FilePath "curl.exe" -ArgumentList $arguments `
        -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $target "$prefix.stdout.log") `
        -RedirectStandardError (Join-Path $target "$prefix.stderr.log")))
    $ranges.Add([pscustomobject]@{
        index = $index
        range = $range
        expected_length = $end - $start + 1
        path = $output
    })
}

foreach ($process in $processes) {
    $process.WaitForExit()
}
$stopwatch.Stop()

$bad = [System.Collections.Generic.List[int]]::new()
for ($index = 0; $index -lt $ranges.Count; $index++) {
    $actual = if (Test-Path -LiteralPath $ranges[$index].path) {
        (Get-Item -LiteralPath $ranges[$index].path).Length
    } else {
        -1
    }
    $ranges[$index] | Add-Member -NotePropertyName exit_code -NotePropertyValue $processes[$index].ExitCode
    $ranges[$index] | Add-Member -NotePropertyName actual_length -NotePropertyValue $actual
    if ($processes[$index].ExitCode -ne 0 -or $actual -ne $ranges[$index].expected_length) {
        $bad.Add($index)
    }
}

if ($bad.Count -gt 0) {
    [ordered]@{
        status = "failed"
        attempt = $Attempt
        failed_subranges = $bad
        ranges = $ranges
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $logDir "bert-seed42-resume-004-official-$Attempt.failed.json") -Encoding utf8
    throw "Failed official subranges: $($bad -join ',')"
}

$combined = Join-Path $target "003.part"
$destination = [System.IO.File]::Open(
    $combined,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::None
)
try {
    foreach ($range in $ranges) {
        $source = [System.IO.File]::OpenRead($range.path)
        try {
            $source.CopyTo($destination)
        } finally {
            $source.Dispose()
        }
    }
} finally {
    $destination.Dispose()
}

$old = "D:\NLPP\resource-cache\trained-artifacts-20261029\bert-len512-seed42.zip.attempt-002.segments\003.part"
$status = [ordered]@{
    status = "passed"
    attempt = $Attempt
    recorded_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss zzz")
    duration_seconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
    segment = 3
    segment_range = "95263140-127017519"
    segment_length = (Get-Item -LiteralPath $combined).Length
    official_segment_sha256 = (Get-FileHash -LiteralPath $combined -Algorithm SHA256).Hash.ToLowerInvariant()
    old_mirror_segment_sha256 = (Get-FileHash -LiteralPath $old -Algorithm SHA256).Hash.ToLowerInvariant()
    subranges = $ranges
    output = $combined
}
$status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $logDir "bert-seed42-resume-004-official-$Attempt.status.json") -Encoding utf8
$status | ConvertTo-Json -Depth 6
