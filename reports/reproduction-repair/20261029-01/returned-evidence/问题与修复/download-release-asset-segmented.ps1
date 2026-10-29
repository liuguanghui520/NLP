param(
    [Parameter(Mandatory = $true)]
    [string]$ArchiveName,
    [int]$SegmentCount = 12,
    [string]$Attempt = "001"
)

$ErrorActionPreference = "Stop"
$assetRoot = "D:\NLPP\resource-cache\trained-artifacts-20261029"
$logRoot = "D:\NLPP\复现证据\命令与原始日志\04_第二层模型重新预测\segmented-downloads"
$catalogPath = "D:\NLPP\复现证据\独立工作副本\提交包复现副本\src\project\artifacts\reproduction-downloads.json"
$mirrorPrefix = "https://ghfast.top/"

New-Item -ItemType Directory -Force -Path $assetRoot | Out-Null
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$catalog = (Get-Content -LiteralPath $catalogPath -Raw | ConvertFrom-Json).trained_artifacts
$asset = $catalog | Where-Object { $_.archive -eq $ArchiveName }
if ($null -eq $asset) {
    throw "Unknown archive: $ArchiveName"
}

$finalPath = Join-Path $assetRoot $ArchiveName
$combinedPath = Join-Path $assetRoot "$ArchiveName.attempt-$Attempt.combined.partial"
$segmentDir = Join-Path $assetRoot "$ArchiveName.attempt-$Attempt.segments"
if (Test-Path -LiteralPath $finalPath) {
    throw "Final archive already exists: $finalPath"
}
if (Test-Path -LiteralPath $combinedPath) {
    throw "Combined partial already exists: $combinedPath"
}
if (Test-Path -LiteralPath $segmentDir) {
    throw "Segment directory already exists: $segmentDir"
}
New-Item -ItemType Directory -Path $segmentDir | Out-Null

$url = $mirrorPrefix + $asset.download_url
$totalBytes = [int64]$asset.archive_bytes
$processes = [System.Collections.Generic.List[object]]::new()
$segments = [System.Collections.Generic.List[object]]::new()
$startedAt = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss zzz")
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

for ($index = 0; $index -lt $SegmentCount; $index++) {
    $start = [int64][math]::Floor($totalBytes * $index / $SegmentCount)
    $end = [int64][math]::Floor($totalBytes * ($index + 1) / $SegmentCount) - 1
    $expectedLength = $end - $start + 1
    $prefix = "{0:D3}" -f $index
    $outputPath = Join-Path $segmentDir "$prefix.part"
    $stdoutPath = Join-Path $segmentDir "$prefix.stdout.log"
    $stderrPath = Join-Path $segmentDir "$prefix.stderr.log"
    $range = "$start-$end"
    $arguments = @(
        "-fL", "--retry", "5", "--retry-all-errors",
        "--connect-timeout", "20", "--max-time", "3600",
        "--range", $range,
        "-o", $outputPath, $url
    )
    $process = Start-Process -FilePath "curl.exe" -ArgumentList $arguments `
        -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath
    $processes.Add($process)
    $segments.Add([pscustomobject]@{
        index = $index
        range = $range
        expected_length = $expectedLength
        output = $outputPath
        stderr = $stderrPath
        process_id = $process.Id
    })
}

foreach ($process in $processes) {
    $process.WaitForExit()
}
$stopwatch.Stop()

$failed = [System.Collections.Generic.List[object]]::new()
for ($index = 0; $index -lt $segments.Count; $index++) {
    $segment = $segments[$index]
    $process = $processes[$index]
    $actualLength = if (Test-Path -LiteralPath $segment.output) {
        (Get-Item -LiteralPath $segment.output).Length
    } else {
        -1
    }
    $segment | Add-Member -NotePropertyName exit_code -NotePropertyValue $process.ExitCode
    $segment | Add-Member -NotePropertyName actual_length -NotePropertyValue $actualLength
    $segment | Add-Member -NotePropertyName length_match -NotePropertyValue ($actualLength -eq $segment.expected_length)
    if ($process.ExitCode -ne 0 -or $actualLength -ne $segment.expected_length) {
        $failed.Add($segment)
    }
}

if ($failed.Count -gt 0) {
    $status = [ordered]@{
        status = "failed"
        archive = $ArchiveName
        attempt = $Attempt
        started_at = $startedAt
        finished_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss zzz")
        duration_seconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
        segment_count = $SegmentCount
        failed_segments = $failed
        segment_dir = $segmentDir
    }
    $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $logRoot "$ArchiveName.attempt-$Attempt.status.json") -Encoding utf8
    throw "Failed segments: $($failed.index -join ', ')"
}

$destination = [System.IO.File]::Open(
    $combinedPath,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::None
)
try {
    foreach ($segment in $segments) {
        $source = [System.IO.File]::OpenRead($segment.output)
        try {
            $source.CopyTo($destination)
        } finally {
            $source.Dispose()
        }
    }
} finally {
    $destination.Dispose()
}

$actualHash = (Get-FileHash -LiteralPath $combinedPath -Algorithm SHA256).Hash.ToLowerInvariant()
$hashMatch = $actualHash -eq $asset.archive_sha256
$status = [ordered]@{
    status = if ($hashMatch) { "passed" } else { "hash_mismatch" }
    archive = $ArchiveName
    attempt = $Attempt
    source_url = $url
    started_at = $startedAt
    finished_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss zzz")
    duration_seconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
    segment_count = $SegmentCount
    expected_bytes = $totalBytes
    actual_bytes = (Get-Item -LiteralPath $combinedPath).Length
    expected_sha256 = $asset.archive_sha256
    actual_sha256 = $actualHash
    hash_match = $hashMatch
    segments = $segments
    combined_path = $combinedPath
    segment_dir = $segmentDir
}

if (-not $hashMatch) {
    $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $logRoot "$ArchiveName.attempt-$Attempt.status.json") -Encoding utf8
    throw "SHA256 mismatch for $ArchiveName"
}

Move-Item -LiteralPath $combinedPath -Destination $finalPath
$status.final_path = $finalPath
$status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $logRoot "$ArchiveName.attempt-$Attempt.status.json") -Encoding utf8
$status | ConvertTo-Json -Depth 8
