param(
    [Parameter(Mandatory = $true)]
    [string]$ArchiveName,
    [Parameter(Mandatory = $true)]
    [string]$PrimarySegmentDir,
    [Parameter(Mandatory = $true)]
    [string]$ResumeSegmentDir,
    [string]$OverrideSegmentDir = "",
    [Parameter(Mandatory = $true)]
    [int]$SegmentCount,
    [string]$Attempt = "001"
)

$ErrorActionPreference = "Stop"
$assetRoot = "D:\NLPP\resource-cache\trained-artifacts-20261029"
$logRoot = "D:\NLPP\复现证据\命令与原始日志\04_第二层模型重新预测\segmented-downloads"
$catalogPath = "D:\NLPP\复现证据\独立工作副本\提交包复现副本\src\project\artifacts\reproduction-downloads.json"
$catalog = (Get-Content -LiteralPath $catalogPath -Raw | ConvertFrom-Json).trained_artifacts
$asset = $catalog | Where-Object { $_.archive -eq $ArchiveName }
if ($null -eq $asset) {
    throw "Unknown archive: $ArchiveName"
}

$finalPath = Join-Path $assetRoot $ArchiveName
$combinedPath = Join-Path $assetRoot "$ArchiveName.integrate-attempt-$Attempt.partial"
if (Test-Path -LiteralPath $finalPath) {
    throw "Final archive already exists: $finalPath"
}
if (Test-Path -LiteralPath $combinedPath) {
    throw "Combined output already exists: $combinedPath"
}

$totalBytes = [int64]$asset.archive_bytes
$segments = [System.Collections.Generic.List[object]]::new()
for ($index = 0; $index -lt $SegmentCount; $index++) {
    $start = [int64][math]::Floor($totalBytes * $index / $SegmentCount)
    $end = [int64][math]::Floor($totalBytes * ($index + 1) / $SegmentCount) - 1
    $expectedLength = $end - $start + 1
    $prefix = "{0:D3}" -f $index
    $primary = Join-Path $PrimarySegmentDir "$prefix.part"
    $resume = Join-Path $ResumeSegmentDir "$prefix.part"
    $override = if ($OverrideSegmentDir) { Join-Path $OverrideSegmentDir "$prefix.part" } else { $null }
    $selected = if ($override -and (Test-Path -LiteralPath $override)) {
        $override
    } elseif (Test-Path -LiteralPath $primary) {
        $primary
    } elseif (Test-Path -LiteralPath $resume) {
        $resume
    } else {
        $null
    }
    if ($null -eq $selected) {
        throw "Missing segment $index"
    }
    $actualLength = (Get-Item -LiteralPath $selected).Length
    if ($actualLength -ne $expectedLength) {
        throw "Segment $index length mismatch: expected $expectedLength, actual $actualLength"
    }
    $segments.Add([pscustomobject]@{
        index = $index
        range = "$start-$end"
        expected_length = $expectedLength
        actual_length = $actualLength
        source_path = $selected
    })
}

$destination = [System.IO.File]::Open(
    $combinedPath,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::None
)
try {
    foreach ($segment in $segments) {
        $source = [System.IO.File]::OpenRead($segment.source_path)
        try {
            $source.CopyTo($destination)
        } finally {
            $source.Dispose()
        }
    }
} finally {
    $destination.Dispose()
}

$actualBytes = (Get-Item -LiteralPath $combinedPath).Length
$actualHash = (Get-FileHash -LiteralPath $combinedPath -Algorithm SHA256).Hash.ToLowerInvariant()
$hashMatch = $actualHash -eq $asset.archive_sha256
$status = [ordered]@{
    status = if ($hashMatch -and $actualBytes -eq $totalBytes) { "passed" } else { "failed" }
    archive = $ArchiveName
    attempt = $Attempt
    recorded_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss zzz")
    segment_count = $SegmentCount
    expected_bytes = $totalBytes
    actual_bytes = $actualBytes
    expected_sha256 = $asset.archive_sha256
    actual_sha256 = $actualHash
    hash_match = $hashMatch
    primary_segment_dir = $PrimarySegmentDir
    resume_segment_dir = $ResumeSegmentDir
    override_segment_dir = $OverrideSegmentDir
    segments = $segments
    combined_path_before_publish = $combinedPath
}

$statusPath = Join-Path $logRoot "$ArchiveName.integration-attempt-$Attempt.status.json"
if (-not ($hashMatch -and $actualBytes -eq $totalBytes)) {
    $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding utf8
    throw "Integrated archive failed byte length or SHA256 validation"
}

Move-Item -LiteralPath $combinedPath -Destination $finalPath
$status.final_path = $finalPath
$status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding utf8
$status | ConvertTo-Json -Depth 8
