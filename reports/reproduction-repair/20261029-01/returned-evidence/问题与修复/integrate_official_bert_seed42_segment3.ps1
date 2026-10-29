$ErrorActionPreference = "Stop"
$source = "D:\NLPP\resource-cache\trained-artifacts-20261029\bert-len512-seed42.zip.resume-004-official.segments"
$output = Join-Path $source "segment-003.part"
$expectedParts = @(
    3969297, 3969298, 3969297, 3969298,
    3969297, 3969298, 3969297, 3969298
)
if (Test-Path -LiteralPath $output) {
    throw "Output already exists: $output"
}

$destination = [System.IO.File]::Open(
    $output,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::None
)
try {
    for ($index = 0; $index -lt $expectedParts.Count; $index++) {
        $part = Join-Path $source ("{0:D3}.part" -f $index)
        $actual = (Get-Item -LiteralPath $part).Length
        if ($actual -ne $expectedParts[$index]) {
            throw "Subrange $index length mismatch: expected $($expectedParts[$index]), actual $actual"
        }
        $stream = [System.IO.File]::OpenRead($part)
        try {
            $stream.CopyTo($destination)
        } finally {
            $stream.Dispose()
        }
    }
} finally {
    $destination.Dispose()
}

$length = (Get-Item -LiteralPath $output).Length
if ($length -ne 31754380) {
    throw "Integrated segment length mismatch: $length"
}
$result = [ordered]@{
    status = "passed"
    segment = 3
    segment_range = "95263140-127017519"
    length = $length
    sha256 = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()
    output = $output
    recorded_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss zzz")
}
$result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $source "integration.status.json") -Encoding utf8
$result | ConvertTo-Json
