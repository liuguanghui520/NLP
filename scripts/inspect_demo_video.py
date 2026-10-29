"""Read MP4 container metadata and SHA256 without copying or editing the video.

Container presence does not prove that speech is intelligible or frames readable.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import struct


def boxes(stream, start, end):
    position = start
    while position + 8 <= end:
        stream.seek(position)
        size, kind = struct.unpack(">I4s", stream.read(8))
        header = 8
        if size == 1:
            size = struct.unpack(">Q", stream.read(8))[0]
            header = 16
        elif size == 0:
            size = end - position
        if size < header or position + size > end:
            raise ValueError("Invalid MP4 box boundary")
        yield kind.decode("ascii", errors="replace"), position + header, position + size
        position += size


def duration(payload):
    version = payload[0]
    if version == 0:
        timescale, ticks = struct.unpack(">II", payload[12:20])
    elif version == 1:
        timescale = struct.unpack(">I", payload[20:24])[0]
        ticks = struct.unpack(">Q", payload[24:32])[0]
    else:
        raise ValueError("Unknown metadata version")
    if not timescale:
        raise ValueError("Invalid timescale")
    return {"timescale": timescale, "ticks": ticks, "seconds": ticks / timescale}


def inspect(path):
    result = {"filename": path.name, "bytes": path.stat().st_size, "tracks": []}
    with path.open("rb") as stream:
        movie = next((start, end) for kind, start, end in boxes(stream, 0, result["bytes"]) if kind == "moov")
        for kind, start, end in boxes(stream, *movie):
            if kind == "mvhd":
                stream.seek(start)
                result["movie_duration"] = duration(stream.read(min(120, end - start)))
            elif kind == "trak":
                track = {}
                for child, a, b in boxes(stream, start, end):
                    if child == "tkhd":
                        stream.seek(b - 8)
                        width, height = struct.unpack(">II", stream.read(8))
                        track.update(width=width / 65536, height=height / 65536)
                    elif child == "mdia":
                        for media, c, d in boxes(stream, a, b):
                            stream.seek(c)
                            if media == "mdhd":
                                track["duration"] = duration(stream.read(min(120, d - c)))
                            elif media == "hdlr":
                                track["handler"] = stream.read(12)[8:12].decode("ascii")
                            elif media == "minf":
                                for sub, e, f in boxes(stream, c, d):
                                    if sub != "stbl":
                                        continue
                                    for entry, g, h in boxes(stream, e, f):
                                        if entry == "stsd":
                                            stream.seek(g)
                                            value = stream.read(min(24, h - g))
                                            track["codec_fourcc"] = value[12:16].decode("ascii", errors="replace")
                result["tracks"].append(track)
    result["has_video_track"] = any(row.get("handler") == "vide" for row in result["tracks"])
    result["has_audio_track"] = any(row.get("handler") == "soun" for row in result["tracks"])
    result["duration_seconds"] = result["movie_duration"]["seconds"]
    result["within_120_seconds"] = result["duration_seconds"] <= 120
    result["size_mb_decimal"] = result["bytes"] / 1000000
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    result["sha256"] = checksum.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    stat = args.path.stat()
    result = inspect(args.path)
    assert args.path.stat().st_size == stat.st_size and args.path.stat().st_mtime_ns == stat.st_mtime_ns
    result.update(checked_at=datetime.now(timezone(timedelta(hours=8))).isoformat(), local_source=str(args.path.resolve()),
                  source_unchanged=True, method="MP4 moov/mvhd/trak/mdhd/tkhd/stsd metadata; streaming SHA256", visual_review=False, audio_listening=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
