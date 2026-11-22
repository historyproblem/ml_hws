"""Bound gzip expansion before parsing tiny, explicitly named result archives.

Only bytes of known regular files are returned. This module never extracts
paths, opens pickle payloads, imports model packages or executes archive data.
"""
from __future__ import annotations

import gzip
from io import BytesIO
from pathlib import Path
import tarfile


def read_known_tar(path, expected_names, *, require_exact=True,
                   max_decompressed_bytes=32_000_000,
                   max_member_bytes=10_000_000,
                   max_total_payload_bytes=30_000_000):
    names=tuple(expected_names)
    if (not names or len(names)!=len(set(names)) or any(
            not isinstance(n,str) or not n or n in {'.','..'}
            or '/' in n or '\\' in n or Path(n).name!=n for n in names)):
        raise ValueError('Expected archive members must be unique known basenames')
    for value in [max_decompressed_bytes,max_member_bytes,max_total_payload_bytes]:
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError('Positive integer archive byte limits required')
    # TarFile.getmembers() can consume arbitrary gzip/PAX/header expansion
    # before per-file limits run. Bound expansion first, including hidden tar
    # headers and concatenated gzip streams, then use uncompressed tar parsing.
    with gzip.open(path,'rb') as compressed:
        raw=compressed.read(max_decompressed_bytes+1)
    if len(raw)>max_decompressed_bytes:
        raise ValueError('Archive decompressed byte limit exceeded')
    allowed=set(names);files={};total=0
    with tarfile.open(fileobj=BytesIO(raw),mode='r:') as archive:
        for count,member in enumerate(archive,start=1):
            if count>len(names):
                raise ValueError('Too many archive members')
            if (member.name not in allowed or member.name in files or not member.isfile()
                    or member.linkname or member.size<0 or member.size>max_member_bytes):
                raise ValueError('Unknown, duplicate, linked, nonregular or oversized archive member')
            total+=member.size
            if total>max_total_payload_bytes:
                raise ValueError('Archive payload byte limit exceeded')
            source=archive.extractfile(member)
            if source is None:raise ValueError('Missing regular archive payload')
            value=source.read(member.size+1)
            if len(value)!=member.size:raise ValueError('Truncated archive payload')
            files[member.name]=value
        # Ignore ordinary zero tar padding, but reject concatenated hidden
        # archives/nonzero data after the first end-of-archive marker.
        if any(raw[archive.offset:]):
            raise ValueError('Unexpected nonzero data after tar end marker')
    if require_exact and set(files)!=allowed:
        raise ValueError('Missing expected archive members')
    return files
