"""Tiny bounded-parser security checks without models or training data."""
import gzip
from io import BytesIO
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory

from safe_archives import read_known_tar


def tar_bytes(items):
    output=BytesIO()
    with tarfile.open(fileobj=output,mode='w:') as archive:
        for name,data,kind in items:
            member=tarfile.TarInfo(name);member.type=kind;member.size=len(data)
            if kind in {tarfile.SYMTYPE,tarfile.LNKTYPE}:member.linkname='../outside'
            archive.addfile(member,BytesIO(data))
    return output.getvalue()


def test_safe_known_tar_exact_subset_regular_and_no_model_deserialization():
    with TemporaryDirectory() as temp:
        path=Path(temp)/'input.tar.gz';path.write_bytes(gzip.compress(tar_bytes([
            ('a.json',b'{}',tarfile.REGTYPE),('b.csv',b'x\n1\n',tarfile.REGTYPE)])))
        assert read_known_tar(path,{'a.json','b.csv'})=={'a.json':b'{}','b.csv':b'x\n1\n'}
        assert set(read_known_tar(path,{'a.json','b.csv','c.csv'},require_exact=False))=={'a.json','b.csv'}
        try:read_known_tar(path,{'a.json','b.csv','c.csv'});raise AssertionError('Missing member accepted')
        except ValueError:pass


def test_safe_known_tar_rejects_unknown_duplicate_traversal_links_directories():
    cases=[ [('a.json',b'{}',tarfile.REGTYPE)]*2,
        [('../a.json',b'{}',tarfile.REGTYPE)], [('a.json',b'',tarfile.SYMTYPE)],
        [('a.json',b'',tarfile.LNKTYPE)], [('a.json',b'',tarfile.DIRTYPE)] ]
    with TemporaryDirectory() as temp:
        path=Path(temp)/'input.tar.gz'
        for items in cases:
            path.write_bytes(gzip.compress(tar_bytes(items)))
            try:read_known_tar(path,{'a.json'});raise AssertionError('Unsafe member accepted')
            except ValueError:pass


def test_safe_known_tar_limits_before_tarparse_and_limits_member_payload():
    with TemporaryDirectory() as temp:
        path=Path(temp)/'input.tar.gz';path.write_bytes(gzip.compress(b'not even tar '*100))
        try:read_known_tar(path,{'a.json'},max_decompressed_bytes=32);raise AssertionError('Expansion accepted')
        except ValueError as error:assert 'decompressed' in str(error)
        path.write_bytes(gzip.compress(tar_bytes([('a.json',b'12345',tarfile.REGTYPE)])))
        for kwargs in [dict(max_member_bytes=4),dict(max_total_payload_bytes=4)]:
            try:read_known_tar(path,{'a.json'},**kwargs);raise AssertionError('Byte cap accepted')
            except ValueError:pass


def test_safe_known_tar_rejects_hidden_concatenated_tar_after_padding():
    with TemporaryDirectory() as temp:
        first=tar_bytes([('a.json',b'{}',tarfile.REGTYPE)])
        hidden=tar_bytes([('hidden.pkl',b'not pickle executed',tarfile.REGTYPE)])
        path=Path(temp)/'input.tar.gz';path.write_bytes(gzip.compress(first+hidden))
        try:read_known_tar(path,{'a.json'});raise AssertionError('Hidden trailing tar accepted')
        except ValueError:pass
