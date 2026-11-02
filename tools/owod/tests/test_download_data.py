import hashlib
import json

import pytest

from tools.owod import download_data as module


def test_voc_reuses_verified_manual_archive_and_preserves_bad_file(tmp_path, monkeypatch):
    target = tmp_path / 'release.tar'
    target.write_bytes(b'original archive')
    expected = hashlib.md5(target.read_bytes()).hexdigest()
    monkeypatch.setattr(module, '_download', lambda *_: pytest.fail('Unexpected network request'))
    assert module.download_voc('https://example.org/release.tar', target, expected) == target
    record = json.loads(target.with_suffix('.tar.source.json').read_text())
    assert record['sha256'] == module.sha256(target)
    assert record['md5'] == expected
    target.with_suffix('.tar.source.json').unlink()
    with pytest.raises(ValueError, match='MD5 mismatch'):
        module.download_voc('https://example.org/release.tar', target, 'wrong')
    assert target.read_bytes() == b'original archive'


@pytest.mark.parametrize('benchmark,expected_voc,expected_coco,expected_weights', [
    ('voc', 3, 0, 0), ('m-owodb', 3, 3, 1), ('both', 3, 3, 1), ('s-owodb', 0, 3, 1),
])
def test_download_selection_and_coco_extraction(tmp_path, monkeypatch, benchmark,
                                               expected_voc, expected_coco, expected_weights):
    calls, extracted = [], []
    def download(url, target, proxy=None):
        calls.append(('standard', url, target))
        return target
    def voc(url, target, md5, proxy=None):
        calls.append(('voc', url, target))
        assert len(md5) == 32
        return target
    monkeypatch.setattr(module, 'download', download)
    monkeypatch.setattr(module, 'download_voc', voc)
    monkeypatch.setattr(module, 'extract', lambda archive, root: extracted.append((archive, root)))
    assert module.main(['--benchmark', benchmark, '--m-source', 'reconstruct',
                        '--root', str(tmp_path), '--weights', str(tmp_path / 'dino.pth')]) == 0
    assert sum(kind == 'voc' for kind, _, _ in calls) == expected_voc
    assert sum(url.startswith(module.COCO) for _, url, _ in calls) == expected_coco
    assert sum(url == module.DINO for _, url, _ in calls) == expected_weights
    assert len(extracted) == expected_coco
    assert all(root == tmp_path / 'coco' for _, root in extracted)


def test_download_failure_stops_readiness(tmp_path, monkeypatch, capsys):
    def fail(*_):
        raise ValueError('fixture: bad VOC archive')
    monkeypatch.setattr(module, 'download_voc', fail)
    assert module.main(['--benchmark', 'voc', '--root', str(tmp_path)]) == 1
    assert 'bad VOC archive' in capsys.readouterr().err
