import io
import itertools
import tarfile
import numpy as np
import pytest
pytest.importorskip('torch')
pytest.importorskip('webdataset',minversion='0.2.73')
from avid_benchmark.data.dataset import build_webdataset


def test_one_subject_tar_is_disjoint_across_ranks(tmp_path,monkeypatch):
    path=tmp_path/'one_subject.tar'
    with tarfile.open(path,'w') as handle:
        for i in range(7):
            for name,value in [('img_name.npy',np.array(f'image_{i}.jpg')),('responses.npy',np.zeros((2,3),dtype=np.float32))]:
                buffer=io.BytesIO(); np.save(buffer,value)
                member=tarfile.TarInfo(f'{i:03d}.{name}'); member.size=len(buffer.getvalue())
                handle.addfile(member,io.BytesIO(buffer.getvalue()))
    monkeypatch.setenv('WORLD_SIZE','2')
    groups=[]
    for rank in range(2):
        monkeypatch.setenv('RANK',str(rank))
        dataset=build_webdataset(path,resampled=False,seed=42,distributed=True)
        groups.append([row[0] for row in dataset])
    assert all(groups)
    assert not set(groups[0]) & set(groups[1])
    assert len(groups[0])+len(groups[1])==7
    assert set(groups[0]+groups[1])=={f'image_{i}.jpg' for i in range(7)}
    repeated=[]
    for rank in range(2):
        monkeypatch.setenv('RANK',str(rank))
        dataset=build_webdataset(path,resampled=True,seed=42,distributed=True)
        repeated.append([row[0] for row in itertools.islice(dataset,20)])
    assert not set(repeated[0]) & set(repeated[1])
