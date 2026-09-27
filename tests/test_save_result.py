import json

import pytest

from experiments.run_benchmark import result_is_valid, save_result


def test_save_result_is_atomic_and_valid(tmp_path):
    path = tmp_path / "r.json"
    save_result({"metrics": {"accuracy": 1.0}, "records": []}, path)
    assert result_is_valid(path)
    assert not list(tmp_path.glob("*.tmp"))


def test_save_result_retries_transient_errors(tmp_path, monkeypatch):
    import experiments.run_benchmark as rb

    calls = {"n": 0}
    real_replace = rb.os.replace

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError(122, "Disk quota exceeded")
        return real_replace(src, dst)

    monkeypatch.setattr(rb.os, "replace", flaky)
    monkeypatch.setattr(rb.time, "sleep", lambda s: None)
    path = tmp_path / "r.json"
    save_result({"metrics": {}, "records": []}, path, retries=5, base_delay=0)
    assert calls["n"] == 3 and result_is_valid(path)

    calls["n"] = -100  # always fail
    with pytest.raises(OSError):
        save_result({"metrics": {}, "records": []}, tmp_path / "s.json", retries=2, base_delay=0)


def test_result_is_valid_rejects_partial_files(tmp_path):
    p = tmp_path / "partial.json"
    p.write_text('{"metrics": {"accuracy": 0.5}, "records": [')
    assert not result_is_valid(p)
    p.write_text(json.dumps({"metrics": {}}))
    assert not result_is_valid(p)  # no records
    assert not result_is_valid(tmp_path / "missing.json")
