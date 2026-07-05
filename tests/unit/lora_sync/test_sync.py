import pytest

import lora_sync.sync as sync


def _setup(monkeypatch, *, registered, local, load_results):
    saved = []
    unloads = []
    loads = []
    load_iter = iter(load_results)

    async def fake_registered(redis):
        return registered

    def fake_get_state():
        return dict(local)

    def fake_save_state(state):
        saved.append(dict(state))

    async def fake_unload(name):
        unloads.append(name)
        return True

    def fake_download(minio, s3_path, name):
        return f"/adapters/{name}"

    async def fake_load(name, path):
        loads.append(name)
        return next(load_iter)

    monkeypatch.setattr(sync, "_get_registered_adapters", fake_registered)
    monkeypatch.setattr(sync, "_get_local_state", fake_get_state)
    monkeypatch.setattr(sync, "_save_local_state", fake_save_state)
    monkeypatch.setattr(sync, "_unload_lora", fake_unload)
    monkeypatch.setattr(sync, "_download_adapter", fake_download)
    monkeypatch.setattr(sync, "_load_lora", fake_load)
    return saved, unloads, loads


@pytest.mark.asyncio
async def test_new_adapter_loads_and_records_version(monkeypatch):
    saved, unloads, loads = _setup(
        monkeypatch,
        registered={
            "B": {"name": "B", "version": "1", "s3_path": "p", "status": "active"}
        },
        local={},
        load_results=[True],
    )
    await sync.sync_once(redis=None, minio_client=None)

    assert unloads == []  # brand new — nothing to unload
    assert loads == ["B"]
    assert saved[-1].get("B") == 1  # version recorded after a successful load


@pytest.mark.asyncio
async def test_version_bump_persists_v0_before_reload(monkeypatch):
    # A is v2 in the registry but v1 locally, and the reload's load FAILS this cycle.
    saved, unloads, loads = _setup(
        monkeypatch,
        registered={
            "A": {"name": "A", "version": "2", "s3_path": "p", "status": "active"}
        },
        local={"A": 1},
        load_results=[False],
    )
    await sync.sync_once(redis=None, minio_client=None)

    assert unloads == ["A"]  # old version unloaded once
    assert loads == ["A"]  # reload attempted
    # v0 was persisted to disk BEFORE the failed load → next cycle self-heals via the
    # load-only path instead of wedging on re-unloading an already-unloaded adapter
    assert saved[0].get("A") == 0


@pytest.mark.asyncio
async def test_unload_failure_skips_reload_without_wedge(monkeypatch):
    saved, unloads, loads = _setup(
        monkeypatch,
        registered={
            "A": {"name": "A", "version": "2", "s3_path": "p", "status": "active"}
        },
        local={"A": 1},
        load_results=[True],
    )

    async def failing_unload(name):
        unloads.append(name)
        return False

    monkeypatch.setattr(sync, "_unload_lora", failing_unload)

    await sync.sync_once(redis=None, minio_client=None)

    assert unloads == ["A"]
    assert loads == []  # skipped this cycle rather than reloading over a live adapter
