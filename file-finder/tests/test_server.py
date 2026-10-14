"""Web 界面后端测试:起真实 HTTP 服务(自动端口),验证各 API 端点。"""

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

from ffinder import server as srv_mod
from ffinder.index import Index


@pytest.fixture()
def tree(tmp_path):
    root = tmp_path / "tree"
    for rel in [
        "projects/app1/main.py",
        "projects/app1/README.md",
        "projects/app1/src/utils/helper.py",
        "projects/app2/MAIN_LOG.txt",
        "data/reports/q1_summary.csv",
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    return root


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        body = r.read()
        ct = r.headers.get("Content-Type", "")
        return r.status, ct, body


def _post(port, path, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


@pytest.fixture()
def live(tree, tmp_path):
    db = str(tmp_path / "web.db")
    idx = Index(db)
    idx.build([str(tree)], workers=1)
    srv = srv_mod.make_server(idx, [str(tree)], host="127.0.0.1", port=0)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield port, idx, tree
    finally:
        srv.shutdown()
        srv.server_close()


def test_index_served(live):
    port, _, _ = live
    st, ct, body = _get(port, "/")
    assert st == 200 and ct.startswith("text/html")
    assert b"ffind" in body and b"<script>" in body


def test_status(live):
    port, _, tree = live
    st, _, body = _get(port, "/api/status")
    d = json.loads(body)
    assert st == 200
    assert d["files"] == 5
    assert d["roots"] == [str(tree)]
    assert d["rebuild"]["running"] is False


def test_search_and_highlights(live):
    port, _, _ = live
    st, _, body = _get(port, "/api/search?q=main&n=10")
    d = json.loads(body)
    names = sorted(h["name"] for h in d["hits"])
    assert names == ["MAIN_LOG.txt", "main.py"]  # 大小写不敏感
    assert d["total"] == 2
    assert all(h["path"] for h in d["hits"])
    # 目录类型过滤
    _, _, b2 = _get(port, "/api/search?q=app1&type=dir")
    assert all(h["dir"] for h in json.loads(b2)["hits"])


def test_search_multiword_and(live):
    port, _, _ = live
    # 按文件名匹配:main+log 只命中 MAIN_LOG.txt(排除 main.py),证明是 AND 语义
    _, _, body = _get(port, "/api/search?q=main%20log")
    d = json.loads(body)
    assert [h["name"] for h in d["hits"]] == ["MAIN_LOG.txt"]


def test_search_empty_returns_no_hits(live):
    port, _, _ = live
    _, _, body = _get(port, "/api/search?q=")
    assert json.loads(body)["hits"] == []


def test_search_scope_filter(live):
    port, _, tree = live
    scope = str(tree / "projects" / "app2")
    _, _, body = _get(port, f"/api/search?q=main&from={urllib.parse.quote(scope)}")
    names = [h["name"] for h in json.loads(body)["hits"]]
    assert names == ["MAIN_LOG.txt"]  # app1/main.py 被范围过滤掉


def test_open_rejects_out_of_root(live):
    port, _, _ = live
    st, _ = _post(port, "/api/open", {"path": "/etc/passwd"})
    assert st == 403


def test_rebuild_via_api(live):
    port, idx, tree = live
    (tree / "projects" / "brand_new_web.html").write_text("x")
    st, body = _post(port, "/api/update", {"roots": [str(tree)]})
    assert st == 200 and body["ok"]
    # 轮询到重建完成
    deadline = time.time() + 30
    while time.time() < deadline:
        _, _, b = _get(port, "/api/status")
        rb = json.loads(b)["rebuild"]
        if not rb["running"] and not rb["error"]:
            break
        time.sleep(0.1)
    else:
        pytest.fail("重建超时")
    assert json.loads(b)["files"] == 6
    _, _, b2 = _get(port, "/api/search?q=brand_new_web")
    assert [h["name"] for h in json.loads(b2)["hits"]] == ["brand_new_web.html"]


def test_rebuild_rejects_while_running(live, monkeypatch):
    """重建进行中再次 POST 应返回 409。"""
    port, idx, tree = live
    # 让 build 卡住:替换 iter_walk 为一个长时间生成器
    import ffinder.scan as scan_mod

    def slow_walk(*a, **k):
        time.sleep(1.5)
        yield [], []

    monkeypatch.setattr(scan_mod, "iter_walk", slow_walk)
    st1, _ = _post(port, "/api/update", {"roots": [str(tree)]})
    assert st1 == 200
    st2, body2 = _post(port, "/api/update", {"roots": [str(tree)]})
    assert st2 == 409 and "重建" in body2["error"]
    # 等它跑完,避免影响其它测试的关闭
    deadline = time.time() + 10
    while time.time() < deadline:
        _, _, b = _get(port, "/api/status")
        if not json.loads(b)["rebuild"]["running"]:
            break
        time.sleep(0.1)
