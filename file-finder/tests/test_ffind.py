"""正确性测试:扫描引擎 + 索引查询 + CLI 冒烟。"""

import os

import pytest

from ffinder import scan
from ffinder.cli import main
from ffinder.index import Index


@pytest.fixture()
def tree(tmp_path):
    """构造一棵混合目录树,含应被默认排除的垃圾目录。"""
    root = tmp_path / "tree"
    files = [
        "projects/app1/README.md",
        "projects/app1/src/main.py",
        "projects/app1/src/utils/helper.py",
        "projects/app1/docs/report_2024.pdf",
        "projects/app1/node_modules/pkg/index.js",      # 默认排除
        "projects/app1/.git/config",                    # 默认排除
        "projects/app1/.cache/junk.bin",                # 默认排除(.cache)
        "projects/app2/Notes.txt",
        "projects/app2/src/MAIN_LOG.txt",
        "data/reports/q1_summary.csv",
        "data/reports/q2_summary.csv",
        "data/reports/annual_report.pdf",
    ]
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    # 若干小文件,便于测 limit
    for i in range(50):
        (root / "data" / f"item_{i:03d}.bin").write_text("x")
    return root


@pytest.fixture()
def idx(tree, tmp_path):
    ix = Index(str(tmp_path / "test.db"))
    st = ix.build([str(tree)], workers=1)
    assert st["files"] == 59  # 9 + 50,不含 node_modules/.git/.cache
    return ix


def paths(hits):
    return [h[0] for h in hits]


def name_of(ix, hits):
    return sorted(os.path.basename(p) for p in paths(hits))


# ---------- scan ----------


def test_normalize_roots(tmp_path):
    a = str(tmp_path / "a")
    os.makedirs(a + "/b", exist_ok=True)
    out = scan.normalize_roots([a, a + "/b", "~", a + "/b/../"])
    assert out == sorted([os.path.expanduser("~"), a])  # 嵌套根被去重


def test_scan_tree_no_stat(tmp_path):
    (tmp_path / "f1").write_text("x")
    (tmp_path / "d1").mkdir()
    (tmp_path / "d1" / "f2").write_text("x")
    res = scan.scan_tree(str(tmp_path))
    assert set(res.files) == {str(tmp_path / "f1"), str(tmp_path / "d1" / "f2")}
    assert res.dirs == [str(tmp_path / "d1")]
    assert res.pending == []


def test_scan_tree_pending_split(tmp_path):
    for i in range(10):
        (tmp_path / f"f{i}").write_text("x")
    res = scan.scan_tree(str(tmp_path), file_limit=4)
    assert len(res.files) >= 4
    assert res.pending  # 提前返回并把剩余目录交还
    # 重扫 pending 能补齐剩余文件
    again = scan.scan_tree(res.pending[0], file_limit=None)
    assert len(res.files) + len(again.files) >= 10


def test_parallel_workers(tree):
    seen_files = []
    for files, _ in scan.iter_walk([str(tree)], workers=2):
        seen_files.extend(files)
    assert len([f for f in seen_files if "node_modules" not in f]) == 59


# ---------- query ----------


def test_substring_and_case(idx):
    hits, total, _ = idx.query(["main"])
    assert name_of(idx, hits) == ["MAIN_LOG.txt", "main.py"]
    hits, _, _ = idx.query(["README"])  # 大小写不敏感
    assert name_of(idx, hits) == ["README.md"]


def test_multi_pattern_and(idx):
    hits, _, _ = idx.query(["report", "2024"])
    assert name_of(idx, hits) == ["report_2024.pdf"]
    hits, _, _ = idx.query(["summary", "q2"])
    assert name_of(idx, hits) == ["q2_summary.csv"]


def test_prefix_ranks_first(idx):
    # 新建一个 summary.py 与 report_summary.txt,前缀命中应排最前
    hits, _, _ = idx.query(["report"])
    first = os.path.basename(hits[0][0])
    assert first.startswith("report")  # report_2024.pdf


def test_type_and_ext_and_roots(idx, tree):
    hits, _, _ = idx.query(["app1"], type_="dir")
    assert all(d for _, d in hits) and hits
    hits, _, _ = idx.query([], exts=["py"])
    assert name_of(idx, hits) == ["helper.py", "main.py"]
    # roots 限定:限定到 projects 之外时搜不到 main.py
    hits, _, _ = idx.query(["main"], roots=[str(tree / "data")])
    assert hits == []
    hits, _, _ = idx.query(["main"], roots=[str(tree / "projects" / "app1")])
    assert name_of(idx, hits) == ["main.py"]


def test_regex_glob_case_sensitive(idx):
    hits, _, _ = idx.query([], regex=r"^q._summary\.csv$")
    assert name_of(idx, hits) == ["q1_summary.csv", "q2_summary.csv"]
    hits, _, _ = idx.query([], glob="*.pdf")
    assert name_of(idx, hits) == ["annual_report.pdf", "report_2024.pdf"]
    hits, _, _ = idx.query(["main"], case_sensitive=True)
    assert name_of(idx, hits) == ["main.py"]  # MAIN_LOG 不命中(大小写敏感且非子串? 'main' 不在 'MAIN_LOG.txt')
    hits, _, _ = idx.query(["MAIN"], case_sensitive=True)
    assert name_of(idx, hits) == ["MAIN_LOG.txt"]


def test_exclude_and_limit(idx):
    hits, total, _ = idx.query(["item"])
    assert total == 50 and len(hits) == 50
    hits, total, _ = idx.query(["item"], limit=10)
    assert len(hits) == 10 and total == 50
    hits, _, _ = idx.query(["item"], excludes=["data"])
    assert hits == []  # 查询期按目录名过滤


def test_empty_index(tmp_path):
    ix = Index(str(tmp_path / "none.db"))
    assert ix.query(["x"]) == ([], 0, False)
    assert ix.age() == float("inf")


def test_rebuild_picks_up_new_file(tree, idx):
    (tree / "projects" / "brand_new_file.dat").write_text("x")
    st = idx.build([str(tree)], workers=1)
    assert st["files"] == 60
    hits, _, _ = idx.query(["brand_new"])
    assert name_of(idx, hits) == ["brand_new_file.dat"]
    assert not os.path.exists(idx.path + ".tmp")  # 原子替换后不留临时库


def test_cache_dir_excluded(idx):
    assert idx.query(["junk"])[0] == []  # .cache 属默认排除


def test_failed_rebuild_keeps_old_index(tree, idx, monkeypatch):
    """建索引中途被打断:旧索引必须完好可查,临时库被清理。"""

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(scan, "iter_walk", boom)
    with pytest.raises(KeyboardInterrupt):
        idx.build([str(tree)], workers=1)
    assert not os.path.exists(idx.path + ".tmp")
    hits, _, _ = idx.query(["main"])  # 旧索引仍可用
    assert name_of(idx, hits) == ["MAIN_LOG.txt", "main.py"]


def test_like_metachars_literal(idx):
    # 模式里的 % _ 按字面处理:'q1_' 中 _ 不是通配
    hits, _, _ = idx.query(["q1_"])
    assert name_of(idx, hits) == ["q1_summary.csv"]
    hits, _, _ = idx.query(["1x_summary"])  # 若 _ 被当通配就会误命中
    assert hits == []


# ---------- CLI ----------


def test_cli_smoke(tmp_path, tree, capsys):
    db = str(tmp_path / "cli.db")
    rc = main(["update", str(tree), "--db", db, "-j", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "已索引 59 文件" in out
    rc = main(["report", "pdf", "--db", db])
    lines = capsys.readouterr().out.strip().splitlines()
    # AND 语义:两条都含 report+pdf;前缀命中的排最前
    assert [os.path.basename(x) for x in lines] == ["report_2024.pdf", "annual_report.pdf"]
    # 无命中 → 退出码 1
    assert main(["zzz_no_such_thing", "--db", db]) == 1
    capsys.readouterr()
    assert main(["-c", "item", "--db", db]) == 0
    assert capsys.readouterr().out.strip() == "50"
    assert main(["status", "--db", db]) == 0
    assert str(tree) in capsys.readouterr().out
