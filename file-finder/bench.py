"""速度实测:ffind(建索引 + 查询) vs find(1)。

用法:
    python bench.py [--files 100000] [--dir PATH] [--keep]

在临时目录生成指定规模的文件树,预热缓存后对比:
  1. find 全量遍历        2. find 子串过滤(每次查询都要走一遍目录树)
  3. ffind 建索引(一次性) 4. ffind 查询(之后每次只碰 SQLite)
"""

import argparse
import os
import random
import shutil
import string
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffinder.index import Index  # noqa: E402

W1 = ["core", "data", "web", "api", "util", "test", "dev", "main", "app", "lib",
      "net", "db", "ui", "engine", "parser", "render", "cache", "proxy", "auth", "log"]
W2 = ["handler", "module", "config", "report", "summary", "notes", "draft", "final",
      "index", "schema", "client", "server", "image", "model", "graph", "trace"]
EXTS = [".py", ".md", ".txt", ".json", ".csv", ".cc", ".h", ".log", ".yaml", ".pdf"]


def make_tree(root: str, n_files: int, seed: int = 42):
    rng = random.Random(seed)
    # 4 层 × 扇出 8 ≈ 4000 个目录
    dirs = [""]
    for _ in range(4):
        dirs = [os.path.join(d, f"{rng.choice(W1)}_{i}")
                for d in dirs for i in range(8)]
    for d in dirs:
        os.makedirs(os.path.join(root, d), exist_ok=True)
    made = 0
    for i in range(n_files):
        d = dirs[i % len(dirs)]
        name = f"{rng.choice(W1)}_{rng.choice(W2)}_{i}{rng.choice(EXTS)}"
        p = os.path.join(root, d, name)
        with open(p, "w"):
            pass
        made += 1
    return made, len(dirs) + 1


def timeit(fn):
    t0 = time.perf_counter()
    r = fn()
    return time.perf_counter() - t0, r


def run_find(root, pat=None):
    cmd = ["find", root, "-name", f"*{pat}*"] if pat else ["find", root]
    def go():
        out = subprocess.run(cmd, capture_output=True, text=True)
        return len(out.stdout.splitlines())
    return go


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=100_000)
    ap.add_argument("--dir", default=None, help="生成树的位置(默认临时目录)")
    ap.add_argument("--keep", action="store_true", help="结束后保留文件树")
    ns = ap.parse_args()

    base = ns.dir or tempfile.mkdtemp(prefix="ffind_bench_")
    root = os.path.join(base, "tree")
    db = os.path.join(base, "bench.db")

    print(f"生成 {ns.files:,} 个文件的树于 {root} …")
    t, n = timeit(lambda: make_tree(root, ns.files))
    print(f"  完成: {n[0]:,} 文件 / {n[1]:,} 目录,用时 {t:.1f}s\n")

    # 预热 dentry/inode 缓存,双方公平
    run_find(root)()

    pats = ["parser", "db_client", "summary_final", "test_report", "auth"]
    for p in pats:  # 预热常见路径
        run_find(root, p)()

    t_find_full, c = timeit(run_find(root))
    print(f"find 全量遍历:            {t_find_full*1000:8.0f} ms   ({c:,} 条目)")

    t_find_pat = sum(timeit(run_find(root, p))[0] for p in pats) / len(pats)
    print(f"find 子串过滤(*pat*):     {t_find_pat*1000:8.0f} ms / 次查询")

    idx = Index(db)
    t_build, st = timeit(lambda: idx.build([root]))
    rate = (st["files"] + st["dirs"]) / t_build
    print(f"ffind 建索引(一次性):     {t_build*1000:8.0f} ms   ({rate:,.0f} 条目/秒)")

    # 每种查询取 5 次中位数,排除 Python GC/缓存噪声
    def bench_query(fn):
        ts = sorted(timeit(fn)[0] for _ in range(5))
        return ts[len(ts) // 2] * 1000

    t_sub = bench_query(lambda: idx.query(["summary_final"], limit=200))
    t_pre = bench_query(lambda: idx.query(["parser"], limit=200))
    t_and = bench_query(lambda: idx.query(["test", "report"], limit=200))
    t_ext = bench_query(lambda: idx.query([], exts=["pdf"], limit=200))
    print(f"ffind 子串查询:           {t_sub:8.2f} ms / 次")
    print(f"ffind 前缀查询(索引路径): {t_pre:8.2f} ms / 次")
    print(f"ffind 双关键词 AND:       {t_and:8.2f} ms / 次")
    print(f"ffind 扩展名过滤:         {t_ext:8.2f} ms / 次")
    print(f"\n子串查询提速:{t_find_pat * 1000 / t_sub:,.0f}×"
          f"   前缀查询提速:{t_find_pat * 1000 / t_pre:,.0f}×")
    print(f"一次性建索引成本 {t_build:.1f}s,可复用任意次;"
          f"每查询一次就省 {(t_find_pat - t_sub/1000)*1000:.0f} ms。")

    if ns.keep:
        print(f"\n保留于 {base}")
    else:
        shutil.rmtree(base, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
