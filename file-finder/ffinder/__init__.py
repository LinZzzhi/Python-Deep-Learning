"""ffind:以快为第一目标的文件查找器。

架构:scan(多进程 scandir 并行扫描)→ index(SQLite 覆盖索引)→ cli。
典型用法::

    from ffinder.index import Index
    idx = Index()
    idx.build(["~/code"])            # 一次
    hits, total, capped = idx.query(["report", "pdf"])   # 之后毫秒级
"""

__version__ = "0.1.0"
