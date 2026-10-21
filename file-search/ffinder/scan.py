"""并行目录扫描引擎。

快的来源:
1. os.scandir():一次 readdir 同时拿到文件名和类型(Linux d_type),
   不对每个条目 stat —— 这是 os.walk / glob 的最大开销。
2. 进程池并行 + 动态任务切分:worker 扫到文件数量达到上限就提前返回,
   把没扫完的子目录交还父进程重新分发,大目录/小目录天然负载均衡。
3. 只收集路径,不做 stat:大小、修改时间等条件在查询结果产生后才处理
   (结果集很小),让建索引和查询本身尽可能快。

不跟随符号链接(符号链接本身按普通文件收录,与 locate/plocate 行为一致)。
"""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Iterator, NamedTuple, Optional, Sequence

__all__ = [
    "DEFAULT_EXCLUDES",
    "FILE_LIMIT_PER_TASK",
    "DIR_LIMIT_PER_TASK",
    "ScanResult",
    "default_workers",
    "normalize_roots",
    "scan_tree",
    "iter_walk",
]

# 默认排除的目录名:垃圾多、体积大、基本不会去搜
DEFAULT_EXCLUDES: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    ".cargo", ".gradle", ".cache", ".Trash", "lost+found",
})

# 单个任务的提前返回阈值,平衡并行度和进程间传输开销
FILE_LIMIT_PER_TASK = 20_000
DIR_LIMIT_PER_TASK = 4_000


def default_workers() -> int:
    return max(4, min(16, os.cpu_count() or 4))


class ScanResult(NamedTuple):
    files: list[str]
    dirs: list[str]
    pending: list[str]  # 因达到阈值没扫完的目录,交还给调度方重排


def normalize_roots(roots: Sequence[str]) -> list[str]:
    """展开 ~、转绝对路径、去重,并丢弃被其它根包含的路径(避免重复扫描)。"""
    out = []
    for r in roots:
        r = os.path.abspath(os.path.expanduser(r))
        if r != os.sep:
            r = r.rstrip(os.sep) or os.sep
        out.append(r)
    kept: list[str] = []
    for r in sorted(set(out)):
        if any(r.startswith(k + os.sep) for k in kept):
            continue  # 已被某个上级根覆盖
        kept.append(r)
    return kept


def _join(cur: str, name: str) -> str:
    return cur + os.sep + name if cur != os.sep else os.sep + name


def scan_tree(
    root: str,
    excludes: frozenset[str] = DEFAULT_EXCLUDES,
    file_limit: Optional[int] = FILE_LIMIT_PER_TASK,
    dir_limit: Optional[int] = DIR_LIMIT_PER_TASK,
) -> ScanResult:
    """迭代扫描 root 子树(不递归、不跟随符号链接)。

    root 本身不写入结果(由调度方负责),返回 root 之下的全部文件与目录;
    达到数量阈值时提前返回,未处理的目录放进 pending。
    """
    files: list[str] = []
    dirs: list[str] = []
    stack: deque[str] = deque((root,))
    hit = False
    cur = root
    while stack and not hit:
        cur = stack.pop()
        try:
            it = os.scandir(cur)
        except OSError:  # 无权限 / 扫描中途消失:跳过
            continue
        with it:
            for entry in it:
                name = entry.name
                path = _join(cur, name)
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:  # 类型探测失败:按文件处理
                    is_dir = False
                if is_dir:
                    if name in excludes:
                        continue
                    dirs.append(path)
                    stack.append(path)
                else:
                    files.append(path)
                if (file_limit is not None and len(files) >= file_limit) or (
                    dir_limit is not None and len(dirs) >= dir_limit
                ):
                    hit = True
                    break
    # 达到阈值:当前目录剩余条目已丢失,连同栈中未扫目录一起交还重扫。
    # (重扫会产生少量重复条目,索引端用 INSERT OR IGNORE 吸收)
    pending = ([cur] + list(stack)) if hit else []
    return ScanResult(files, dirs, pending)


def _scan_blob(root, excludes, file_limit, dir_limit):
    """scan_tree 的进程间传输封装:路径用 NUL 拼成单个字符串传。

    pickle 大 list(每元素一个小对象)的开销远大于传单个大 str;
    路径按 POSIX 规则不可能含 NUL,可安全 split。
    """
    files, dirs, pending = scan_tree(root, excludes, file_limit, dir_limit)
    return "\0".join(files), "\0".join(dirs), pending


def _seed(
    roots: list[str],
    excludes: frozenset[str],
    min_tasks: int,
) -> tuple[list[str], list[str], list[str]]:
    """把 roots 预展开成尽量多的一级任务目录,尽量均衡。

    返回 (loose_files, discovered_dirs, task_dirs):
    展开过程中遇到的文件、全部新发现目录、最终未展开的任务目录。
    roots 自身不在 discovered_dirs 里(由 build 记录)。
    """
    loose: list[str] = []
    disc: list[str] = []
    level = list(roots)
    while level and len(level) < min_tasks:
        nxt: list[str] = []
        for d in level:
            try:
                it = os.scandir(d)
            except OSError:
                continue
            with it:
                for e in it:
                    if e.name in excludes:
                        continue
                    p = _join(d, e.name)
                    try:
                        is_dir = e.is_dir(follow_symlinks=False)
                    except OSError:
                        is_dir = False
                    if is_dir:
                        nxt.append(p)
                    else:
                        loose.append(p)
        disc.extend(nxt)
        level = nxt
    return loose, disc, level


def iter_walk(
    roots: Sequence[str],
    *,
    excludes: frozenset[str] = DEFAULT_EXCLUDES,
    workers: Optional[int] = None,
) -> Iterator[tuple[list[str], list[str]]]:
    """并行扫描多个根目录,逐批 yield (files, dirs)。不含 roots 自身。"""
    roots = normalize_roots(roots)
    if workers is None:
        workers = default_workers()
    loose, disc, tasks = _seed(roots, excludes, min_tasks=workers * 6)
    if loose or disc:
        yield loose, disc

    if workers <= 1 or not tasks:
        for t in tasks:
            res = scan_tree(t, excludes, None, None)
            if res.files or res.dirs:
                yield res.files, res.dirs
        return

    fl, dl = FILE_LIMIT_PER_TASK, DIR_LIMIT_PER_TASK
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_scan_blob, t, excludes, fl, dl) for t in tasks}
        while futs:
            done, _ = wait(futs, return_when=FIRST_COMPLETED)
            for fut in done:
                futs.discard(fut)
                fblob, dblob, pending = fut.result()
                files = fblob.split("\0") if fblob else []
                dirs = dblob.split("\0") if dblob else []
                if files or dirs:
                    yield files, dirs
                for p in pending:
                    futs.add(ex.submit(_scan_blob, p, excludes, fl, dl))
