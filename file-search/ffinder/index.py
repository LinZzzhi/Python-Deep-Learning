"""SQLite 文件名索引:一次构建,毫秒查询。

快的来源:
- 查询只碰索引文件,不再遍历磁盘;
- WITHOUT ROWID 表 + (name, path) 覆盖索引:子串全扫时只读索引页,不回表;
- 前缀匹配(name LIKE 'x%')走索引范围扫描,毫秒内完成,不够再补子串扫;
- 建索引期间 journal OFF + 批量 executemany,单事务落盘。

查询语法:多个关键字为「同时包含」(AND);默认 ASCII 大小写不敏感。
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import sqlite3
import time
from typing import Optional, Sequence

from . import scan

__all__ = ["Index", "default_db_path"]

_CAP = 20_000  # 单次查询最多取回并排序的候选数,防止极端模式吃爆内存
_E = " ESCAPE '\\'"


def default_db_path() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "ffind", "index.sqlite")


def _esc_like(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _basename(p: str) -> str:
    b = os.path.basename(p)
    return b or p


def _glob_re(glob: str):
    if not any(c in glob for c in "*?["):
        glob = f"*{glob}*"  # 没写通配符时按子串语义处理
    return re.compile(fnmatch.translate(glob))


class Index:
    """文件路径索引。path 为 None 时用 ~/.cache/ffind/index.sqlite。"""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = os.path.abspath(os.path.expanduser(path or default_db_path()))

    # ---------- 连接 ----------

    def _connect(self, path: Optional[str] = None) -> sqlite3.Connection:
        p = path or self.path
        parent = os.path.dirname(p)
        if parent:
            os.makedirs(parent, exist_ok=True)
        con = sqlite3.connect(p, timeout=15)
        con.execute("PRAGMA synchronous=OFF")
        con.execute("PRAGMA mmap_size=268435456")  # 256MB:查询走内存映射,省 read syscall
        con.execute("PRAGMA cache_size=-16384")    # 16MB 页缓存
        return con

    def exists(self) -> bool:
        if not os.path.exists(self.path):
            return False
        try:
            con = self._connect()
            try:
                return (
                    con.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='files'"
                    ).fetchone()
                    is not None
                )
            finally:
                con.close()
        except sqlite3.Error:
            return False

    # ---------- 元信息 ----------

    def stats(self) -> dict:
        out: dict = {
            "path": self.path, "size": 0, "built_at": 0.0,
            "roots": [], "excludes": set(), "files": 0, "dirs": 0,
        }
        if not os.path.exists(self.path):
            # 尚无索引:存储的排除集视为默认(自动重建路径依赖这个语义)
            out["excludes"] = set(scan.DEFAULT_EXCLUDES)
            return out
        out["size"] = os.path.getsize(self.path)
        raw: dict[str, str] = {}
        try:
            con = self._connect()
            try:
                for k, v in con.execute("SELECT key, value FROM meta"):
                    raw[k] = v
            finally:
                con.close()
        except sqlite3.Error:
            return out
        try:
            out["roots"] = json.loads(raw.get("roots", "[]"))
        except ValueError:
            pass
        ex = raw.get("excludes")
        if ex is not None:
            try:
                out["excludes"] = set(json.loads(ex))
            except ValueError:
                out["excludes"] = set(scan.DEFAULT_EXCLUDES)
        else:
            out["excludes"] = set(scan.DEFAULT_EXCLUDES)
        out["built_at"] = float(raw.get("built_at", 0) or 0)
        out["files"] = int(raw.get("n_files", 0) or 0)
        out["dirs"] = int(raw.get("n_dirs", 0) or 0)
        return out

    # ---------- 删除 ----------

    def delete(self) -> dict:
        """删除索引文件及其附属文件(-wal/-shm/-journal/残留 .tmp)。

        连接都是用完即关,没有常驻句柄,所以直接 unlink 即可。
        先删半成品再删主库:重建残留的 .tmp 删不掉时,主索引仍完好可查。
        返回 {"freed": 释放字节数, "removed": [被删的路径]}。
        """
        freed = 0
        removed: list[str] = []
        for p in (
            self.path + ".tmp",
            self.path + "-journal",
            self.path + "-wal",
            self.path + "-shm",
            self.path,
        ):
            try:
                size = os.path.getsize(p)
            except OSError:
                continue  # 文件本来就不存在
            try:
                os.remove(p)
            except FileNotFoundError:
                continue
            except PermissionError as e:  # Windows 下被搜索/重建占用
                raise PermissionError(f"索引文件被占用,暂时无法删除: {p}") from e
            freed += size
            removed.append(p)
        return {"freed": freed, "removed": removed}

    def stored_roots(self) -> list[str]:
        return self.stats()["roots"]

    def stored_excludes(self) -> frozenset[str]:
        return frozenset(self.stats()["excludes"])

    def age(self) -> float:
        """距上次建索引的秒数;无索引时返回 inf。"""
        st = self.stats()
        if not st["built_at"]:
            return float("inf")
        return max(0.0, time.time() - st["built_at"])

    # ---------- 建索引 ----------

    def build(
        self,
        roots: Sequence[str],
        *,
        excludes: Optional[frozenset[str]] = None,
        workers: Optional[int] = None,
        progress=None,
    ) -> dict:
        """全量重建索引。progress(files, dirs, seconds) 逐批回调。"""
        excl = scan.DEFAULT_EXCLUDES if excludes is None else frozenset(excludes)
        roots = scan.normalize_roots(roots)
        if not roots:
            raise ValueError("没有有效的扫描根目录")
        t0 = time.perf_counter()
        # 先建到临时库、成功后原子替换:重建被中断(Ctrl-C 等)时旧索引
        # 完好可查,也让激进的 journal_mode=OFF 变得安全
        tmp = self.path + ".tmp"
        con = self._connect(tmp)
        try:
            con.executescript(
                """
                PRAGMA journal_mode=OFF;
                DROP TABLE IF EXISTS files;
                DROP TABLE IF EXISTS meta;
                CREATE TABLE files(
                    path TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE,
                    is_dir INTEGER NOT NULL DEFAULT 0
                ) WITHOUT ROWID;
                CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                """
            )
            con.execute("BEGIN")
            ins = "INSERT OR IGNORE INTO files(path, name, is_dir) VALUES(?,?,?)"
            seen = [0, 0]

            def put(paths: list[str], is_dir: int) -> None:
                if not paths:
                    return
                con.executemany(ins, ((p, _basename(p), is_dir) for p in paths))
                seen[is_dir] += len(paths)

            put(list(roots), 1)
            for files, dirs in scan.iter_walk(
                roots, excludes=excl, workers=workers
            ):
                put(files, 0)
                put(dirs, 1)
                if progress is not None:
                    progress(seen[0], seen[1], time.perf_counter() - t0)
            con.commit()
            # 数据全部落盘后再一次性建索引:退化为一次排序,远快于每行增量维护
            con.execute("BEGIN")
            con.execute("CREATE INDEX IF NOT EXISTS files_name ON files(name, path)")
            con.commit()
            # 精确计数(重扫批次可能有重复,以库内为准)
            n_f = n_d = 0
            for d, c in con.execute(
                "SELECT is_dir, COUNT(*) FROM files GROUP BY is_dir"
            ):
                if d:
                    n_d = c
                else:
                    n_f = c
            for k, v in (
                ("roots", json.dumps(roots)),
                ("excludes", json.dumps(sorted(excl))),
                ("built_at", str(time.time())),
                ("n_files", str(n_f)),
                ("n_dirs", str(n_d)),
            ):
                con.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, v))
            con.commit()
        except BaseException:
            con.close()
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        con.close()
        os.replace(tmp, self.path)
        return {
            "files": n_f,
            "dirs": n_d,
            "seconds": time.perf_counter() - t0,
            "roots": roots,
            "db": self.path,
        }

    # ---------- 查询 ----------

    def query(
        self,
        patterns: Sequence[str] = (),
        *,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
        exts: Sequence[str] = (),
        type_: str = "any",  # any | file | dir
        roots: Sequence[str] = (),
        excludes: Sequence[str] = (),
        case_sensitive: bool = False,
        limit: int = 200,
    ) -> tuple[list[tuple[str, bool]], int, bool]:
        """返回 (hits, total, capped)。hits=[(path, is_dir)] 按相关度排序。"""
        patterns = [p for p in patterns if p]
        if not self.exists():
            return [], 0, False
        con = self._connect()
        try:
            base: list[str] = []
            bparams: list[object] = []
            if type_ == "file":
                base.append("is_dir=0")
            elif type_ == "dir":
                base.append("is_dir=1")
            if exts:
                es = [_esc_like(e.strip().lstrip(".").lower()) for e in exts if e.strip(".")]
                if es:
                    base.append("(" + " OR ".join(["name LIKE ?" + _E] * len(es)) + ")")
                    bparams += ["%." + e for e in es]
            if roots:
                frags = []
                for r in scan.normalize_roots(roots):
                    frags.append("(path=? OR path LIKE ?" + _E + ")")
                    bparams += [r, _esc_like(r + os.sep) + "%"]
                base.append("(" + " OR ".join(frags) + ")")

            mid_c = ["name LIKE ?" + _E] * len(patterns)
            mid_p = ["%" + _esc_like(p) + "%" for p in patterns]
            pre_c = ["name LIKE ?" + _E] * len(patterns)
            pre_p = [_esc_like(p) + "%" for p in patterns]

            py_filter = bool(regex or glob or excludes or case_sensitive)
            rows: list[tuple[str, str, int]] = []
            capped = False

            if patterns and not py_filter:
                # 快路径:前缀匹配走 (name,path) 索引,几乎瞬时;
                # 不够 limit 条再补一次子串全扫,合并去重。
                seen_paths: set[str] = set()
                for r in self._iter(con, base + pre_c, bparams + pre_p,
                                    min(_CAP, max(limit * 4, 80))):
                    seen_paths.add(r[0])
                    rows.append(r)
                if len(rows) < limit:
                    for r in self._iter(con, base + mid_c, bparams + mid_p, _CAP):
                        if r[0] in seen_paths:
                            continue
                        rows.append(r)
                        if len(rows) >= _CAP:
                            capped = True
                            break
            else:
                rx = re.compile(regex) if regex else None
                gx = _glob_re(glob) if glob else None
                exl = set(excludes) if excludes else None
                # SQL 层不设 LIMIT(截断会漏结果);流式扫描,匹配数到上限即停
                for path, name, isd in self._iter(con, base + mid_c, bparams + mid_p):
                    if rx is not None and not rx.search(name):
                        continue
                    if gx is not None and not gx.match(name):
                        continue
                    if case_sensitive and any(p not in name for p in patterns):
                        continue
                    if exl and (set(path.split(os.sep)) & exl):
                        continue
                    rows.append((path, name, isd))
                    if len(rows) >= _CAP:
                        capped = True
                        break
            total = len(rows)
            pats_l = [p.lower() for p in patterns]
            rows.sort(key=lambda r: (-_score(r[0], r[1], pats_l), r[0]))
            hits = [(p, bool(d)) for p, _, d in rows[:limit]]
            return hits, total, capped
        finally:
            con.close()

    @staticmethod
    def _iter(con, conds, params, lim: Optional[int] = None):
        where = " AND ".join(conds) if conds else "1=1"
        sql = f"SELECT path, name, is_dir FROM files WHERE {where}"
        p = list(params)
        if lim is not None:
            sql += " LIMIT ?"
            p.append(lim)
        cur = con.execute(sql, p)
        while True:
            chunk = cur.fetchmany(20_000)
            if not chunk:
                return
            yield from chunk


def _score(path: str, name: str, pats_l: list[str]) -> int:
    """简单相关度:精确=前缀>词首>任意位置;路径越短越靠前。"""
    s = 0
    nl = name.lower()
    for p in pats_l:
        i = nl.find(p)
        if i < 0:
            continue
        if i == 0:
            s += 600 if nl == p else 150
        else:
            s += 80 if nl[i - 1] in "-_. " else 25
            s -= min(i, 60)
    return s - len(path) // 60
