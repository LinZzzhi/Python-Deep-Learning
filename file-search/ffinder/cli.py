"""ffind 命令行。

子命令:
    ffind <关键词>...          搜索(可带正则/通配/扩展名等过滤)
    ffind update [目录]...     并行扫描建索引(全量重建)
    ffind status               查看索引状态
    ffind serve                启动本地 Web 界面(浏览器实时搜索)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from . import __version__, scan
from .index import Index

_HOME = os.path.expanduser("~")

_EPILOG = """\
示例:
  ffind update ~/code /data          # 建索引(默认 ~);update 可指定多个根
  ffind report pdf                   # 文件名同时包含 report 和 pdf
  ffind -e md readme                 # 文件名含 readme 的 .md 文件
  ffind -t dir config                # 只找目录
  ffind -r '^test_.*\\.py$'           # 正则(匹配文件名)
  ffind -g '*.cc' --from src         # 通配符,限定在 ./src 下
  ffind --rebuild notes              # 强制重建后再搜
  ffind --fresh 30 notes             # 索引超过 30 分钟则自动重建
  ffind serve --open                 # 启动 Web 界面并打开浏览器(127.0.0.1:8765)
"""


def _size_human(n: float) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.0f}G"


def _time_ago(ts: float) -> str:
    s = max(0.0, time.time() - ts)
    if s < 90:
        return f"{s:.0f} 秒前"
    if s < 5400:
        return f"{s / 60:.0f} 分钟前"
    if s < 172800:
        return f"{s / 3600:.1f} 小时前"
    return f"{s / 86400:.1f} 天前"


def _ticker():
    """交互式进度回调;非 tty 时返回 None。"""
    if not sys.stderr.isatty():
        return None
    state = {"last": 0.0}

    def cb(nf: int, nd: int, secs: float) -> None:
        now = time.monotonic()
        if now - state["last"] < 0.2:
            return
        state["last"] = now
        sys.stderr.write(f"\r  已收录 {nf:,} 文件 / {nd:,} 目录  {secs:.1f}s ")
        sys.stderr.flush()

    return cb


def _rebuild(idx: Index, roots, workers=None, why: str = "") -> None:
    excl = idx.stored_excludes()
    sys.stderr.write(
        f"ffind: {why}正在为 {', '.join(roots)} 建索引…\n" if why
        else f"ffind: 正在为 {', '.join(roots)} 建索引…\n"
    )
    cb = _ticker()
    st = idx.build(roots, excludes=excl, workers=workers, progress=cb)
    if cb is not None:
        sys.stderr.write("\n")
    sys.stderr.write(
        f"ffind: 完成 —— {st['files']:,} 文件 / {st['dirs']:,} 目录,"
        f"用时 {st['seconds']:.2f}s\n"
    )


# ---------------- update ----------------


def _cmd_update(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="ffind update", description="并行扫描,全量重建文件名索引"
    )
    p.add_argument("roots", nargs="*", help=f"扫描根目录(默认沿用上次,再默认 {_HOME})")
    p.add_argument("-j", "--jobs", type=int, default=None, help="并行进程数(默认自适应)")
    p.add_argument("-x", "--exclude", action="append", default=[],
                   help="追加排除的目录名(可多次;默认排除 .git/node_modules/.cache 等)")
    p.add_argument("--no-default-excludes", action="store_true",
                   help="关闭默认排除集,全量收录(-x 仍生效)")
    p.add_argument("--db", default=None, help="索引文件路径")
    ns = p.parse_args(argv)

    idx = Index(ns.db)
    roots = ns.roots or idx.stored_roots() or [_HOME]
    excl = (frozenset() if ns.no_default_excludes else scan.DEFAULT_EXCLUDES) | set(ns.exclude)
    cb = _ticker()
    st = idx.build(roots, excludes=excl, workers=ns.jobs, progress=cb)
    if cb is not None:
        sys.stderr.write("\n")
    size = os.path.getsize(st["db"])
    rate = (st["files"] + st["dirs"]) / max(st["seconds"], 1e-6)
    print(
        f"已索引 {st['files']:,} 文件 / {st['dirs']:,} 目录"
        f"  [{rate:,.0f} 条目/秒, {_size_human(size)}, {st['seconds']:.2f}s]\n"
        f"索引位置: {st['db']}\n搜索根:   {', '.join(st['roots'])}"
    )
    return 0


# ---------------- status ----------------


def _cmd_status(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="ffind status", description="查看索引状态")
    p.add_argument("--db", default=None)
    ns = p.parse_args(argv)
    idx = Index(ns.db)
    st = idx.stats()
    if not st["built_at"]:
        print(f"尚无索引({st['path']})。先运行: ffind update <目录>")
        return 1
    print(f"索引文件: {st['path']}  ({_size_human(st['size'])}, SQLite)")
    print(f"构建时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st['built_at']))}"
          f"  ({_time_ago(st['built_at'])})")
    print(f"收录条目: {st['files']:,} 文件 / {st['dirs']:,} 目录")
    print(f"搜索根:   {', '.join(st['roots'])}")
    print(f"排除目录: {', '.join(sorted(st['excludes']))}")
    return 0


# ---------------- search ----------------


def _search_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ffind",
        description="快文件查找器:一次并行建索引,之后毫秒级查询",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("patterns", nargs="*", help="关键词(多个为 AND,子串匹配,默认大小写不敏感)")
    p.add_argument("-r", "--regex", metavar="RE", help="按文件名正则搜索")
    p.add_argument("-g", "--glob", metavar="PAT", help="按文件名通配符搜索(如 '*.cc')")
    p.add_argument("-e", "--ext", action="append", default=[], metavar="EXT",
                   help="限定扩展名(可多次,如 -e py -e md)")
    p.add_argument("-t", "--type", choices=["file", "dir", "any"], default="any",
                   help="结果类型(默认 any)")
    p.add_argument("-p", "--from", dest="from_", action="append", default=[], metavar="PATH",
                   help="限定在某个路径下搜索(可多次)")
    p.add_argument("-x", "--exclude", action="append", default=[], metavar="NAME",
                   help="查询时过滤掉路径含该目录名的结果(可多次)")
    p.add_argument("-i", "--case-sensitive", action="store_true", help="区分大小写")
    p.add_argument("-n", "--limit", type=int, default=200, help="最多显示条数(0=不限,默认 200)")
    p.add_argument("-c", "--count", action="store_true", help="只显示命中数")
    p.add_argument("-l", "--long", action="store_true", help="附带大小与修改时间(结果集上 stat,不影响速度)")
    p.add_argument("-0", "--null", action="store_true", help="以 NUL 分隔输出(便于 xargs -0)")
    p.add_argument("--rebuild", action="store_true", help="搜索前强制重建索引")
    p.add_argument("--fresh", type=float, metavar="MIN", help="索引超过 MIN 分钟则先自动重建")
    p.add_argument("-j", "--jobs", type=int, default=None, help="重建时的并行进程数")
    p.add_argument("--db", default=None, help="索引文件路径")
    p.add_argument("-V", "--version", action="version", version=f"ffind {__version__}")
    return p


def _cmd_search(argv: list[str]) -> int:
    ns = _search_parser().parse_args(argv)
    if not (ns.patterns or ns.regex or ns.glob):
        print("ffind: 请给出关键词(或 -r/--regex、-g/--glob),-h 查看帮助。", file=sys.stderr)
        return 2

    idx = Index(ns.db)
    # 建索引的根:显式 --from 路径 > 上次存的根 > 家目录。
    # 首次搜索就带 -p 时只索引该路径,避免不动声色去扫整个 ~。
    build_roots = scan.normalize_roots(ns.from_) or idx.stored_roots() or [_HOME]
    if ns.rebuild or not idx.exists():
        _rebuild(idx, build_roots, ns.jobs,
                 why="按 --rebuild " if ns.rebuild else "尚无索引,")
    elif ns.fresh is not None and idx.age() > ns.fresh * 60:
        _rebuild(idx, build_roots, ns.jobs,
                 why=f"索引已过期({ns.fresh:.0f} 分钟),")

    limit = 10**9 if ns.limit == 0 else ns.limit
    hits, total, capped = idx.query(
        ns.patterns,
        regex=ns.regex,
        glob=ns.glob,
        exts=ns.ext,
        type_=ns.type,
        roots=ns.from_,
        excludes=ns.exclude,
        case_sensitive=ns.case_sensitive,
        limit=limit,
    )

    if ns.count:
        print(f"{total}{'+' if capped else ''}")
        return 0 if hits else 1

    sep = "\0" if ns.null else "\n"
    lines = []
    for path, is_dir in hits:
        if ns.long:
            try:
                st = os.stat(path)
                mt = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
                size = "     -" if is_dir else _size_human(st.st_size).rjust(6)
                lines.append(f"{size}  {mt}  {path}")
                continue
            except OSError:
                lines.append(f"  ??  {path}")
                continue
        lines.append(path)
    if lines:
        sys.stdout.write(sep.join(lines) + ("\0" if ns.null else "\n"))
    if total > len(hits):
        hint = "+" if capped else ""
        print(f"(共 {total}{hint} 条,已显示前 {len(hits)} 条;-n 调整条数,-c 看总数)",
              file=sys.stderr)
    return 0 if hits else 1


# ---------------- serve (Web 界面) ----------------


def _cmd_serve(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="ffind serve", description="启动本地 Web 界面(浏览器实时搜索 + 可视化重建索引)"
    )
    p.add_argument("--port", type=int, default=8765, help="监听端口(默认 8765,仅 127.0.0.1)")
    p.add_argument("--db", default=None, help="索引文件路径")
    p.add_argument("-p", "--from", dest="roots", action="append", default=[], metavar="PATH",
                   help="限定搜索/打开的根目录(可多次;默认沿用索引已存根)")
    p.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    ns = p.parse_args(argv)
    from .server import serve
    return serve(db=ns.db, port=ns.port, roots=ns.roots, open_browser=ns.open)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "update":
        return _cmd_update(argv[1:])
    if argv and argv[0] == "status":
        return _cmd_status(argv[1:])
    if argv and argv[0] == "serve":
        return _cmd_serve(argv[1:])
    if not argv:
        _search_parser().print_help()
        return 2
    return _cmd_search(argv)
