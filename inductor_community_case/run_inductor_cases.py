#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inductor 社区用例批量执行脚本。

用法:
    python run_inductor_cases.py                  # 使用脚本内置的 CASE_TEXT 字符串
    python run_inductor_cases.py cases.txt        # 使用 txt 文件作为用例清单
    python run_inductor_cases.py cases.txt -r D:\\pytorch\\test -r D:\\other

说明:
    - 用例清单每行格式为:  py文件名   用例名   (中间用空格/tab 隔开)
    - 用例名不带参数、也不知道类名，脚本通过 pytest --collect-only 收集出完整的
      执行用例名 (nodeid)，并跑完该用例的所有参数组合。
    - 每个参数组合单独起线程 + 单独子进程执行，个别用例 core 不影响其它用例。
    - pytest 输出实时流式写入日志文件（每行都 flush）。
"""

import datetime
import os
import subprocess
import sys
import threading

# ============================ 用户配置区 ============================

# 用例清单：命令行未传入 txt 文件时，使用本字符串。
CASE_TEXT = """test_async_compile.py 	 test_bad_kernel
test_async_compile.py 	 test_wait_pool_ready
test_mmdecomp.py 	 test_simple_mm
test_mmdecomp.py 	 test_batched_mm
test_mmdecomp.py 	 test_bmm_batch2_last_dim_size_is_one
test_mmdecomp.py 	 test_bmm_outer_product_k_is_one_with_unbacked_k
test_mmdecomp.py 	 test_some
test_mmdecomp.py 	 test_some_batched
test_dropout_align_random_eager.py 	 test_inductor_generated_code_contains_dropout
test_dropout_align_random_eager.py 	 test_perf_smoke_cuda
"""

# 搜索根目录（按顺序搜索，找到即停止）。请改成你自己的路径。
SEARCH_ROOTS = [
    r"D:\pytorch\test\inductor",
    r"D:\pytorch\test",
]

# 日志输出目录（固定路径，由用户指定）
LOG_DIR = r"D:\0Code\my_code\inductor_community_case\logs"

# Python 解释器（默认使用当前解释器）
PYTHON = sys.executable

# 每个小用例（单个参数组合）的超时时间（秒），<= 0 表示不限制
PER_CASE_TIMEOUT = 0

# 并行线程数（同时执行的 pytest 子进程个数），<= 0 表示不限制（有几个参数组合起几个）
MAX_PARALLEL = 4

# pytest 附加参数
EXTRA_PYTEST_ARGS = ["-v"]

# ============================ 分隔符/图案 ============================

STAR_SEP = "*" * 100          # 大用例之间的分隔（3 行）
EQ_SEP = "=" * 200            # 不同参数组合之间的分隔
THIN_SEP = "-" * 100          # 参数块内部的分隔

# 状态图案（emoji 圆圈，自带颜色）
EMOJI_PASS = "🟢"
EMOJI_FAIL = "🔴"
EMOJI_SKIP = "🟡"
STATUS_EMOJI = {"PASS": EMOJI_PASS, "FAIL": EMOJI_FAIL, "SKIP": EMOJI_SKIP}

SUMMARY_SEP = "─" * 72        # summary 内部分隔线


# ============================ 工具函数 ============================

def parse_argv(argv):
    """解析命令行参数，返回 (txt路径, 额外搜索根目录列表)。"""
    txt = None
    roots = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-r", "--root"):
            i += 1
            if i < len(argv):
                roots.append(argv[i])
        elif a in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        else:
            txt = a
        i += 1
    return txt, roots


def parse_cases(text):
    """把清单文本解析成 [(py文件名, 用例名), ...]，去重且保持顺序。"""
    cases = []
    seen = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        key = (parts[0], parts[1])
        if key in seen:
            continue
        seen.add(key)
        cases.append(key)
    return cases


def find_file(pyfile, roots):
    """按顺序在多个根目录中递归查找 py 文件，找到即停止。

    返回 (文件绝对路径, 工作目录)；找不到返回 (None, None)。
    """
    for root in roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            continue
        direct = os.path.join(root, pyfile)
        if os.path.isfile(direct):
            return direct, os.path.dirname(direct)
        for dirpath, _dirnames, filenames in os.walk(root):
            if pyfile in filenames:
                full = os.path.join(dirpath, pyfile)
                return full, dirpath
    return None, None


def _child_env():
    """子进程环境：python 不缓冲，保证输出实时。"""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return env


def collect_nodeids(pyfile, workdir, testname):
    """收集 py 文件中指定用例的全部参数组合 nodeid。"""
    args = ["--collect-only", "-q", pyfile, "-p", "no:cacheprovider"]
    try:
        proc = subprocess.run(
            [PYTHON, "-m", "pytest"] + args,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_child_env(),
        )
        out = proc.stdout.decode("utf-8", "replace")
        rc = proc.returncode
    except Exception as e:  # noqa: BLE001
        return [], -1, "[ERROR] 收集失败: %r" % e

    nodeids = []
    for line in out.splitlines():
        line = line.strip()
        if "::" not in line:
            continue
        base = line.split("::")[-1].split("[")[0]
        if base == testname and line not in nodeids:
            nodeids.append(line)
    return nodeids, rc, out


def _stream_pytest(args, workdir, on_line, timeout=None):
    """执行 pytest，边跑边把每一行交给 on_line 输出（实时）。

    返回 (returncode, 合并后的输出文本)。
    """
    cmd = [PYTHON, "-m", "pytest"] + args
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
            env=_child_env(),
        )
    except Exception as e:  # noqa: BLE001
        text = "[ERROR] 启动 pytest 进程失败: %r" % e
        on_line(text)
        return -1, text

    timed_out = {"v": False}
    timer = None
    if timeout and timeout > 0:
        def _kill():
            timed_out["v"] = True
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

        timer = threading.Timer(timeout, _kill)
        timer.start()

    collected = []
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            collected.append(line)
            on_line(line)
    except Exception as e:  # noqa: BLE001
        msg = "[ERROR] 读取输出异常: %r" % e
        collected.append(msg)
        on_line(msg)
    finally:
        if timer is not None:
            timer.cancel()
        try:
            proc.stdout.close()
        except Exception:  # noqa: BLE001
            pass

    rc = proc.wait()
    if timed_out["v"]:
        msg = "[TIMEOUT] 超过 %s 秒未结束，已强制终止" % timeout
        collected.append(msg)
        on_line(msg)
        rc = -1
    return rc, "\n".join(collected)


def decide_status(rc, output):
    """根据返回码和输出判断 3 种状态。"""
    if rc != 0:
        return "FAIL"
    if "SKIPPED" in output:
        return "SKIP"
    return "PASS"


def status_label(status):
    """带 emoji 的状态标签，例如 '🟢 PASS'。"""
    return "%s %s" % (STATUS_EMOJI[status], status)


def build_reproduce(nodeid, workdir):
    """生成一键复现命令。"""
    args = " ".join(EXTRA_PYTEST_ARGS)
    return 'cd /d "%s" && "%s" -m pytest "%s" %s' % (workdir, PYTHON, nodeid, args)


# ============================ 用例执行 ============================

_print_lock = threading.Lock()
_log_lock = threading.Lock()


def make_writer(handle):
    """构造一个线程安全的日志写入函数（每行立即 flush，实时落盘）。"""
    def write(text):
        with _log_lock:
            handle.write(text + "\n")
            handle.flush()
    return write


def _worker(idx, total, nodeid, workdir, write, box, sem):
    """单个参数组合：单独线程 + 单独子进程执行，输出实时写日志。"""
    if sem is not None:
        sem.acquire()
    try:
        with _print_lock:
            print("[开始] (%d/%d) %s" % (idx, total, nodeid))

        write(EQ_SEP)
        write("[小用例 %d/%d] %s" % (idx, total, nodeid))
        write("一键复现命令:")
        write(build_reproduce(nodeid, workdir))
        write(THIN_SEP)
        write("pytest 输出 (实时):")

        def on_line(line):
            write(line)

        rc, out = _stream_pytest([nodeid] + EXTRA_PYTEST_ARGS + ["-p", "no:cacheprovider"],
                                 workdir, on_line, PER_CASE_TIMEOUT)
        status = decide_status(rc, out)

        write(THIN_SEP)
        write("状态: %s    (returncode=%d)" % (status_label(status), rc))

        box[idx] = status
        with _print_lock:
            print("[结束] (%d/%d) %-40s -> %s" % (idx, total, nodeid, status))
    finally:
        if sem is not None:
            sem.release()


def build_summary(nodeids, results):
    """生成简洁美观的 summary 文本。"""
    groups = {"PASS": [], "FAIL": [], "SKIP": []}
    for i, nodeid in enumerate(nodeids, 1):
        groups[results.get(i, "FAIL")].append(nodeid)

    lines = [
        SUMMARY_SEP,
        "  SUMMARY   %s PASS %d   %s FAIL %d   %s SKIP %d" % (
            EMOJI_PASS, len(groups["PASS"]),
            EMOJI_FAIL, len(groups["FAIL"]),
            EMOJI_SKIP, len(groups["SKIP"])),
        SUMMARY_SEP,
    ]
    for status in ("PASS", "FAIL", "SKIP"):
        lines.append("%s %d" % (status_label(status), len(groups[status])))
        if groups[status]:
            for nid in groups[status]:
                lines.append("    - %s" % nid)
        else:
            lines.append("    (无)")
    lines.append(SUMMARY_SEP)
    return "\n".join(lines)


def run_big_case(pyfile, testname, roots, write):
    """执行一个大用例的所有参数组合，实时写入日志。

    返回 (nodeid -> status 映射, summary 文本)。
    """
    write(STAR_SEP)
    write(STAR_SEP)
    write(STAR_SEP)
    write("大用例: %s :: %s" % (pyfile, testname))

    filepath, workdir = find_file(pyfile, roots)
    if not filepath:
        write("[错误] 未在搜索根目录中找到文件 %s" % pyfile)
        write("搜索根目录: %s" % roots)
        write(STAR_SEP)
        return {}, ""

    write("文件路径: %s" % filepath)
    write("工作目录: %s" % workdir)

    nodeids, rc, collect_out = collect_nodeids(pyfile, workdir, testname)
    if not nodeids:
        write("[错误] 未收集到用例 %s 的任何参数组合 (returncode=%d)" % (testname, rc))
        write(THIN_SEP)
        write("pytest --collect-only 输出:")
        write(collect_out.rstrip("\n"))
        write(STAR_SEP)
        return {}, ""

    write("参数组合数: %d    并行线程数: %s" % (
        len(nodeids), MAX_PARALLEL if MAX_PARALLEL > 0 else "不限"))

    total = len(nodeids)
    box = {}
    sem = threading.Semaphore(MAX_PARALLEL) if MAX_PARALLEL > 0 else None
    threads = []
    for i, nodeid in enumerate(nodeids, 1):
        t = threading.Thread(target=_worker, args=(i, total, nodeid, workdir, write, box, sem))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    summary_text = build_summary(nodeids, box)
    write(summary_text)
    write(STAR_SEP)

    status_map = {nodeid: box[i] for i, nodeid in enumerate(nodeids, 1)}
    return status_map, summary_text


# ============================ 入口 ============================

def main():
    txt_path, extra_roots = parse_argv(sys.argv[1:])

    if txt_path:
        if not os.path.isfile(txt_path):
            print("[错误] 用例清单文件不存在: %s" % txt_path)
            sys.exit(1)
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        source = "清单文件: %s" % os.path.abspath(txt_path)
    else:
        text = CASE_TEXT
        source = "清单来源: 脚本内置字符串"

    roots = extra_roots if extra_roots else SEARCH_ROOTS
    cases = parse_cases(text)
    if not cases:
        print("[错误] 用例清单为空")
        sys.exit(1)

    os.makedirs(LOG_DIR, exist_ok=True)
    run_ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    print("=" * 100)
    print(source)
    print("搜索根目录: %s" % roots)
    print("日志目录  : %s" % LOG_DIR)
    print("用例数量  : %d" % len(cases))
    print("=" * 100)

    handles = {}
    overall = []  # (大用例标题, {nodeid: status})
    for pyfile, testname in cases:
        stem = os.path.splitext(os.path.basename(pyfile))[0]
        if stem not in handles:
            log_path = os.path.join(LOG_DIR, "%s_%s.log" % (stem, run_ts))
            handle = open(log_path, "w", encoding="utf-8")
            handle.write("清单来源: %s\n搜索根目录: %s\n\n" % (source, roots))
            handle.flush()
            handles[stem] = handle
        write = make_writer(handles[stem])

        status_map, summary_text = run_big_case(pyfile, testname, roots, write)
        if summary_text:
            print(summary_text)

        overall.append(("%s :: %s" % (pyfile, testname), status_map))

    for h in handles.values():
        h.close()

    # 整体汇总
    all_pass = all_fail = all_skip = 0
    print()
    print("=" * 100)
    print("整体汇总")
    print("=" * 100)
    for title, status_map in overall:
        p = sum(1 for v in status_map.values() if v == "PASS")
        f = sum(1 for v in status_map.values() if v == "FAIL")
        s = sum(1 for v in status_map.values() if v == "SKIP")
        all_pass += p
        all_fail += f
        all_skip += s
        print("%-60s %s %d  %s %d  %s %d"
              % (title, EMOJI_PASS, p, EMOJI_FAIL, f, EMOJI_SKIP, s))
    print("-" * 100)
    print("总计: %s %d  %s %d  %s %d"
          % (EMOJI_PASS, all_pass, EMOJI_FAIL, all_fail, EMOJI_SKIP, all_skip))
    print("日志文件已保存到: %s" % LOG_DIR)


if __name__ == "__main__":
    main()
