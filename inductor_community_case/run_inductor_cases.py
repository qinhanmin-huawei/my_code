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

# pytest 附加参数
EXTRA_PYTEST_ARGS = ["-v"]

# ============================ 分隔符/图案 ============================

STAR_SEP = "*" * 100          # 大用例之间的分隔（3 行）
EQ_SEP = "=" * 200            # 不同参数组合之间的分隔
THIN_SEP = "-" * 100          # 参数块内部的分隔

SUMMARY_ART_TOP = "@" * 84
SUMMARY_ART_MID = "@" * 36 + "  S U M M A R Y  " + "@" * 36
SUMMARY_ART_BOT = "@" * 84

# ============================ 颜色（ANSI） ============================

ANSI_RESET = "\033[0m"
COLOR_ART = "\033[95m"    # 亮品红
COLOR_PASS = "\033[92m"   # 亮绿
COLOR_FAIL = "\033[91m"   # 亮红
COLOR_SKIP = "\033[93m"   # 亮黄
COLOR_PALETTE = {"PASS": COLOR_PASS, "FAIL": COLOR_FAIL, "SKIP": COLOR_SKIP}


def colorize(text, color):
    """给文本加上 ANSI 颜色。"""
    return "%s%s%s" % (color, text, ANSI_RESET)


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


def _run_pytest(args, workdir, timeout=None):
    """执行 pytest 子进程，返回 (returncode, 合并后的输出文本)。"""
    cmd = [PYTHON, "-m", "pytest"] + args
    try:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout if timeout and timeout > 0 else None,
        )
        out = proc.stdout.decode("utf-8", "replace")
        return proc.returncode, out
    except subprocess.TimeoutExpired as e:
        out = (e.output or b"").decode("utf-8", "replace")
        out += "\n[TIMEOUT] 超过 %s 秒未结束" % timeout
        return -1, out
    except Exception as e:  # noqa: BLE001
        return -1, "[ERROR] 执行异常: %r" % e


def collect_nodeids(pyfile, workdir, testname):
    """收集 py 文件中指定用例的全部参数组合 nodeid。"""
    args = ["--collect-only", "-q", pyfile, "-p", "no:cacheprovider"]
    rc, out = _run_pytest(args, workdir)
    nodeids = []
    for line in out.splitlines():
        line = line.strip()
        if "::" not in line:
            continue
        base = line.split("::")[-1].split("[")[0]
        if base == testname and line not in nodeids:
            nodeids.append(line)
    return nodeids, rc, out


def decide_status(rc, output):
    """根据返回码和输出判断 3 种状态。"""
    if rc != 0:
        return "FAIL"
    if "SKIPPED" in output:
        return "SKIP"
    return "PASS"


def build_reproduce(nodeid, workdir):
    """生成一键复现命令。"""
    args = " ".join(EXTRA_PYTEST_ARGS)
    return 'cd /d "%s" && "%s" -m pytest "%s" %s' % (workdir, PYTHON, nodeid, args)


# ============================ 用例执行 ============================

_print_lock = threading.Lock()


def _worker(idx, total, nodeid, workdir, box):
    """单个参数组合：单独线程 + 单独子进程执行。"""
    with _print_lock:
        print("[开始] (%d/%d) %s" % (idx, total, nodeid))
    rc, out = _run_pytest([nodeid] + EXTRA_PYTEST_ARGS + ["-p", "no:cacheprovider"],
                          workdir, PER_CASE_TIMEOUT)
    status = decide_status(rc, out)

    block = []
    block.append(EQ_SEP)
    block.append("[小用例 %d/%d] %s" % (idx, total, nodeid))
    block.append("状态: %s" % colorize(status, COLOR_PALETTE[status]))
    block.append("一键复现命令:")
    block.append(build_reproduce(nodeid, workdir))
    block.append(THIN_SEP)
    block.append("pytest 输出 (returncode=%d):" % rc)
    block.append(out.rstrip("\n"))
    box[idx] = (status, "\n".join(block))

    with _print_lock:
        print("[结束] (%d/%d) %-40s -> %s" % (idx, total, nodeid, status))


def build_summary(nodeids, results):
    """生成带颜色图案的 summary 文本。"""
    groups = {"PASS": [], "FAIL": [], "SKIP": []}
    for i, nodeid in enumerate(nodeids, 1):
        status = results.get(i, ("FAIL", ""))[0]
        groups[status].append(nodeid)

    lines = [
        EQ_SEP,
        colorize(SUMMARY_ART_TOP, COLOR_ART),
        colorize(SUMMARY_ART_MID, COLOR_ART),
        colorize(SUMMARY_ART_BOT, COLOR_ART),
    ]
    lines.append("总参数组合: %d    %s    %s    %s" % (
        len(nodeids),
        colorize("PASS: %d" % len(groups["PASS"]), COLOR_PASS),
        colorize("FAIL: %d" % len(groups["FAIL"]), COLOR_FAIL),
        colorize("SKIP: %d" % len(groups["SKIP"]), COLOR_SKIP),
    ))
    lines.append(THIN_SEP)
    for status in ("PASS", "FAIL", "SKIP"):
        lines.append(colorize("%s (%d):" % (status, len(groups[status])), COLOR_PALETTE[status]))
        if groups[status]:
            for nid in groups[status]:
                lines.append("    - %s" % nid)
        else:
            lines.append("    (无)")
        lines.append(THIN_SEP)
    lines.append(colorize(SUMMARY_ART_TOP, COLOR_ART))
    lines.append(colorize(SUMMARY_ART_MID, COLOR_ART))
    lines.append(colorize(SUMMARY_ART_BOT, COLOR_ART))
    return "\n".join(lines), groups


def run_big_case(pyfile, testname, roots):
    """执行一个大用例（py 文件 + 用例名）的所有参数组合。

    返回 (日志文本, {nodeid: status}, summary 文本)。
    """
    lines = [STAR_SEP, STAR_SEP, STAR_SEP]
    lines.append("大用例: %s :: %s" % (pyfile, testname))

    filepath, workdir = find_file(pyfile, roots)
    if not filepath:
        lines.append("[错误] 未在搜索根目录中找到文件 %s" % pyfile)
        lines.append("搜索根目录: %s" % roots)
        lines.append(STAR_SEP)
        return "\n".join(lines), {}, ""

    lines.append("文件路径: %s" % filepath)
    lines.append("工作目录: %s" % workdir)

    nodeids, rc, collect_out = collect_nodeids(pyfile, workdir, testname)
    if not nodeids:
        lines.append("[错误] 未收集到用例 %s 的任何参数组合 (returncode=%d)" % (testname, rc))
        lines.append(THIN_SEP)
        lines.append("pytest --collect-only 输出:")
        lines.append(collect_out.rstrip("\n"))
        lines.append(STAR_SEP)
        return "\n".join(lines), {}, ""

    lines.append("参数组合数: %d" % len(nodeids))

    total = len(nodeids)
    box = {}
    threads = []
    for i, nodeid in enumerate(nodeids, 1):
        t = threading.Thread(target=_worker, args=(i, total, nodeid, workdir, box))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    for i in range(1, total + 1):
        lines.append(box[i][1])

    summary_text, _groups = build_summary(nodeids, box)
    lines.append(summary_text)
    lines.append(STAR_SEP)

    status_map = {nodeid: box[i][0] for i, nodeid in enumerate(nodeids, 1)}
    return "\n".join(lines), status_map, summary_text


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
        block, status_map, summary_text = run_big_case(pyfile, testname, roots)

        stem = os.path.splitext(os.path.basename(pyfile))[0]
        if stem not in handles:
            log_path = os.path.join(LOG_DIR, "%s_%s.log" % (stem, run_ts))
            handles[stem] = open(log_path, "w", encoding="utf-8")
            handles[stem].write("清单来源: %s\n搜索根目录: %s\n\n" % (source, roots))
        handles[stem].write(block + "\n\n")
        handles[stem].flush()

        # 控制台同步打印 summary（含颜色图案）
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
        print("%-60s PASS=%-3d FAIL=%-3d SKIP=%-3d" % (title, p, f, s))
    print("-" * 100)
    print("总计: PASS=%d  FAIL=%d  SKIP=%d" % (all_pass, all_fail, all_skip))
    print("日志文件已保存到: %s" % LOG_DIR)


if __name__ == "__main__":
    main()
