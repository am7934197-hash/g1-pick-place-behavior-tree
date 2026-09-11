#!/usr/bin/env python3
"""Analyze VLA server logs for RTC stalls, buffer underruns and lost chunks.

The analyzer accepts the console output of ``server.py`` (the most complete
source) and the JSON-lines files written by ``VLAInferenceLogger``.  It only
uses the Python standard library so it can be copied to the robot/server.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


INITIAL_START_RE = re.compile(r"RTC: running initial inference")
INITIAL_DONE_RE = re.compile(
    r"RTC: initial inference done in (?P<ms>[\d.]+)ms, buffer=(?P<buffer>\d+) frames"
)
STEP_RE = re.compile(r"RTC step (?P<step>\d+): buffer_remaining=(?P<buffer>\d+)")
TRIGGER_RE = re.compile(
    r"RTC: triggering async inference at step (?P<step>\d+) \(remaining=(?P<remaining>\d+)\)"
)
DONE_RE = re.compile(
    r"RTC async infer done: obs=(?P<obs>[\d.]+)ms, infer=(?P<infer>[\d.]+)ms, actions=(?P<actions>\d+)"
)
FUSED_RE = re.compile(r"RTC: fused new chunk at step (?P<step>\d+), buffer=(?P<buffer>\d+)")
EMPTY_RE = re.compile(r"RTC: buffer empty at step (?P<step>\d+)")
STOP_RE = re.compile(r"RTC: stop requested at step (?P<step>\d+)")
RTC_ERROR_RE = re.compile(
    r"RTC(?: async)?(?::| inference).*?(?:error|failed|returned no actions|observation is None|schema invalid)",
    re.IGNORECASE,
)
RECEIVED_ZERO_RE = re.compile(r"Received 0 actions")
FAULT_RE = re.compile(r"set_joint_commands status=\S*FAULT")
ACTION_FAIL_RE = re.compile(r"Action execution failed at step (?P<step>\d+)")
SYNC_STEP_RE = re.compile(
    r"Step (?P<step>\d+): (?P<n>\d+) actions, (?P<ms>[\d.]+)ms"
)
ISO_TIME_RE = re.compile(r"(?P<iso>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)")
BRACKET_TIME_RE = re.compile(r"\[(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})(?P<f>[.,]\d+)?\]")


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    path: str
    line: Optional[int] = None
    run_id: Optional[str] = None


@dataclass
class FileResult:
    path: str
    format: str = "unknown"
    rtc_events: int = 0
    rtc_sessions: int = 0
    steps: int = 0
    triggers: int = 0
    completions: int = 0
    fusions: int = 0
    chunks: list[int] = field(default_factory=list)
    infer_ms: list[float] = field(default_factory=list)
    obs_ms: list[float] = field(default_factory=list)
    fault_events: int = 0
    empty_get_actions: int = 0
    issues: list[Issue] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class Trigger:
    step: int
    remaining: int
    line: int
    timestamp: Optional[float]
    done_actions: Optional[int] = None
    done_line: Optional[int] = None


def _timestamp(line: str, previous: Optional[float]) -> tuple[Optional[float], float]:
    """Return comparable seconds and source resolution in seconds."""
    match = ISO_TIME_RE.search(line)
    if match:
        raw = match.group("iso").replace(",", ".").replace(" ", "T")
        try:
            value = datetime.fromisoformat(raw).timestamp()
            resolution = 0.001 if "." in raw else 1.0
            return value, resolution
        except ValueError:
            pass
    match = BRACKET_TIME_RE.search(line)
    if not match:
        return None, 0.0
    fraction = match.group("f")
    value = int(match.group("h")) * 3600 + int(match.group("m")) * 60 + int(match.group("s"))
    if fraction:
        value += float(fraction.replace(",", "."))
    if previous is not None:
        # Console logs contain only time-of-day; handle a run crossing midnight.
        while value + 43200 < previous:
            value += 86400
    return float(value), 0.001 if fraction else 1.0


def _add(
    result: FileResult,
    severity: str,
    code: str,
    message: str,
    line: Optional[int] = None,
    run_id: Optional[str] = None,
) -> None:
    result.issues.append(Issue(severity, code, message, result.path, line, run_id))


def analyze_console(lines: list[str], result: FileResult, args: argparse.Namespace) -> None:
    result.format = "console"
    last_step: Optional[int] = None
    last_buffer: Optional[int] = None
    last_step_time: Optional[float] = None
    last_time: Optional[float] = None
    pending: Optional[Trigger] = None
    fused_buffer: Optional[int] = None
    expected_chunk: Optional[int] = args.expected_chunk_size
    stopped = False

    for lineno, line in enumerate(lines, 1):
        timestamp, resolution = _timestamp(line, last_time)
        if timestamp is not None:
            last_time = timestamp

        if INITIAL_START_RE.search(line):
            if pending and not stopped:
                _add(result, "ERROR", "UNFINISHED_INFERENCE", f"上一个异步推理触发于 step {pending.step}，但未完成融合", pending.line)
            result.rtc_sessions += 1
            result.rtc_events += 1
            last_step = last_buffer = None
            last_step_time = None
            pending = None
            fused_buffer = None
            stopped = False
            continue

        match = INITIAL_DONE_RE.search(line)
        if match:
            result.rtc_events += 1
            chunk = int(match.group("buffer"))
            expected_chunk = expected_chunk or chunk
            result.chunks.append(chunk)
            continue

        match = STEP_RE.search(line)
        if match:
            result.rtc_events += 1
            result.steps += 1
            step = int(match.group("step"))
            buffer = int(match.group("buffer"))
            if last_step is not None:
                if step != last_step + 1:
                    code = "STEP_GAP" if step > last_step + 1 else "STEP_REORDER"
                    _add(result, "ERROR", code, f"RTC step 不连续：{last_step} -> {step}", lineno)
                expected_buffer = (fused_buffer if fused_buffer is not None else last_buffer)
                if expected_buffer is not None and buffer != max(0, expected_buffer - 1):
                    _add(
                        result,
                        "ERROR",
                        "BUFFER_DISCONTINUITY",
                        f"buffer 无法解释的跳变：上一状态 {expected_buffer}，执行一帧后应为 {max(0, expected_buffer - 1)}，实际 {buffer}",
                        lineno,
                    )
                if timestamp is not None and last_step_time is not None:
                    gap_ms = (timestamp - last_step_time) * 1000
                    # A second-only logger quantizes normal 33 ms gaps to 0/1000 ms.
                    effective_limit = max(args.stall_ms, 1500.0 if resolution >= 1.0 else 0.0)
                    if gap_ms > effective_limit:
                        _add(result, "ERROR", "DATA_STALL", f"相邻 RTC step 间隔 {gap_ms:.0f}ms，超过阈值 {effective_limit:.0f}ms", lineno)
            last_step, last_buffer = step, buffer
            last_step_time = timestamp
            fused_buffer = None
            continue

        match = TRIGGER_RE.search(line)
        if match:
            result.rtc_events += 1
            result.triggers += 1
            if pending is not None:
                _add(result, "ERROR", "OVERLAPPING_TRIGGER", f"step {pending.step} 的推理尚未融合，又在 step {match.group('step')} 触发", lineno)
            pending = Trigger(int(match.group("step")), int(match.group("remaining")), lineno, timestamp)
            continue

        match = DONE_RE.search(line)
        if match:
            result.rtc_events += 1
            result.completions += 1
            obs_ms = float(match.group("obs"))
            infer_ms = float(match.group("infer"))
            actions = int(match.group("actions"))
            result.obs_ms.append(obs_ms)
            result.infer_ms.append(infer_ms)
            result.chunks.append(actions)
            if pending is None:
                _add(result, "ERROR", "ORPHAN_COMPLETION", "收到异步推理结果，但找不到对应 trigger", lineno)
            else:
                pending.done_actions = actions
                pending.done_line = lineno
                budget_ms = pending.remaining * args.dt_ms
                cost_ms = obs_ms + infer_ms
                if cost_ms > budget_ms:
                    _add(result, "ERROR", "UNDERFLOW_RISK", f"异步观测+推理耗时 {cost_ms:.0f}ms，超过 buffer 预算 {budget_ms:.0f}ms（{pending.remaining} 帧）", lineno)
                if timestamp is not None and pending.timestamp is not None:
                    wall_ms = (timestamp - pending.timestamp) * 1000
                    if wall_ms > budget_ms + max(10.0, args.dt_ms):
                        _add(result, "WARN", "ASYNC_WALLTIME_HIGH", f"trigger 到完成的日志墙钟耗时 {wall_ms:.0f}ms，超过 buffer 预算 {budget_ms:.0f}ms", lineno)
            if actions == 0:
                _add(result, "ERROR", "EMPTY_CHUNK", "异步推理返回 0 个动作", lineno)
            elif expected_chunk and actions < expected_chunk:
                _add(result, "WARN", "SHORT_CHUNK", f"异步 chunk 只有 {actions} 帧，期望 {expected_chunk} 帧；可能发生 chunk 截断/丢失", lineno)
            continue

        match = FUSED_RE.search(line)
        if match:
            result.rtc_events += 1
            result.fusions += 1
            fusion_step = int(match.group("step"))
            fused_buffer = int(match.group("buffer"))
            if pending is None:
                _add(result, "ERROR", "ORPHAN_FUSION", "发生 chunk 融合，但找不到对应 trigger", lineno)
            elif pending.done_actions is None:
                _add(result, "ERROR", "FUSION_BEFORE_COMPLETION", f"step {pending.step} 的推理尚无完成日志就发生融合", lineno)
                pending = None
            else:
                elapsed_frames = max(0, fusion_step - pending.step)
                expected_after_fusion = max(0, pending.done_actions - elapsed_frames)
                if fused_buffer != expected_after_fusion:
                    _add(
                        result,
                        "ERROR",
                        "CHUNK_ACCOUNTING_MISMATCH",
                        f"融合后 buffer={fused_buffer}，按 chunk={pending.done_actions}、推理期间执行={elapsed_frames} 帧计算应为 {expected_after_fusion}；疑似 chunk/日志丢失",
                        lineno,
                    )
                pending = None
            continue

        match = EMPTY_RE.search(line)
        if match:
            result.rtc_events += 1
            if pending and pending.done_actions is None:
                state = "异步推理仍在等待"
            elif pending:
                state = "异步结果已完成但尚未融合"
            else:
                state = "没有可用的新 chunk"
            _add(result, "ERROR", "BUFFER_UNDERRUN", f"step {match.group('step')} buffer 耗尽，{state}；动作流发生阻塞", lineno)
            continue

        if STOP_RE.search(line):
            result.rtc_events += 1
            stopped = True
            pending = None
            continue

        if RTC_ERROR_RE.search(line):
            result.rtc_events += 1
            _add(result, "ERROR", "RTC_RUNTIME_ERROR", line.strip(), lineno)

    if pending is not None and not stopped:
        state = "已完成但未融合" if pending.done_actions is not None else "未收到完成结果"
        _add(result, "ERROR", "UNFINISHED_INFERENCE", f"文件结束时 step {pending.step} 的异步推理{state}", pending.line)
    _analyze_sync_console(lines, result, args)
    if result.rtc_events == 0:
        result.notes.append(
            "RTC 未启用（未找到 RTC 控制台事件）。本次按同步推理检查空 GetActions / 推理超时 / FAULT。"
        )


def _analyze_sync_console(lines: list[str], result: FileResult, args: argparse.Namespace) -> None:
    """Sync-mode stalls: empty GetActions, FAULT streaks, Step latency near timeout."""
    timeout_ms = float(getattr(args, "infer_timeout_ms", 20000.0))
    fault_streak = 0
    streak_start: Optional[int] = None
    for lineno, line in enumerate(lines, 1):
        if RECEIVED_ZERO_RE.search(line):
            result.empty_get_actions += 1
            _add(result, "ERROR", "EMPTY_CHUNK", f"GetActions 返回 0 个动作：{line.strip()}", lineno)
        match = SYNC_STEP_RE.search(line)
        if match:
            ms = float(match.group("ms"))
            result.infer_ms.append(ms)
            if ms >= timeout_ms:
                _add(
                    result,
                    "ERROR",
                    "INFER_TIMEOUT",
                    f"同步 Step {match.group('step')} 推理耗时 {ms:.0f}ms，接近/超过阈值 {timeout_ms:.0f}ms",
                    lineno,
                )
        if FAULT_RE.search(line):
            result.fault_events += 1
            if fault_streak == 0:
                streak_start = lineno
            fault_streak += 1
        else:
            if fault_streak >= 3:
                _add(
                    result,
                    "ERROR",
                    "ROBOT_FAULT",
                    f"连续 {fault_streak} 次 set_joint_commands FAULT（自 line {streak_start}）",
                    streak_start,
                )
            fault_streak = 0
            streak_start = None
        match = ACTION_FAIL_RE.search(line)
        if match:
            _add(
                result,
                "ERROR",
                "ACTION_EXEC_FAILED",
                f"step {match.group('step')} 动作执行失败，同步循环中止",
                lineno,
            )
    if fault_streak >= 3:
        _add(
            result,
            "ERROR",
            "ROBOT_FAULT",
            f"连续 {fault_streak} 次 set_joint_commands FAULT（自 line {streak_start}）",
            streak_start,
        )


def analyze_jsonl(lines: list[str], result: FileResult, args: argparse.Namespace) -> None:
    result.format = "jsonl"
    runs: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            _add(result, "ERROR", "MALFORMED_JSON", f"JSON 解析失败：{exc.msg}", lineno)
            continue
        run_id = str(record.get("run_id", "unknown"))
        runs[run_id].append((lineno, record))

    for run_id, records in runs.items():
        starts = [r for _, r in records if r.get("event") == "run_start"]
        ends = [r for _, r in records if r.get("event") == "run_end"]
        if starts and not ends:
            _add(result, "WARN", "INCOMPLETE_RUN", "有 run_start 但没有 run_end，日志可能截断或服务异常退出", records[-1][0], run_id)

        step_records = [(n, r) for n, r in records if isinstance(r.get("step"), int)]
        result.steps += len({r["step"] for _, r in step_records})
        action_records = [(n, r) for n, r in step_records if "num_actions" in r]
        sizes = [int(r.get("num_actions", 0)) for _, r in action_records]
        expected = args.expected_chunk_size
        if expected is None and sizes:
            expected = Counter(sizes).most_common(1)[0][0]
        for lineno, record in action_records:
            count = int(record.get("num_actions", 0))
            result.chunks.append(count)
            if count <= 0:
                _add(result, "ERROR", "EMPTY_CHUNK", f"step {record['step']} 的 chunk 为空", lineno, run_id)
            elif expected and count < expected:
                _add(result, "WARN", "SHORT_CHUNK", f"step {record['step']} 只有 {count} 个动作，常见/期望值为 {expected}", lineno, run_id)
            if isinstance(record.get("elapsed_ms"), (int, float)):
                result.infer_ms.append(float(record["elapsed_ms"]))
                timeout_ms = float(getattr(args, "infer_timeout_ms", 20000.0))
                if float(record["elapsed_ms"]) >= timeout_ms:
                    _add(
                        result,
                        "ERROR",
                        "INFER_TIMEOUT",
                        f"step {record['step']} elapsed_ms={record['elapsed_ms']:.0f} 接近/超过阈值 {timeout_ms:.0f}ms",
                        lineno,
                        run_id,
                    )

        ordered = sorted(step_records, key=lambda item: item[0])
        previous_time: Optional[datetime] = None
        previous_line: Optional[int] = None
        for lineno, record in ordered:
            raw = record.get("timestamp")
            if not isinstance(raw, str):
                continue
            try:
                current = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if previous_time is not None:
                gap_ms = (current - previous_time).total_seconds() * 1000
                if gap_ms > args.json_stall_ms:
                    _add(result, "WARN", "LOG_DATA_GAP", f"连续结构化记录相隔 {gap_ms:.0f}ms（line {previous_line}->{lineno}）；可能阻塞，也可能是同步执行耗时", lineno, run_id)
            previous_time, previous_line = current, lineno

    result.notes.append(
        "JSONL 不含 RTC trigger/fusion/buffer 事件；当前抓取配置 rtc.enabled=false，"
        "num_actions 众数是 execute_actions_per_chunk 截断后的同步执行长度，不是 PolicyServer 丢了 50 帧 chunk。"
        "判断 RTC buffer 阻塞请同时传入 infer_server 控制台日志。"
    )


def analyze_file(path: Path, args: argparse.Namespace) -> FileResult:
    result = FileResult(str(path))
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        _add(result, "ERROR", "READ_ERROR", str(exc))
        return result
    first = next((line.lstrip() for line in lines if line.strip()), "")
    if first.startswith("{"):
        analyze_jsonl(lines, result, args)
    else:
        analyze_console(lines, result, args)
    return result


def resolve_inputs(values: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = [Path(item) for item in glob.glob(value)]
        if not matches:
            matches = [Path(value)]
        for path in matches:
            if path.is_dir():
                paths.extend(sorted(p for p in path.iterdir() if p.is_file() and p.suffix in {".log", ".txt", ".jsonl"}))
            else:
                paths.append(path)
    return list(dict.fromkeys(paths))


def _stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return f"avg={statistics.fmean(values):.0f}ms p95={p95:.0f}ms max={max(values):.0f}ms"


def print_text(results: list[FileResult]) -> None:
    severity_rank = {"ERROR": 0, "WARN": 1, "INFO": 2}
    for result in results:
        print(f"\n=== {result.path} [{result.format}] ===")
        print(
            f"RTC sessions={result.rtc_sessions}, steps={result.steps}, triggers={result.triggers}, "
            f"done={result.completions}, fused={result.fusions}"
        )
        if result.chunks:
            print(f"chunk frames: min={min(result.chunks)}, median={statistics.median(result.chunks):g}, max={max(result.chunks)}")
        if result.infer_ms:
            print(f"infer latency: {_stats(result.infer_ms)}")
        if result.obs_ms:
            print(f"obs latency:   {_stats(result.obs_ms)}")
        if result.fault_events or result.empty_get_actions:
            print(f"sync faults={result.fault_events}, empty GetActions={result.empty_get_actions}")
        for note in result.notes:
            print(f"NOTE: {note}")
        if not result.issues:
            print("OK: 未发现可由当前日志证实的异常")
        for issue in sorted(result.issues, key=lambda item: (severity_rank[item.severity], item.line or 0)):
            location = f"line {issue.line}" if issue.line else "file"
            run = f", run={issue.run_id}" if issue.run_id else ""
            print(f"{issue.severity} [{issue.code}] ({location}{run}) {issue.message}")

    errors = sum(issue.severity == "ERROR" for result in results for issue in result.issues)
    warnings = sum(issue.severity == "WARN" for result in results for issue in result.issues)
    print(f"\nSUMMARY: files={len(results)}, errors={errors}, warnings={warnings}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="分析 VLA server 控制台/JSONL 日志中的 RTC 阻塞、buffer 耗尽和 chunk 丢失迹象"
    )
    parser.add_argument("paths", nargs="+", help="日志文件、目录或 glob（目录会读取 *.log/*.txt/*.jsonl）")
    parser.add_argument("--dt-ms", type=float, default=33.0, help="RTC 单动作周期，默认 33ms")
    parser.add_argument("--stall-ms", type=float, default=250.0, help="RTC step 阻塞阈值，默认 250ms")
    parser.add_argument("--json-stall-ms", type=float, default=3000.0, help="JSONL 相邻记录间隔告警阈值，默认 3000ms")
    parser.add_argument("--expected-chunk-size", type=int, help="期望每个推理 chunk 的动作数；默认从 initial/众数推断")
    parser.add_argument(
        "--infer-timeout-ms",
        type=float,
        default=20000.0,
        help="同步推理 elapsed_ms 告警阈值，默认 20000（action_response_timeout 为 30s）",
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    parser.add_argument("--strict", action="store_true", help="有 WARN 时也返回退出码 1")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dt_ms <= 0 or args.stall_ms <= 0 or args.json_stall_ms <= 0 or args.infer_timeout_ms <= 0:
        print("阈值必须大于 0", file=sys.stderr)
        return 2
    paths = resolve_inputs(args.paths)
    if not paths:
        print("没有匹配到日志文件", file=sys.stderr)
        return 2
    results = [analyze_file(path, args) for path in paths]
    if args.json:
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    else:
        print_text(results)
    has_error = any(issue.severity == "ERROR" for result in results for issue in result.issues)
    has_warning = any(issue.severity == "WARN" for result in results for issue in result.issues)
    return 1 if has_error or (args.strict and has_warning) else 0


if __name__ == "__main__":
    raise SystemExit(main())
