#!/usr/bin/env python3
"""分析 Codex Desktop JSONL 会话耗时。

日志不会暴露私有 reasoning 内容，也不会暴露模型内部精确执行时间。
本工具只报告可观测事件、工具调用区间、从 shell 输出解析出的 wall time，
以及根据事件空档推断出的非工具时间。
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SESSIONS_ROOT = Path.home() / ".codex" / "sessions"
WALL_TIME_RE = re.compile(r"^Wall time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds?", re.MULTILINE)
EXIT_CODE_RE = re.compile(r"^Exit code:\s*(-?[0-9]+)", re.MULTILINE)
MAX_LABEL_LENGTH = 92
LARGE_TURN_OUTPUT_CHARS = 100_000
LARGE_SINGLE_OUTPUT_CHARS = 50_000
SLOW_TOOL_UNION_SECONDS = 30.0
LONG_POST_TOOL_GAP_SECONDS = 20.0
HIGH_INPUT_TOKENS = 120_000
HIGH_REASONING_TOKENS = 8_000
SLOW_FIRST_ACTION_SECONDS = 20.0
MANY_TOOL_CALLS_PER_TURN = 50
BOTTLENECK_LABELS_ZH = {
    "tool_timeout": "命令超时",
    "tool_runtime": "命令运行耗时",
    "tool_output_size": "工具输出过大",
    "post_tool_analysis": "工具输出后分析耗时",
    "model_context_or_reasoning": "上下文或推理成本高",
    "retries_or_failures": "失败重试偏多",
    "first_action_latency": "开始行动偏慢",
    "balanced_or_low_signal": "暂无明显单一瓶颈",
    "no_turn_data": "没有轮次数据",
}
RISK_LEVEL_LABELS_ZH = {
    "green": "绿色：只读观察",
    "yellow": "黄色：人工确认",
    "red": "红色：暂不自动执行",
}
BOTTLENECK_ADVICE_ZH = {
    "tool_timeout": {
        "advice": "先查失败命令、timeout 和 runner preflight；必要时把长任务拆成短步骤并把完整日志落盘。",
        "quality_risk": "不能因为超时就跳过验证；修复后仍要复跑同等覆盖的测试或 runner。",
    },
    "tool_runtime": {
        "advice": "优先优化真实慢命令本身，例如 Godot runner、截图生成或外部进程启动；先保留 baseline 再 A/B 对比。",
        "quality_risk": "不能用更少测试场景换速度；若缩短 runner，必须证明覆盖范围不变。",
    },
    "tool_output_size": {
        "advice": "采用完整清单/计数 + 限量预览 + 必要时定点展开；日志和大报告写文件，HTML 只显示摘要。",
        "quality_risk": "最大风险是漏查。被截断的 rg、Select-String、git diff 预览不能作为全量无问题证据。",
    },
    "post_tool_analysis": {
        "advice": "让工具直接输出结构化摘要、失败原因和下一步；减少需要模型二次筛选的大段状态输出。",
        "quality_risk": "摘要必须能追溯到原始证据路径；不能隐藏失败日志或丢掉 touched files。",
    },
    "model_context_or_reasoning": {
        "advice": "把长上下文压成短 handoff：事实源、决策、证据路径、阻塞项、touched files；后续按需回读。",
        "quality_risk": "压缩不能丢关键约束、用户决策或验证证据；有疑问时回读原文档。",
    },
    "retries_or_failures": {
        "advice": "先分类失败原因，修正命令、路径或环境，再继续主线；重复失败应查错题集并沉淀。",
        "quality_risk": "不要把失败输出静默忽略；失败路径本身可能是根因证据。",
    },
    "first_action_latency": {
        "advice": "减少开局重复读大上下文；用知识索引定位后只读相关段落，再开始行动。",
        "quality_risk": "不能省略 AGENTS、知识索引和必要 SOP；只减少重复全文读取。",
    },
    "balanced_or_low_signal": {
        "advice": "暂不优化，继续观察更多样本；把精力放到高输出、超时或失败明显的 turn。",
        "quality_risk": "不要为了追求指标改动低信号流程，避免制造新风险。",
    },
    "no_turn_data": {
        "advice": "补齐日志轮次元数据后再判断；当前只能看会话级工具和空档。",
        "quality_risk": "缺轮次数据时不要下过细结论。",
    },
}

READ_ONLY_DASHBOARD_NOTICE = (
    "风险看板只做只读提示，不改变命令、不跳过检查、不自动限流；红色/黄色项需要人工判断。"
)


@dataclass
class ObservableEvent:
    timestamp: str
    offset_seconds: float
    record_type: str
    payload_type: str
    label: str
    turn_id: str = ""
    detail: str = ""


@dataclass
class ToolCall:
    call_id: str
    name: str
    started_at: str
    start_offset_seconds: float
    arguments: str
    turn_id: str = ""
    command: str = ""
    workdir: str = ""
    timeout_ms: int | None = None
    ended_at: str | None = None
    end_offset_seconds: float | None = None
    logged_duration_seconds: float | None = None
    shell_wall_seconds: float | None = None
    exit_code: int | None = None
    timed_out: bool = False
    output_char_count: int = 0
    output_line_count: int = 0
    output_preview: str = ""


@dataclass
class TimelineBucket:
    second: int
    start_offset_seconds: float
    end_offset_seconds: float
    category: str
    label: str
    call_ids: list[str]


def parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def seconds_between(start: datetime, end: datetime) -> float:
    return max(0.0, (end - start).total_seconds())


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {remainder:.1f}s"


def shorten(text: str, limit: int = MAX_LABEL_LENGTH) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record: {exc}") from exc
            if "timestamp" not in record:
                continue
            records.append(record)
    records.sort(key=lambda record: parse_timestamp(str(record["timestamp"])))
    return records


def find_latest_session_log(sessions_root: Path) -> Path:
    candidates = list_session_logs(sessions_root)
    if not candidates:
        raise FileNotFoundError(f"{sessions_root} 下没有 rollout-*.jsonl 文件")
    return candidates[0]


def list_session_logs(sessions_root: Path, limit: int | None = None) -> list[Path]:
    candidates = [path for path in sessions_root.rglob("rollout-*.jsonl") if path.is_file()]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if limit is not None:
        return candidates[:limit]
    return candidates


def parse_arguments(arguments: str) -> dict[str, Any]:
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"raw": arguments}
    return parsed if isinstance(parsed, dict) else {"raw": parsed}


def command_from_arguments(name: str, arguments: str) -> tuple[str, str, int | None]:
    if name == "apply_patch":
        return summarize_patch_input(arguments), "", None

    parsed = parse_arguments(arguments)
    workdir = str(parsed.get("workdir", ""))
    timeout = parsed.get("timeout_ms")
    timeout_ms = timeout if isinstance(timeout, int) else None

    if name == "shell_command":
        command = str(parsed.get("command", ""))
    elif "command" in parsed:
        command = str(parsed["command"])
    elif "prompt" in parsed:
        command = str(parsed["prompt"])
    elif "raw" in parsed:
        command = str(parsed["raw"])
    else:
        command = arguments
    return command, workdir, timeout_ms


def summarize_patch_input(patch_text: str) -> str:
    paths: list[str] = []
    for line in patch_text.splitlines():
        for marker in ("*** Add File: ", "*** Update File: ", "*** Delete File: "):
            if line.startswith(marker):
                paths.append(line.removeprefix(marker).strip())
                break
    if not paths:
        return "apply_patch"
    if len(paths) == 1:
        return f"apply_patch {paths[0]}"
    return f"apply_patch {len(paths)} files: {', '.join(paths[:3])}"


def parse_shell_output(output: str) -> tuple[int | None, float | None, bool, int, int, str]:
    exit_match = EXIT_CODE_RE.search(output)
    wall_match = WALL_TIME_RE.search(output)
    exit_code = int(exit_match.group(1)) if exit_match else None
    wall_seconds = float(wall_match.group(1)) if wall_match else None
    timed_out = exit_code == 124 or "command timed out" in output.lower()
    char_count = len(output)
    line_count = len(output.splitlines())
    preview = shorten(output, 180)
    return exit_code, wall_seconds, timed_out, char_count, line_count, preview


def extract_turn_id(record: dict[str, Any]) -> str:
    payload = record.get("payload")
    if isinstance(payload, dict):
        metadata = payload.get("internal_chat_message_metadata_passthrough")
        if isinstance(metadata, dict) and metadata.get("turn_id"):
            return str(metadata["turn_id"])
        if payload.get("turn_id"):
            return str(payload["turn_id"])
    return ""


def annotate_records_with_turns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    current_turn_id = ""
    for record in records:
        payload = record.get("payload")
        payload_type = payload.get("type") if isinstance(payload, dict) else ""
        explicit_turn_id = extract_turn_id(record)
        if payload_type == "task_started" and explicit_turn_id:
            current_turn_id = explicit_turn_id
        turn_id = explicit_turn_id or current_turn_id
        annotated.append({"record": record, "turn_id": turn_id})
        if payload_type in ("task_complete", "turn_aborted") and explicit_turn_id == current_turn_id:
            current_turn_id = ""
    return annotated


def event_label(record: dict[str, Any]) -> tuple[str, str]:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return str(record.get("type", "")), ""

    payload_type = str(payload.get("type", ""))
    if payload_type == "user_message":
        return "user_message", shorten(payload.get("message", ""))
    if payload_type == "agent_message":
        return "agent_message", shorten(payload.get("message", ""))
    if payload_type == "reasoning":
        return "reasoning_marker", "加密 reasoning 条目的时间戳"
    if payload_type == "function_call":
        name = str(payload.get("name", "tool"))
        command, _, _ = command_from_arguments(name, str(payload.get("arguments", "")))
        return f"tool_call:{name}", shorten(command or name)
    if payload_type == "custom_tool_call":
        name = str(payload.get("name", "tool"))
        command, _, _ = command_from_arguments(name, str(payload.get("input", "")))
        return f"tool_call:{name}", shorten(command or name)
    if payload_type == "function_call_output":
        return "tool_output", str(payload.get("call_id", ""))
    if payload_type == "custom_tool_call_output":
        return "tool_output", str(payload.get("call_id", ""))
    if payload_type == "token_count":
        total = payload.get("info", {}).get("total_token_usage", {}) if isinstance(payload.get("info"), dict) else {}
        return "token_count", f"total_tokens={total.get('total_tokens', 'unknown')}"
    if payload_type:
        return payload_type, ""
    return str(record.get("type", "")), ""


def build_events(annotated_records: list[dict[str, Any]], base_time: datetime) -> list[ObservableEvent]:
    events: list[ObservableEvent] = []
    for annotated in annotated_records:
        record = annotated["record"]
        timestamp = parse_timestamp(str(record["timestamp"]))
        payload = record.get("payload")
        payload_type = str(payload.get("type", "")) if isinstance(payload, dict) else ""
        label, detail = event_label(record)
        events.append(
            ObservableEvent(
                timestamp=isoformat_utc(timestamp),
                offset_seconds=seconds_between(base_time, timestamp),
                record_type=str(record.get("type", "")),
                payload_type=payload_type,
                label=label,
                turn_id=str(annotated.get("turn_id", "")),
                detail=detail,
            )
        )
    return events


def build_tool_calls(annotated_records: list[dict[str, Any]], base_time: datetime) -> list[ToolCall]:
    calls: list[ToolCall] = []
    by_id: dict[str, ToolCall] = {}

    for annotated in annotated_records:
        record = annotated["record"]
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        payload_type = payload.get("type")
        timestamp = parse_timestamp(str(record["timestamp"]))

        if payload_type in ("function_call", "custom_tool_call"):
            call_id = str(payload.get("call_id", ""))
            name = str(payload.get("name", "tool"))
            arguments = str(payload.get("arguments", payload.get("input", "")))
            command, workdir, timeout_ms = command_from_arguments(name, arguments)
            call = ToolCall(
                call_id=call_id,
                name=name,
                started_at=isoformat_utc(timestamp),
                start_offset_seconds=seconds_between(base_time, timestamp),
                arguments=shorten(arguments, 500),
                turn_id=str(annotated.get("turn_id", "")),
                command=command,
                workdir=workdir,
                timeout_ms=timeout_ms,
            )
            calls.append(call)
            by_id[call_id] = call
            continue

        if payload_type in ("function_call_output", "custom_tool_call_output"):
            call_id = str(payload.get("call_id", ""))
            call = by_id.get(call_id)
            output = str(payload.get("output", ""))
            exit_code, wall_seconds, timed_out, char_count, line_count, preview = parse_shell_output(output)
            if call is None:
                call = ToolCall(
                    call_id=call_id,
                    name="unknown_tool",
                    started_at=isoformat_utc(timestamp),
                    start_offset_seconds=seconds_between(base_time, timestamp),
                    arguments="",
                    turn_id=str(annotated.get("turn_id", "")),
                )
                calls.append(call)
                by_id[call_id] = call
            call.ended_at = isoformat_utc(timestamp)
            call.end_offset_seconds = seconds_between(base_time, timestamp)
            call.logged_duration_seconds = call.end_offset_seconds - call.start_offset_seconds
            call.shell_wall_seconds = wall_seconds
            call.exit_code = exit_code
            call.timed_out = timed_out
            call.output_char_count = char_count
            call.output_line_count = line_count
            call.output_preview = preview

    return calls


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    valid = sorted((start, end) for start, end in intervals if end > start)
    if not valid:
        return []
    merged = [valid[0]]
    for start, end in valid[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def build_timeline(
    calls: list[ToolCall],
    total_seconds: float,
    bucket_seconds: int,
    events: list[ObservableEvent],
    max_buckets: int | None = None,
) -> list[TimelineBucket]:
    if bucket_seconds <= 0:
        raise ValueError("bucket_seconds must be > 0")

    tool_spans = [
        call
        for call in calls
        if call.end_offset_seconds is not None and call.end_offset_seconds > call.start_offset_seconds
    ]
    buckets: list[TimelineBucket] = []
    count = int(total_seconds // bucket_seconds) + 1
    generated_count = min(count, max_buckets) if max_buckets is not None else count
    for index in range(generated_count):
        start = index * bucket_seconds
        end = min(total_seconds, start + bucket_seconds)
        if end <= start:
            continue
        overlapping = [
            call
            for call in tool_spans
            if overlap_seconds(start, end, call.start_offset_seconds, call.end_offset_seconds or call.start_offset_seconds)
        ]
        instant_events = [
            event
            for event in events
            if start <= event.offset_seconds < end and not event.label.startswith("tool_")
        ]
        if len(overlapping) > 1:
            labels = [shorten(call.command or call.name, 38) for call in overlapping[:3]]
            buckets.append(
                TimelineBucket(
                    second=index * bucket_seconds,
                    start_offset_seconds=start,
                    end_offset_seconds=end,
                    category="tool_parallel",
                    label=" | ".join(labels),
                    call_ids=[call.call_id for call in overlapping],
                )
            )
        elif len(overlapping) == 1:
            call = overlapping[0]
            buckets.append(
                TimelineBucket(
                    second=index * bucket_seconds,
                    start_offset_seconds=start,
                    end_offset_seconds=end,
                    category=f"tool:{call.name}",
                    label=shorten(call.command or call.name),
                    call_ids=[call.call_id],
                )
            )
        elif instant_events:
            labels = [event.label for event in instant_events[:3]]
            buckets.append(
                TimelineBucket(
                    second=index * bucket_seconds,
                    start_offset_seconds=start,
                    end_offset_seconds=end,
                    category="observable_event",
                    label=", ".join(labels),
                    call_ids=[],
                )
            )
        else:
            buckets.append(
                TimelineBucket(
                    second=index * bucket_seconds,
                    start_offset_seconds=start,
                    end_offset_seconds=end,
                    category="agent_thinking_or_waiting_inferred",
                    label="这个时间段没有可观测工具事件",
                    call_ids=[],
                )
            )
    return buckets


def build_gaps(
    events: list[ObservableEvent],
    calls: list[ToolCall],
    threshold_seconds: float,
) -> list[dict[str, Any]]:
    tool_intervals = merge_intervals(
        [
            (call.start_offset_seconds, call.end_offset_seconds)
            for call in calls
            if call.end_offset_seconds is not None
        ]
    )
    gaps: list[dict[str, Any]] = []
    for previous, current in zip(events, events[1:]):
        duration = current.offset_seconds - previous.offset_seconds
        if duration < threshold_seconds:
            continue
        covered_by_tool = any(
            overlap_seconds(previous.offset_seconds, current.offset_seconds, start, end) > 0
            for start, end in tool_intervals
        )
        if covered_by_tool:
            continue
        gaps.append(
            {
                "start_offset_seconds": previous.offset_seconds,
                "end_offset_seconds": current.offset_seconds,
                "duration_seconds": duration,
                "after": previous.label,
                "before": current.label,
            }
        )
    gaps.sort(key=lambda item: item["duration_seconds"], reverse=True)
    return gaps


def session_metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for record in records:
        if record.get("type") == "session_meta" and isinstance(record.get("payload"), dict):
            payload = record["payload"]
            metadata = {
                "session_id": payload.get("session_id") or payload.get("id", ""),
                "cwd": payload.get("cwd", ""),
                "originator": payload.get("originator", ""),
                "cli_version": payload.get("cli_version", ""),
                "source": payload.get("source", ""),
                "model_provider": payload.get("model_provider", ""),
            }
            break
    turn_ids = sorted({turn_id for turn_id in (extract_turn_id(record) for record in records) if turn_id})
    metadata["turn_ids"] = turn_ids
    return metadata


def _empty_token_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0,
    }


def _token_usage_from_payload(payload: dict[str, Any]) -> dict[str, int]:
    info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
    usage = info.get("last_token_usage") if isinstance(info.get("last_token_usage"), dict) else {}
    if not usage:
        usage = info.get("total_token_usage") if isinstance(info.get("total_token_usage"), dict) else {}
    parsed = _empty_token_usage()
    for key in parsed:
        value = usage.get(key)
        parsed[key] = int(value) if isinstance(value, int) else 0
    return parsed


def _add_token_usage(total: dict[str, int], increment: dict[str, int]) -> None:
    for key, value in increment.items():
        total[key] = total.get(key, 0) + value


def _first_action_offset(events: list[ObservableEvent], anchor_offset: float) -> float | None:
    for event in events:
        if event.offset_seconds < anchor_offset:
            continue
        if event.label == "agent_message" or event.label.startswith("tool_call:"):
            return event.offset_seconds
    return None


def _post_tool_gaps(events: list[ObservableEvent], turn_end_offset: float) -> tuple[float, float, int]:
    intervals: list[tuple[float, float]] = []
    for index, event in enumerate(events):
        if event.label != "tool_output":
            continue
        next_offset: float | None = None
        for next_event in events[index + 1 :]:
            if next_event.label == "agent_message" or next_event.label.startswith("tool_call:"):
                next_offset = next_event.offset_seconds
                break
        if next_offset is None:
            next_offset = turn_end_offset
        if next_offset > event.offset_seconds:
            intervals.append((event.offset_seconds, next_offset))
    merged = merge_intervals(intervals)
    total = sum(end - start for start, end in merged)
    maximum = max((end - start for start, end in merged), default=0.0)
    return total, maximum, len(merged)


def _diagnose_turn(turn: dict[str, Any]) -> tuple[str, str]:
    active = float(turn.get("active_duration_seconds") or 0.0)
    tool_union = float(turn.get("tool_occupied_union_seconds") or 0.0)
    post_tool = float(turn.get("post_tool_gap_seconds") or 0.0)
    output_chars = int(turn.get("output_char_count") or 0)
    max_output_chars = int(turn.get("max_output_char_count") or 0)
    tokens = turn.get("token_usage", {}) if isinstance(turn.get("token_usage"), dict) else {}
    input_tokens = int(tokens.get("input_tokens") or 0)
    reasoning_tokens = int(tokens.get("reasoning_output_tokens") or 0)
    time_to_first_action = turn.get("time_to_first_action_seconds")

    if int(turn.get("timeout_count") or 0) > 0:
        return "tool_timeout", "至少有一个工具或命令超时。"
    if tool_union >= SLOW_TOOL_UNION_SECONDS and tool_union >= active * 0.35:
        return "tool_runtime", "工具执行占用了这一轮的大部分活跃时间。"
    if output_chars >= LARGE_TURN_OUTPUT_CHARS or max_output_chars >= LARGE_SINGLE_OUTPUT_CHARS:
        return "tool_output_size", "命令输出太大，后续模型读取和分析会变慢。"
    if post_tool >= LONG_POST_TOOL_GAP_SECONDS and post_tool >= tool_union * 0.75:
        return "post_tool_analysis", "主要非工具耗时出现在工具输出之后，通常是模型在消化结果。"
    if input_tokens >= HIGH_INPUT_TOKENS or reasoning_tokens >= HIGH_REASONING_TOKENS:
        return "model_context_or_reasoning", "输入上下文或推理 token 偏高，模型处理成本可能较大。"
    if int(turn.get("failed_tool_call_count") or 0) >= 2:
        return "retries_or_failures", "多个工具调用失败，可能造成重复排查和重试。"
    if isinstance(time_to_first_action, (int, float)) and time_to_first_action >= SLOW_FIRST_ACTION_SECONDS:
        return "first_action_latency", "从用户消息到第一次可观测动作等待较久。"
    return "balanced_or_low_signal", "从可观测日志看不出单一主导瓶颈。"


def _tool_output_pattern(call: dict[str, Any]) -> tuple[str, str]:
    name = str(call.get("name") or "")
    command = str(call.get("command") or "")
    text = f"{name} {command}".lower()
    if "view_image" in text:
        return "image_review", "图片查看"
    if "git diff" in text:
        return "git_diff", "git diff"
    if "git status" in text:
        return "git_status", "git status"
    if "search_knowledge" in text:
        return "knowledge_search", "知识索引搜索"
    if "check_commit_readiness" in text:
        return "commit_readiness", "提交门禁检查"
    if "check_knowledge_capture" in text:
        return "knowledge_capture", "知识沉淀检查"
    if "validate_knowledge_governance" in text:
        return "knowledge_governance", "知识治理校验"
    if "select-string" in text:
        return "select_string", "Select-String 文本查询"
    if re.search(r"(^|\s)rg(\.exe)?(\s|$)", command.lower()):
        return "rg_search", "rg 文本查询"
    if "godot" in text and ("capture" in text or "screenshot" in text):
        return "godot_capture", "Godot 截图/捕获 runner"
    if "godot" in text:
        return "godot_runner", "Godot runner"
    if "get-content" in text:
        return "get_content", "Get-Content 文件读取"
    return "other", "其他工具/命令"


def _aggregate_tool_output_patterns(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patterns: dict[str, dict[str, Any]] = {}
    for call in calls:
        code, label = _tool_output_pattern(call)
        item = patterns.setdefault(
            code,
            {
                "code": code,
                "label": label,
                "call_count": 0,
                "output_char_count": 0,
                "output_line_count": 0,
                "failed_call_count": 0,
                "timeout_count": 0,
                "shell_wall_seconds": 0.0,
                "logged_duration_seconds": 0.0,
            },
        )
        item["call_count"] += 1
        item["output_char_count"] += int(call.get("output_char_count") or 0)
        item["output_line_count"] += int(call.get("output_line_count") or 0)
        if call.get("exit_code") not in (None, 0):
            item["failed_call_count"] += 1
        if call.get("timed_out"):
            item["timeout_count"] += 1
        item["shell_wall_seconds"] += float(call.get("shell_wall_seconds") or 0.0)
        item["logged_duration_seconds"] += float(call.get("logged_duration_seconds") or 0.0)
    return sorted(
        patterns.values(),
        key=lambda item: (
            int(item.get("output_char_count") or 0),
            float(item.get("shell_wall_seconds") or 0.0),
            int(item.get("call_count") or 0),
        ),
        reverse=True,
    )


def risk_level_label_zh(level: Any) -> str:
    return RISK_LEVEL_LABELS_ZH.get(str(level), str(level) if level else "未知")


def _risk_level_for_turn(turn: dict[str, Any], patterns: list[dict[str, Any]]) -> dict[str, str]:
    bottleneck = str(turn.get("suspected_bottleneck") or "balanced_or_low_signal")
    top_pattern = str(patterns[0].get("code") if patterns else "")
    if bottleneck in {"tool_timeout", "tool_runtime"}:
        return {
            "level": "red",
            "reason": "优化可能改变验证路径、runner 行为或超时处理，不能自动执行。",
        }
    if top_pattern in {"image_review", "godot_capture", "godot_runner"}:
        return {
            "level": "red",
            "reason": "涉及截图、Godot runner 或可见验收证据，必须保留完整日志和 full-size PNG。",
        }
    if bottleneck == "tool_output_size":
        return {
            "level": "yellow",
            "reason": "输出优化可能把预览误当全集；只能人工确认完整清单和命中数。",
        }
    if bottleneck in {"post_tool_analysis", "model_context_or_reasoning", "first_action_latency", "retries_or_failures"}:
        return {
            "level": "yellow",
            "reason": "优化可能改变信息组织方式；必须能追溯原始证据和 touched files。",
        }
    return {
        "level": "green",
        "reason": "当前只建议继续观察或增强报告，不需要改变执行路径。",
    }


def _optimization_advice_for_turn(turn: dict[str, Any]) -> dict[str, Any]:
    bottleneck = str(turn.get("suspected_bottleneck") or "balanced_or_low_signal")
    advice = dict(BOTTLENECK_ADVICE_ZH.get(bottleneck, BOTTLENECK_ADVICE_ZH["balanced_or_low_signal"]))
    patterns = turn.get("tool_output_patterns", []) if isinstance(turn.get("tool_output_patterns"), list) else []
    risk = _risk_level_for_turn(turn, patterns)
    safeguards: list[str] = []
    if int(turn.get("output_char_count") or 0) >= LARGE_TURN_OUTPUT_CHARS:
        safeguards.append("先保留完整命中文件清单/命中数，再限量展开片段。")
    if int(turn.get("max_output_char_count") or 0) >= LARGE_SINGLE_OUTPUT_CHARS:
        safeguards.append("单次大输出应落盘或结构化摘要，不能只看截断预览。")
    if int(turn.get("timeout_count") or 0) > 0:
        safeguards.append("超时命令修复后必须复跑同等覆盖的验证。")
    if int(turn.get("failed_tool_call_count") or 0) > 0:
        safeguards.append("失败输出必须保留为根因证据，不要静默忽略。")
    if int(turn.get("tool_call_count") or 0) >= MANY_TOOL_CALLS_PER_TURN:
        safeguards.append("工具调用过碎时先合并读取计划，但不能减少影响范围。")
    if not safeguards:
        safeguards.append("保持现有品质门禁，先观察更多样本。")
    return {
        "bottleneck": bottleneck,
        "advice": advice["advice"],
        "quality_risk": advice["quality_risk"],
        "risk_level": risk["level"],
        "risk_label": risk_level_label_zh(risk["level"]),
        "risk_reason": risk["reason"],
        "read_only_notice": READ_ONLY_DASHBOARD_NOTICE,
        "safeguards": safeguards,
    }


def build_turn_summaries(
    annotated_records: list[dict[str, Any]],
    events: list[ObservableEvent],
    calls: list[ToolCall],
) -> list[dict[str, Any]]:
    turns: dict[str, dict[str, Any]] = {}

    def ensure_turn(turn_id: str) -> dict[str, Any]:
        if turn_id not in turns:
            turns[turn_id] = {
                "turn_id": turn_id,
                "status": "unknown",
                "started_at": "",
                "ended_at": "",
                "start_offset_seconds": None,
                "end_offset_seconds": None,
                "user_message_offset_seconds": None,
                "token_usage": _empty_token_usage(),
                "token_event_count": 0,
            }
        return turns[turn_id]

    for annotated in annotated_records:
        turn_id = str(annotated.get("turn_id") or "")
        if not turn_id:
            continue
        record = annotated["record"]
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        timestamp = parse_timestamp(str(record["timestamp"]))
        first_timestamp = parse_timestamp(str(annotated_records[0]["record"]["timestamp"]))
        offset = seconds_between(first_timestamp, timestamp)
        payload_type = str(payload.get("type", ""))
        turn = ensure_turn(turn_id)
        if turn["start_offset_seconds"] is None or offset < turn["start_offset_seconds"]:
            turn["start_offset_seconds"] = offset
            turn["started_at"] = isoformat_utc(timestamp)
        if turn["end_offset_seconds"] is None or offset > turn["end_offset_seconds"]:
            turn["end_offset_seconds"] = offset
            turn["ended_at"] = isoformat_utc(timestamp)
        if payload_type == "task_started":
            turn["status"] = "active"
            turn["start_offset_seconds"] = offset
            turn["started_at"] = isoformat_utc(timestamp)
        elif payload_type == "user_message" and turn["user_message_offset_seconds"] is None:
            turn["user_message_offset_seconds"] = offset
        elif payload_type == "task_complete":
            turn["status"] = "complete"
            turn["end_offset_seconds"] = offset
            turn["ended_at"] = isoformat_utc(timestamp)
        elif payload_type == "turn_aborted":
            turn["status"] = str(payload.get("reason") or "aborted")
            turn["end_offset_seconds"] = offset
            turn["ended_at"] = isoformat_utc(timestamp)
        elif payload_type == "token_count":
            _add_token_usage(turn["token_usage"], _token_usage_from_payload(payload))
            turn["token_event_count"] += 1

    events_by_turn: dict[str, list[ObservableEvent]] = {}
    for event in events:
        if event.turn_id:
            events_by_turn.setdefault(event.turn_id, []).append(event)

    calls_by_turn: dict[str, list[ToolCall]] = {}
    for call in calls:
        if call.turn_id:
            calls_by_turn.setdefault(call.turn_id, []).append(call)

    summaries: list[dict[str, Any]] = []
    for turn_id, turn in turns.items():
        turn_events = sorted(events_by_turn.get(turn_id, []), key=lambda event: event.offset_seconds)
        turn_calls = calls_by_turn.get(turn_id, [])
        start_offset = float(turn["start_offset_seconds"] or 0.0)
        end_offset = float(turn["end_offset_seconds"] if turn["end_offset_seconds"] is not None else start_offset)
        active_duration = max(0.0, end_offset - start_offset)
        anchor_offset = float(turn["user_message_offset_seconds"] if turn["user_message_offset_seconds"] is not None else start_offset)
        first_action = _first_action_offset(turn_events, anchor_offset)
        tool_intervals = merge_intervals(
            [
                (call.start_offset_seconds, call.end_offset_seconds)
                for call in turn_calls
                if call.end_offset_seconds is not None
            ]
        )
        tool_union = sum(end - start for start, end in tool_intervals)
        tool_cumulative = sum(call.logged_duration_seconds or 0.0 for call in turn_calls)
        shell_wall = sum(call.shell_wall_seconds or 0.0 for call in turn_calls)
        output_chars = sum(call.output_char_count for call in turn_calls)
        output_lines = sum(call.output_line_count for call in turn_calls)
        max_output = max((call.output_char_count for call in turn_calls), default=0)
        failed = [call for call in turn_calls if call.exit_code not in (None, 0)]
        timed_out = [call for call in turn_calls if call.timed_out]
        post_tool_total, post_tool_max, post_tool_count = _post_tool_gaps(turn_events, end_offset)
        slow_calls = sorted(
            [asdict(call) for call in turn_calls],
            key=lambda call: max(call.get("shell_wall_seconds") or 0.0, call.get("logged_duration_seconds") or 0.0),
            reverse=True,
        )
        large_outputs = sorted(
            [asdict(call) for call in turn_calls],
            key=lambda call: call.get("output_char_count") or 0,
            reverse=True,
        )
        summary = {
            **turn,
            "active_duration_seconds": active_duration,
            "time_to_first_action_seconds": None if first_action is None else max(0.0, first_action - anchor_offset),
            "tool_occupied_union_seconds": tool_union,
            "tool_cumulative_logged_seconds": tool_cumulative,
            "tool_cumulative_shell_wall_seconds": shell_wall,
            "inferred_non_tool_active_seconds": max(0.0, active_duration - tool_union),
            "post_tool_gap_seconds": post_tool_total,
            "max_post_tool_gap_seconds": post_tool_max,
            "post_tool_gap_count": post_tool_count,
            "tool_call_count": len(turn_calls),
            "failed_tool_call_count": len(failed),
            "timeout_count": len(timed_out),
            "output_char_count": output_chars,
            "output_line_count": output_lines,
            "max_output_char_count": max_output,
            "top_slow_tool_calls": slow_calls[:5],
            "top_large_output_calls": large_outputs[:5],
        }
        bottleneck, reason = _diagnose_turn(summary)
        summary["suspected_bottleneck"] = bottleneck
        summary["bottleneck_reason"] = reason
        summary["tool_output_patterns"] = _aggregate_tool_output_patterns([asdict(call) for call in turn_calls])
        summary["optimization_advice"] = _optimization_advice_for_turn(summary)
        summaries.append(summary)

    summaries.sort(key=lambda turn: float(turn.get("start_offset_seconds") or 0.0))
    previous_end: float | None = None
    for turn in summaries:
        start = float(turn.get("start_offset_seconds") or 0.0)
        turn["idle_gap_before_seconds"] = 0.0 if previous_end is None else max(0.0, start - previous_end)
        previous_end = float(turn.get("end_offset_seconds") if turn.get("end_offset_seconds") is not None else start)
    return summaries


def analyze_session(
    path: Path,
    bucket_seconds: int,
    gap_threshold_seconds: float,
    timeline_bucket_limit: int | None = None,
) -> dict[str, Any]:
    records = load_jsonl(path)
    if not records:
        raise ValueError(f"{path} has no timestamped records")

    base_time = parse_timestamp(str(records[0]["timestamp"]))
    end_time = parse_timestamp(str(records[-1]["timestamp"]))
    total_seconds = seconds_between(base_time, end_time)
    annotated_records = annotate_records_with_turns(records)
    events = build_events(annotated_records, base_time)
    calls = build_tool_calls(annotated_records, base_time)
    total_timeline_bucket_count = int(total_seconds // bucket_seconds) + 1 if total_seconds > 0 else 0
    timeline = build_timeline(
        calls,
        total_seconds,
        bucket_seconds,
        events,
        max_buckets=timeline_bucket_limit,
    )
    gaps = build_gaps(events, calls, gap_threshold_seconds)

    logged_intervals = merge_intervals(
        [
            (call.start_offset_seconds, call.end_offset_seconds)
            for call in calls
            if call.end_offset_seconds is not None
        ]
    )
    tool_union_seconds = sum(end - start for start, end in logged_intervals)
    cumulative_logged_seconds = sum(call.logged_duration_seconds or 0.0 for call in calls)
    cumulative_shell_wall_seconds = sum(call.shell_wall_seconds or 0.0 for call in calls)
    failed_tool_calls = [call for call in calls if call.exit_code not in (None, 0)]
    turn_summaries = build_turn_summaries(annotated_records, events, calls)
    tool_calls_as_dicts = [asdict(call) for call in calls]
    token_usage_total = _empty_token_usage()
    for turn in turn_summaries:
        token_usage = turn.get("token_usage") if isinstance(turn.get("token_usage"), dict) else {}
        _add_token_usage(token_usage_total, {key: int(token_usage.get(key) or 0) for key in token_usage_total})

    return {
        "log_path": str(path),
        "metadata": session_metadata(records),
        "limits": [
            "Reasoning 内容是加密/私有的；非工具时间只能根据事件空档推断。",
            "工具耗时使用 function_call 到 function_call_output 的时间戳计算。",
            "shell_wall_seconds 只在工具输出包含 Wall time 行时才能解析。",
        ],
        "summary": {
            "started_at": isoformat_utc(base_time),
            "ended_at": isoformat_utc(end_time),
            "elapsed_seconds": total_seconds,
            "observable_event_count": len(events),
            "tool_call_count": len(calls),
            "failed_tool_call_count": len(failed_tool_calls),
            "tool_occupied_union_seconds": tool_union_seconds,
            "tool_cumulative_logged_seconds": cumulative_logged_seconds,
            "tool_cumulative_shell_wall_seconds": cumulative_shell_wall_seconds,
            "inferred_non_tool_seconds": max(0.0, total_seconds - tool_union_seconds),
            "timeline_bucket_count": total_timeline_bucket_count,
            "timeline_bucket_size_seconds": bucket_seconds,
            "output_char_count": sum(call.output_char_count for call in calls),
            "output_line_count": sum(call.output_line_count for call in calls),
            "token_usage": token_usage_total,
            "turn_count": len(turn_summaries),
        },
        "tool_output_patterns": _aggregate_tool_output_patterns(tool_calls_as_dicts),
        "tool_calls": tool_calls_as_dicts,
        "turn_summaries": turn_summaries,
        "longest_inferred_gaps": gaps,
        "timeline": [asdict(bucket) for bucket in timeline],
    }


def render_report(report: dict[str, Any], top: int, timeline_limit: int) -> str:
    metadata = report["metadata"]
    summary = report["summary"]
    tool_calls = report["tool_calls"]
    slow_calls = sorted(
        tool_calls,
        key=lambda call: max(
            call.get("shell_wall_seconds") or 0.0,
            call.get("logged_duration_seconds") or 0.0,
        ),
        reverse=True,
    )

    lines: list[str] = []
    lines.append("Codex 对话耗时报告")
    lines.append(f"日志: {report['log_path']}")
    if metadata.get("session_id"):
        lines.append(f"会话: {metadata['session_id']}")
    if metadata.get("cwd"):
        lines.append(f"工作目录: {metadata['cwd']}")
    if metadata.get("turn_ids"):
        lines.append(f"对话轮次: {len(metadata['turn_ids'])}")
    lines.append("")
    lines.append("限制说明:")
    for limit in report["limits"]:
        lines.append(f"- {limit}")
    lines.append("")
    lines.append("汇总:")
    lines.append(f"- 总耗时: {format_duration(summary['elapsed_seconds'])}")
    lines.append(f"- 工具占用时间（并集）: {format_duration(summary['tool_occupied_union_seconds'])}")
    lines.append(f"- 工具日志累计耗时: {format_duration(summary['tool_cumulative_logged_seconds'])}")
    lines.append(f"- Shell 实际累计耗时: {format_duration(summary['tool_cumulative_shell_wall_seconds'])}")
    lines.append(f"- 推断非工具时间: {format_duration(summary['inferred_non_tool_seconds'])}")
    lines.append(f"- 工具调用: {summary['tool_call_count']}（失败: {summary['failed_tool_call_count']}）")
    lines.append(f"- 工具输出: {summary.get('output_char_count', 0)} 字符 / {summary.get('output_line_count', 0)} 行")
    token_usage = summary.get("token_usage", {}) if isinstance(summary.get("token_usage"), dict) else {}
    if token_usage:
        lines.append(
            "- Token 用量: 输入={input_tokens} 缓存输入={cached_input_tokens} 推理={reasoning_output_tokens} "
            "输出={output_tokens}".format(**token_usage)
        )
    lines.append("")
    lines.append(f"输出源排行榜（前 {top} 条）:")
    for index, pattern in enumerate(report.get("tool_output_patterns", [])[:top], start=1):
        lines.append(
            f"{index}. {pattern.get('label')} 调用={pattern.get('call_count')} "
            f"输出={pattern.get('output_char_count', 0)} 字符 "
            f"Shell={format_duration(pattern.get('shell_wall_seconds'))} "
            f"失败/超时={pattern.get('failed_call_count', 0)}/{pattern.get('timeout_count', 0)}"
        )
    if not report.get("tool_output_patterns"):
        lines.append("（没有工具输出源数据）")
    lines.append("")
    risk_counts = risk_counts_for_turns(report.get("turn_summaries", []))
    lines.append(
        "只读风险看板: 红色={red} 黄色={yellow} 绿色={green}；{notice}".format(
            red=risk_counts.get("red", 0),
            yellow=risk_counts.get("yellow", 0),
            green=risk_counts.get("green", 0),
            notice=READ_ONLY_DASHBOARD_NOTICE,
        )
    )
    lines.append("")
    turn_summaries = sorted(
        report.get("turn_summaries", []),
        key=lambda turn: max(
            float(turn.get("active_duration_seconds") or 0.0),
            float(turn.get("tool_occupied_union_seconds") or 0.0),
            float(turn.get("post_tool_gap_seconds") or 0.0),
        ),
        reverse=True,
    )
    lines.append(f"按对话轮次诊断性能（前 {top} 条）:")
    for index, turn in enumerate(turn_summaries[:top], start=1):
        bottleneck_code = turn.get("suspected_bottleneck", "")
        lines.append(
            f"{index}. {bottleneck_label_zh(bottleneck_code)} active={format_duration(turn.get('active_duration_seconds'))} "
            f"开始行动={format_duration(turn.get('time_to_first_action_seconds'))} "
            f"工具占用={format_duration(turn.get('tool_occupied_union_seconds'))} "
            f"输出后等待={format_duration(turn.get('post_tool_gap_seconds'))} "
            f"输入tokens={turn.get('token_usage', {}).get('input_tokens', 0)} "
            f"输出字符={turn.get('output_char_count', 0)} :: {turn.get('bottleneck_reason')}"
        )
        advice = turn.get("optimization_advice") if isinstance(turn.get("optimization_advice"), dict) else {}
        if advice:
            lines.append(f"   建议: {advice.get('advice')}")
            lines.append(f"   品质风险: {advice.get('quality_risk')}")
    if not turn_summaries:
        lines.append("（没有轮次汇总）")
    lines.append("")
    lines.append(f"最慢工具调用（前 {top} 条）:")
    for index, call in enumerate(slow_calls[:top], start=1):
        logged = format_duration(call.get("logged_duration_seconds"))
        wall = format_duration(call.get("shell_wall_seconds"))
        exit_code = call.get("exit_code")
        command = shorten(call.get("command") or call.get("name") or call.get("call_id"))
        lines.append(f"{index}. {call.get('name')} 日志耗时={logged} Shell实际={wall} 退出码={exit_code} :: {command}")
    if not slow_calls:
        lines.append("（没有工具调用）")
    lines.append("")
    lines.append(f"最长推断非工具空档（前 {top} 条）:")
    for index, gap in enumerate(report["longest_inferred_gaps"][:top], start=1):
        lines.append(
            f"{index}. {format_duration(gap['duration_seconds'])} "
            f"after={gap['after']} before={gap['before']} "
            f"at +{gap['start_offset_seconds']:.1f}s"
        )
    if not report["longest_inferred_gaps"]:
        lines.append("（没有超过阈值的推断空档）")
    lines.append("")

    timeline = report["timeline"]
    total_timeline_count = report["summary"].get("timeline_bucket_count", len(timeline))
    if timeline_limit >= 0:
        shown = timeline[:timeline_limit]
    else:
        shown = timeline
    lines.append(f"时间线分桶（显示 {len(shown)}/{total_timeline_count}）:")
    for bucket in shown:
        start = format_duration(bucket["start_offset_seconds"])
        end = format_duration(bucket["end_offset_seconds"])
        lines.append(f"+{start:>8}..+{end:<8} {timeline_category_label(str(bucket['category']))}: {bucket['label']}")
    if len(shown) < total_timeline_count:
        lines.append(f"... 还有 {total_timeline_count - len(shown)} 个分桶被隐藏；传 --timeline-limit -1 可完整输出。")
    return "\n".join(lines)


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def percentage(value: float, total: float) -> str:
    if total <= 0:
        return "0.0%"
    return f"{(value / total * 100):.1f}%"


def format_count(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def bottleneck_label_zh(code: Any) -> str:
    return BOTTLENECK_LABELS_ZH.get(str(code), str(code) if code else "未知")


def format_tool_status(call: dict[str, Any]) -> str:
    if call.get("timed_out"):
        return "超时"
    exit_code = call.get("exit_code")
    if exit_code is None:
        return ""
    if exit_code == 0:
        return "成功"
    return f"失败 {exit_code}"


def timeline_category_label(category: str) -> str:
    if category.startswith("tool:"):
        return "工具：" + category.removeprefix("tool:")
    if category == "tool_parallel":
        return "并行工具"
    if category == "observable_event":
        return "可观测事件"
    if category == "agent_thinking_or_waiting_inferred":
        return "推断非工具时间"
    return category


def sorted_tool_calls(report: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        report["tool_calls"],
        key=lambda call: max(
            call.get("shell_wall_seconds") or 0.0,
            call.get("logged_duration_seconds") or 0.0,
        ),
        reverse=True,
    )


def turn_cost_score(turn: dict[str, Any]) -> float:
    tokens = turn.get("token_usage", {}) if isinstance(turn.get("token_usage"), dict) else {}
    return (
        float(turn.get("active_duration_seconds") or 0.0)
        + float(turn.get("tool_occupied_union_seconds") or 0.0)
        + float(turn.get("post_tool_gap_seconds") or 0.0)
        + (int(turn.get("output_char_count") or 0) / 10_000)
        + (int(tokens.get("input_tokens") or 0) / 50_000)
    )


def sorted_turn_summaries(report: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(report.get("turn_summaries", []), key=turn_cost_score, reverse=True)


def report_title(report: dict[str, Any]) -> str:
    metadata = report.get("metadata", {})
    session_id = metadata.get("session_id") or Path(str(report.get("log_path", ""))).stem
    started_at = report.get("summary", {}).get("started_at", "")
    cwd = metadata.get("cwd", "")
    if cwd:
        return f"{started_at} | {Path(cwd).name} | {session_id}"
    return f"{started_at} | {session_id}"


def report_matches_project_filter(report: dict[str, Any], project_filter: str) -> bool:
    if not project_filter:
        return True
    needle = project_filter.lower()
    metadata = report.get("metadata", {}) if isinstance(report.get("metadata"), dict) else {}
    haystacks = [
        str(metadata.get("cwd") or ""),
        str(metadata.get("session_id") or ""),
        str(report.get("log_path") or ""),
        report_title(report),
    ]
    return any(needle in value.lower() for value in haystacks)


def filter_reports_by_project(reports: list[dict[str, Any]], project_filter: str) -> list[dict[str, Any]]:
    if not project_filter:
        return reports
    return [report for report in reports if report_matches_project_filter(report, project_filter)]


def render_metric_card(label: str, value: str, note: str = "") -> str:
    note_html = f"<small>{escape(note)}</small>" if note else ""
    return f'<div class="metric"><span>{escape(label)}</span><strong>{escape(value)}</strong>{note_html}</div>'


def render_tool_table(calls: list[dict[str, Any]], top: int) -> str:
    if not calls:
        return '<p class="muted">没有工具调用。</p>'
    rows: list[str] = []
    for call in calls[:top]:
        exit_code = call.get("exit_code")
        timed_out = bool(call.get("timed_out"))
        row_class = "failed" if timed_out or exit_code not in (None, 0) else ""
        status = format_tool_status(call)
        command = call.get("command") or call.get("name") or call.get("call_id")
        rows.append(
            "<tr class=\"%s\"><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td class=\"nowrap\">%s 字符 / %s 行</td><td><code>%s</code></td></tr>"
            % (
                row_class,
                escape(call.get("name", "")),
                escape(format_duration(call.get("logged_duration_seconds"))),
                escape(format_duration(call.get("shell_wall_seconds"))),
                escape(status),
                escape(format_count(call.get("output_char_count"))),
                escape(format_count(call.get("output_line_count"))),
                escape(command),
            )
        )
    return (
        '<table><thead><tr><th>工具</th><th>日志耗时</th><th>Shell 实际耗时</th>'
        '<th>状态</th><th>输出大小</th><th>命令 / 动作</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )


def render_turn_diagnosis_table(turns: list[dict[str, Any]], top: int) -> str:
    if not turns:
        return '<p class="muted">这份日志里没有找到任务轮次元数据。</p>'
    rows: list[str] = []
    for turn in turns[:top]:
        tokens = turn.get("token_usage", {}) if isinstance(turn.get("token_usage"), dict) else {}
        top_calls = turn.get("top_slow_tool_calls", []) if isinstance(turn.get("top_slow_tool_calls"), list) else []
        top_call = top_calls[0] if top_calls and isinstance(top_calls[0], dict) else {}
        top_action = top_call.get("command") or top_call.get("name") or ""
        failed = int(turn.get("failed_tool_call_count") or 0)
        timeout = int(turn.get("timeout_count") or 0)
        row_class = "failed" if failed or timeout else ""
        bottleneck_code = str(turn.get("suspected_bottleneck", ""))
        rows.append(
            "<tr class=\"%s\" data-bottleneck=\"%s\"><td><strong>%s</strong><br><span class=\"muted\">%s</span></td>"
            "<td class=\"nowrap\">%s</td><td class=\"nowrap\">%s</td><td class=\"nowrap\">%s</td>"
            "<td class=\"nowrap\">%s</td><td class=\"nowrap\">%s / %s</td><td class=\"nowrap\">%s / %s / %s</td>"
            "<td class=\"nowrap\">%s 字符<br>%s 行</td><td><code>%s</code></td></tr>"
            % (
                row_class,
                escape(bottleneck_code),
                escape(bottleneck_label_zh(bottleneck_code)),
                escape(turn.get("bottleneck_reason", "")),
                escape(format_duration(turn.get("active_duration_seconds"))),
                escape(format_duration(turn.get("time_to_first_action_seconds"))),
                escape(format_duration(turn.get("tool_occupied_union_seconds"))),
                escape(format_duration(turn.get("post_tool_gap_seconds"))),
                escape(turn.get("tool_call_count", 0)),
                escape(f"{failed} / {timeout}"),
                escape(format_count(tokens.get("input_tokens"))),
                escape(format_count(tokens.get("output_tokens"))),
                escape(format_count(tokens.get("reasoning_output_tokens"))),
                escape(format_count(turn.get("output_char_count"))),
                escape(format_count(turn.get("output_line_count"))),
                escape(top_action),
            )
        )
    return (
        '<table class="diagnosis-table"><thead><tr><th>疑似瓶颈</th><th>本轮活跃耗时</th><th>开始行动</th>'
        '<th>工具占用</th><th>输出后等待</th><th>工具数 / 失败-超时</th><th>Token 输入/输出/推理</th>'
        '<th>输出大小</th><th>最慢动作</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )


def render_optimization_advice_table(turns: list[dict[str, Any]], top: int) -> str:
    if not turns:
        return '<p class="muted">没有可生成建议的轮次。</p>'
    rows: list[str] = []
    for turn in turns[:top]:
        advice = turn.get("optimization_advice") if isinstance(turn.get("optimization_advice"), dict) else {}
        safeguards = advice.get("safeguards", []) if isinstance(advice.get("safeguards"), list) else []
        patterns = turn.get("tool_output_patterns", []) if isinstance(turn.get("tool_output_patterns"), list) else []
        top_pattern = patterns[0] if patterns and isinstance(patterns[0], dict) else {}
        rows.append(
            "<tr><td><strong>%s</strong><br><span class=\"muted\">%s</span></td>"
            "<td>%s</td><td>%s</td><td>%s</td><td class=\"nowrap\">%s<br>%s 字符</td></tr>"
            % (
                escape(bottleneck_label_zh(turn.get("suspected_bottleneck"))),
                escape(turn.get("turn_id", "")),
                escape(advice.get("advice", "")),
                escape(advice.get("quality_risk", "")),
                escape("；".join(str(item) for item in safeguards)),
                escape(top_pattern.get("label", "")),
                escape(format_count(top_pattern.get("output_char_count"))),
            )
        )
    return (
        '<table class="advice-table"><thead><tr><th>瓶颈</th><th>优化建议</th><th>品质风险</th>'
        '<th>护栏</th><th>最大输出源</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )


def risk_counts_for_turns(turns: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"red": 0, "yellow": 0, "green": 0}
    for turn in turns:
        advice = turn.get("optimization_advice") if isinstance(turn.get("optimization_advice"), dict) else {}
        level = str(advice.get("risk_level") or "green")
        counts[level] = counts.get(level, 0) + 1
    return counts


def render_read_only_risk_dashboard(turns: list[dict[str, Any]], top: int) -> str:
    if not turns:
        return '<p class="muted">没有可评估的风险项。</p>'
    counts = risk_counts_for_turns(turns)
    rows: list[str] = []
    sorted_turns = sorted(
        turns,
        key=lambda turn: {"red": 0, "yellow": 1, "green": 2}.get(
            str((turn.get("optimization_advice") or {}).get("risk_level")), 3
        ),
    )
    for turn in sorted_turns[:top]:
        advice = turn.get("optimization_advice") if isinstance(turn.get("optimization_advice"), dict) else {}
        level = str(advice.get("risk_level") or "green")
        rows.append(
            "<tr class=\"risk-%s\"><td><span class=\"risk-pill %s\">%s</span><br><span class=\"muted\">%s</span></td>"
            "<td><strong>%s</strong><br><span class=\"muted\">%s</span></td><td>%s</td><td>%s</td></tr>"
            % (
                escape(level),
                escape(level),
                escape(advice.get("risk_label", risk_level_label_zh(level))),
                escape(turn.get("turn_id", "")),
                escape(bottleneck_label_zh(turn.get("suspected_bottleneck"))),
                escape(turn.get("bottleneck_reason", "")),
                escape(advice.get("risk_reason", "")),
                escape(READ_ONLY_DASHBOARD_NOTICE),
            )
        )
    return (
        '<p class="warning">%s</p>' % escape(READ_ONLY_DASHBOARD_NOTICE)
        + '<div class="metrics">'
        + render_metric_card("红色风险", str(counts.get("red", 0)), "暂不自动执行")
        + render_metric_card("黄色风险", str(counts.get("yellow", 0)), "人工确认")
        + render_metric_card("绿色风险", str(counts.get("green", 0)), "只读观察")
        + "</div>"
        + '<table class="risk-table"><thead><tr><th>风险等级</th><th>瓶颈</th><th>为什么有风险</th><th>执行边界</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )


def render_tool_output_pattern_table(patterns: list[dict[str, Any]], top: int) -> str:
    if not patterns:
        return '<p class="muted">没有工具输出源数据。</p>'
    rows: list[str] = []
    for pattern in patterns[:top]:
        rows.append(
            "<tr><td><strong>%s</strong><br><code>%s</code></td><td class=\"nowrap\">%s</td>"
            "<td class=\"nowrap\">%s 字符<br>%s 行</td><td class=\"nowrap\">%s</td><td class=\"nowrap\">%s / %s</td></tr>"
            % (
                escape(pattern.get("label", "")),
                escape(pattern.get("code", "")),
                escape(pattern.get("call_count", 0)),
                escape(format_count(pattern.get("output_char_count"))),
                escape(format_count(pattern.get("output_line_count"))),
                escape(format_duration(pattern.get("shell_wall_seconds"))),
                escape(pattern.get("failed_call_count", 0)),
                escape(pattern.get("timeout_count", 0)),
            )
        )
    return (
        '<table><thead><tr><th>输出源</th><th>调用数</th><th>输出大小</th><th>Shell 实际耗时</th>'
        '<th>失败 / 超时</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )


def render_session_bottleneck_table(reports: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for report in reports:
        turns = sorted_turn_summaries(report)
        top_turn = turns[0] if turns else {}
        advice = top_turn.get("optimization_advice") if isinstance(top_turn.get("optimization_advice"), dict) else {}
        summary = report.get("summary", {})
        tokens = summary.get("token_usage", {}) if isinstance(summary.get("token_usage"), dict) else {}
        bottleneck_code = str(top_turn.get("suspected_bottleneck", "no_turn_data"))
        rows.append(
            "<tr data-bottleneck=\"%s\"><td>%s<br><code>%s</code></td><td><strong>%s</strong><br><span class=\"muted\">%s</span></td>"
            "<td>%s</td><td class=\"nowrap\">%s</td><td class=\"nowrap\">%s</td><td class=\"nowrap\">%s</td>"
            "<td class=\"nowrap\">%s / %s</td><td class=\"nowrap\">%s 字符</td><td class=\"nowrap\">%s / %s</td></tr>"
            % (
                escape(bottleneck_code),
                escape(report_title(report)),
                escape(report.get("log_path", "")),
                escape(bottleneck_label_zh(bottleneck_code)),
                escape(top_turn.get("bottleneck_reason", "")),
                escape(advice.get("advice", "")),
                escape(format_duration(summary.get("elapsed_seconds"))),
                escape(format_duration(summary.get("tool_occupied_union_seconds"))),
                escape(format_duration(top_turn.get("post_tool_gap_seconds"))),
                escape(summary.get("tool_call_count", 0)),
                escape(summary.get("failed_tool_call_count", 0)),
                escape(format_count(summary.get("output_char_count"))),
                escape(format_count(tokens.get("input_tokens"))),
                escape(format_count(tokens.get("reasoning_output_tokens"))),
            )
        )
    if not rows:
        return '<p class="muted">没有会话。</p>'
    return (
        '<table class="session-table"><thead><tr><th>会话</th><th>最可疑轮次</th><th>建议</th><th>总耗时</th>'
        '<th>工具占用</th><th>最大输出后等待</th><th>工具 / 失败</th><th>输出</th>'
        '<th>Token 输入/推理</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )


def render_gap_table(gaps: list[dict[str, Any]], top: int) -> str:
    if not gaps:
        return '<p class="muted">没有超过阈值的推断空档。</p>'
    rows: list[str] = []
    for gap in gaps[:top]:
        rows.append(
            "<tr><td>%s</td><td>+%.1fs</td><td>%s</td><td>%s</td></tr>"
            % (
                escape(format_duration(gap["duration_seconds"])),
                gap["start_offset_seconds"],
                escape(gap["after"]),
                escape(gap["before"]),
            )
        )
    return (
        "<table><thead><tr><th>持续时间</th><th>开始位置</th><th>前一个事件</th><th>后一个事件</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def timeline_class(category: str) -> str:
    if category.startswith("tool:"):
        return "tool"
    if category == "tool_parallel":
        return "parallel"
    if category == "observable_event":
        return "event"
    return "inferred"


def render_timeline_table(report: dict[str, Any], limit: int) -> str:
    timeline = report["timeline"]
    total_count = report["summary"].get("timeline_bucket_count", len(timeline))
    shown = timeline if limit < 0 else timeline[:limit]
    if not shown:
        return '<p class="muted">没有显示时间线分桶。</p>'
    rows: list[str] = []
    for bucket in shown:
        category = str(bucket["category"])
        rows.append(
            '<tr data-category="%s"><td class="time">+%s</td><td><span class="pill %s">%s</span></td><td>%s</td></tr>'
            % (
                escape(category),
                escape(format_duration(bucket["start_offset_seconds"])),
                timeline_class(category),
                escape(timeline_category_label(category)),
                escape(bucket["label"]),
            )
    )
    hidden = ""
    if len(shown) < total_count:
        hidden = f'<p class="muted">还有 {total_count - len(shown)} 个分桶被隐藏。使用 --timeline-limit -1 可生成完整时间线。</p>'
    return (
        '<table class="timeline"><thead><tr><th>时间偏移</th><th>类别</th><th>内容</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
        + hidden
    )


def html_styles() -> str:
    return """
    :root { color-scheme: light; --bg:#f7f8fa; --panel:#fff; --text:#17202a; --muted:#667085; --line:#d8dde6; --tool:#0f766e; --parallel:#9a3412; --event:#1d4ed8; --inferred:#6b7280; --bad:#b42318; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.5 "Segoe UI", Arial, sans-serif; }
    main { max-width: 1280px; margin: 0 auto; padding: 24px; }
    h1 { margin: 0 0 6px; font-size: 28px; }
    h2 { margin: 22px 0 10px; font-size: 18px; }
    h3 { margin: 18px 0 8px; font-size: 15px; }
    code { font-family: Consolas, "SFMono-Regular", monospace; font-size: 12px; white-space: pre-wrap; word-break: break-word; }
    .muted { color: var(--muted); }
    .panel, details { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; margin: 14px 0; }
    details summary { cursor: pointer; font-weight: 700; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin: 14px 0; }
    .metric { background: #f2f4f7; border: 1px solid var(--line); border-radius: 8px; padding: 10px; min-height: 82px; }
    .metric span, .metric small { display: block; color: var(--muted); }
    .metric strong { display: block; font-size: 20px; margin: 4px 0; }
    table { width: 100%; border-collapse: collapse; margin: 8px 0 16px; table-layout: fixed; }
    th, td { border-top: 1px solid var(--line); padding: 7px 8px; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 600; background: #f9fafb; }
    tr.failed td { background: #fff1f0; }
    .diagnosis-table, .session-table, .advice-table, .risk-table { table-layout: auto; }
    .nowrap { white-space: nowrap; }
    .time { width: 90px; color: var(--muted); }
    .pill { display: inline-block; border-radius: 999px; padding: 2px 8px; color: #fff; font-size: 12px; }
    .tool { background: var(--tool); }
    .parallel { background: var(--parallel); }
    .event { background: var(--event); }
    .inferred { background: var(--inferred); }
    .bar { height: 10px; border-radius: 999px; overflow: hidden; display: flex; background: #e5e7eb; margin: 8px 0; }
    .bar span { display: block; }
    .bar .tool-time { background: var(--tool); }
    .bar .non-tool-time { background: var(--inferred); }
    .warning { border-left: 4px solid var(--parallel); padding-left: 10px; color: #713f12; }
    .risk-pill { display: inline-block; border-radius: 999px; padding: 2px 8px; color: #fff; font-size: 12px; font-weight: 700; }
    .risk-pill.red { background: #b42318; }
    .risk-pill.yellow { background: #b54708; }
    .risk-pill.green { background: #027a48; }
    tr.risk-red td { background: #fff1f0; }
    tr.risk-yellow td { background: #fffaeb; }
    tr.risk-green td { background: #f0fdf4; }
    """


def render_html_report(report: dict[str, Any], top: int, timeline_limit: int, embedded: bool = False) -> str:
    metadata = report["metadata"]
    summary = report["summary"]
    elapsed = summary["elapsed_seconds"]
    tool_union = summary["tool_occupied_union_seconds"]
    non_tool = summary["inferred_non_tool_seconds"]
    failed_calls = [call for call in report["tool_calls"] if call.get("exit_code") not in (None, 0)]
    token_usage = summary.get("token_usage", {}) if isinstance(summary.get("token_usage"), dict) else {}

    body: list[str] = []
    if not embedded:
        body.append("<!doctype html><html><head><meta charset=\"utf-8\">")
        body.append("<title>Codex 对话耗时报告</title>")
        body.append(f"<style>{html_styles()}</style></head><body><main>")
    body.append(f"<h1>{escape(report_title(report))}</h1>")
    body.append(f"<p class=\"muted\">日志: <code>{escape(report['log_path'])}</code></p>")
    body.append('<p class="warning">Codex 的私有 reasoning 内容不可见；这里的“非工具时间”是根据日志事件空档推断出来的。</p>')
    body.append('<div class="metrics">')
    body.append(render_metric_card("总耗时", format_duration(elapsed), f"{summary['observable_event_count']} 个可观测事件"))
    body.append(render_metric_card("工具占用", format_duration(tool_union), percentage(tool_union, elapsed)))
    body.append(render_metric_card("推断非工具", format_duration(non_tool), percentage(non_tool, elapsed)))
    body.append(render_metric_card("工具调用", str(summary["tool_call_count"]), f"失败: {summary['failed_tool_call_count']}"))
    body.append(render_metric_card("Shell 实际耗时", format_duration(summary["tool_cumulative_shell_wall_seconds"]), "从工具输出解析"))
    body.append(render_metric_card("工具输出", f"{format_count(summary.get('output_char_count'))} 字符", f"{format_count(summary.get('output_line_count'))} 行"))
    body.append(render_metric_card("Token 输入", format_count(token_usage.get("input_tokens")), f"推理: {format_count(token_usage.get('reasoning_output_tokens'))}"))
    body.append(render_metric_card("对话轮次", str(summary.get("turn_count", len(metadata.get("turn_ids", [])))), metadata.get("cwd", "")))
    body.append("</div>")
    body.append('<div class="bar"><span class="tool-time" style="width:%s"></span><span class="non-tool-time" style="width:%s"></span></div>' % (percentage(tool_union, elapsed), percentage(non_tool, elapsed)))
    body.append("<h2>只读风险看板</h2>")
    body.append(render_read_only_risk_dashboard(sorted_turn_summaries(report), top))
    body.append("<h2>优化建议与品质风险</h2>")
    body.append(render_optimization_advice_table(sorted_turn_summaries(report), top))
    body.append("<h2>输出源排行榜</h2>")
    body.append(render_tool_output_pattern_table(report.get("tool_output_patterns", []), top))
    body.append("<h2>按对话轮次诊断性能</h2>")
    body.append(render_turn_diagnosis_table(sorted_turn_summaries(report), top))
    body.append("<h2>最慢工具调用</h2>")
    body.append(render_tool_table(sorted_tool_calls(report), top))
    body.append("<h2>失败工具调用</h2>")
    body.append(render_tool_table(failed_calls, top))
    body.append("<h2>最长推断非工具空档</h2>")
    body.append(render_gap_table(report["longest_inferred_gaps"], top))
    body.append("<h2>时间线</h2>")
    body.append(render_timeline_table(report, timeline_limit))
    if not embedded:
        body.append("</main></body></html>")
    return "\n".join(body)


def render_html_index(reports: list[dict[str, Any]], top: int, timeline_limit: int) -> str:
    total_elapsed = sum(report["summary"]["elapsed_seconds"] for report in reports)
    total_tool = sum(report["summary"]["tool_occupied_union_seconds"] for report in reports)
    total_failed = sum(report["summary"]["failed_tool_call_count"] for report in reports)
    total_output = sum(report["summary"].get("output_char_count", 0) for report in reports)
    body: list[str] = [
        "<!doctype html><html><head><meta charset=\"utf-8\">",
        "<title>Codex 对话耗时总览</title>",
        f"<style>{html_styles()}</style></head><body><main>",
        "<h1>Codex 对话耗时总览</h1>",
        '<p class="warning">此页面只展示本地 Codex JSONL 日志里可观测到的工具活动。私有 reasoning 不可见；非工具时间只能根据事件空档推断。</p>',
        '<div class="metrics">',
        render_metric_card("会话数", str(len(reports))),
        render_metric_card("总耗时", format_duration(total_elapsed)),
        render_metric_card("工具占用", format_duration(total_tool), percentage(total_tool, total_elapsed)),
        render_metric_card("失败工具调用", str(total_failed)),
        render_metric_card("工具输出", f"{format_count(total_output)} 字符"),
        "</div>",
    ]
    body.append("<h2>会话瓶颈总览</h2>")
    body.append(render_session_bottleneck_table(reports))
    all_turns: list[dict[str, Any]] = []
    for report in reports:
        all_turns.extend(report.get("turn_summaries", []))
    body.append("<h2>跨会话只读风险看板</h2>")
    body.append(render_read_only_risk_dashboard(sorted(all_turns, key=turn_cost_score, reverse=True), top))
    all_calls: list[dict[str, Any]] = []
    for report in reports:
        all_calls.extend(report.get("tool_calls", []))
    body.append("<h2>跨会话输出源排行榜</h2>")
    body.append(render_tool_output_pattern_table(_aggregate_tool_output_patterns(all_calls), top))
    for index, report in enumerate(reports, start=1):
        summary = report["summary"]
        failed = summary["failed_tool_call_count"]
        title = report_title(report)
        body.append("<details%s>" % (" open" if index == 1 else ""))
        body.append(
            "<summary>%s | 总耗时 %s | 工具占用 %s | 失败 %s</summary>"
            % (
                escape(title),
                escape(format_duration(summary["elapsed_seconds"])),
                escape(format_duration(summary["tool_occupied_union_seconds"])),
                escape(failed),
            )
        )
        body.append(render_html_report(report, top=top, timeline_limit=timeline_limit, embedded=True))
        body.append("</details>")
    body.append("</main></body></html>")
    return "\n".join(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_log", nargs="?", type=Path, help="Codex rollout-*.jsonl 日志路径。")
    parser.add_argument(
        "--latest",
        action="store_true",
        help="分析 --sessions-root 下最新的 rollout-*.jsonl；不传 session_log 时默认使用。",
    )
    parser.add_argument(
        "--sessions-root",
        type=Path,
        default=DEFAULT_SESSIONS_ROOT,
        help=f"Codex sessions 根目录。默认: {DEFAULT_SESSIONS_ROOT}",
    )
    parser.add_argument("--bucket-seconds", type=int, default=1, help="时间线分桶大小。默认: 1 秒。")
    parser.add_argument("--gap-threshold", type=float, default=1.0, help="报告推断空档的最小秒数。默认: 1 秒。")
    parser.add_argument("--top", type=int, default=10, help="显示前 N 条慢调用、空档和轮次诊断。")
    parser.add_argument(
        "--timeline-limit",
        type=int,
        default=180,
        help="输出多少个时间线分桶；使用 -1 输出完整时间线。",
    )
    parser.add_argument("--json-out", type=Path, help="可选：完整 JSON 报告输出路径。")
    parser.add_argument("--html-out", type=Path, help="可选：单会话 HTML 报告输出路径。")
    parser.add_argument(
        "--html-index-out",
        type=Path,
        help="可选：最近多个会话 HTML 总览输出路径，使用 --sessions-root 下的最新日志。",
    )
    parser.add_argument(
        "--session-limit",
        type=int,
        default=20,
        help="HTML 总览包含最近多少个会话。默认: 20。",
    )
    parser.add_argument(
        "--project-filter",
        default="",
        help="只保留 cwd、session_id、日志路径或标题包含该字符串的会话，例如 my-oss-project。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        timeline_bucket_limit = None if args.timeline_limit < 0 else args.timeline_limit
        if args.html_index_out:
            session_logs = list_session_logs(args.sessions_root, limit=args.session_limit)
            if not session_logs:
                raise FileNotFoundError(f"{args.sessions_root} 下没有 rollout-*.jsonl 文件")
            reports = [
                analyze_session(
                    path=session_log,
                    bucket_seconds=args.bucket_seconds,
                    gap_threshold_seconds=args.gap_threshold,
                    timeline_bucket_limit=timeline_bucket_limit,
                )
                for session_log in session_logs
            ]
            reports = filter_reports_by_project(reports, args.project_filter)
            if not reports:
                raise ValueError(f"没有会话匹配 --project-filter {args.project_filter!r}")
            args.html_index_out.parent.mkdir(parents=True, exist_ok=True)
            args.html_index_out.write_text(
                render_html_index(reports, top=args.top, timeline_limit=args.timeline_limit),
                encoding="utf-8",
            )
            print(f"Wrote HTML index: {args.html_index_out}")

        run_single_report = bool(args.session_log or args.latest or args.json_out or args.html_out or not args.html_index_out)
        if not run_single_report:
            return 0

        if args.session_log:
            session_log = args.session_log
        else:
            session_log = find_latest_session_log(args.sessions_root)
        report = analyze_session(
            path=session_log,
            bucket_seconds=args.bucket_seconds,
            gap_threshold_seconds=args.gap_threshold,
            timeline_bucket_limit=timeline_bucket_limit,
        )
        if args.project_filter and not report_matches_project_filter(report, args.project_filter):
            raise ValueError(f"当前会话不匹配 --project-filter {args.project_filter!r}")
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.html_out:
            args.html_out.parent.mkdir(parents=True, exist_ok=True)
            args.html_out.write_text(render_html_report(report, top=args.top, timeline_limit=args.timeline_limit), encoding="utf-8")
        print(render_report(report, top=args.top, timeline_limit=args.timeline_limit))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
