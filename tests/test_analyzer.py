from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codex_session_timeline.analyzer import (
    analyze_session,
    filter_reports_by_project,
    find_latest_session_log,
    render_html_index,
    render_html_report,
)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


class AnalyzeCodexSessionTimelineTests(unittest.TestCase):
    def test_analyze_session_pairs_tool_call_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollout-test.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "timestamp": "2026-06-30T00:00:00.000Z",
                        "type": "session_meta",
                        "payload": {
                            "session_id": "session-1",
                            "cwd": "/workspace/sample-project",
                            "originator": "Codex Desktop",
                        },
                    },
                    {
                        "timestamp": "2026-06-30T00:00:01.000Z",
                        "type": "event_msg",
                        "payload": {"type": "user_message", "message": "run slow command"},
                    },
                    {
                        "timestamp": "2026-06-30T00:00:02.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "shell_command",
                            "call_id": "call_1",
                            "arguments": json.dumps(
                                {
                                    "command": "Start-Sleep -Seconds 3",
                                    "workdir": "/workspace/sample-project",
                                    "timeout_ms": 10000,
                                }
                            ),
                        },
                    },
                    {
                        "timestamp": "2026-06-30T00:00:05.500Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": "Exit code: 0\nWall time: 3.2 seconds\nOutput:\n",
                        },
                    },
                    {
                        "timestamp": "2026-06-30T00:00:07.000Z",
                        "type": "event_msg",
                        "payload": {"type": "agent_message", "message": "done"},
                    },
                ],
            )

            report = analyze_session(path, bucket_seconds=1, gap_threshold_seconds=1)

        self.assertEqual(report["metadata"]["session_id"], "session-1")
        self.assertEqual(report["summary"]["tool_call_count"], 1)
        self.assertAlmostEqual(report["summary"]["tool_occupied_union_seconds"], 3.5)
        self.assertAlmostEqual(report["summary"]["tool_cumulative_shell_wall_seconds"], 3.2)

        call = report["tool_calls"][0]
        self.assertEqual(call["command"], "Start-Sleep -Seconds 3")
        self.assertEqual(call["workdir"], "/workspace/sample-project")
        self.assertEqual(call["timeout_ms"], 10000)
        self.assertEqual(call["exit_code"], 0)

        categories = [bucket["category"] for bucket in report["timeline"]]
        self.assertIn("tool:shell_command", categories)
        self.assertIn("agent_thinking_or_waiting_inferred", categories)

    def test_parallel_tool_spans_are_counted_as_union_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollout-parallel.jsonl"
            records: list[dict[str, object]] = [
                {"timestamp": "2026-06-30T00:00:00.000Z", "type": "event_msg", "payload": {"type": "user_message"}},
                {
                    "timestamp": "2026-06-30T00:00:01.000Z",
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "name": "shell_command",
                        "call_id": "call_a",
                        "arguments": json.dumps({"command": "a"}),
                    },
                },
                {
                    "timestamp": "2026-06-30T00:00:01.500Z",
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "name": "shell_command",
                        "call_id": "call_b",
                        "arguments": json.dumps({"command": "b"}),
                    },
                },
                {
                    "timestamp": "2026-06-30T00:00:04.000Z",
                    "type": "response_item",
                    "payload": {"type": "function_call_output", "call_id": "call_a", "output": "Exit code: 0\n"},
                },
                {
                    "timestamp": "2026-06-30T00:00:05.000Z",
                    "type": "response_item",
                    "payload": {"type": "function_call_output", "call_id": "call_b", "output": "Exit code: 0\n"},
                },
            ]
            write_jsonl(path, records)

            report = analyze_session(path, bucket_seconds=1, gap_threshold_seconds=1)

        self.assertAlmostEqual(report["summary"]["tool_occupied_union_seconds"], 4.0)
        self.assertAlmostEqual(report["summary"]["tool_cumulative_logged_seconds"], 6.5)
        self.assertTrue(any(bucket["category"] == "tool_parallel" for bucket in report["timeline"]))

    def test_custom_tool_call_is_included_as_tool_span(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollout-custom.jsonl"
            write_jsonl(
                path,
                [
                    {"timestamp": "2026-06-30T00:00:00.000Z", "type": "event_msg", "payload": {"type": "user_message"}},
                    {
                        "timestamp": "2026-06-30T00:00:01.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "custom_tool_call",
                            "name": "apply_patch",
                            "call_id": "call_patch",
                            "input": "*** Begin Patch\n*** Add File: tools/example.py\n+pass\n*** End Patch\n",
                        },
                    },
                    {
                        "timestamp": "2026-06-30T00:00:02.250Z",
                        "type": "response_item",
                        "payload": {
                            "type": "custom_tool_call_output",
                            "call_id": "call_patch",
                            "output": "Exit code: 0\nWall time: 0 seconds\nOutput:\nSuccess.\n",
                        },
                    },
                ],
            )

            report = analyze_session(path, bucket_seconds=1, gap_threshold_seconds=1)

        self.assertEqual(report["summary"]["tool_call_count"], 1)
        self.assertAlmostEqual(report["summary"]["tool_occupied_union_seconds"], 1.25)
        self.assertEqual(report["tool_calls"][0]["name"], "apply_patch")
        self.assertEqual(report["tool_calls"][0]["command"], "apply_patch tools/example.py")
        self.assertTrue(any(bucket["category"] == "tool:apply_patch" for bucket in report["timeline"]))

    def test_find_latest_session_log_uses_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_path = root / "2026" / "06" / "30" / "rollout-old.jsonl"
            new_path = root / "2026" / "06" / "30" / "rollout-new.jsonl"
            old_path.parent.mkdir(parents=True)
            old_path.write_text("{}\n", encoding="utf-8")
            new_path.write_text("{}\n", encoding="utf-8")
            old_path.touch()
            new_path.touch()

            self.assertEqual(find_latest_session_log(root), new_path)

    def test_render_html_report_contains_session_details(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollout-html.jsonl"
            write_jsonl(
                path,
                [
                    {"timestamp": "2026-06-30T00:00:00.000Z", "type": "event_msg", "payload": {"type": "user_message"}},
                    {
                        "timestamp": "2026-06-30T00:00:01.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "shell_command",
                            "call_id": "call_fail",
                            "arguments": json.dumps({"command": "bad command"}),
                        },
                    },
                    {
                        "timestamp": "2026-06-30T00:00:03.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call_fail",
                            "output": "Exit code: 1\nWall time: 2.0 seconds\nOutput:\nfailed\n",
                        },
                    },
                ],
            )
            report = analyze_session(path, bucket_seconds=1, gap_threshold_seconds=1)

        html = render_html_report(report, top=10, timeline_limit=-1)

        self.assertIn("<!doctype html>", html)
        self.assertIn("最慢工具调用", html)
        self.assertIn("失败工具调用", html)
        self.assertIn("bad command", html)
        self.assertIn("tool:shell_command", html)

    def test_turn_performance_diagnosis_tracks_tokens_output_and_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollout-turn-diagnosis.jsonl"
            large_output = "Exit code: 0\nWall time: 2.0 seconds\nOutput:\n" + ("x" * 100_100)
            write_jsonl(
                path,
                [
                    {"timestamp": "2026-06-30T00:00:00.000Z", "type": "session_meta", "payload": {"session_id": "turn-session"}},
                    {"timestamp": "2026-06-30T00:00:01.000Z", "type": "event_msg", "payload": {"type": "task_started", "turn_id": "turn-1"}},
                    {"timestamp": "2026-06-30T00:00:01.500Z", "type": "event_msg", "payload": {"type": "user_message", "message": "why slow"}},
                    {
                        "timestamp": "2026-06-30T00:00:02.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "shell_command",
                            "call_id": "call_big_output",
                            "arguments": json.dumps({"command": "rg noisy"}),
                        },
                    },
                    {
                        "timestamp": "2026-06-30T00:00:02.500Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "shell_command",
                            "call_id": "call_small_output",
                            "arguments": json.dumps({"command": "git status --short"}),
                        },
                    },
                    {
                        "timestamp": "2026-06-30T00:00:04.000Z",
                        "type": "response_item",
                        "payload": {"type": "function_call_output", "call_id": "call_big_output", "output": large_output},
                    },
                    {
                        "timestamp": "2026-06-30T00:00:05.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call_small_output",
                            "output": "Exit code: 0\nWall time: 1.0 seconds\nOutput:\n M tools/example.py\n",
                        },
                    },
                    {
                        "timestamp": "2026-06-30T00:00:29.000Z",
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "last_token_usage": {
                                    "input_tokens": 12345,
                                    "cached_input_tokens": 1000,
                                    "output_tokens": 456,
                                    "reasoning_output_tokens": 789,
                                    "total_tokens": 13590,
                                }
                            },
                        },
                    },
                    {"timestamp": "2026-06-30T00:00:30.000Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "done"}},
                    {"timestamp": "2026-06-30T00:00:31.000Z", "type": "event_msg", "payload": {"type": "task_complete", "turn_id": "turn-1"}},
                ],
            )

            report = analyze_session(path, bucket_seconds=1, gap_threshold_seconds=1)

        turn = report["turn_summaries"][0]
        self.assertEqual(turn["turn_id"], "turn-1")
        self.assertEqual(turn["suspected_bottleneck"], "tool_output_size")
        self.assertAlmostEqual(turn["time_to_first_action_seconds"], 0.5)
        self.assertAlmostEqual(turn["post_tool_gap_seconds"], 26.0)
        self.assertEqual(turn["post_tool_gap_count"], 1)
        self.assertEqual(turn["tool_call_count"], 2)
        self.assertGreater(turn["output_char_count"], 100_000)
        self.assertEqual(turn["token_usage"]["input_tokens"], 12345)
        self.assertIn("完整清单/计数", turn["optimization_advice"]["advice"])
        self.assertIn("漏查", turn["optimization_advice"]["quality_risk"])
        self.assertEqual(turn["optimization_advice"]["risk_level"], "yellow")
        self.assertIn("人工确认", turn["optimization_advice"]["risk_label"])
        self.assertEqual(turn["tool_output_patterns"][0]["code"], "rg_search")
        self.assertEqual(report["tool_output_patterns"][0]["code"], "rg_search")
        self.assertEqual(report["summary"]["turn_count"], 1)
        self.assertEqual(report["summary"]["token_usage"]["reasoning_output_tokens"], 789)

        html = render_html_report(report, top=10, timeline_limit=-1)
        self.assertIn("只读风险看板", html)
        self.assertIn("黄色：人工确认", html)
        self.assertIn("不改变命令", html)
        self.assertIn("优化建议与品质风险", html)
        self.assertIn("输出源排行榜", html)
        self.assertIn("rg 文本查询", html)
        self.assertIn("完整清单/计数", html)
        self.assertIn("按对话轮次诊断性能", html)
        self.assertIn("工具输出过大", html)
        self.assertIn("tool_output_size", html)
        self.assertIn("rg noisy", html)

    def test_risk_dashboard_marks_timeout_as_red(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollout-timeout-risk.jsonl"
            write_jsonl(
                path,
                [
                    {"timestamp": "2026-06-30T00:00:00.000Z", "type": "session_meta", "payload": {"session_id": "risk-session"}},
                    {"timestamp": "2026-06-30T00:00:01.000Z", "type": "event_msg", "payload": {"type": "task_started", "turn_id": "turn-timeout"}},
                    {"timestamp": "2026-06-30T00:00:01.500Z", "type": "event_msg", "payload": {"type": "user_message", "message": "run timeout"}},
                    {
                        "timestamp": "2026-06-30T00:00:02.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "shell_command",
                            "call_id": "call_timeout",
                            "arguments": json.dumps({"command": "pytest tests/e2e/test_release_flow.py"}),
                        },
                    },
                    {
                        "timestamp": "2026-06-30T00:00:35.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call_timeout",
                            "output": "Exit code: 124\nWall time: 30.0 seconds\nOutput:\ncommand timed out\n",
                        },
                    },
                    {"timestamp": "2026-06-30T00:00:36.000Z", "type": "event_msg", "payload": {"type": "task_complete", "turn_id": "turn-timeout"}},
                ],
            )

            report = analyze_session(path, bucket_seconds=1, gap_threshold_seconds=1)

        turn = report["turn_summaries"][0]
        self.assertEqual(turn["suspected_bottleneck"], "tool_timeout")
        self.assertEqual(turn["optimization_advice"]["risk_level"], "red")
        self.assertIn("暂不自动执行", turn["optimization_advice"]["risk_label"])

        html = render_html_report(report, top=10, timeline_limit=-1)
        self.assertIn("红色：暂不自动执行", html)
        self.assertIn("只读提示", html)

    def test_render_html_index_contains_multiple_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "rollout-first.jsonl"
            second = Path(temp_dir) / "rollout-second.jsonl"
            write_jsonl(
                first,
                [
                    {"timestamp": "2026-06-30T00:00:00.000Z", "type": "session_meta", "payload": {"session_id": "one"}},
                    {"timestamp": "2026-06-30T00:00:01.000Z", "type": "event_msg", "payload": {"type": "user_message"}},
                ],
            )
            write_jsonl(
                second,
                [
                    {"timestamp": "2026-06-30T00:00:00.000Z", "type": "session_meta", "payload": {"session_id": "two"}},
                    {"timestamp": "2026-06-30T00:00:02.000Z", "type": "event_msg", "payload": {"type": "agent_message"}},
                ],
            )
            reports = [
                analyze_session(first, bucket_seconds=1, gap_threshold_seconds=1),
                analyze_session(second, bucket_seconds=1, gap_threshold_seconds=1),
            ]

        html = render_html_index(reports, top=3, timeline_limit=2)

        self.assertIn("Codex 对话耗时总览", html)
        self.assertIn("会话数", html)
        self.assertIn("会话瓶颈总览", html)
        self.assertIn("跨会话只读风险看板", html)
        self.assertIn("跨会话输出源排行榜", html)
        self.assertIn("one", html)
        self.assertIn("two", html)
        self.assertIn("<details", html)

    def test_filter_reports_by_project_matches_cwd(self) -> None:
        reports = [
            {"metadata": {"cwd": "/workspace/sample-project", "session_id": "sample"}, "summary": {"started_at": ""}, "log_path": "sample.jsonl"},
            {"metadata": {"cwd": "/workspace/other-project", "session_id": "other"}, "summary": {"started_at": ""}, "log_path": "other.jsonl"},
        ]

        filtered = filter_reports_by_project(reports, "sample-project")

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["metadata"]["session_id"], "sample")

    def test_timeline_bucket_limit_avoids_full_span_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollout-long-gap.jsonl"
            write_jsonl(
                path,
                [
                    {"timestamp": "2026-06-30T00:00:00.000Z", "type": "event_msg", "payload": {"type": "user_message"}},
                    {"timestamp": "2026-06-30T01:00:00.000Z", "type": "event_msg", "payload": {"type": "agent_message"}},
                ],
            )

            report = analyze_session(path, bucket_seconds=1, gap_threshold_seconds=1, timeline_bucket_limit=3)

        self.assertEqual(len(report["timeline"]), 3)
        self.assertEqual(report["summary"]["timeline_bucket_count"], 3601)


if __name__ == "__main__":
    unittest.main()
