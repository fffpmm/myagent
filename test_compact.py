"""
上下文压缩功能测试（纯本地，不调 API）
"""
import json
import sys
import os
from pathlib import Path

# 确保能 import main 里的函数
sys.path.insert(0, str(Path(__file__).parent))

import main

# ========== 构造模拟对话 ==========
def make_long_output(n_chars: int) -> str:
    """生成指定长度的模拟输出"""
    base = "line " * 100  # ~500 chars
    repeats = (n_chars // len(base)) + 1
    return (base * repeats)[:n_chars]

def build_test_messages(n_tools: int, output_size: int) -> list:
    """构造对话历史：n_tools 轮工具调用，每个工具返回 output_size 字符"""
    msgs = [{"role": "system", "content": "You are a test agent."}]
    for i in range(n_tools):
        msgs.append({"role": "user", "content": f"task step {i+1}"})
        msgs.append({
            "role": "assistant",
            "content": f"Doing step {i+1}",
            "tool_calls": [{
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": "bash", "arguments": f'{{"command":"echo step{i}"}}'}
            }]
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": make_long_output(output_size)
        })
    msgs.append({"role": "user", "content": "final task"})
    msgs.append({"role": "assistant", "content": "I've completed all tasks."})
    return msgs


# ========== 测试用例 ==========

def test_l1_snip():
    """L1: 消息数量裁剪（snip_compact）"""
    print("\n" + "="*60)
    print("L1 测试: snip_compact — 消息数量超过 max_messages 时裁剪中间")
    msgs = build_test_messages(n_tools=30, output_size=200)  # 30轮=90+条消息
    print(f"  裁剪前消息数: {len(msgs)}, 估算大小: {main.estimate_size(msgs)} chars")
    result = main.snip_compact(msgs, max_messages=20)
    print(f"  裁剪后消息数: {len(result)}, 估算大小: {main.estimate_size(result)} chars")
    # 检查中间有 snipped 标记
    snipped_msgs = [m for m in result if isinstance(m.get("content"), str) and "snipped" in m["content"]]
    print(f"  snipped 标记: {snipped_msgs[0]['content'] if snipped_msgs else '无'}")


def test_l2_micro():
    """L2: 工具结果压缩（micro_compact）"""
    print("\n" + "="*60)
    print("L2 测试: micro_compact — 只保留最近 KEEP_RECENT 条完整工具结果")
    msgs = build_test_messages(n_tools=10, output_size=500)
    tool_count_before = sum(1 for m in msgs if m.get("role") == "tool")
    result = main.micro_compact(msgs)
    # 检查前面的工具结果是否被替换
    replaced = 0
    kept = 0
    for m in result:
        if m.get("role") == "tool":
            c = m.get("content", "")
            if "紧凑" in c:
                replaced += 1
            else:
                kept += 1
    print(f"  工具消息总数: {tool_count_before}")
    print(f"  被替换为占位符: {replaced}")
    print(f"  保留完整内容: {kept}  (KEEP_RECENT={main.KEEP_RECENT})")


def test_l3_budget():
    """L3: 工具结果写入文件（tool_result_budget）"""
    print("\n" + "="*60)
    print("L3 测试: tool_result_budget — 超长工具输出持久化到文件")
    # 只有一轮对话，构造多个超大 tool 结果
    msgs = [{"role": "system", "content": "test"}]
    msgs.append({"role": "user", "content": "run a big command"})
    msgs.append({
        "role": "assistant",
        "content": "running",
        "tool_calls": [
            {"id": "big1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"cat huge1"}'}},
            {"id": "big2", "type": "function", "function": {"name": "bash", "arguments": '{"command":"cat huge2"}'}},
        ]
    })
    msgs.append({"role": "tool", "tool_call_id": "big1", "content": make_long_output(40000)})
    msgs.append({"role": "tool", "tool_call_id": "big2", "content": make_long_output(50000)})

    total_before = main.estimate_size(msgs)
    print(f"  压缩前本轮 tool 总大小: {total_before} chars")

    result = main.tool_result_budget(msgs, max_bytes=50000)
    total_after = main.estimate_size(result)
    print(f"  压缩后本轮 tool 总大小: {total_after} chars")

    # 检查文件是否生成
    files = list(main.TOOL_RESULTS_DIR.glob("*.txt"))
    print(f"  持久化文件数: {len(files)}")
    for f in files:
        content = f.read_text()
        print(f"    {f.name}: {len(content)} chars")


def test_l4_compact():
    """L4: 自动压缩（compact_history）— 只测保存，不调 AI"""
    print("\n" + "="*60)
    print("L4 测试: compact_history — 保存对话 + AI 总结")
    msgs = build_test_messages(n_tools=20, output_size=500)
    # 直接调 summarize_history 会调 API，这里只测 write_transcript
    path = main.write_transcript(msgs)
    print(f"  对话已保存: {path}")
    print(f"  文件大小: {path.stat().st_size} bytes")
    # 清理
    path.unlink()


def test_l5_reactive():
    """L5: 紧急压缩（reactive_compact）— 只测保存，不调 AI"""
    print("\n" + "="*60)
    print("L5 测试: reactive_compact — 上下文超限时紧急压缩")
    msgs = build_test_messages(n_tools=25, output_size=500)
    print(f"  原始消息数: {len(msgs)}, 大小: {main.estimate_size(msgs)} chars")

    # 测 transcript 路径（reactive_compact 内部调 write_transcript）
    # 不实际调 AI，手动验证切片逻辑
    tail_start = max(0, len(msgs) - 5)
    if (tail_start > 0 and tail_start < len(msgs)
            and main._is_tool_result_message(msgs[tail_start])
            and main._message_has_tool_calls(msgs[tail_start - 1])):
        tail_start -= 1
    print(f"  保留尾部消息数: {len(msgs) - tail_start}")
    print(f"  送去总结的消息数: {tail_start}")


def test_full_pipeline():
    """完整 pipeline: L3 → L1 → L2（模拟 agent_loop 的顺序）"""
    print("\n" + "="*60)
    print("完整 pipeline 测试: L3 → L1 → L2")
    msgs = build_test_messages(n_tools=35, output_size=35000)
    before = main.estimate_size(msgs)
    print(f"  原始: {len(msgs)} 条消息, {before} chars")

    # 模拟 agent_loop 中的压缩顺序
    main.tool_result_budget(msgs)
    print(f"  L3 后: {main.estimate_size(msgs)} chars")

    msgs[:] = main.snip_compact(msgs)
    print(f"  L1 后: {len(msgs)} 条消息, {main.estimate_size(msgs)} chars")

    msgs[:] = main.micro_compact(msgs)
    print(f"  L2 后: {main.estimate_size(msgs)} chars")

    after = main.estimate_size(msgs)
    reduction = (1 - after / before) * 100 if before else 0
    print(f"  总压缩率: {reduction:.1f}%")


def test_context_limit():
    """测试超 CONTEXT_LIMIT 触发条件"""
    print("\n" + "="*60)
    print(f"CONTEXT_LIMIT 触发测试: 阈值={main.CONTEXT_LIMIT}")
    msgs_low = build_test_messages(n_tools=5, output_size=200)
    msgs_high = build_test_messages(n_tools=60, output_size=2000)
    print(f"  少量消息: {main.estimate_size(msgs_low)} chars → {'触发' if main.estimate_size(msgs_low) > main.CONTEXT_LIMIT else '不触发'}")
    print(f"  大量消息: {main.estimate_size(msgs_high)} chars → {'触发' if main.estimate_size(msgs_high) > main.CONTEXT_LIMIT else '不触发'}")


# ========== 运行 ==========
if __name__ == "__main__":
    print("上下文压缩功能测试")
    print(f"CONTEXT_LIMIT={main.CONTEXT_LIMIT}, KEEP_RECENT={main.KEEP_RECENT}, PERSIST_THRESHOLD={main.PERSIST_THRESHOLD}")

    test_l1_snip()
    test_l2_micro()
    test_l3_budget()
    test_l4_compact()
    test_l5_reactive()
    test_full_pipeline()
    test_context_limit()

    # 清理持久化文件
    print(f"\n清理 {main.TOOL_RESULTS_DIR}")
    for f in main.TOOL_RESULTS_DIR.glob("*.txt"):
        f.unlink()
    print("测试完成")
