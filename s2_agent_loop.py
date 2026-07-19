"""""
def agent_loop(messages: list, context: dict):
    system = get_system_prompt(context)
    while True:
        # Consume fired cron jobs → inject as messages
        fired = consume_cron_queue()
        for job in fired:
            messages.append({"role": "user",
                             "content": f"[Scheduled] {job.prompt}"})
            print(f"  \033[35m[inject cron] {job.prompt[:50]}\033[0m")

        try:
            response = client.messages.create(
                model=MODEL, system=system, messages=messages,
                tools=TOOLS, max_tokens=8000)
        except Exception as e:
            messages.append({"role": "assistant", "content": [
                {"type": "text",
                 "text": f"[Error] {type(e).__name__}: {e}"}]})
            return

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\033[36m> {block.name}\033[0m")

            if should_run_background(block.name, block.input):
                bg_id = start_background_task(block)
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"[Background task {bg_id} started] "
                                           f"Result will be available when complete."})
            else:
                output = execute_tool(block)
                print(str(output)[:300])
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": output})

        # Merge background tool results + notifications into one user message
        user_content = list(results)
        bg_notifications = collect_background_results()
        if bg_notifications:
            for notif in bg_notifications:
                user_content.append({"type": "text", "text": notif})
        messages.append({"role": "user", "content": user_content})
        context = update_context(context, messages)
        system = get_system_prompt(context)
"""

"""
if __name__ == "__main__":
    print("s15: agent teams")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history = []
    context = update_context({}, [])

    # input() and a 1s poller (teammate inbox or background results) feed one
    # event queue (issues #291, #46).
    events = queue.Queue()

    def input_reader():
        while True:
            try:
                line = input("\033[36ms15 >> \033[0m")
            except (EOFError, KeyboardInterrupt):
                events.put(("quit", None))
                return
            events.put(("user", line))

    def inbox_poller():
        # Poll ~1s and wake the Lead when async results are ready: teammate
        # inbox messages or completed background tasks. Don't gate on
        # active_teammates: a teammate sends its result and then removes itself,
        # so the final message can outlive its registry entry.
        while True:
            time.sleep(1)
            if BUS.peek("lead") or has_pending_background():
                events.put(("wake", None))

    threading.Thread(target=input_reader, daemon=True).start()
    threading.Thread(target=inbox_poller, daemon=True).start()

    had_teammates = False
    while True:
        kind, payload = events.get()
        if kind == "quit":
            break
        if kind == "user":
            if payload.strip().lower() in ("q", "exit", ""):
                break
            history.append({"role": "user", "content": payload})
        else:  # "wake": teammate inbox or background results are ready
            parts = []
            inbox = BUS.read_inbox("lead")
            if inbox:
                parts.append("[Inbox]\n" + "\n".join(
                    f"From {m['from']}: {m['content'][:200]}" for m in inbox))
            bg = collect_background_results()
            parts.extend(bg)
            if not parts:
                continue  # already drained by an earlier wake (idempotent)
            history.append({"role": "user", "content": "\n".join(parts)})
            print(f"\n\033[33m[wake: {len(inbox)} inbox + {len(bg)} background "
                  f"-> new turn]\033[0m")

        # One turn for whichever source woke us.
        agent_loop(history, context)
        context = update_context(context, history)
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                print(block.get("text", ""))

        # Announce once when every teammate has finished and its output drained.
        if active_teammates:
            had_teammates = True
        elif had_teammates and not BUS.peek("lead") and not has_pending_background():
            print("\033[32m[all teammates done]\033[0m")
            had_teammates = False
        print()
"""