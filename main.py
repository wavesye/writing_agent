"""AI Agent 命令行入口。"""

import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

from llm import run_agent


def _prompt_session() -> PromptSession:
    bindings = KeyBindings()

    @bindings.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    return PromptSession(multiline=True, key_bindings=bindings)


def _error_details(error: BaseException) -> str:
    """Flatten TaskGroup/ExceptionGroup so CLI shows the actionable cause."""
    children = getattr(error, "exceptions", None)
    if children:
        details = [_error_details(child) for child in children]
        return "；".join(dict.fromkeys(detail for detail in details if detail))
    cause = error.__cause__ or error.__context__
    own = str(error).strip()
    if cause and str(cause).strip() != own:
        nested = _error_details(cause)
        return f"{own}：{nested}" if own else nested
    return own or error.__class__.__name__


def main() -> None:
    print(
        "Writing Agent · 本地论文风格知识库\n"
        "支持整段多行粘贴；Enter 发送，Option+Enter 换行，输入 exit 退出"
    )
    session = _prompt_session()
    while True:
        try:
            user_input = session.prompt("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("再见！")
            break
        if not user_input:
            continue

        try:
            print(f"\nAgent: {asyncio.run(run_agent(user_input))}")
        except Exception as error:
            print(f"\n调用失败: {_error_details(error)}")


if __name__ == "__main__":
    main()
