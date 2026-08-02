"""AI Agent 命令行入口。"""

import asyncio

from llm import run_agent


def main() -> None:
    print("VIN Agent v6.3 · 本地知识库 RAG（输入 exit 退出）")
    while True:
        try:
            user_input = input("\n你: ").strip()
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
            print(f"\n调用失败: {error}")


if __name__ == "__main__":
    main()
