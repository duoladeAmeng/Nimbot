# import typer
#
# app = typer.Typer(
#     name="myapp",                    # 应用名
#     help="应用帮助文本",              # 应用描述
#     add_completion=True,             # 启用自动补全
#     no_args_is_help=True,            # 无参数显示帮助
# )
# def hello_world(i):
#     print("[CLI] hello world")
# app.command(name="hello")(hello_world)
#
#
# if __name__ == "__main__":
#     app()


import typer

# 1. 创建一个 Typer 应用实例（相当于一个容器）
app = typer.Typer()

# 2. 注册第一个命令：hello
@app.command()
def hello(name: str, count: int = 1):
    """
    向某人打招呼 [count] 次
    """
    for _ in range(count):
        print(f"Hello {name}!")

# 3. 注册第二个命令：goodbye
@app.command()
def goodbye(name: str):
    """
    跟某人说再见
    """
    print(f"Goodbye {name}!")

if __name__ == "__main__":
    # 运行这个应用
    app()