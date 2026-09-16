import typer

from duolaAgent.cli.agent import agent

app = typer.Typer(
    name="duolaAgent",
    help="duola AI Agent",
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        agent()

app.command("agent")(agent)
