from duolaAgent.cli.agent import agent
import typer

app = typer.Typer(
    name="duolaAgent",
    help="duola AI Agent",
    no_args_is_help=False,
)

@app.callback()
def main():
    pass

app.command("agent")(agent)
