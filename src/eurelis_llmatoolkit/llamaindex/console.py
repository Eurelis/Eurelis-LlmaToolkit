import click

from eurelis_llmatoolkit.llamaindex.config_loader import ConfigLoader
from eurelis_llmatoolkit.llamaindex.ingestion_wrapper import IngestionWrapper


@click.group()
@click.option(
    "-config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the configuration file.",
)
@click.pass_context
def cli(ctx: click.Context, config: str):
    """
    Root command to handle configuration options.

    Args:
        ctx: Click context
        config: Path to the configuration file
    """
    config_dict = ConfigLoader.load_config(config)
    ctx.obj["wrapper"] = IngestionWrapper(config_dict)


@click.group()
@click.option("--id", default=None, help="Dataset ID")
@click.pass_context
def dataset(ctx: click.Context, id: str):
    """
    Group of commands for dataset management.

    Args:
        ctx: Click context
        id: Dataset ID
    """
    ctx.obj["dataset_id"] = id


@dataset.command("ingest")
@click.pass_context
def dataset_ingest(ctx: click.Context):
    """Launch ingestion"""
    dataset_id = ctx.obj["dataset_id"]

    wrapper: IngestionWrapper = ctx.obj["wrapper"]
    wrapper.run(dataset_id)
    click.echo(f"End of ingestion !")


# Register the dataset group under the main CLI
cli.add_command(dataset)

if __name__ == "__main__":
    cli(obj={})
