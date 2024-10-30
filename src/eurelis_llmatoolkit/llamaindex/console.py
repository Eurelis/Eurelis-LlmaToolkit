import click

from eurelis_llmatoolkit.llamaindex.chatbot_wrapper import ChatbotWrapper
from eurelis_llmatoolkit.llamaindex.config_loader import ConfigLoader
from eurelis_llmatoolkit.llamaindex.ingestion_wrapper import IngestionWrapper
from eurelis_llmatoolkit.llamaindex.search_wrapper import SearchWrapper


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
    ctx.obj["search_wrapper"] = SearchWrapper(config_dict)
    ctx.obj["chatbot_wrapper"] = ChatbotWrapper(config_dict)


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
@click.option(
    "--from_cache", is_flag=True, default=False, help="Load documents from cache."
)
@click.pass_context
def dataset_ingest(ctx: click.Context, from_cache: bool):
    """Launch ingestion"""
    dataset_id = ctx.obj["dataset_id"]

    wrapper: IngestionWrapper = ctx.obj["wrapper"]
    wrapper.run(dataset_id=dataset_id, use_cache=from_cache)
    click.echo(f"End of ingestion!")


@dataset.command("cache")
@click.pass_context
def dataset_cache(ctx: click.Context):
    """Generate cache for the dataset"""
    dataset_id = ctx.obj["dataset_id"]

    wrapper: IngestionWrapper = ctx.obj["wrapper"]
    wrapper.generate_cache(dataset_id)
    click.echo(f"End of cache generation!")


@click.group()
@click.option("--query", required=True, help="Search query")
@click.pass_context
def search(ctx: click.Context, query: str):
    """
    Group of commands for dataset management.

    Args:
        ctx: Click context
        query: Search query
    """
    ctx.obj["query"] = query


@search.command("nodes")
@click.pass_context
def search_nodes(ctx: click.Context):
    """Search the vector store"""
    wrapper: SearchWrapper = ctx.obj["search_wrapper"]
    query = ctx.obj["query"]
    results = wrapper.search_nodes(query)
    for result in results:
        click.echo(result)


@search.command("documents")
@click.pass_context
def search_docs(ctx: click.Context):
    """Search the vector store"""
    wrapper: SearchWrapper = ctx.obj["search_wrapper"]
    query = ctx.obj["query"]
    results = wrapper.search_documents(query)
    for result in results:
        click.echo(result)


@click.group()
@click.option("--query", required=True, help="Chat query")
@click.option("--id_conversation", required=True, help="ID de la conversation")
@click.pass_context
def chatbot(ctx: click.Context, query: str, id_conversation: str):
    """
    Group of commands for chatbot management.

    Args:
        ctx: Click context
        query: Chat query
    """
    ctx.obj["query"] = query
    ctx.obj["id_conversation"] = id_conversation


@chatbot.command("chat")
@click.pass_context
def chat(ctx: click.Context):
    """Chat with chatbot"""
    wrapper: ChatbotWrapper = ctx.obj["chatbot_wrapper"]
    query = ctx.obj["query"]
    id_conversation = ctx.obj["id_conversation"]
    result = wrapper.run(id_conversation, query)
    click.echo(result)


# Register the dataset group under the main CLI
cli.add_command(dataset)
cli.add_command(search)
cli.add_command(chatbot)

if __name__ == "__main__":
    cli(obj={})


def main_cli():
    cli(obj={})
