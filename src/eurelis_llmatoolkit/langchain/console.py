import json
import os
from typing import TYPE_CHECKING, cast

import click

from eurelis_llmatoolkit.utils.output import Verbosity

if TYPE_CHECKING:
    from eurelis_llmatoolkit.langchain.langchain_wrapper import LangchainWrapper


# FIXME: Move to TOP level package
@click.group()
@click.option("--verbose/--no-verbose", default=False)
@click.option("-config", default=None)
@click.pass_context
def cli(ctx, **kwargs):
    """
    Root command, will handle retrieving verbose and config options values
    Args:
        ctx: click context
        **kwargs: arguments

    Returns:

    """

    # singleton method to instantiate the LangchainWrapper
    def wrapper() -> "LangchainWrapper":
        instance = getattr(wrapper, "instance")
        if not instance:
            # Get and prepare the factory
            from eurelis_llmatoolkit.langchain import LangchainWrapperFactory
            from eurelis_llmatoolkit.langchain.langchain_wrapper import BaseContext

            factory = LangchainWrapperFactory()

            verbose = kwargs.get("verbose") if "verbose" in kwargs else False
            logger_log = os.getenv('LANGCHAIN_LOG', 'False').lower() == 'true'

            if verbose and logger_log:
                verbosity_level = Verbosity.LOG_DEBUG
            elif verbose:
                verbosity_level = Verbosity.CONSOLE_DEBUG
            elif logger_log:
                verbosity_level = Verbosity.LOG_INFO
            else:
                verbosity_level = Verbosity.CONSOLE_INFO

            factory.set_verbose(verbosity_level)

            if "config" in kwargs and kwargs["config"]:
                factory.set_config_path(kwargs["config"])

            instance = factory.build(
                cast(BaseContext, None)
            )  # casting None to BaseContext is wanted
            setattr(factory, "instance", instance)

        return instance

    wrapper.instance = None
    ctx.obj["singleton"] = wrapper


@cli.group()
@click.option("--id", default=None, help="Dataset ID")
@click.option("--content", default=None, help="Content path")
@click.pass_context
def dataset(ctx, **kwargs):
    """
    Method handling dataset options
    Args:
        ctx: click context
        **kwargs: options
    Returns:

    """
    dataset_id = kwargs["id"] if "id" in kwargs else None
    content_path = kwargs["content"] if "content" in kwargs else None
    wrapper = ctx.obj["singleton"]()
    
    ctx.obj["wrapper"] = wrapper
    ctx.obj["dataset_id"] = dataset_id
    ctx.obj["content_path"] = content_path


@dataset.command("index")
@click.pass_context
def dataset_index(ctx, **kwargs):
    """
    Launch indexation
    Args:
        ctx: click context
        **kwargs: options

    Returns:

    """
    wrapper = ctx.obj["wrapper"]

    wrapper.index_documents(ctx.obj["dataset_id"], ctx.obj["content_path"])

@dataset.command("metadata")
@click.pass_context
def dataset_metadata(ctx, **kwargs):
    """
    Print first doc metadata
    Args:
        ctx: click context
        **kwargs: options

    Returns:

    """
    wrapper = ctx.obj["wrapper"]
    wrapper.print_metadata(ctx.obj["dataset_id"])


@dataset.command("cache")
@click.pass_context
def dataset_cache(ctx, **kwargs):
    """
    Write cache files for metadata
    Args:
        ctx: click context
        **kwargs: options

    Returns:

    """
    wrapper = ctx.obj["wrapper"]
    wrapper.write_files(ctx.obj["dataset_id"])


@dataset.command("ls")
@click.pass_context
def dataset_list(ctx):
    """
    List dataset
    Args:
        ctx: click context

    Returns:

    """
    wrapper = ctx.obj["wrapper"]
    wrapper.list_datasets()


@dataset.command("clear")
@click.pass_context
def dataset_clear(ctx):
    """
    Clear dataset
    Args:
        ctx: click context

    Returns:

    """
    wrapper = ctx.obj["wrapper"]
    wrapper.clear_datasets(ctx.obj["dataset_id"])


@cli.command()
@click.option("--id", default=None, help="Dataset ID")
@click.option("filters", "--filter", multiple=True, type=str)
@click.pass_context
def delete(ctx, filters, **kwargs):
    """
    Delete content from database using a query
    Args:
        ctx: click context
        filters: filters to apply to search query
        **kwargs: options

    Returns:

    """
    dataset_id = kwargs["id"] if "id" in kwargs else None
    wrapper = ctx.obj["singleton"]()

    filter_args = {}

    for filter_arg in filters:
        if not filter_arg:
            continue
        filter_split = filter_arg.split(":")
        filter_key = filter_split[0]
        filter_value = ":".join(filter_split[1:])

        filter_args[filter_key] = filter_value

    if not filter_args:  # if empty dict we consider a None value
        filter_args = None

    wrapper.delete(filter_args, dataset_id)

@cli.command()
@click.option("filters", "--filter", multiple=True, type=str)
@click.pass_context
def unitarydelete(ctx, filters, **kwargs):
    """
    Delete the contents of a database using a query
    Args:
        ctx: click context
        filters : filters to be applied to the query
        **kwargs: options

    Returns:

    """
    wrapper = ctx.obj["singleton"]()

    filter_args = {}

    for filter_arg in filters:
        if not filter_arg:
            continue
        filter_split = filter_arg.split(":")
        filter_key = filter_split[0]
        filter_value = ":".join(filter_split[1:])

        filter_args[filter_key] = filter_value

    if not filter_args:  # if empty dict we consider a None value
        filter_args = None

    wrapper.unitaryDelete(filter_args)


@cli.command()
@click.argument("query")
@click.option("filters", "--filter", multiple=True, type=str)
@click.pass_context
def search(ctx, query, filters):
    """
    Method to handle search
    Args:
        ctx: click context
        query: the text to look for
        filters: list of filters to apply

    Returns:

    """
    filter_args = {}

    for filter_arg in filters:
        if not filter_arg:
            continue
        filter_split = filter_arg.split(":")
        filter_key = filter_split[0]
        filter_value = ":".join(filter_split[1:])

        filter_args[filter_key] = filter_value

    if not filter_args:  # if empty dict we consider a None value
        filter_args = None

    wrapper = ctx.obj["singleton"]()
    wrapper.search_documents(query, for_print=True, search_filter=filter_args)


@cli.command()
@click.option("--selfcheck/--no-selfcheck", default=False)
@click.pass_context
def gradio(ctx, selfcheck: bool):
    wrapper = ctx.obj["singleton"]()

    from eurelis_llmatoolkit.langchain import gradiochat

    gradiochat.define_chatbot(wrapper, selfcheck).launch()


# enable the dataset and search commands
if __name__ == "__main__":
    cli(obj={})


def main_cli():
    cli(obj={})
