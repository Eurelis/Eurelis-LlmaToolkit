from typing import Iterator, List, Sequence

from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document

from eurelis_llmatoolkit.utils.base_factory import BaseFactory
from eurelis_llmatoolkit.types import FACTORY


class ListLoader(BaseLoader):
    """
    List loader class to transform a single target based loader into a multi target one
    """

    def __init__(
        self,
        targets: Sequence[str],
        loader: FACTORY,
        varname: str,
        parameters: dict,
        context,
    ):
        """
        Constructor
        Args:
            targets (Sequence[str]: list of targets (strings)
            loader (FACTORY): parameter to get the under the hood loader factory
            varname (str): name of the parameter on the sub-factory to provide target value with
            parameters (dict): parameters for the under the hood loader factory
            context: the context object, usually the current langchain wrapper instance
        """
        self.loader = loader
        self.targets = targets
        self.varname = varname
        self.context = context
        self.parameters = parameters

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """
        Lazy load method
        Returns:
            iterator over documents
        """

        for target in self.targets:
            params = dict()
            params[self.varname] = target

            # TODO: instantiate factory only one time, and clear params during the loop
            loader_factory = self.context.loader.instantiate_factory(
                "eurelis_llmatoolkit.langchain.document_loaders",
                "GenericLoaderFactory",
                self.loader.copy() if isinstance(self.loader, dict) else self.loader,
            )
            loader_factory.set_params(params)

            final_loader = loader_factory.build(self.context)

            try:
                # preferred method to use
                documents = final_loader.lazy_load()
            except NotImplementedError:
                # fallback if it isn't implemented
                documents = final_loader.load()

            yield from documents

    def load(self) -> List[Document]:
        """
        Load method
        Returns:
            list of documents
        """
        return list(self.lazy_load())
