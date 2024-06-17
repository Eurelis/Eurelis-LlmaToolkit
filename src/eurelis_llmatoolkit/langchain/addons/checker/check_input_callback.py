from typing import Any, Dict, List, Optional
from uuid import UUID

from eurelis_llmatoolkit.langchain.addons.checker import CheckInput, Method
from eurelis_llmatoolkit.langchain.addons.checker.chat_checker import ChatChecker
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage


class CheckInputCallback(BaseCallbackHandler):
    def __init__(
        self, checker, method: Optional[Method] = Method.NLI, language: str = "en"
    ):
        super().__init__()
        self.messages: List[BaseMessage] = []
        self.checker: ChatChecker = checker
        self.method: Optional[Method] = method
        self.language: str = language

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> Any:
        self.messages = messages[
            0
        ]  # we have only one message in the chain at a given time

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None and "answer" in outputs:  # last chain_end
            answer = outputs.get("answer")

            if not isinstance(answer, str):
                raise RuntimeError(
                    f"answer value should be a string not {type(answer)}"
                )

            values = self.checker.check(
                CheckInput(self.messages, answer, self.language), self.method
            )
            outputs.update({"selfcheck": values})
