from enum import Enum

from eurelis_llmatoolkit.langchain.addons.checker.check_input import CheckInput


class Method(Enum):
    MQAG = "mqag"
    BERTSCORE = "bertscore"
    NGRAM = "ngram"
    NLI = "nli"
