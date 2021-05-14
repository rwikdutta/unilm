from transformers import CONFIG_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_NAMES_MAPPING, TOKENIZER_MAPPING, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, BertConverter
from transformers.models.auto.modeling_auto import auto_class_factory

from .modeling.layoutlmv2 import (
    LayoutLMv2Config,
    LayoutLMv2ForTokenClassification,
    LayoutLMv2Tokenizer,
    LayoutLMv2TokenizerFast,
    LayoutLMv2ForSequenceClassification
)


CONFIG_MAPPING.update([("layoutlmv2", LayoutLMv2Config)])
MODEL_NAMES_MAPPING.update([("layoutlmv2", "LayoutLMv2")])
TOKENIZER_MAPPING.update([(LayoutLMv2Config, (LayoutLMv2Tokenizer, LayoutLMv2TokenizerFast))])
SLOW_TO_FAST_CONVERTERS.update({"LayoutLMv2Tokenizer": BertConverter})
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update([(LayoutLMv2Config, LayoutLMv2ForTokenClassification)])
AutoModelForTokenClassification = auto_class_factory(
    "AutoModelForTokenClassification", MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, head_doc="token classification"
)
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.update([LayoutLMv2Config,LayoutLMv2ForSequenceClassification])
AutoModelForSequenceClassification = auto_class_factory(
  "AutoModelForSequenceClassification",MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,head_doc="sequence classification"
)
