from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

from allennlp.data.tokenizers import Token
from allennlp.data.fields import Field, TextField, LabelField, SpanField, MetadataField

from allennlp.data.instance import Instance

from typing import Dict, List, Iterator
import json

@DatasetReader.register("uds_srl_reader")
class UDSDatasetReader(DatasetReader):
    
    def __init__(self, 
                token_indexers: Dict[str, TokenIndexer] = None, 
                lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterator[Instance]:
        #YOUR CODE HERE
        with open(file_path, "r") as f:
            data = json.load(f)
            for example in data:
                #import pdb; pdb.set_trace()            
                tokens = data[example]["tokens"]
                predicate_index = int(data[example]["predicate_head_idx"])
                argument_index = int(data[example]["argument_head_idx"])
                label = data[example]["label"]
                yield self.text_to_instance(tokens, predicate_index, argument_index, label)

    def text_to_instance(self,
                tokens: List[str],
                predicate_index: int,
                argument_index: int,
                label: str) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = TextField([Token(t) for t in tokens], self._token_indexers)
        fields["tokens"] = tokens
        fields["predicate_index"] = SpanField(span_start=predicate_index - 1, span_end=predicate_index - 1, sequence_field=tokens)
        fields["argument_index"] = SpanField(span_start=argument_index - 1, span_end=argument_index - 1, sequence_field=tokens)
        fields["label"] = LabelField(label)
        
        return Instance(fields)

