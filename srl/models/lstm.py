from allennlp.models import Model

import torch
import torch.nn as nn

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.metrics import SpanBasedF1Measure, F1Measure

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from typing import Dict, Optional

@Model.register("srl_lstm")
class SRLLSTM(Model):
    def __init__(self,
            vocab: Vocabulary,
            embedder: TextFieldEmbedder,
            encoder: Seq2SeqEncoder) -> None:
        super().__init__(vocab)
        self._embedder = embedder
        self._encoder = encoder
        self._classifier = nn.Linear(in_features=encoder.get_output_dim()*2,
                out_features=vocab.get_vocab_size('labels'))
        self._metric = F1Measure(positive_label=vocab.get_token_index(token="positive", namespace='labels'))
        #self._metric = SpanBasedF1Measure(vocab, 'labels')

    def forward(self,
            tokens: Dict[str, torch.Tensor],
            predicate_index: torch.Tensor,
            argument_index: torch.Tensor,
            label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        
        # get embeddings, run through seq2seq
        embedded = self._embedder(tokens)
        encoded = self._encoder(embedded, mask)

        # extract predicate and argument token encodings
        batch_size = encoded.size()[0]
        hidden_size = encoded.size()[2]
        #pred_idxs = predicate_index.gather(1, torch.ones(batch_size, dtype=torch.int64, device=torch.device("cuda:0")).view(-1,1))
        pred_idxs = predicate_index.gather(1, torch.ones(batch_size, dtype=torch.int64).view(-1,1))
        #arg_idxs = argument_index.gather(1, torch.ones(batch_size, dtype=torch.int64, device=torch.device("cuda:0")).view(-1,1)) 
        arg_idxs = argument_index.gather(1, torch.ones(batch_size, dtype=torch.int64).view(-1,1)) 
        pred_mask = pred_idxs.repeat(1, hidden_size).unsqueeze(1)
        arg_mask = arg_idxs.repeat(1, hidden_size).unsqueeze(1)
        pred_tensors = encoded.gather(1, pred_mask) 
        arg_tensors = encoded.gather(1, arg_mask)

        # concatenate predicate and argument token encodings
        catted = torch.cat((pred_tensors, arg_tensors), dim=2)

        logits = self._classifier(catted)
        
        #preds = torch.argmax(logits, dim=2)
        #import pdb; pdb.set_trace()


        self._metric(logits.detach(), label.view(-1,1))
        
        output: Dict[str, torch.Tensor] = {}
        output['logits'] = logits

        if label is not None:
            output["loss"] = sequence_cross_entropy_with_logits(logits, label.view(-1,1), mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._metric.get_metric(reset)

