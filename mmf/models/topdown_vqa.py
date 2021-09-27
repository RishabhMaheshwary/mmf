from copy import deepcopy

import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import ClassifierLayer, Flatten, GatedTanh
from mmf.modules.attention import ConcatenationAttention
from mmf.modules.embeddings import BiLSTMTextEmbedding
from torch import nn


_TEMPLATES = {
    "question_vocab_size": "{}_text_vocab_size",
    "number_of_answers": "{}_num_final_outputs",
}

_CONSTANTS = {"hidden_state_warning": "hidden state (final) should have 1st dim as 2"}


@registry.register_model("topdown_vqa")
class TopDownVQA(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets.split(",")
        self.data_state = {}

    @classmethod
    def config_path(cls):
        return "configs/models/topdown_vqa/defaults.yaml"


    def build(self):
        assert len(self._datasets) > 0
        num_question_choices = registry.get(
            _TEMPLATES["question_vocab_size"].format(self._datasets[0])
        )
        num_answer_choices = registry.get(
            _TEMPLATES["number_of_answers"].format(self._datasets[0])
        )
        print(num_answer_choices)
        print(self._datasets)
        self.text_embedding = nn.Embedding(
            num_question_choices, self.config.text_embedding.embedding_dim
        )

        self.lstm = nn.GRU(**self.config.lstm)

        self.cnn1 = nn.Conv2d(in_channels=2560 ,out_channels=512, kernel_size=1)
        self.cnn2 = nn.Conv2d(in_channels=2560 ,out_channels=512, kernel_size=1)
        self.cnn_scores = nn.Conv2d(in_channels=512 ,out_channels=1, kernel_size=1)
        self.gated_tanh = GatedTanh(512, 512)
        self.gated_tanh_v = GatedTanh(2048, 512)
        self.final_gated_tanh = GatedTanh(512, 1024)
        # As we generate output dim dynamically, we need to copy the config
        # to update it
        classifier_config = deepcopy(self.config.classifier)
        classifier_config.params.out_dim = num_answer_choices
        self.classifier = ClassifierLayer(
            classifier_config.type, **classifier_config.params
        )

    def _build_word_embedding(self):
        assert len(self._datasets) > 0
        text_processor = registry.get(self._datasets[0] + "_text_processor")
        vocab = text_processor.vocab
        self.word_embedding = vocab.get_embedding(torch.nn.Embedding, embedding_dim=300)

    def forward(self, sample_list):
        self.lstm.flatten_parameters()
        question = sample_list.text
        image_feat = sample_list.image_feature_0
        batch_size = image_feat.shape[0]
        image_feat = torch.reshape(image_feat, (batch_size, 2048, 10, 10))
        #breakpoint()
        out, hidden = self.lstm(self.text_embedding(question))
        out = hidden[-1,:,:]
        out_reshaped = torch.reshape(out, (batch_size, out.shape[1], 1, 1))
        q_embs = torch.tile(out_reshaped, (1, 1, image_feat.shape[2], image_feat.shape[3]))

        qv_embs1 = torch.sigmoid(self.cnn1(torch.cat([image_feat, q_embs], dim=1)))
        qv_embs2 = torch.tanh(self.cnn2(torch.cat([image_feat, q_embs], dim=1)))

        qv_embs = qv_embs1 * qv_embs2
        attn_scores = self.cnn_scores(qv_embs)
        softmax = torch.nn.Softmax(dim=0)

        attn_scores = softmax(attn_scores)
        attn_scores = torch.reshape(attn_scores, (batch_size, 10, 10, 1))
        image_feat = torch.reshape(image_feat, (batch_size, 10, 10, 2048))
        attn_feat = attn_scores * image_feat
        #breakpoint()
        attn_feat = torch.mean(attn_feat, dim=(1,2))

        v_final = self.gated_tanh_v(attn_feat)
        q_final = self.gated_tanh(out)

        qv_combine = q_final * v_final

        scores = torch.sigmoid(self.classifier(self.final_gated_tanh(qv_combine)))
        return {"scores": scores}
