import torch
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class FourEncoderClassifier(torch.nn.Module):

    def __init__(
        self,
        model_name,
        num_labels=3,
        class_weights=None,
        dropout=0.0
    ):
        super().__init__()

        self.encoder_high = AutoModel.from_pretrained(model_name)
        self.encoder_medium = AutoModel.from_pretrained(model_name)
        self.encoder_low = AutoModel.from_pretrained(model_name)
        self.encoder_background = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder_high.config.hidden_size

        self.gate_high = torch.nn.Linear(
            hidden_size,
            hidden_size
        )

        self.gate_medium = torch.nn.Linear(
            hidden_size,
            hidden_size
        )

        self.gate_low = torch.nn.Linear(
            hidden_size,
            hidden_size
        )

        self.gate_background = torch.nn.Linear(
            hidden_size,
            hidden_size
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(
                hidden_size * 4,
                hidden_size
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(
                hidden_size,
                num_labels
            )
        )

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                class_weights
            )
        else:
            self.class_weights = None

    def mean_pool(
        self,
        hidden_states,
        attention_mask
    ):
        mask = (
            attention_mask
            .unsqueeze(-1)
            .expand(hidden_states.size())
            .float()
        )

        summed = (
            hidden_states * mask
        ).sum(dim=1)

        counts = (
            mask.sum(dim=1)
            .clamp(min=1e-9)
        )

        return summed / counts

    def apply_gate(
        self,
        representation,
        gate_layer
    ):
        return (
            torch.sigmoid(
                gate_layer(representation)
            )
            * representation
        )

    def forward(
        self,
        input_ids_high,
        attention_mask_high,
        input_ids_medium,
        attention_mask_medium,
        input_ids_low,
        attention_mask_low,
        input_ids_background,
        attention_mask_background,
        labels=None
    ):
        high_output = self.encoder_high(
            input_ids=input_ids_high,
            attention_mask=attention_mask_high
        )

        medium_output = self.encoder_medium(
            input_ids=input_ids_medium,
            attention_mask=attention_mask_medium
        )

        low_output = self.encoder_low(
            input_ids=input_ids_low,
            attention_mask=attention_mask_low
        )

        background_output = self.encoder_background(
            input_ids=input_ids_background,
            attention_mask=attention_mask_background
        )

        high = self.mean_pool(
            high_output.last_hidden_state,
            attention_mask_high
        )

        medium = self.mean_pool(
            medium_output.last_hidden_state,
            attention_mask_medium
        )

        low = self.mean_pool(
            low_output.last_hidden_state,
            attention_mask_low
        )

        background = self.mean_pool(
            background_output.last_hidden_state,
            attention_mask_background
        )

        high = self.apply_gate(
            high,
            self.gate_high
        )

        medium = self.apply_gate(
            medium,
            self.gate_medium
        )

        low = self.apply_gate(
            low,
            self.gate_low
        )

        background = self.apply_gate(
            background,
            self.gate_background
        )

        fused = torch.cat(
            [
                high,
                medium,
                low,
                background
            ],
            dim=1
        )

        logits = self.classifier(fused)

        loss = None

        if labels is not None:
            loss_function = torch.nn.CrossEntropyLoss(
                weight=self.class_weights
            )

            loss = loss_function(
                logits,
                labels
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )