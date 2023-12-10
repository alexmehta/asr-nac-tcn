from typing import Any
import pytorch_lightning as pl
from nt_tcn import TemporalConvNet
from torch import nn
import torch


class ASRModel(pl.LightningModule):
    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.params['lr'])

    def __init__(self, params) -> None:
        super().__init__()
        self.temporal = TemporalConvNet(
            params["input_dim"], params["num_channels"], params["kernel_size"], params["dropout"], params["dilation_size"])
        self.spatial_transform = nn.Linear(
            params["num_channels"][-1], params["output_dim"])
        self.temporal_transform = nn.Linear(
            params['output_len'], params['text_length'])

        self.loss = nn.CrossEntropyLoss()
        self.params = params

    def forward(self, x):
        x = self.temporal(x)
        x = x.permute(0, 2, 1)
        x = self.spatial_transform(x)
        x = x.permute(0, 2, 1)
        x = self.temporal_transform(x)
        x = x.permute(0, 2, 1)
        return x

    def process(self, batch):
        x, y = batch['audio'], batch['text']
        y = [[ord(item) - 32 for item in array] for array in y]
        y = torch.as_tensor(y, device=self.device, dtype=torch.long)

        return x, y

    def training_step(self, batch, batch_idx):
        # if 1st of epoch, pause for 10 minutes
        print(self.current_epoch)
        if batch_idx == 0 and self.current_epoch == 0:
            print("Waiting for 10 minutes")
            import time
            time.sleep(600)
        loss, logits, x, y = self.epoch(batch)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True)
        return loss

    def transform(self, logits):
        logits = torch.softmax(logits, dim=-1)
        selected = logits.argmax(dim=-1)
        selected = selected + 32
        phrases = []
        for phrase in selected:
            phrases.append(''.join([chr(character) for character in phrase]))
        return phrases

    def epoch(self, batch):
        x, y = self.process(batch)
        logits = self(x)
        y_hot = torch.functional.F.one_hot(y, num_classes=91).float()
        loss = self.loss(logits, y_hot)
        return loss, logits, x, y

    def validation_step(self, batch, batch_idx):
        loss, logits, x, y = self.epoch(batch)

        self.log('val_loss_epoch', loss, on_step=False, on_epoch=True)
        self.log('val_loss_step', loss, on_step=True, on_epoch=False)
        predicted_text = self.transform(logits)
        # print("Predicted text: ", predicted_text[0])
        # print("True text: ", batch['text'][0])
        # print("True text from one hot", self.transform(torch.functional.F.one_hot(y, num_classes=91).float()))

        self.logger.experiment.add_text(
            'val_text_prediction', predicted_text[0], self.global_step)
        self.logger.experiment.add_text(
            'val_text_true', self.transform(torch.functional.F.one_hot(y, num_classes=91).float())[0], self.global_step)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, x, y = self.epoch(batch)
        self.log('test_loss', loss)
        return loss
