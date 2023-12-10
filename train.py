
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from model import ASRModel
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
max_len = 600_000
max_text_len = 300


def calculate_mean_std(batch):
    mean = 0.
    std = 0.
    mean += torch.mean(batch['audio']['array'], dim=-1)
    std += torch.std(batch['audio']['array'], dim=-1)
    mean /= len(batch)
    std /= len(batch)
    return mean, std


def z_score_norm(dataset, mean, std):
    for data in dataset:
        data['audio']['array'] = (data['audio']['array'] - mean) / std
    return dataset


def custom_collate(batch):
    filtered_audio = []
    transformed_text = []
    max_v = 0
    for item in batch:
        if item['audio']['array'].shape[0] < max_len:
            item['audio']['array'] = torch.functional.F.pad(
                item['audio']['array'], (0, max_len - item['audio']['array'].shape[0]))
        item['audio']['array'] = item['audio']['array'][::10]
        filtered_audio.append(item['audio']['array'].unsqueeze(-2))
        max_v = max(max_v, item['audio']['array'].shape[0])
        if len(item['text']) < max_text_len:
            transformed_text.append(
                (item['text'] + ' ' * (max_text_len - len(item['text']))).lower())
        else:
            transformed_text.append(item['text'][:max_text_len])

    audio = torch.stack(filtered_audio)
    return {"audio": audio, "text": transformed_text}


def setup_datset():
    dataset_train = load_dataset("librispeech_asr", split="train.clean.360")
    dataset_train.set_format("torch")

    dataset_test = load_dataset("librispeech_asr", split="test.clean")
    dataset_test.set_format("torch")

    dataset_validation = load_dataset(
        "librispeech_asr", split="validation.clean")
    dataset_validation.set_format("torch")

    train_loader = DataLoader(
        dataset_train, batch_size=8, shuffle=True, collate_fn=custom_collate, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=8,
                             shuffle=True, collate_fn=custom_collate, num_workers=4,  pin_memory=True)
    validation_loader = DataLoader(
        dataset_validation, batch_size=8, shuffle=False, collate_fn=custom_collate, num_workers=8, pin_memory=True)
    return train_loader, test_loader, validation_loader


# , logger=TensorBoardLogger(save_dir='.tensorboard', name='asr')
if __name__ == "__main__":
    pl.seed_everything(42)
    train_loader, test_loader, validation_loader = setup_datset()
    trainer = pl.Trainer(accelerator="gpu", gpus=2, precision='16',max_epochs=10, callbacks=[RichProgressBar(), ModelCheckpoint(), EarlyStopping(
        patience=10, monitor='val_loss', mode="min"), LearningRateMonitor(logging_interval='step')], log_every_n_steps=1, logger=TensorBoardLogger(save_dir='.tensorboard', name='asr'))
    model = ASRModel({"input_dim": 1, "num_channels": [64, 128, 256, 512], "kernel_size": 13,
                      "dropout": 0.3, "dilation_size": 4, "output_dim": 91, "lr": 0.001, "output_len": 60000, 'text_length': max_text_len} )
    trainer.fit(model, train_loader, validation_loader)
    trainer.test(test_dataloaders=test_loader)
