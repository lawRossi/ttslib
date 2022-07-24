"""
@Author: Rossi
Created At: 2022-07-24
"""

from collections import defaultdict
import glob
import os
import re

from loguru import logger
import pickle
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import tqdm
import yaml

from ttslib.audio_processing import TacotronSTFT, inv_mel_spec
from ttslib.dataset import load_align_dataset, BucketIterator
from ttslib.optimizer import ScheduledOptimizer
from ttslib.utils import plot_spectrogram, find_class


class Trainer:
    def __init__(self, train_config) -> None:
        self.train_steps = train_config["train_steps"]
        self.batch_size = train_config["batch_size"]
        self.save_steps = train_config["save_steps"]
        self.log_steps = train_config["log_steps"]
        self.output_dir = train_config["output_dir"]
        self.device = train_config["device"]
        self.train_logger = SummaryWriter(self.output_dir)
        self.train_config = train_config

    def train(self):
        model = self._prepare_model()
        optimizer = self._prepare_optimizer(model)
        train_dataset, eval_dataset = self._load_datasets()
        train_data_iter = BucketIterator(train_dataset, self.batch_size, shuffle=True,
                                         sort_key="mels", device=self.device)
        step = 0
        losses = defaultdict(float)
        for epoch in range(10000):
            logger.info(f"epoch {epoch}")
            model.train()
            for batch in tqdm.tqdm(train_data_iter, "train interations"):
                batch_inputs = {field: getattr(batch, field) for field in batch.fields}
                model_outputs = model(**batch_inputs)
                total_loss = model_outputs["total_loss"]
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                for key, loss in model_outputs.items():
                    losses[key] += loss.item()
                if step % self.log_steps == 0:
                    self._log_step(optimizer, losses, step)
                    losses.clear()
                if step % self.save_steps == 0:
                    self._save_checkpoint(model, optimizer)
                    break
                if step == self.train_steps:
                    break

            self._evaluate(model, eval_dataset, step)
            if step == self.train_steps:
                logger.info("training finished")
                break

    def _evaluate(self, model, eval_dataset, step):
        logger.info("evaluation")
        eval_data_iter = BucketIterator(eval_dataset, self.batch_size, train=False, sort=False)
        model.eval()
        with torch.no_grad():
            ctc_loss = 0
            alignment_loss = 0
            n = 0
            for batch in tqdm.tqdm(eval_data_iter, "eval interations"):
                batch_inputs = {field: getattr(batch, field) for field in batch.fields}
                losses = model(**batch_inputs)
                ctc_loss += losses["ctc_loss"].item()
                alignment_loss += losses["alignment_loss"].item()
                n += 1
            ctc_loss /= n
            alignment_loss /= n
            self.train_logger.add_scalar("eval/ctc_loss", ctc_loss, step)
            self.train_logger.add_scalar("eval/alignment_loss", alignment_loss, step)

    def _prepare_model(self):
        model_config_file = self.train_config["model_config_file"]
        model_config = yaml.load(open(model_config_file), Loader=yaml.FullLoader)
        model_cls = find_class(model_config["model"])
        model = model_cls(model_config)

        if self.train_config["restore_path"] is not None:
            checkpoint_path = self.train_config["restore_path"]
            logger.info(f"restore from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model"])
            step = int(checkpoint_path[checkpoint_path.find("_")+1:])
            if step >= self.fine_tuning_start:
                logger.info("fine tuning")
                model.start_fine_tuning()

        model.to(self.device)

        return model

    def _prepare_optimizer(self, model):
        lr = self.train_config["optimizer"]["lr"]
        optimizer = Adam(model.parameters(), lr=lr)
        return optimizer

    def _load_datasets(self):
        metadata_file = self.train_config["metadata_file"]
        data_dir = self.train_config["data_dir"]
        train_dataset, eval_dataset = load_align_dataset(
            metadata_file,
            data_dir
        )

        fields = train_dataset.fields
        phonemes = fields.get("phonemes")
        with open(os.path.join(self.output_dir, "phonemes.pkl"), "wb") as fo:
            pickle.dump(phonemes, fo)

        return train_dataset, eval_dataset

    def _log_step(self, optimizer, losses, step):
        for loss_name, loss in losses.items():
            loss /= self.log_steps
            self.train_logger.add_scalar(loss_name, loss, step)

    def _save_checkpoint(self, model, optimizer):
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
             },
            os.path.join(self.output_dir, "checkpoint")
        )


if __name__ == "__main__":
    config = yaml.load(open("data/train_aligner_config.yaml"), Loader=yaml.FullLoader)
    trainer = Trainer(config)
    trainer.train()
