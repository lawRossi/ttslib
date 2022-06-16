"""
@Author: Rossi
Created At: 2022-04-18
"""

from collections import defaultdict
import os

from loguru import logger
import pickle
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm
import yaml

from ttslib.audio_processing import TacotronSTFT, inv_mel_spec
from ttslib.dataset import load_adaptation_dataset, load_dataset, BucketIterator
from ttslib.optimizer import FineTuningOptimizer, ScheduledOptimizer
from ttslib.utils import plot_spectrogram, find_class


class Trainer:
    def __init__(self, train_config) -> None:
        self.train_steps = train_config["train_steps"]
        self.batch_size = train_config["batch_size"]
        self.save_steps = train_config["save_steps"]
        self.log_steps = train_config["log_steps"]
        self.eval_epoch = train_config["eval_epoch"]
        self.fine_tuning_start = train_config["fine_tuning_start"]
        self.adaptive_training = train_config["adaptive_training"]
        self.data_dir = train_config["data_dir"]
        self.output_dir = train_config["output_dir"]
        self.device = train_config["device"]
        self.train_logger = SummaryWriter(self.output_dir)
        self.train_config = train_config
        model_config_file = self.train_config["model_config_file"]
        self.model_config = yaml.load(open(model_config_file), Loader=yaml.FullLoader)
        self._init_stft()

    def _init_stft(self):
        config_file = self.train_config["preprocess_config_file"]
        config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        
        self.sampling_rate = config["audio"]["sampling_rate"]
        filter_length = config["stft"]["filter_length"]
        hop_length = config["stft"]["hop_length"]
        win_length = config["stft"]["win_length"]  
        n_channels = config["mel"]["n_mel_channels"]
        f_min = config["mel"]["mel_fmin"]
        f_max = config["mel"]["mel_fmax"]
        self.stft = TacotronSTFT(filter_length, hop_length, win_length, n_channels,
                                 self.sampling_rate, f_min, f_max)

    def train(self):
        model = self._prepare_model()
        optimizer = self._prepare_optimizer(model)
        train_dataset, eval_dataset = self._load_datasets()
        train_data_iter = BucketIterator(train_dataset, self.batch_size, shuffle=True, 
                                         sort_key="mels", device=self.device)
        step = optimizer.current_step
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
                    self._log_model(model, batch, step)
                if step == self.train_steps:
                    break
                if step == self.fine_tuning_start:
                    logger.info("start fine tuning")
                    optimizer = FineTuningOptimizer(model, self.train_config)
                    optimizer.current_step = step

            if epoch >= self.eval_epoch:
                self._evaluate(model, eval_dataset, step)

            if step == self.train_steps:
                logger.info("training finished")
                break

    def _evaluate(self, model, eval_dataset, step):
        logger.info("evaluation")
        eval_data_iter = BucketIterator(eval_dataset, self.batch_size, train=False)
        model.eval()
        with torch.no_grad():
            mel_loss = 0
            n = 0
            for batch in tqdm.tqdm(eval_data_iter, "eval interations"):
                batch_inputs = {field: getattr(batch, field) for field in batch.fields}
                model_outputs = model(**batch_inputs)
                mel_loss += model_outputs["mel_loss"].item()
                n += 1
            mel_loss /= n
            self.train_logger.add_scalar("eval_mel_loss", mel_loss, step)

    def _prepare_model(self):
        model_cls = find_class(self.train_config["model"])
        model = model_cls(self.model_config)

        if self.train_config["restore_path"] is not None:
            checkpoint_path = self.train_config["restore_path"]
            logger.info(f"restore from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model"])

        model.to(self.device)

        return model

    def _prepare_optimizer(self, model):
        optimizer = ScheduledOptimizer(model, self.train_config)

        if self.train_config["restore_path"] is None:
            return optimizer

        checkpoint_path = self.train_config["restore_path"]
        step = int(checkpoint_path[checkpoint_path.find("_")+1:])
        
        if step > self.train_config["fine_tuning_start"] or self.adaptive_training:
            logger.info("fine tuning")
            optimizer = FineTuningOptimizer(model, self.train_config)
            optimizer.current_step = step
        else:   
            checkpoint = torch.load(checkpoint_path)
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer.current_step = step

        return optimizer

    def _load_datasets(self):
        if self.model_config["use_existing_speaker_vectors"]:
            speaker_vector_file = self.model_config["speaker_encoder"]["vector_file"]
        else:
            speaker_vector_file = None

        if not self.adaptive_training:
            use_pitch = self.model_config.get("use_pitch", True)
            train_dataset, eval_dataset = load_dataset(self.data_dir, speaker_vector_file, use_pitch)
        else:
            train_dataset, eval_dataset = load_adaptation_dataset(
                self.data_dir,
                os.path.dirname(self.train_config["restore_path"]),
                speaker_vector_file,
                self.train_config["adaptation_speakers"]
            )            

        fields = train_dataset.fields
        phonemes = fields.get("phonemes")
        with open(os.path.join(self.output_dir, "phonemes.pkl"), "wb") as fo:
            pickle.dump(phonemes, fo)
        speakers = fields.get("speakers")
        with open(os.path.join(self.output_dir, "speakers.pkl"), "wb") as fo:
            pickle.dump(speakers, fo)

        return train_dataset, eval_dataset

    def _log_step(self, optimizer, losses, step):
        self.train_logger.add_scalar("lr", optimizer.get_lr(), step)
        for loss_name, loss in losses.items():
            loss /= self.log_steps
            self.train_logger.add_scalar(loss_name, loss, step)

    def _log_model(self, model, batch, step):
        if "durations" in batch.fields:
            duration = int(batch.durations[0].sum().item())
        else:
            duration = batch.mel_lens[0].item()
        target_mel = batch.mels[0][:duration].cpu()
        fig = plot_spectrogram(target_mel)
        self.train_logger.add_figure("target_mel", fig, step)

        model.eval()
        with torch.no_grad():
            batch_inputs = {field: getattr(batch, field) for field in batch.fields}
            mels = model.inference(**batch_inputs)
        model.train()

        try:
            fig = plot_spectrogram(mels[0])
            self.train_logger.add_figure("pred_mel", fig, step)
        except:
            logger.warning("failed to plot pred mel")

        target_audio = inv_mel_spec(target_mel.transpose(1, 0), self.stft)
        target_audio = target_audio / max(abs(target_audio))

        try:
            pred_mel = mels[0].transpose(1, 0).cpu()
            pred_audio = inv_mel_spec(pred_mel, self.stft)
            pred_audio = pred_audio / max(abs(pred_audio))
            self.train_logger.add_audio(f"target_audio-{step}", target_audio, step, sample_rate=self.sampling_rate)
            self.train_logger.add_audio(f"pred_audio-{step}", pred_audio, step, sample_rate=self.sampling_rate)
        except:
            logger.warning("fail to generate predicted audio")

    def _save_checkpoint(self, model, optimizer):
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
             },
            os.path.join(self.output_dir, f"checkpoint_{optimizer.current_step}")
        )


if __name__ == "__main__":
    config = yaml.load(open("data/train_config.yaml"), Loader=yaml.FullLoader)
    trainer = Trainer(config)
    trainer.train()
