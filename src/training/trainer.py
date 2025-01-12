import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import os
from typing import Dict, Optional

from ..models.classifier import NarrativeClassifier
from ..utils.config import Config

class Trainer:
    def __init__(
        self,
        model: NarrativeClassifier,
        config: Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda"
    ):
        """Initialize trainer."""
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        lr = float(config.training.learning_rate)
        eps = float(config.optimizer.epsilon)


        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=config.training.weight_decay,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            eps=eps
        )

        # Setup scheduler
        total_steps = len(train_loader) * config.training.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.training.warmup_steps,
            num_training_steps=total_steps
        )

        # Initialize tracking
        self.global_step = 0
        self.best_val_f1 = 0.0

    def train(self):
        """Main training loop."""
        # Initialize wandb
        if self.config.logging.wandb_project:
            wandb.init(project=self.config.logging.wandb_project)
            wandb.config.update(self.config)

        for epoch in range(self.config.training.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")

            # Training
            self.model.train()
            train_loss = 0
            train_steps = 0

            progress_bar = tqdm(self.train_loader, desc="Training")
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels={
                        "narratives": batch["narrative_labels"],
                        "subnarratives": batch["subnarrative_labels"]
                    }
                )

                loss = outputs["loss"]

                # Scale loss if using gradient accumulation
                if self.config.training.gradient_accumulation_steps > 1:
                    loss = loss / self.config.training.gradient_accumulation_steps

                # Backward pass
                loss.backward()
                train_loss += loss.item()
                train_steps += 1

                # Update weights
                if train_steps % self.config.training.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm
                    )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'lr': self.scheduler.get_last_lr()[0]
                })

                # Logging
                if self.global_step % self.config.logging.logging_steps == 0:
                    avg_loss = train_loss / train_steps
                    self._log_metrics({
                        'train/loss': avg_loss,
                        'train/learning_rate': self.scheduler.get_last_lr()[0]
                    })

                # Evaluation
                if self.val_loader and self.global_step % self.config.logging.eval_steps == 0:
                    val_metrics = self.evaluate()
                    self._log_metrics(val_metrics)

                    # Save best model
                    if val_metrics['val/f1'] > self.best_val_f1:
                        self.best_val_f1 = val_metrics['val/f1']
                        self.save_model('best_model.pt')

                # Save checkpoint
                if self.global_step % self.config.logging.save_steps == 0:
                    self.save_model(f'checkpoint-{self.global_step}.pt')

    def evaluate(self):
        """Evaluate the model on validation set."""
        self.model.eval()
        val_loss = 0
        val_steps = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels={
                        "narratives": batch["narrative_labels"],
                        "subnarratives": batch["subnarrative_labels"]
                    }
                )

                val_loss += outputs["loss"].item()
                val_steps += 1

                # Get predictions
                preds = self.model.predict(
                    batch["input_ids"],
                    batch["attention_mask"]
                )
                all_preds.append(preds)
                all_labels.append((batch["narrative_labels"], batch["subnarrative_labels"]))

        # Calculate metrics
        metrics = self._calculate_metrics(all_preds, all_labels)
        metrics['val/loss'] = val_loss / val_steps

        return metrics

    def save_model(self, filename: str):
        """Save model checkpoint."""
        save_path = os.path.join(self.config.logging.output_dir, filename)
        os.makedirs(self.config.logging.output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to wandb if enabled."""
        if self.config.logging.wandb_project:
            wandb.log(metrics, step=self.global_step)
        print(metrics)

    def _calculate_metrics(self, preds, labels):
        """Calculate evaluation metrics."""
        # TODO: Implement detailed metrics calculation for both narratives and subnarratives
        # This should include F1 scores as per task requirements
        return {'val/f1': 0.0}  # Placeholder