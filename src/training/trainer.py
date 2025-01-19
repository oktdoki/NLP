# src/training/trainer.py

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Dict, Optional

from src.models.classifier import NarrativeClassifier
from src.models.config import ModelConfig

class Trainer:
    """Trainer class for the narrative classification model."""
    
    def __init__(
        self,
        model: NarrativeClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ModelConfig,
        device: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Model configuration
            device: Device to use for training (will use cuda if available if not specified)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Set up device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Set up optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Set up learning rate scheduler
        num_training_steps = len(train_loader) * config.max_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Initialize best metrics for model saving
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        # Use tqdm for progress bar
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                narrative_labels=batch['narrative_labels'],
                subnarrative_labels=batch['subnarrative_labels']
            )
            
            loss = outputs['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            batch_size = batch['input_ids'].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item()
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / total_samples
        
        return {
            'loss': avg_loss
        }
    
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validation')
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    narrative_labels=batch['narrative_labels'],
                    subnarrative_labels=batch['subnarrative_labels']
                )
                
                # Update metrics
                batch_size = batch['input_ids'].size(0)
                total_loss += outputs['total_loss'].item() * batch_size
                total_samples += batch_size
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': outputs['total_loss'].item()
                })
        
        # Calculate validation metrics
        avg_loss = total_loss / total_samples
        
        return {
            'loss': avg_loss
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], path: str, save_full: bool = False):
         """Save a checkpoint.

         Args:
             epoch: Current epoch number
             metrics: Current metrics
             path: Path to save checkpoint
             save_full: If True, save full training state, otherwise just model weights
         """
         Path(path).parent.mkdir(parents=True, exist_ok=True)

         if save_full:
             # Save full training state
             checkpoint = {
                 'epoch': epoch,
                 'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'scheduler_state_dict': self.scheduler.state_dict(),
                 'metrics': metrics,
                 'config': self.config
             }
             full_path = str(Path(path).parent / 'full_checkpoint.pt')
             torch.save(checkpoint, full_path)
             self.logger.info(f"Saved full checkpoint to {full_path}")

         # Always save inference version
         inference_checkpoint = {
             'model_state_dict': self.model.state_dict(),
             'config': self.config
         }
         inference_path = path
         torch.save(inference_checkpoint, inference_path)
         self.logger.info(f"Saved inference checkpoint to {inference_path}")
    
    def train(self, checkpoint_dir: str = 'checkpoints'):
        """Train the model."""
        self.logger.info("Starting training...")

        for epoch in range(self.config.max_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")

            # Training
            train_metrics = self.train_epoch()
            self.logger.info(f"Training Loss: {train_metrics['loss']:.4f}")

            # Validation
            val_metrics = self.validate()
            self.logger.info(f"Validation Loss: {val_metrics['loss']:.4f}")

            # Save checkpoint if best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch

                # Save both full and inference checkpoints for best model
                checkpoint_path = Path(checkpoint_dir) / 'best_model.pt'
                self.save_checkpoint(
                    epoch=epoch,
                    metrics=val_metrics,
                    path=str(checkpoint_path),
                    save_full=True  # This will save both versions
                )

            # Early stopping
            if epoch - self.best_epoch >= 5:  # 5 epochs patience
                self.logger.info("Early stopping triggered!")
                break

        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch + 1}")