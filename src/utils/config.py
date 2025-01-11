import yaml
from dataclasses import dataclass
from typing import List, Dict, Any
import os

@dataclass
class ModelConfig:
    name: str
    max_length: int
    hidden_dropout_prob: float
    attention_dropout_prob: float

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int
    warmup_steps: int
    weight_decay: float
    gradient_accumulation_steps: int
    max_grad_norm: float
    seed: int

@dataclass
class DataConfig:
    train_ratio: float
    validation_ratio: float
    supported_languages: List[str]
    label_categories: Dict[str, List[str]]

@dataclass
class OptimizerConfig:
    type: str
    beta1: float
    beta2: float
    epsilon: float

@dataclass
class LoggingConfig:
    wandb_project: str
    save_steps: int
    eval_steps: int
    logging_steps: int
    output_dir: str

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    optimizer: OptimizerConfig
    logging: LoggingConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found at: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            data=DataConfig(**config_dict['data']),
            optimizer=OptimizerConfig(**config_dict['optimizer']),
            logging=LoggingConfig(**config_dict['logging'])
        )

    def save(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'optimizer': self.optimizer.__dict__,
            'logging': self.logging.__dict__
        }

        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)

def load_config(config_path: str) -> Config:
    """Helper function to load configuration."""
    return Config.from_yaml(config_path)