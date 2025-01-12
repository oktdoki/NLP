# src/utils/config.py
from dataclasses import dataclass
from typing import List, Dict, Optional

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

def load_config(config_path: str) -> Config:
    """Load config from YAML file."""
    import yaml

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    model_config = ModelConfig(**config_dict['model'])
    training_config = TrainingConfig(**config_dict['training'])
    data_config = DataConfig(**config_dict['data'])
    optimizer_config = OptimizerConfig(**config_dict['optimizer'])
    logging_config = LoggingConfig(**config_dict['logging'])

    return Config(
        model=model_config,
        training=training_config,
        data=data_config,
        optimizer=optimizer_config,
        logging=logging_config
    )