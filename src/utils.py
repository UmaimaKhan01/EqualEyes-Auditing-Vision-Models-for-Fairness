import os
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import torch

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "gender_bias_analysis.log"),
            logging.StreamHandler()
        ]
    )

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def load_environment() -> None:
    """Load environment variables from .env file."""
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
    else:
        logging.warning("No .env file found. Make sure to set required environment variables.")

def get_device() -> torch.device:
    """Get the appropriate device (GPU/CPU) for processing."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU for processing")
    
    return device

def create_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories for the project."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/annotations",
        "models",
        "results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def validate_environment() -> bool:
    """Validate that required environment variables are set."""
    required_vars = ["HUGGINGFACE_TOKEN"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logging.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    return True

def save_results(data: Any, filename: str, format: str = "json") -> None:
    """Save results in specified format."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    filepath = results_dir / f"{filename}.{format}"
    
    if format == "json":
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    elif format == "csv":
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            pd.DataFrame(data).to_csv(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logging.info(f"Results saved to {filepath}")

class ProgressTracker:
    """Track progress of long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        from tqdm import tqdm
        self.pbar = tqdm(total=total, desc=description)
        self.current = 0
    
    def update(self, n: int = 1) -> None:
        self.pbar.update(n)
        self.current += n
    
    def close(self) -> None:
        self.pbar.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()