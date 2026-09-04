import yaml
import os
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path='content/root-main/config.yaml'):
        self.config = {}
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        # Default configuration values
        self.config = {
            "experiment_settings": {
                "random_seed": 42,
                "num_npus": 2,
                "num_time_steps": 2,
                "npu_default_compute_capacity": 500,
                "latency_constraint": 1000,
                "max_energy_budget": 10000,
                "num_apr_iterations": 3
            },
            "energy_model_penalties": {
                "alpha": 1.0, "beta": 0.1, "gamma": 1.0, "delta": 0.5, "epsilon": 0.1,
                "zeta": 0.01, "eta": 0.01, "theta": 0.1, "phi": 0.1, "rho": 0.0,
                "tau": 0.01, "kappa": 0.01, "lam": 0.01, "mu": 0.01, "nu": 0.01,
                "psi": 0.0, "omega": 0.0
            },
            "qaoa_solver": {
                "p_layers": 1
            },
            "penalty_tuner": {
                "eta1": 0.1,
                "eta2": 0.05,
                "lam_min": 0.01,
                "lam_max": 10.0
            },
            "file_paths": {
                "example_workload_json": "content/root-main/example_workload.json",
                "output_directory": "outputs",
                "plot_prefix": "fig_"
            }
        }

        # Attempt to load from config.yaml if it exists to override defaults
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                if loaded_config:
                    self._merge_dicts(self.config, loaded_config)
                print(f"Loaded configuration from {self.config_path}.")
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}. Using default configuration.")

    def _merge_dicts(self, dict1: Dict, dict2: Dict) -> Dict:
        for k, v in dict2.items():
            if k in dict1 and isinstance(dict1[k], dict) and isinstance(v, dict):
                dict1[k] = self._merge_dicts(dict1[k], v)
            else:
                dict1[k] = v
        return dict1

    def get(self, key: str, default: Any = None) -> Any:
        # Simple getter, can be expanded for nested access
        parts = key.split('.')
        current = self.config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def __getattr__(self, name: str) -> Any:
        # Allow attribute-like access, e.g., config.experiment_settings
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"'ConfigManager' object has no attribute '{name}'")
