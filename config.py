import json
import os
import torch

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "stt": {
        "model": "base",
        "device": "cpu",
        "compute_type": "float32"
    },
    "nlp": {
        "model": "all-MiniLM-L6-v2",
        "device": "cpu"
    }
}

def get_default_config():
    config = DEFAULT_CONFIG.copy()
    if torch.cuda.is_available():
        config["stt"]["device"] = "cuda"
        config["stt"]["compute_type"] = "float16"
        config["stt"]["model"] = "large-v3"
        config["nlp"]["device"] = "cuda"
    return config

def load_config():
    if not os.path.exists(CONFIG_FILE):
        config = get_default_config()
        save_config(config)
        return config
    
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
            # Merge defaults if keys are missing
            default = get_default_config()
            for key, val in default.items():
                if key not in config:
                    config[key] = val
                elif isinstance(val, dict):
                    for subkey, subval in val.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subval
            return config
    except Exception as e:
        print(f"Error loading config.json: {e}. Using defaults.")
        return get_default_config()

def save_config(config):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving config.json: {e}")
