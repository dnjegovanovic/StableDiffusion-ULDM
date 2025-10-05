from pathlib import Path
from typing import Dict
from pydantic import BaseModel
from strictyaml import YAML, load
from yaml.loader import FullLoader
import yaml
import ulsd_model as UCDM

# Project Directories
PACKAGE_ROOT = Path(UCDM.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"


class AppConfig(BaseModel):
    package_name: str


class UNet(BaseModel):
    UnetParams: Dict


class UCDModel(BaseModel):
    UCDMParams: Dict


class Config(BaseModel):
    """Master config object."""

    unet_model: UNet
    ucdm_model: UCDModel
    app_config: AppConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    with open(cfg_path, "r") as stream:
        try:
            # Converts yaml document to python object
            parsed_config = yaml.load(stream, Loader=FullLoader)
            return parsed_config
        except yaml.YAMLError as e:
            print(e)


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()
        for k, v in parsed_config.items():
            if k == "UnetParams":
                parsed_config[k]["im_channels"] = int(parsed_config[k]["im_channels"])
                parsed_config[k]["im_size"] = int(parsed_config[k]["im_size"])
                parsed_config[k]["down_channels"] = parsed_config[k]["down_channels"]
                parsed_config[k]["mid_channels"] = parsed_config[k]["mid_channels"]
                parsed_config[k]["down_sample"] = parsed_config[k]["down_sample"]
                parsed_config[k]["time_emb_dim"] = int(parsed_config[k]["time_emb_dim"])
                parsed_config[k]["num_down_layers"] = int(
                    parsed_config[k]["num_down_layers"]
                )
                parsed_config[k]["num_mid_layers"] = int(
                    parsed_config[k]["num_mid_layers"]
                )
                parsed_config[k]["num_up_layers"] = int(
                    parsed_config[k]["num_up_layers"]
                )
                parsed_config[k]["num_heads"] = int(parsed_config[k]["num_heads"])
            elif k == "DDPMParams":
                parsed_config[k]["num_timesteps"] = int(
                    parsed_config[k]["num_timesteps"]
                )
                parsed_config[k]["beta_start"] = float(parsed_config[k]["beta_start"])
                parsed_config[k]["beta_end"] = float(parsed_config[k]["beta_end"])
            else:
                Exception("No configuration in config file.")

    _config = Config(
        app_config=AppConfig(**parsed_config),
        unet_model=UNet(**parsed_config),
        ucdm_model=UCDModel(**parsed_config),
    )

    return _config


config = create_and_validate_config()