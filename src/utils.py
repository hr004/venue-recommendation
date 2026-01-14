import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def parse_config_files(
    config_folder_paths_to_search: list[str] | None = [
        "/etc/service",
        "./target",
        "/var/lib/vault",
    ]
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    config_file_paths: list[str] = []
    if type(config_folder_paths_to_search) is list:
        for config_path in config_folder_paths_to_search:
            config_file_paths = config_file_paths + [
                os.path.join(dirpath, f)
                for (dirpath, dirnames, filenames) in os.walk(config_path)
                for f in filenames
            ]
        # Loading configuration
        # Ordered list of config files with precedence
        config_file_names = ["/config-default.yml", "/config.yml", "/secrets.yml"]
        for config_file_name in config_file_names:
            found_config_path = [
                config_file_path
                for config_file_path in config_file_paths
                if config_file_path.endswith(config_file_name)
            ]
            if len(found_config_path) > 0:
                logger.debug("[get_config] Config file found: %s", found_config_path[0])
                with open(found_config_path[0]) as f:
                    config_from_file = yaml.safe_load(f)
                    if config_from_file is not None:
                        config = config | config_from_file
    return config