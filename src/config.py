import logging
import ecs_logging
from colorlog import ColoredFormatter

from venue_rec.utils import parse_config_files

# TODO: use enum
DEFAULT_VECTORSTORE_TYPE = "opensearch"


class Configuration:
    def __init__(self):
        configs = parse_config_files()
        self.server = Server(configs["server"])

        self.opensearch = OpenSearch(configs["opensearch"])
        self.llm_config = LLMConfig(configs["llm_config"])
        self.logger = LoggerConfig(configs["logger"])


class Server:
    def __init__(self, configs: dict):
        self.host = configs["host"]
        self.port = configs["port"]


class OpenSearch:
    def __init__(self, configs: dict) -> None:
        self.url = configs["url"]
        self.index_name = configs["index_name"]


class LLMConfig:
    def __init__(self, configs: dict):
        self.url = configs["url"]
        self.azure_endpoint = configs["azure_endpoint"]
        self.azure_deployments_endpoint = configs["azure_deployments_endpoint"]
        self.azure_api_version = configs["azure_api_version"]
        self.openai_endpoint = configs["openai_endpoint"]
        self.identity_name = configs["identity_name"]


class LoggerConfig:
    def __init__(self, configs: dict):
        self.level = configs["level"].upper()
        self.enable_structured_logging = configs.get(
            "enable_structured_logging", "True"
        )


config = Configuration()