def default_configs() -> dict:
    return {
        "environment": "local",
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
        },
        "opensearch": {
            "url": "http://localhost:9200",
            "index_name": "venue_event_history",
        },
        "llm_config": {
            "model": "gpt-4o-mini",
        },
        "logger": {
            "level": "INFO",
            "enable_structured_logging": True,
            "service_name": "venue-recommendation-service",
        },
    }


class Configuration:
    def __init__(self, configs: dict):
        self.server = Server(configs["server"])
        self.environment = configs["environment"]
        self.opensearch = OpenSearch(configs["opensearch"])
        self.llm_config = LLMConfig(configs["llm_config"])
        self.logger = LoggerConfig(configs["logger"])


class Server:
    def __init__(self, configs: dict):
        self.host = configs["host"]
        self.port = configs["port"]

    def get_server_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class OpenSearch:
    def __init__(self, configs: dict) -> None:
        self.url = configs["url"]
        self.index_name = configs["index_name"]


class LLMConfig:
    def __init__(self, configs: dict):
        self.model = configs["model"]


class LoggerConfig:
    def __init__(self, configs: dict):
        self.level = configs["level"].upper()
        self.enable_structured_logging = configs.get(
            "enable_structured_logging", "True"
        )
        self.service_name = configs["service_name"]


config = Configuration(default_configs())
