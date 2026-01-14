status_tag = "Liveness and Readiness Probe"
bot_config_file_tag = "LLM Bot Configuration - Knowledge files"

openapi_tags = [
    {
        "name": bot_config_file_tag,
        "description": "Files used to provide knowledge to the LLM Bot, such as FAQs",
        "notUsed": {},
    },
    {
        "name": status_tag,
        "externalDocs": {
            "description": "Configure Liveness, Readiness and Startup Probes",
            "url": "https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/",
        },
    },
]
