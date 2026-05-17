import os

# 默认指向本地 vLLM (OpenAI 兼容)；可用环境变量覆盖到任意 OpenAI 兼容端点。
_DEFAULT_BASE_URL = os.environ.get("MINJA_BASE_URL", "http://localhost:8000/v1")
_DEFAULT_API_KEY = os.environ.get("MINJA_API_KEY", "EMPTY")
_DEFAULT_MODEL = os.environ.get("MINJA_MODEL", "qwen2.5-72b")


def openai_config(model):
    # `model` 来自 CLI --llm；保留它以兼容历史命令行，但实际请求都打到本地 vLLM。
    # 如果用户显式传入了一个本地 served-model-name，就用它，否则用 MINJA_MODEL 环境变量。
    served_name = model if model and not model.startswith("gpt-") and model != "o1-preview" else _DEFAULT_MODEL
    return {
        "model": served_name,
        "api_key": _DEFAULT_API_KEY,
        "base_url": _DEFAULT_BASE_URL,
        "api_type": "openai",   # autogen 0.2.x 通过此字段路由到 OpenAI 兼容 client
        "price": [0.0, 0.0],    # 本地推理，跳过 autogen 的成本警告
    }

def llm_config_list(seed, config_list):
    llm_config_list = {
        "functions": [
            {
                "name": "python",
                "description": "run the entire code and return the execution result. Only generate the code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell": {
                            "type": "string",
                            "description": "Valid Python code to execute.",
                        }
                    },
                    "required": ["cell"],
                },
            },
        ],
        "config_list": config_list,
        "timeout": 120,
        "cache_seed": seed,
        "temperature": 0,
    }
    return llm_config_list