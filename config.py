import configparser


configs = configparser.ConfigParser()
configs.read("./configs.ini", encoding='utf-8')

# ==============================================================================
# 服务配置选项
# ==============================================================================
SERVICE_HOST = configs.get("SERVICE", "host")
SERVICE_PORT = configs.getint("SERVICE", "port")
API_PREFIX = configs.get("SERVICE", "prefix")
EMBEDDING_ROUTE = configs.get("SERVICE", "embedding_route")

# ==============================================================================
# 模型配置选项
# ==============================================================================
EMBED_MODEL_NAME = configs.get("EMBED_MODEL", "embed_model_name")
EMBED_MODEL_PATH = configs.get("EMBED_MODEL", "embed_model_path")
EMBED_SIZE = configs.getint("EMBED_MODEL", "embed_size")
BATCH_SIZE = configs.getint("EMBED_MODEL", "batch_size")
DEVICE = configs.get("EMBED_MODEL", "device")
GPU_ID = configs.get("EMBED_MODEL", "gpu_id")
# NUM_GPUS = configs.get("EMBED_MODEL", "num_gpus")