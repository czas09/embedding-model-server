import base64

import numpy as np
from fastapi import APIRouter
from loguru import logger

from models import EMBEDDING_MODEL
from config import (
    EMBED_MODEL_NAME, 
    EMBED_MODEL_PATH, 
    EMBED_SIZE, 
)
from protocol import (
    ModelCard,
    ModelList,
    # ModelPermission,
    UsageInfo, 
    EmbeddingsResponse, 
    EmbeddingsRequest, 
)


logger.add("./service.log", level='INFO')

model_router = APIRouter()
embedding_router = APIRouter()

@model_router.get("/models")
async def show_available_models() -> ModelList: 
    logger.info("当前模型服务：")
    logger.info("    {}".format(EMBED_MODEL_NAME))
    return ModelList(data=[ModelCard(id=EMBED_MODEL_NAME, root=EMBED_MODEL_NAME)])


@embedding_router.post("/embeddings")
async def create_embeddings(request: EmbeddingsRequest, model_name: str = None) -> EmbeddingsResponse:  
    """Creates embeddings for the text"""
    if request.model is None:
        request.model = model_name

    inputs = request.input
    if isinstance(inputs, str):
        inputs = [inputs]
    
    data, token_num = [], 0
    batches = [
        inputs[i: min(i + 1024, len(inputs))]
        for i in range(0, len(inputs), 1024)
    ]

    for num_batch, batch in enumerate(batches):
        token_num = sum([len(i) for i in batch])

        logger.info("当前处理批次中的文本：")
        for text in batch: 
            logger.info("    {}".format(text))
        embeds = EMBEDDING_MODEL.encode(batch, normalize_embeddings=True)

        batch_size, embed_dim = embeds.shape
        if EMBED_SIZE is not None and EMBED_SIZE > embed_dim:
            zeros = np.zeros((batch_size, EMBED_SIZE - embed_dim))
            embeds = np.c_[embeds, zeros]

        if request.encoding_format == "base64":
            embeds = [base64.b64encode(embed.tobytes()).decode("utf-8") for embed in embeds]
        else:
            embeds = embeds.tolist()

        data += [
            {
                "object": "embedding",
                "embedding": embed,
                "index": num_batch * 1024 + i,
            }
            for i, embed in enumerate(embeds)
        ]
        token_num += token_num

    return EmbeddingsResponse(
        data=data,
        model=request.model,
        usage=UsageInfo(
            prompt_tokens=token_num,
            total_tokens=token_num,
            completion_tokens=None,
        ),
    ).dict(exclude_none=True)