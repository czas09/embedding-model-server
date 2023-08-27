import sys
sys.path.insert(0, ".")

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from routes import model_router, embedding_router
from config import SERVICE_HOST, SERVICE_PORT, API_PREFIX


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_router, prefix=API_PREFIX, tags=["Model"])
app.include_router(embedding_router, prefix=API_PREFIX, tags=["Embedding"])



if __name__ == '__main__': 
    import uvicorn
    
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT, log_level='info')