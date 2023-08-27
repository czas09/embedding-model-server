# embedding-model-server
参考 OpenAI API 格式以及开源框架 xusenlinzy/api-for-open-llm 的文本 embedding 模型服务接口

调用方法：
```bash
curl http://localhost:10270/v1/embeddings \
  -H "Authorization: Bearer xxx" \
  -H "Content-Type: application/json" \
  -d '{"input": "你好", "model": "m3e-base-20230608"}'
```

响应体结构：
```json
{
  "object": "list", 
  "data": [
    {
      "object": "embedding", 
      "embedding": [
        0.028495032340288162, 
        0.02246084064245224, 
        ... (768 floats total for m3e-base)
        -0.012482207268476486, 
      ], 
      "index": 0
    }
  ], 
  "model": "m3e-base-20230608", 
  "usage": {
    "prompt_tokens": 4, 
    "total_tokens": 4
  }
}
```
