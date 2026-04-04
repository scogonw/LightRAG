# Scogo x LightRAG — active `.env` configuration

This document lists **uncommented (active)** variables from the project `.env` and the **purpose** described in that file’s comments. Secret values (API keys, passwords) are not reproduced here; set them only in `.env`.

---

## Server configuration

| Variable | Value (non-secret) | Purpose (from `.env`) |
|----------|-------------------|------------------------|
| `HOST` | `0.0.0.0` | Server bind address. |
| `PORT` | `9621` | Server port. |
| `WEBUI_TITLE` | `SIA` | Web UI title. |
| `WEBUI_DESCRIPTION` | Simple and Fast Graph Based RAG System | Web UI description. |
| `WORKERS` | `8` | Worker count (Gunicorn / server). |
| `TIMEOUT` | `600` | Gunicorn worker timeout; also acts as default LLM request timeout when `LLM_TIMEOUT` is not set. |


## Logging

| Variable | Value | Purpose (from `.env`) |
|----------|-------|------------------------|
| `LOG_LEVEL` | `INFO` | Logging level. |

---

## Reranking

| Variable | Value (non-secret) | Purpose (from `.env`) |
|----------|-------------------|------------------------|
| `RERANK_BINDING` | `cohere` | Rerank backend: `null`, `cohere`, `jina`, `aliyun`; use `cohere` binding for vLLM-deployed rerank models. |
| `RERANK_MODEL` | `cohere-rerank-v4.0-fast` | Rerank model name (Cohere AI section). |
| `RERANK_BINDING_HOST` | *(Azure Cohere endpoint)* | Rerank API URL; in Docker use `host.docker.internal` instead of `localhost` when applicable. |
| `RERANK_BINDING_API_KEY` | *(secret)* | Rerank API key. |
| `RERANK_ENABLE_CHUNKING` | `true` | Cohere rerank chunking; useful for models with token limits (e.g. ColBERT). |

---

## Document processing

| Variable | Value | Purpose (from `.env`) |
|----------|-------|------------------------|
| `ENABLE_LLM_CACHE_FOR_EXTRACT` | `true` | Enables LLM cache during extraction / document processing. |
| `SUMMARY_LANGUAGE` | `English` | Language for document processing output (e.g. English, Chinese, French, German). |
| `CHUNK_SIZE` | `1200` | Chunk size for document splitting; 500–1500 recommended. |

---

## Concurrency

| Variable | Value | Purpose (from `.env`) |
|----------|-------|------------------------|
| `MAX_ASYNC` | `16` | Max concurrent LLM requests (query and document processing). |
| `MAX_PARALLEL_INSERT` | `8` | Parallel documents being processed; between 2–10; `MAX_ASYNC/3` recommended. |
| `EMBEDDING_FUNC_MAX_ASYNC` | `16` | Max concurrent embedding requests. |
| `EMBEDDING_BATCH_NUM` | `5` | Number of chunks sent to embedding in a single request. |

---

## LLM (Azure OpenAI)

| Variable | Value (non-secret) | Purpose (from `.env`) |
|----------|-------------------|------------------------|
| `LLM_TIMEOUT` | `600` | LLM request timeout for all LLMs (`0` = no timeout for Ollama). |
| `LLM_BINDING` | `azure_openai` | Provider type: `openai`, `ollama`, `lollms`, `azure_openai`, `aws_bedrock`, `gemini`. |
| `LLM_BINDING_HOST` | *(Azure endpoint)* | Service endpoint; in Docker use `host.docker.internal` instead of `localhost` when applicable. |
| `LLM_BINDING_API_KEY` | *(secret)* | API key for the LLM service. |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4.1-nano` | Azure deployment name (or use as model name per Azure OpenAI example comments). |

---

## Embedding (Azure OpenAI)

| Variable | Value (non-secret) | Purpose (from `.env`) |
|----------|-------------------|------------------------|
| `EMBEDDING_TIMEOUT` | `180` | Embedding request timeout. |
| `EMBEDDING_BINDING` | `azure_openai` | Embedding provider; **should not be changed after the first file is processed**. |
| `EMBEDDING_BINDING_HOST` | *(Azure endpoint)* | Embedding service endpoint; Docker note same as LLM. |
| `EMBEDDING_BINDING_API_KEY` | *(secret)* | Embedding API key. |
| `AZURE_EMBEDDING_DEPLOYMENT` | `text-embedding-3-small` | Azure embedding deployment (or use deployment name as model name). |
| `EMBEDDING_DIM` | `1536` | Embedding vector dimension (must stay consistent with the chosen model). |

---

## Langfuse (observability)

| Variable | Value (non-secret) | Purpose (from `.env` / usage) |
|----------|-------------------|-------------------------------|
| `LANGFUSE_SECRET_KEY` | *(secret)* | Langfuse secret API key (server-side ingestion). |
| `LANGFUSE_PUBLIC_KEY` | *(set in `.env`)* | Langfuse public key (client/SDK identification). |
| `LANGFUSE_HOST` | *(your Langfuse base URL)* | Langfuse API host (e.g. `https://cloud.langfuse.com` or your self-hosted origin). |
| `LANGFUSE_ENABLE_TRACE` | `true` | Turn Langfuse tracing on when set to `true`. |

---

## LightRAG storage selection

| Variable | Value | Purpose (from `.env`) |
|----------|-------|------------------------|
| `LIGHTRAG_KV_STORAGE` | `JsonKVStorage` | KV storage; default JSON (recommended for test deployment). |
| `LIGHTRAG_DOC_STATUS_STORAGE` | `JsonDocStatusStorage` | Document status storage. |
| `LIGHTRAG_GRAPH_STORAGE` | `NetworkXStorage` | Graph storage. |
| `LIGHTRAG_VECTOR_STORAGE` | `NanoVectorDBStorage` | Vector storage. |

---

## PostgreSQL

| Variable | Value (non-secret) | Purpose (from `.env`) |
|----------|-------------------|------------------------|
| `POSTGRES_HOST` | `localhost` | PostgreSQL host. |
| `POSTGRES_PORT` | `5432` | PostgreSQL port. |
| `POSTGRES_USER` | *(configured)* | Database user. |
| `POSTGRES_PASSWORD` | *(secret)* | Database password. |
| `POSTGRES_DATABASE` | `rag` | Database name. |
| `POSTGRES_MAX_CONNECTIONS` | `25` | Max pool connections. |
| `POSTGRES_ENABLE_VECTOR` | `true` | Enable pgvector / vector ops; set `false` if using PostgreSQL only for KV/graph/doc-status with another vector backend. |
| `POSTGRES_VECTOR_INDEX_TYPE` | `HNSW` | Vector index: `HNSW`, `HNSW_HALFVEC` (2000+ dim, pgvector ≥ 0.7), `IVFFlat`, `VCHORDRQ`. |
| `POSTGRES_HNSW_M` | `16` | HNSW *M* parameter. |
| `POSTGRES_HNSW_EF` | `200` | HNSW *ef* parameter. |
| `POSTGRES_IVFFLAT_LISTS` | `100` | IVFFlat lists. |
| `POSTGRES_VCHORDRQ_BUILD_OPTIONS` | *(empty)* | VCHORDRQ build options. |
| `POSTGRES_VCHORDRQ_PROBES` | *(empty)* | VCHORDRQ probes. |
| `POSTGRES_VCHORDRQ_EPSILON` | `1.9` | VCHORDRQ epsilon. |

---

## Neo4j

| Variable | Value (non-secret) | Purpose (from `.env`) |
|----------|-------------------|------------------------|
| `NEO4J_URI` | *(Neo4j Aura-style URI)* | Neo4j connection URI. |
| `NEO4J_USERNAME` | `neo4j` | Neo4j user. |
| `NEO4J_PASSWORD` | *(secret)* | Neo4j password. |
| `NEO4J_DATABASE` | `neo4j` | Database name. |
| `NEO4J_MAX_CONNECTION_POOL_SIZE` | `100` | Connection pool size. |
| `NEO4J_CONNECTION_TIMEOUT` | `30` | Connection timeout (seconds). |
| `NEO4J_CONNECTION_ACQUISITION_TIMEOUT` | `30` | Time to acquire connection from pool. |
| `NEO4J_MAX_TRANSACTION_RETRY_TIME` | `30` | Max transaction retry time. |
| `NEO4J_MAX_CONNECTION_LIFETIME` | `300` | Max connection lifetime (seconds). |
| `NEO4J_LIVENESS_CHECK_TIMEOUT` | `30` | Liveness check timeout. |
| `NEO4J_KEEP_ALIVE` | `true` | TCP keep-alive. |


## OpenSearch

| Variable | Value (non-secret) | Purpose (from `.env`) |
|----------|-------------------|------------------------|
| `OPENSEARCH_HOSTS` | `localhost:9200` | Comma-separated `host:port` (no `http://` or `https://`). |
| `OPENSEARCH_USER` | `admin` | Cluster user (wizard: authenticated clusters only). |
| `OPENSEARCH_PASSWORD` | *(secret)* | Cluster password. |
| `OPENSEARCH_USE_SSL` | `true` | Use TLS to reach hosts. |
| `OPENSEARCH_VERIFY_CERTS` | `false` | Whether to verify TLS certificates. |


## Redis

| Variable | Value | Purpose (from `.env`) |
|----------|-------|------------------------|
| `REDIS_URI` | `redis://localhost:6379` | Redis connection URI. |
| `REDIS_SOCKET_TIMEOUT` | `30` | Socket timeout (seconds). |
| `REDIS_CONNECT_TIMEOUT` | `10` | Connect timeout (seconds). |
| `REDIS_MAX_CONNECTIONS` | `100` | Max connections. |
| `REDIS_RETRY_ATTEMPTS` | `3` | Retry attempts. |

---

## Notes

- **Source of truth**: Values and any additions live in `.env`; regenerate this doc if you change which lines are active.
- **Security**: Never commit real API keys or database passwords to git; rotate any keys that appeared in a shared or committed `.env`.
