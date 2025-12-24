# Eval-Gated RAG Platform

Production-ready RAG (Retrieval-Augmented Generation) system with evaluation-driven CI/CD, gold-set testing, and automated quality gates. Self-hosted on Kubernetes.

**Case Study:** [yrgenkuci.com/case-studies/eval-gated-rag-platform](https://yrgenkuci.com/case-studies/eval-gated-rag-platform)

## Features

- **Evaluation-First Development**: Every change tested against gold-set before merge
- **Automated Quality Gates**: CI blocks changes that drop accuracy below 85% SLO
- **Self-Hosted**: Complete stack runs on Kubernetes (no external API dependencies)
- **Observability**: Prometheus metrics + Grafana dashboards + alerting rules
- **Cost Efficient**: ~93% cost reduction vs OpenAI API at scale

## Architecture

```
                    +------------------+
                    |   FastAPI API    |
                    |  /query /ingest  |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+         +---------v---------+
    |  Semantic Search  |         |   LLM Generation  |
    |    (Retriever)    |         |  (Ollama/vLLM)    |
    +---------+---------+         +-------------------+
              |
    +---------v---------+
    |   Vector Store    |
    |     (Qdrant)      |
    +-------------------+
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| API | FastAPI | REST endpoints |
| Vector DB | Qdrant | Semantic search |
| Embeddings | BGE-large (configurable) | Text vectorization |
| LLM (dev) | Ollama | Local inference |
| LLM (prod) | vLLM | GPU inference |
| Evaluation | ROUGE, BLEU, Semantic | Quality metrics |
| CI | GitHub Actions | Testing + eval gate |
| CD | ArgoCD | GitOps deployment |
| Orchestration | Kubernetes + Helm | Infrastructure |
| Observability | Prometheus + Grafana | Monitoring |

## Quick Start

### Prerequisites

- Python 3.12+
- Poetry
- Docker (for local services)
- kubectl + Helm (for Kubernetes deployment)

### Local Development

```bash
# Clone repository
git clone https://github.com/yrgenkuci/eval-gated-rag-platform.git
cd eval-gated-rag-platform

# Install dependencies
poetry install

# Run tests
poetry run pytest tests/ -v

# Start API server
poetry run uvicorn src.api.app:app --reload
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/health/ready` | GET | Kubernetes readiness probe |
| `/health/live` | GET | Kubernetes liveness probe |
| `/metrics` | GET | Prometheus metrics |
| `/api/v1/query` | POST | Query the RAG system |
| `/api/v1/ingest` | POST | Ingest documents |

### Query Example

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our data retention policy?"}'
```

## Evaluation Harness

### Gold-Set Format

Create test cases in `data/gold/`:

```json
{
  "name": "my_gold_set",
  "version": "1.0.0",
  "items": [
    {
      "id": "q1",
      "question": "What is our data retention policy?",
      "expected_answer": "Data is retained for 90 days for logs and 7 years for financial records.",
      "context": ["docs/policies/data-retention.md"],
      "metadata": {"category": "compliance"}
    }
  ]
}
```

### Run Evaluation

```bash
# Run evaluation against gold-set
python -m scripts.run_eval \
  --gold-set data/gold/test.json \
  --threshold 0.85 \
  --output results.json
```

### Metrics

- **ROUGE-L**: Longest common subsequence F1 score
- **BLEU**: N-gram precision with brevity penalty
- **Semantic Similarity**: Jaccard word overlap (production: embedding cosine)

## CI/CD Integration

### GitHub Actions Workflows

**`.github/workflows/ci.yml`** - Runs on every push:
- Linting (ruff)
- Type checking (mypy)
- Unit tests (pytest)
- Coverage reporting

**`.github/workflows/eval-gate.yml`** - Runs on PRs:
- Loads gold-set
- Runs evaluation suite
- Blocks merge if accuracy < 85%

### SLO Configuration

Evaluation thresholds in workflow:

```yaml
env:
  EVAL_THRESHOLD: "0.85"  # 85% pass rate required
  EVAL_GOLD_SET: "data/gold/test.json"
```

### ArgoCD GitOps

ArgoCD provides continuous delivery with automatic sync from Git:

```
GitHub Actions (CI)          ArgoCD (CD)
      │                           │
      ▼                           ▼
  Tests pass ──► Merge ──► Detect change ──► Deploy to K8s
      │                           │
      ▼                           ▼
  Eval gate                  Auto-sync
```

**Install ArgoCD:**

```bash
# Install ArgoCD in cluster
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Wait for ArgoCD to be ready
kubectl wait --for=condition=available deployment/argocd-server -n argocd --timeout=300s

# Get initial admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

# Access ArgoCD UI
kubectl port-forward svc/argocd-server -n argocd 8080:443
# Open https://localhost:8080 (admin / <password from above>)
```

**Deploy the Application:**

```bash
# Create the ArgoCD project (optional, for RBAC)
kubectl apply -f argocd/project.yaml

# Create the ArgoCD application
kubectl apply -f argocd/application.yaml

# Check sync status
kubectl get applications -n argocd
```

**ArgoCD Features:**

| Feature | Description |
|---------|-------------|
| Auto-sync | Deploys on every Git push to main |
| Self-heal | Reverts manual cluster changes |
| Prune | Removes deleted resources |
| Rollback | One-click rollback to previous revision |
| Health checks | Monitors deployment health |

## Kubernetes Deployment

### Helm Installation

```bash
# Development (with Ollama)
helm install rag ./helm/eval-gated-rag \
  --set global.environment=development \
  --set vllm.enabled=false \
  --set ollama.enabled=true

# Production (with vLLM + GPU)
helm install rag ./helm/eval-gated-rag \
  --set global.environment=production \
  --set vllm.enabled=true \
  --set ollama.enabled=false \
  --set ragApi.replicaCount=3
```

### Components Deployed

| Component | Type | Description |
|-----------|------|-------------|
| rag-api | Deployment | FastAPI application |
| qdrant | StatefulSet | Vector database |
| ollama/vllm | Deployment | LLM inference |
| embedding | Deployment | Text embeddings |
| prometheus | Deployment | Metrics collection |
| grafana | Deployment | Dashboards |

### Access Services

```bash
# API
kubectl port-forward svc/rag-eval-gated-rag-api 8000:8000

# Prometheus
kubectl port-forward svc/rag-eval-gated-rag-prometheus 9090:9090

# Grafana (admin/admin)
kubectl port-forward svc/rag-eval-gated-rag-grafana 3000:3000
```

## Observability

### Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `http_request_duration_seconds` | Histogram | Request latency |
| `http_requests_total` | Counter | Request count |
| `llm_tokens_total` | Counter | Token usage |
| `llm_request_duration_seconds` | Histogram | LLM latency |
| `retrieval_chunks_returned` | Histogram | Chunks per query |
| `retrieval_top_score` | Histogram | Relevance scores |
| `eval_pass_rate` | Gauge | Evaluation pass rate |

### Alerting Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| HighRequestLatency | p95 > 2s for 5m | warning |
| CriticalErrorRate | Error rate > 10% for 2m | critical |
| EvalPassRateDegraded | Pass rate < 85% for 10m | warning |
| RAGServiceDown | Service unavailable 1m | critical |

### Grafana Dashboard

Pre-configured dashboard includes:
- HTTP latency (p50/p95/p99)
- Request rate by endpoint
- LLM token usage
- Retrieval metrics
- Evaluation pass rate gauge
- Error rate monitoring

## Configuration

### Environment Variables

```bash
# Application
ENVIRONMENT=development  # development, staging, production
LOG_LEVEL=INFO
DEBUG=false

# LLM
LLM_BASE_URL=http://localhost:11434/v1  # Ollama default
LLM_MODEL=llama3:8b
LLM_TIMEOUT=120

# Embeddings
EMBEDDING_BASE_URL=http://localhost:8080/v1
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=documents
```

### Helm Values

See `helm/eval-gated-rag/values.yaml` for all configuration options.

## Project Structure

```
eval-gated-rag-platform/
├── src/
│   ├── api/            # FastAPI application
│   ├── documents/      # Document loading and chunking
│   ├── embeddings/     # Embedding service client
│   ├── evaluation/     # Gold-set and metrics
│   ├── llm/            # LLM client and prompts
│   ├── observability/  # Prometheus metrics
│   ├── rag/            # Pipeline orchestrator
│   ├── retrieval/      # Semantic search
│   └── vectorstore/    # Qdrant client
├── tests/              # Unit and integration tests
├── scripts/            # CLI tools
├── helm/               # Kubernetes Helm charts
├── argocd/             # ArgoCD GitOps manifests
├── data/gold/          # Evaluation gold-sets
└── .github/workflows/  # CI/CD pipelines
```

## Development

### Code Quality

```bash
# Lint
poetry run ruff check src/ tests/

# Type check
poetry run mypy src/ tests/

# Format
poetry run ruff format src/ tests/

# All tests
poetry run pytest tests/ -v --cov=src
```

### Adding New Metrics

1. Define metric in `src/observability/metrics.py`
2. Add tracking function
3. Call from relevant service
4. Add to Grafana dashboard

### Adding Evaluation Metrics

1. Implement `EvaluationMetric` interface in `src/evaluation/metrics.py`
2. Add to `EvaluationRunner` metrics list
3. Update gold-set schema if needed

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Ensure all tests pass and eval gate succeeds
4. Submit a pull request

## Author

**Yrgen Kuci** - [yrgenkuci.com](https://yrgenkuci.com)

---

*Reference implementation for evaluation-driven RAG development. See the [case study](https://yrgenkuci.com/case-studies/eval-gated-rag-platform) for detailed architecture decisions and results.*
