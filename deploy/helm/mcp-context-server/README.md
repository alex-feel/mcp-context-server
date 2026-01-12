# MCP Context Server Helm Chart

A Helm chart for deploying MCP Context Server on Kubernetes.

## Prerequisites

- Kubernetes 1.21+
- Helm 3.8+
- PV provisioner support (for SQLite persistence)

## Installation

### Quick Start (SQLite)

```bash
helm install mcp ./deploy/helm/mcp-context-server
```

### SQLite with Persistence

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  -f ./deploy/helm/mcp-context-server/values-sqlite.yaml
```

### PostgreSQL

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  -f ./deploy/helm/mcp-context-server/values-postgresql.yaml \
  --set storage.postgresql.host=your-postgres-host \
  --set storage.postgresql.password=your-password
```

### With Semantic Search (Ollama Sidecar)

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  --set search.semantic.enabled=true \
  --set ollama.enabled=true
```

## Configuration

### Key Values

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | Image repository | `ghcr.io/alex-feel/mcp-context-server` |
| `image.tag` | Image tag | `Chart.appVersion` |
| `replicaCount` | Number of replicas | `1` |
| `service.type` | Kubernetes service type | `ClusterIP` |
| `service.port` | Service port | `8000` |
| `storage.backend` | Storage backend (sqlite/postgresql) | `sqlite` |
| `search.fts.enabled` | Enable full-text search | `true` |
| `search.semantic.enabled` | Enable semantic search | `false` |
| `search.hybrid.enabled` | Enable hybrid search | `false` |
| `ollama.enabled` | Enable Ollama sidecar | `false` |

### Storage Configuration

#### SQLite

```yaml
storage:
  backend: sqlite
  sqlite:
    enabled: true
    path: /data/context_storage.db
    persistence:
      enabled: true
      size: 1Gi
      storageClassName: ""
```

#### PostgreSQL

```yaml
storage:
  backend: postgresql
  postgresql:
    enabled: true
    host: "postgresql-host"
    port: "5432"
    user: "postgres"
    password: "your-password"
    database: "mcp_context"
    sslMode: "prefer"
```

For production, use an existing secret:

```yaml
storage:
  postgresql:
    existingSecret: "my-postgres-secret"
    existingSecretKey: "password"
```

### Search Features

Enable all search features:

```yaml
search:
  fts:
    enabled: true
    language: "english"
  semantic:
    enabled: true
    model: "qwen3-embedding:0.6b"
    dim: 1024
  hybrid:
    enabled: true
    rrfK: 60

ollama:
  enabled: true
```

### Ingress

Enable ingress with TLS:

```yaml
ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: mcp.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: mcp-tls
      hosts:
        - mcp.example.com
```

## Upgrading

```bash
helm upgrade mcp ./deploy/helm/mcp-context-server
```

## Uninstalling

```bash
helm uninstall mcp
```

Note: PersistentVolumeClaims are not deleted automatically. To remove data:

```bash
kubectl delete pvc -l app.kubernetes.io/instance=mcp
```

## Resources

- [MCP Context Server Documentation](https://github.com/alex-feel/mcp-context-server)
- [Helm Documentation](https://helm.sh/docs/)
