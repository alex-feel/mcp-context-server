# Helm Deployment Guide

This guide covers deploying the MCP Context Server using the official Helm chart.

## Prerequisites

- Kubernetes 1.21+
- Helm 3.8+
- PV provisioner support (for SQLite persistence)
- Optional: PostgreSQL database for production deployments

## Installation

### Add Repository (Future)

```bash
# Once published to a Helm repository
helm repo add mcp-context https://alex-feel.github.io/mcp-context-server
helm repo update
```

### Install from Local Chart

```bash
# Clone the repository
git clone https://github.com/alex-feel/mcp-context-server.git
cd mcp-context-server

# Install with default values (SQLite)
helm install mcp ./deploy/helm/mcp-context-server

# Install with custom values
helm install mcp ./deploy/helm/mcp-context-server -f my-values.yaml
```

## Configuration Profiles

### SQLite (Development)

Best for single-user deployments and development:

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  -f ./deploy/helm/mcp-context-server/values-sqlite.yaml
```

### PostgreSQL (Production)

Best for multi-user and production deployments:

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  -f ./deploy/helm/mcp-context-server/values-postgresql.yaml \
  --set storage.postgresql.host=your-postgres-host \
  --set storage.postgresql.password=your-password
```

### With Semantic Search

Enable AI-powered semantic search with Ollama sidecar:

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  --set search.semantic.enabled=true \
  --set ollama.enabled=true
```

### Full-Featured Production

All features enabled with PostgreSQL:

```bash
helm install mcp ./deploy/helm/mcp-context-server \
  --set storage.backend=postgresql \
  --set storage.postgresql.enabled=true \
  --set storage.postgresql.host=postgres.example.com \
  --set storage.postgresql.password=secure-password \
  --set search.fts.enabled=true \
  --set search.semantic.enabled=true \
  --set search.hybrid.enabled=true \
  --set ollama.enabled=true \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=mcp.example.com
```

## Values Reference

### Image Configuration

```yaml
image:
  repository: ghcr.io/alex-feel/mcp-context-server
  tag: ""  # Defaults to Chart.appVersion
  pullPolicy: IfNotPresent

imagePullSecrets: []
```

### Service Configuration

```yaml
service:
  type: ClusterIP  # or LoadBalancer, NodePort
  port: 8000
```

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
      storageClassName: ""  # Use default
      accessModes:
        - ReadWriteOnce
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
    password: ""  # Set via --set or secret
    database: "mcp_context"
    sslMode: "prefer"
    # Use existing secret instead of password
    existingSecret: ""
    existingSecretKey: "postgresql-password"
```

### Search Configuration

```yaml
search:
  fts:
    enabled: true
    language: "english"
  semantic:
    enabled: false
    model: "embeddinggemma:latest"
    dim: 768
  hybrid:
    enabled: false
    rrfK: 60
```

### Ollama Sidecar

```yaml
ollama:
  enabled: false
  image:
    repository: ollama/ollama
    tag: "latest"
    pullPolicy: IfNotPresent
  resources:
    requests:
      cpu: "500m"
      memory: "2Gi"
    limits:
      cpu: "2000m"
      memory: "4Gi"
  persistence:
    enabled: true
    size: 5Gi
    storageClassName: ""
```

### Ingress Configuration

```yaml
ingress:
  enabled: false
  className: ""
  annotations: {}
  hosts:
    - host: mcp-context-server.local
      paths:
        - path: /
          pathType: Prefix
  tls: []
```

### Resource Limits

```yaml
resources:
  requests:
    cpu: "100m"
    memory: "256Mi"
  limits:
    cpu: "500m"
    memory: "512Mi"
```

### Health Probes

```yaml
probes:
  liveness:
    enabled: true
    path: /health
    initialDelaySeconds: 30
    periodSeconds: 30
    failureThreshold: 3
  readiness:
    enabled: true
    path: /health
    initialDelaySeconds: 10
    periodSeconds: 10
    failureThreshold: 3
  startup:
    enabled: true
    path: /health
    initialDelaySeconds: 5
    periodSeconds: 10
    failureThreshold: 30
```

### Security Context

```yaml
podSecurityContext:
  runAsNonRoot: true
  runAsUser: 10001
  runAsGroup: 10001
  fsGroup: 10001

securityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: false
  capabilities:
    drop:
      - ALL
```

### Service Account

```yaml
serviceAccount:
  create: true
  name: ""
  annotations: {}
```

## Common Operations

### Upgrade

```bash
helm upgrade mcp ./deploy/helm/mcp-context-server
```

### Rollback

```bash
helm rollback mcp 1
```

### Uninstall

```bash
helm uninstall mcp
```

**Note:** PersistentVolumeClaims are not deleted automatically. To remove data:

```bash
kubectl delete pvc -l app.kubernetes.io/instance=mcp
```

### View Rendered Templates

```bash
helm template mcp ./deploy/helm/mcp-context-server
```

### Debug Installation

```bash
helm install mcp ./deploy/helm/mcp-context-server --debug --dry-run
```

## Examples

### External PostgreSQL with Existing Secret

```yaml
# values-production.yaml
storage:
  backend: postgresql
  sqlite:
    enabled: false
  postgresql:
    enabled: true
    host: "postgres.production.svc.cluster.local"
    port: "5432"
    user: "mcp_user"
    database: "mcp_context"
    sslMode: "require"
    existingSecret: "postgres-credentials"
    existingSecretKey: "password"

search:
  fts:
    enabled: true
  semantic:
    enabled: true
  hybrid:
    enabled: true

ollama:
  enabled: true
  resources:
    requests:
      memory: "4Gi"
    limits:
      memory: "8Gi"

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
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

Install with:

```bash
helm install mcp ./deploy/helm/mcp-context-server -f values-production.yaml
```

### Development with Port Forward

```bash
# Install minimal setup
helm install mcp ./deploy/helm/mcp-context-server

# Port forward to local machine
kubectl port-forward svc/mcp 8000:8000

# Test connection
curl http://localhost:8000/health
```

## Troubleshooting

### Pod Stuck in Pending

Check PersistentVolumeClaim:
```bash
kubectl get pvc
kubectl describe pvc mcp-data
```

### Ollama Out of Memory

Increase resources:
```yaml
ollama:
  resources:
    limits:
      memory: "8Gi"
```

### Health Check Failing

Check logs:
```bash
kubectl logs -l app.kubernetes.io/name=mcp-context-server
```

## Additional Resources

### Related Documentation

- [Kubernetes Deployment Guide](kubernetes.md) - General Kubernetes deployment
- [Docker Deployment Guide](docker.md) - Alternative Docker Compose deployment
- [Database Backends](../database-backends.md) - Database configuration details
- [Semantic Search](../semantic-search.md) - Ollama and embedding configuration
