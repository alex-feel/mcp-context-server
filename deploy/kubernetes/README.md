# Raw Kubernetes Manifests

This directory is a placeholder for raw Kubernetes manifests for users who prefer `kubectl apply` over Helm.

## Current Status

Raw Kubernetes manifests are not yet available. For Kubernetes deployments, please use the Helm chart:

```bash
helm install mcp ./deploy/helm/mcp-context-server
```

## Generating Manifests from Helm

If you need raw manifests, you can generate them from the Helm chart:

```bash
# Generate manifests with default values
helm template mcp ./deploy/helm/mcp-context-server > manifests.yaml

# Generate with custom values
helm template mcp ./deploy/helm/mcp-context-server \
  -f ./deploy/helm/mcp-context-server/values-postgresql.yaml \
  --set storage.postgresql.password=your-password \
  > manifests.yaml

# Apply generated manifests
kubectl apply -f manifests.yaml
```

## Alternative: Docker Compose

For simpler deployments without Kubernetes, see the Docker Compose configurations:

- [Docker Deployment Guide](../../docs/deployment/docker.md)
- Configuration files in `deploy/docker/`

## Contributing

If you would like to contribute raw Kubernetes manifests, please open a pull request.
