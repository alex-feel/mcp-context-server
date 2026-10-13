{{/*
Expand the name of the chart.
*/}}
{{- define "mcp-context-server.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "mcp-context-server.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "mcp-context-server.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "mcp-context-server.labels" -}}
helm.sh/chart: {{ include "mcp-context-server.chart" . }}
{{ include "mcp-context-server.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "mcp-context-server.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mcp-context-server.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "mcp-context-server.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "mcp-context-server.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the image reference
*/}}
{{- define "mcp-context-server.image" -}}
{{- $tag := default .Chart.AppVersion .Values.image.tag }}
{{- printf "%s:%s" .Values.image.repository $tag }}
{{- end }}

{{/*
Create Ollama image reference
*/}}
{{- define "mcp-context-server.ollamaImage" -}}
{{- printf "%s:%s" .Values.ollama.image.repository .Values.ollama.image.tag }}
{{- end }}

{{/*
Refuse values the server would refuse at startup, mirroring its settings validators:
row-level security is a PostgreSQL feature, and it filters only a role that does not own
the tables, so it needs the migration role that does.
*/}}
{{- define "mcp-context-server.validateValues" -}}
{{- if .Values.storage.postgresql.rls.enabled }}
{{- if ne .Values.storage.backend "postgresql" }}
{{- fail "storage.postgresql.rls.enabled needs storage.backend=postgresql: ENABLE_POSTGRESQL_RLS requires STORAGE_BACKEND=postgresql" }}
{{- end }}
{{- if not .Values.storage.postgresql.migration.user }}
{{- fail "storage.postgresql.rls.enabled needs a separate migration role that owns the tables: set storage.postgresql.migration.user and its password (POSTGRESQL_MIGRATION_USER and POSTGRESQL_MIGRATION_PASSWORD)" }}
{{- end }}
{{- end }}
{{- end }}
