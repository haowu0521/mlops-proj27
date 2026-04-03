# Infrastructure Requirements Table

## Cluster Overview

- **Site**: KVM@TACC
- **Node**: proj27-k8s-node
- **Flavor**: m1.large (4 vCPUs, 8GB RAM, 40GB disk)
- **OS**: CC-Ubuntu24.04
- **Kubernetes**: k3s v1.34.6+k3s1

## Node-Level Resource Usage (Observed)

| Metric | Value |
|--------|-------|
| CPU Usage | 173m (4%) |
| Memory Usage | 1838Mi (23%) |

## Per-Service Resource Usage and Limits

| Service | Namespace | CPU Observed | Memory Observed | CPU Request | CPU Limit | Memory Request | Memory Limit | Notes |
|---------|-----------|-------------|----------------|-------------|-----------|----------------|--------------|-------|
| prosody | jitsi | 2m | 32Mi | 50m | 200m | 64Mi | 256Mi | XMPP server, lightweight |
| jicofo | jitsi | 4m | 170Mi | 50m | 300m | 200Mi | 400Mi | Conference manager, moderate memory |
| jvb | jitsi | 5m | 201Mi | 50m | 500m | 256Mi | 512Mi | Video bridge, usage increases under load |
| web | jitsi | 0m | 22Mi | 10m | 200m | 32Mi | 128Mi | Nginx frontend, very lightweight |
| mlflow | platform | 1m | 523Mi | 50m | 500m | 512Mi | 1Gi | Tracking server, high baseline memory |

## Right-Sizing Methodology

- Observed values were collected using `kubectl top pods -A` and `kubectl top nodes` on Chameleon under idle conditions (no active meetings or training runs).
- **CPU Requests** are set above observed usage to accommodate normal variance.
- **CPU Limits** are set with headroom for load spikes (e.g., JVB during active video conferences).
- **Memory Requests** are set close to observed usage to ensure scheduling.
- **Memory Limits** include buffer for peak usage.
- JVB resource usage is expected to increase significantly during active video conferences with multiple participants.
- MLflow memory usage may grow as more experiment data is stored.