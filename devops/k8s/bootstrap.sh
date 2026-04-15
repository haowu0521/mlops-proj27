#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# bootstrap.sh — one-command cluster setup for Jitsi + MLflow on k3s
#
# Usage:
#   bash bootstrap.sh <FLOATING_IP> <JICOFO_PASSWORD> <JVB_PASSWORD>
#
# Example:
#   bash bootstrap.sh 129.114.27.109 myJicofoPass myJvbPass
# ---------------------------------------------------------------------------

FLOATING_IP="${1:?Error: FLOATING_IP is required (arg 1)}"
JICOFO_PASS="${2:?Error: JICOFO_PASSWORD is required (arg 2)}"
JVB_PASS="${3:?Error: JVB_PASSWORD is required (arg 3)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
K8S_DIR="$SCRIPT_DIR"
KUBECTL="sudo kubectl"

NIP_DOMAIN="${FLOATING_IP//./-}.nip.io"

echo "============================================"
echo " Bootstrap: Jitsi + MLflow on k3s"
echo " Floating IP : $FLOATING_IP"
echo " nip.io domain: $NIP_DOMAIN"
echo " Manifests dir: $K8S_DIR"
echo "============================================"

# ------------------------------------------------------------------
# 1. Install k3s (skip if already installed)
# ------------------------------------------------------------------
if ! command -v k3s &> /dev/null; then
    echo "[1/7] Installing k3s ..."
    curl -sfL https://get.k3s.io | sh -
    echo "Waiting for k3s node to register ..."
    until $KUBECTL get nodes 2>/dev/null | grep -q " Ready"; do
        sleep 3
    done
    echo "k3s node is Ready."
else
    echo "[1/7] k3s already installed, skipping."
fi

# ------------------------------------------------------------------
# 2. Create namespaces
# ------------------------------------------------------------------
echo "[2/7] Creating namespaces ..."
$KUBECTL create namespace jitsi    --dry-run=client -o yaml | $KUBECTL apply -f -
$KUBECTL create namespace platform --dry-run=client -o yaml | $KUBECTL apply -f -

# ------------------------------------------------------------------
# 3. Create secrets (values from arguments, never stored in Git)
# ------------------------------------------------------------------
echo "[3/7] Creating secrets ..."
$KUBECTL create secret generic jitsi-secrets \
    --from-literal=JICOFO_AUTH_PASSWORD="$JICOFO_PASS" \
    --from-literal=JVB_AUTH_PASSWORD="$JVB_PASS" \
    -n jitsi --dry-run=client -o yaml | $KUBECTL apply -f -

# ------------------------------------------------------------------
# 4. Generate TLS certificate (self-signed via k3s/traefik default)
#    and create the tls secret referenced by Ingress
# ------------------------------------------------------------------
echo "[4/7] Generating TLS certificate for $NIP_DOMAIN ..."
TLS_DIR=$(mktemp -d)
openssl req -x509 -nodes -days 365 \
    -newkey rsa:2048 \
    -keyout "$TLS_DIR/tls.key" \
    -out "$TLS_DIR/tls.crt" \
    -subj "/CN=$NIP_DOMAIN" \
    -addext "subjectAltName=DNS:$NIP_DOMAIN" 2>/dev/null

$KUBECTL create secret tls jitsi-tls \
    --cert="$TLS_DIR/tls.crt" \
    --key="$TLS_DIR/tls.key" \
    -n jitsi --dry-run=client -o yaml | $KUBECTL apply -f -
rm -rf "$TLS_DIR"

# ------------------------------------------------------------------
# 5. Deploy Jitsi (substitute placeholders then apply)
# ------------------------------------------------------------------
echo "[5/7] Deploying Jitsi ..."

PROSODY_CLUSTER_IP=""

apply_jitsi_manifest() {
    local file="$1"
    sed \
        -e "s|A-B-C-D.nip.io|${NIP_DOMAIN}|g" \
        -e "s|https://A-B-C-D.nip.io|https://${NIP_DOMAIN}|g" \
        -e "s|<FLOATING_IP>|${FLOATING_IP}|g" \
        -e "s|<PROSODY_CLUSTER_IP>|${PROSODY_CLUSTER_IP}|g" \
        "$file" | $KUBECTL apply -f -
}

apply_jitsi_manifest "$K8S_DIR/jitsi/configmap.yaml"

apply_jitsi_manifest "$K8S_DIR/jitsi/prosody.yaml"
echo "  Waiting for prosody pod to be ready ..."
until $KUBECTL get deployment prosody -n jitsi 2>/dev/null | grep -q "1/1"; do
    sleep 3
done
echo "  Prosody deployment is ready."

PROSODY_CLUSTER_IP=$($KUBECTL get svc prosody -n jitsi -o jsonpath='{.spec.clusterIP}')
echo "  Prosody ClusterIP: $PROSODY_CLUSTER_IP"

apply_jitsi_manifest "$K8S_DIR/jitsi/jicofo.yaml"
apply_jitsi_manifest "$K8S_DIR/jitsi/jvb.yaml"
apply_jitsi_manifest "$K8S_DIR/jitsi/web.yaml"
apply_jitsi_manifest "$K8S_DIR/jitsi/ingress.yaml"

# ------------------------------------------------------------------
# 6. Deploy MLflow
# ------------------------------------------------------------------
echo "[6/7] Deploying MLflow ..."
$KUBECTL apply -f "$K8S_DIR/mlflow/pv.yaml"
$KUBECTL apply -f "$K8S_DIR/mlflow/mlflow.yaml"

# ------------------------------------------------------------------
# 7. Verify
# ------------------------------------------------------------------
echo "[7/7] Waiting for all deployments ..."

echo "  Waiting for jitsi namespace ..."
for deploy in prosody jicofo jvb web; do
    until $KUBECTL get deployment "$deploy" -n jitsi 2>/dev/null | grep -q "1/1"; do
        sleep 5
    done
    echo "    $deploy: ready"
done

echo "  Waiting for platform namespace ..."
until $KUBECTL get deployment mlflow -n platform 2>/dev/null | grep -q "1/1"; do
    sleep 5
done
echo "    mlflow: ready"

echo ""
echo "============================================"
echo " Deployment complete!"
echo "============================================"
echo ""
$KUBECTL get pods -A -o wide
echo ""
echo " Jitsi  : https://$NIP_DOMAIN"
echo " MLflow : http://$FLOATING_IP:30500"
echo "============================================"
