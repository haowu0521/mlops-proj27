import logging
import os
from datetime import UTC, datetime
from typing import Any

import requests
from kubernetes import client, config


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("model-refresher")

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://mlflow.platform.svc.cluster.local:5000",
).rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "jitsi-summarizer")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "production")
K8S_NAMESPACE = os.environ.get("K8S_NAMESPACE", "platform")
DEPLOYMENT_NAME = os.environ.get("DEPLOYMENT_NAME", "serving-baseline-mlflow")
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "30"))

VERSION_ANNOTATION = os.environ.get(
    "VERSION_ANNOTATION",
    "mlops.proj27/model-version",
)
REFRESHED_AT_ANNOTATION = os.environ.get(
    "REFRESHED_AT_ANNOTATION",
    "mlops.proj27/model-refreshed-at",
)
ALIAS_ANNOTATION = os.environ.get(
    "ALIAS_ANNOTATION",
    "mlops.proj27/model-alias",
)


def _load_kube_config() -> None:
    try:
        config.load_incluster_config()
        return
    except config.ConfigException:
        config.load_kube_config()


def _get_alias_model_version() -> dict[str, Any]:
    resp = requests.get(
        f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/registered-models/alias",
        params={"name": MODEL_NAME, "alias": MODEL_ALIAS},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    payload = resp.json()
    model_version = payload.get("model_version", payload)
    version = model_version.get("version")
    if not version:
        raise RuntimeError(f"MLflow alias response did not include a version: {payload}")
    return model_version


def _current_deployment_annotations(apps: client.AppsV1Api) -> dict[str, str]:
    deployment = apps.read_namespaced_deployment(
        name=DEPLOYMENT_NAME,
        namespace=K8S_NAMESPACE,
    )
    template = deployment.spec.template
    if not template.metadata:
        return {}
    return template.metadata.annotations or {}


def _patch_deployment_for_version(apps: client.AppsV1Api, version: str) -> None:
    now = datetime.now(UTC).isoformat()
    body = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        VERSION_ANNOTATION: version,
                        ALIAS_ANNOTATION: f"{MODEL_NAME}@{MODEL_ALIAS}",
                        REFRESHED_AT_ANNOTATION: now,
                    }
                }
            }
        }
    }
    apps.patch_namespaced_deployment(
        name=DEPLOYMENT_NAME,
        namespace=K8S_NAMESPACE,
        body=body,
    )
    log.info(
        "Triggered rollout restart for %s/%s at %s",
        K8S_NAMESPACE,
        DEPLOYMENT_NAME,
        now,
    )


def main() -> None:
    model_version = _get_alias_model_version()
    version = str(model_version["version"])
    run_id = model_version.get("run_id", "unknown")
    log.info(
        "MLflow alias %s@%s currently points to version %s (run_id=%s)",
        MODEL_NAME,
        MODEL_ALIAS,
        version,
        run_id,
    )

    _load_kube_config()
    apps = client.AppsV1Api()
    annotations = _current_deployment_annotations(apps)
    deployed_version = annotations.get(VERSION_ANNOTATION)

    if deployed_version == version:
        log.info(
            "Deployment %s/%s already records model version %s; no restart needed.",
            K8S_NAMESPACE,
            DEPLOYMENT_NAME,
            version,
        )
        return

    log.info(
        "Deployment %s/%s records version %s, but MLflow alias is version %s.",
        K8S_NAMESPACE,
        DEPLOYMENT_NAME,
        deployed_version or "<missing>",
        version,
    )
    _patch_deployment_for_version(apps, version)


if __name__ == "__main__":
    main()
