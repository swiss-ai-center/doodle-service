from uuid import uuid4

import pytest
from pydantic import ValidationError

from common_code.tasks.models import ServiceTask


def service_task_payload():
    return {
        "storage_url": "https://core-engine.example/storage",
        "task": {
            "data_in": ["input.jpg"],
            "service_id": str(uuid4()),
            "id": str(uuid4()),
            "status": "pending",
        },
        "callback_url": "https://core-engine.example/tasks/task-id",
    }


def test_service_task_uses_core_engine_storage():
    service_task = ServiceTask.model_validate(service_task_payload())

    assert service_task.storage_url == "https://core-engine.example/storage"
    assert not any(
        field.startswith("s3_") for field in ServiceTask.model_fields
    )


def test_service_task_rejects_legacy_storage_payload():
    payload = service_task_payload()
    payload.pop("storage_url")
    payload.update({
        "s3_access_key_id": "access-key",
        "s3_secret_access_key": "secret-key",
        "s3_region": "region",
        "s3_host": "https://storage.example",
        "s3_bucket": "bucket",
    })

    with pytest.raises(ValidationError):
        ServiceTask.model_validate(payload)
