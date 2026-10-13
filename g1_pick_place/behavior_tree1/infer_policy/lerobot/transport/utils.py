"""Lightweight LeRobot transport helpers for the robot-side runtime."""

import io
import json
import pickle  # nosec B403: trusted internal client/server protocol
from typing import Any

from . import services_pb2


TransferState = services_pb2.TransferState
CHUNK_SIZE = 2 * 1024 * 1024
MAX_MESSAGE_SIZE = 4 * 1024 * 1024


def send_bytes_in_chunks(buffer: bytes, message_class: Any):
    bytes_buffer = io.BytesIO(buffer)
    size_in_bytes = len(buffer)
    sent_bytes = 0
    while sent_bytes < size_in_bytes:
        transfer_state = TransferState.TRANSFER_MIDDLE
        if sent_bytes + CHUNK_SIZE >= size_in_bytes:
            transfer_state = TransferState.TRANSFER_END
        elif sent_bytes == 0:
            transfer_state = TransferState.TRANSFER_BEGIN
        size_to_read = min(CHUNK_SIZE, size_in_bytes - sent_bytes)
        yield message_class(
            transfer_state=transfer_state,
            data=bytes_buffer.read(size_to_read),
        )
        sent_bytes += size_to_read


def python_object_to_bytes(python_object: Any) -> bytes:
    return pickle.dumps(python_object)


def bytes_to_python_object(buffer: bytes) -> Any:
    return pickle.loads(buffer)  # nosec B301: trusted internal protocol


def grpc_channel_options(
    max_receive_message_length: int = MAX_MESSAGE_SIZE,
    max_send_message_length: int = MAX_MESSAGE_SIZE,
    enable_retries: bool = True,
    initial_backoff: str = "0.1s",
    max_attempts: int = 5,
    backoff_multiplier: float = 2,
    max_backoff: str = "2s",
):
    service_config = {
        "methodConfig": [{
            "name": [{}],
            "retryPolicy": {
                "maxAttempts": max_attempts,
                "initialBackoff": initial_backoff,
                "maxBackoff": max_backoff,
                "backoffMultiplier": backoff_multiplier,
                "retryableStatusCodes": ["UNAVAILABLE", "DEADLINE_EXCEEDED"],
            },
        }]
    }
    return [
        ("grpc.max_receive_message_length", max_receive_message_length),
        ("grpc.max_send_message_length", max_send_message_length),
        ("grpc.enable_retries", 1 if enable_retries else 0),
        ("grpc.service_config", json.dumps(service_config)),
    ]
