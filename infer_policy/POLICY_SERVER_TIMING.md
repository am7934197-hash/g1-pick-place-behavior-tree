# PolicyServer RTC and timing protocol

The client sends one `trace_id` both in gRPC metadata and in the pickled
`TimedObservation`. The same observation also carries:

```text
previous_actions             float32 [remaining_steps, action_dim]
previous_action_start_index  index of the first remaining action in the old chunk
control_step                 client control-loop counter at capture
trajectory_step              number of policy actions executed at capture
fps                          policy/control frequency
```

`previous_actions` are postprocessed absolute joint targets. The server trims
frames that expired between capture and inference with LeRobot `ActionQueue`,
then uses `reanchor_relative_rtc_prefix` plus the checkpoint normalizer to put
the prefix back in Pi0.5 model space. The resulting tensor is passed to
`PI05Policy.predict_action_chunk` as `prev_chunk_left_over`; no action-chunk
interpolation occurs after generation.

The pickled response envelope is:

```text
trace_id
observation_timestep
inference_delay_frames
prefix_frames_used
expired_prefix_frames
previous_action_start_index
actions                     list[TimedAction]
```

The client rejects a response whose `trace_id` or `observation_timestep` does
not match the request. Session-close checks remain in the asynchronous RTC
worker, so a result from an earlier robot session is still discarded.

## Server trace events

All server timestamps use `time.perf_counter()` and are correlated by
`trace_id`; they must not be subtracted from client-host `perf_counter()`
values.

```text
SERVER_REQUEST_RECEIVED
SERVER_OBSERVATION_DECODED
SERVER_QUEUE_DONE
SERVER_INFER_START
SERVER_INFER_END
SERVER_RESPONSE_SEND
```

Example:

```text
SERVER TRACE trace_id=abc:4:120 event=SERVER_REQUEST_RECEIVED perf_s=8421.100000
SERVER TRACE trace_id=abc:4:120 event=SERVER_OBSERVATION_DECODED perf_s=8421.118000 server_receive_ms=18.000
SERVER TRACE trace_id=abc:4:120 event=SERVER_QUEUE_DONE perf_s=8421.119000 server_queue_ms=1.000
SERVER TRACE trace_id=abc:4:120 event=SERVER_INFER_START perf_s=8421.119100
SERVER TRACE trace_id=abc:4:120 event=SERVER_INFER_END perf_s=8421.421100 server_model_infer_ms=276.000 prefix_frames_used=10 inference_delay_frames=10
SERVER TRACE trace_id=abc:4:120 event=SERVER_RESPONSE_SEND perf_s=8421.423100 server_receive_ms=18.000 server_queue_ms=1.000 server_preprocess_ms=19.000 server_model_infer_ms=276.000 server_postprocess_ms=7.000 server_serialize_ms=2.000
```

`GetActions` trailing metadata contains:

```text
x-lerobot-trace-id
x-lerobot-server-receive-ms
x-lerobot-server-queue-ms
x-lerobot-server-preprocess-ms
x-lerobot-server-model-ms
x-lerobot-server-postprocess-ms
x-lerobot-server-serialize-ms
x-lerobot-server-get-actions-ms
x-lerobot-inference-delay-frames
x-lerobot-prefix-frames-used
```

The six requested timing values map to `server_receive_ms`, `server_queue_ms`,
`server_preprocess_ms`, `server_model_infer_ms`, `server_postprocess_ms`, and
`server_serialize_ms`.
