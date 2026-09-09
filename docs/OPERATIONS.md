# Operation and deployment

This local prototype calls a shared Python pipeline synchronously from Streamlit.
FastAPI is unnecessary at this scale. Add an API and durable worker queue for
remote clients, concurrent jobs or live camera ingestion.

## Alert policy

- YOLO feeds ByteTrack; every run creates fresh model/tracker state.
- Model class names identify bags, including `bag`, `luggage` and `travel bag`.
- The default monitoring region covers the frame; the dashboard supports a rectangular ROI.
- A bag is stationary when its center stays within 1.5% of the frame diagonal of
  an anchored position for 3 observed seconds. Slow drift eventually leaves that radius.
- A person is nearby when the distance from the bag center to their bounding box
  is at most 0.65 times their box height. This configurable pixel-space heuristic
  does not measure meters or establish ownership.
- Stationary bags accumulate unattended time without a detected nearby person.
  The first absent frame does not count the preceding attended interval.
- Alert at 300 unattended seconds. Nearby people or bag motion reset the episode.
  Any nearby person can reset it, including a passerby.
- Missing detections pause the clock. Gaps over 2 seconds discard the history.
  Disappearance resolves as loss of visibility, not a confirmed pickup.
- New tracker IDs start new histories. No guessed identity transfers between IDs.
- Time is frame index / FPS, independent of inference speed or player seeking.
  Use constant-frame-rate video; normalize variable-rate input first with FFmpeg
  `-r 25 -fps_mode cfr`. Edited footage cannot establish continuous real-world time.

The policy can miss bags after detection gaps or ID switches, and can falsely
alert when people are missed. An alert flags an attendance pattern, not danger.

## Local services

Run from the repository root in separate terminals:

```powershell
uv run streamlit run app.py --server.address 127.0.0.1
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Dashboard: http://127.0.0.1:8501; MLflow: http://127.0.0.1:5000.
Set `MLFLOW_TRACKING_URI` in the process environment for remote tracking.
`.env.example` is a reference and is not loaded automatically. Back up SQLite and
`mlartifacts/` together. Streamlit limits uploads to 200 MB; inference rejects over
4K resolution. Output video is silent H.264. Frames are processed incrementally;
outputs are retained locally and copied to MLflow, using additional disk space.

Loopback binding is deliberate. Before shared deployment, configure authentication,
TLS, access control, retention, resource limits and a worker queue. No external
email/SMS/Slack notifications are configured.

## Containers

Start Docker Desktop, then:

```sh
docker compose --profile setup run --rm prepare
docker compose up --build -d dashboard mlflow
docker compose logs -f dashboard
docker compose down
```

The named volume retains data, weights, runs and MLflow. `down` preserves it.
The image uses a non-root user and health checks, with only loopback ports exposed.
CPU PyTorch is locked for reproducibility; CUDA is not configured.

## CI and delivery

CI runs Ruff, formatting and pytest on Ubuntu/Windows. A Linux smoke job downloads
verified assets, runs real YOLO inference and trains one COCO8 epoch. Release tags
(`v*`) trigger quality checks, a container build and publication to
`ghcr.io/AI-King/abandoned_bag_detection`. This delivers an image; it does not update
a production host. Actions uses `GITHUB_TOKEN`; no credentials are embedded.

Git versions code/configuration, `uv.lock` versions dependencies, checksums version
downloads, and MLflow records experiments and candidate weights. Training and
evaluation log exact image/label hashes. Candidates are named by run ID and do
not automatically replace the pretrained default. Code releases do not certify models.
