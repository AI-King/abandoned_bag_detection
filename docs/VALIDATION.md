# Validation record

Local execution on 2026-09-09, Windows, Python 3.12, CPU PyTorch and YOLO11n.
The exact dependency versions are in `uv.lock`. These results describe the
prototype and are not a security-system acceptance test.

| Check | Result |
| --- | --- |
| Alert rules | Five-minute boundary, nearby person, return, movement, occlusion, ID changes, multiple bags, ROI and invalid timestamps tested |
| Media pipeline | Actual H.264 output, CSV rows, MLflow metrics/artifacts and failure cleanup tested |
| Data checks | SHA-256 corruption detection, YOLO labels and duplicate image leakage tested |
| Automated suite | 41 pytest tests; Ruff lint/format and all pre-commit hooks pass locally |
| COCO8 training | One epoch completed and checkpoint/metrics saved in MLflow |
| Luggage subset training | Three epochs completed on 95 train / 33 validation images; candidate retained separately |
| Synthetic five-minute video | Real YOLO + ByteTrack inference produced an alert at 303s: 3s stationary + 300s unattended |
| CAVIAR LeftBag | Full 1,440-frame clip processed; pretrained YOLO detected people but missed the small bag, producing no bag alert |
| Streamlit | Browser checked video player, analytics below it, alert table and timeline |
| Service health | Dashboard and local MLflow returned HTTP 200 |
| Docker | Compose configuration validated; local container execution unavailable because Docker engine was stopped |

The three-epoch luggage candidate achieved only about 0.068 mAP50 on its tiny
development validation split. It is not promoted as the default. The default
remains the COCO-pretrained YOLO11n. COCO8 smoke-test scores do not measure luggage
accuracy. COCO-pretrained weights have seen the underlying COCO training images,
so the derived luggage validation split is not an independent benchmark.

The synthetic fixture is a repeated cropped photograph with no person-leaving
event. It validates the complete timer/output path, not real-world abandonment.
The CAVIAR miss is preserved as a known failure, not hidden by using annotations
as detections. The system cannot alert on a bag the model fails to recognize.

Before station deployment, use independent station clips with varied bag forms,
camera angles, lighting, occlusion, attended bags and pickup events. Measure bag
recall, false alarms per video hour, missed events, alert delay and ID stability.
Calibrate proximity and stationarity for the camera. Do not claim ownership or
danger from the proximity heuristic.
