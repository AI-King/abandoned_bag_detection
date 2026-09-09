from vision_lab.training import metric_name


def test_ultralytics_metrics_have_mlflow_safe_names():
    assert metric_name("metrics/precision(B)") == "metrics/precision_B_"
    assert metric_name("metrics/mAP50-95(B)") == "metrics/mAP50-95_B_"
