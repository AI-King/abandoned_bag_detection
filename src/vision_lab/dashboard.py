"""Video-first unattended luggage review dashboard."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from vision_lab.config import InferenceConfig, workspace
from vision_lab.pipeline import process_video
from vision_lab.unattended import MonitorConfig


def chart_style(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=20, b=0),
        font=dict(color="#AFBED0"),
        height=260,
    )
    return fig


def render_result(output: Path):
    summary = json.loads((output / "summary.json").read_text())
    config = json.loads((output / "config.json").read_text())
    st.caption(
        f"{summary['source_name']}  ·  Run {output.name}  ·  "
        f"{summary['duration_seconds']:.1f}s analyzed"
    )
    st.video(str(output / "annotated.mp4"))
    # Analytics intentionally appear immediately below the player.
    cols = st.columns(4)
    cols[0].metric("Unattended alerts", summary["alert_count"])
    cols[1].metric("Object tracks", summary["unique_tracks"])
    cols[2].metric("Frames analyzed", f"{summary['frames_processed']:,}")
    cols[3].metric("Processing speed", f"{summary['processing_fps']:.1f} fps")
    st.caption(
        "Track IDs can split after occlusion. "
        "Counts are tracker histories, not verified unique objects."
    )
    if summary.get("truncated"):
        st.warning(
            "This run analyzed only the configured frame limit. "
            "The remaining video was not checked."
        )
    if summary["source_name"].startswith("Synthetic_"):
        st.info(
            "Synthetic repeated-photo fixture: validates the timer and alert pipeline, "
            "not real-world abandonment detection."
        )
    threshold = summary.get("alert_threshold_seconds")
    if threshold is not None and summary["duration_seconds"] < threshold:
        st.info(
            f"This clip is shorter than the {threshold:g}s alert threshold. "
            "Use a longer video to test that threshold, or explicitly lower it for a demo."
        )
    alert_tab, analytics_tab, export_tab = st.tabs(
        ["Luggage & alerts", "Detection analytics", "Run & exports"]
    )
    with alert_tab:
        events = json.loads((output / "events.json").read_text())
        bags = pd.read_csv(output / "bags.csv")
        if events:
            st.dataframe(pd.DataFrame(events), hide_index=True, width="stretch")
        else:
            st.info("No unattended-bag alerts were generated in this analyzed interval.")
        if not bags.empty:
            st.plotly_chart(
                chart_style(
                    px.line(
                        bags,
                        x="timestamp_seconds",
                        y="unattended_seconds",
                        color=bags.track_id.astype(str),
                        labels={
                            "timestamp_seconds": "Video time (s)",
                            "unattended_seconds": "Unattended (s)",
                            "color": "Bag ID",
                        },
                        color_discrete_sequence=px.colors.qualitative.Safe,
                    ).add_hline(y=threshold, line_dash="dot", line_color="#FF7474")
                ),
                width="stretch",
            )
            latest = bags.sort_values("timestamp_seconds").groupby("track_id").tail(1)
            st.caption("Last observed state of each bag track")
            st.dataframe(latest, hide_index=True, width="stretch")
        else:
            st.warning(
                "No supported bags were tracked. "
                "A missed or unrecognized bag cannot trigger an alert."
            )
    with analytics_tab:
        timeline = pd.read_csv(output / "timeline.csv")
        left, right = st.columns([2, 1])
        with left:
            st.markdown("#### Objects over time")
            st.plotly_chart(
                chart_style(
                    px.area(
                        timeline,
                        x="timestamp_seconds",
                        y="detections",
                        labels={
                            "timestamp_seconds": "Video time (s)",
                            "detections": "Objects / frame",
                        },
                        color_discrete_sequence=["#36D6AE"],
                    )
                ),
                width="stretch",
            )
        with right:
            st.markdown("#### Detection observations")
            counts = pd.DataFrame(
                list(summary["class_counts"].items()), columns=["Class", "Observations"]
            )
            if not counts.empty:
                st.plotly_chart(
                    chart_style(
                        px.bar(
                            counts, x="Observations", y="Class", color_discrete_sequence=["#76A6FF"]
                        )
                    ),
                    width="stretch",
                )
            else:
                st.info("No detections at this confidence threshold.")
        st.caption(
            "An observation is one detection in one frame; the same bag may appear many times."
        )
    with export_tab:
        st.code(f"MLflow run: {summary['mlflow_run_id']}")
        st.json(config, expanded=False)
        for filename, label, mime in [
            ("annotated.mp4", "Download annotated video", "video/mp4"),
            ("events.json", "Download alert events", "application/json"),
            ("bags.csv", "Download luggage timeline", "text/csv"),
            ("detections.csv", "Download all detections", "text/csv"),
            ("summary.json", "Download summary", "application/json"),
        ]:
            with (output / filename).open("rb") as stream:
                st.download_button(label, stream, file_name=f"{output.name}-{filename}", mime=mime)


def main():
    st.set_page_config(page_title="Vision Lab · Luggage Watch", page_icon="🧳", layout="wide")
    st.markdown(
        """<style>
    .block-container{max-width:1250px;padding-top:2rem}
    [data-testid="stMetric"]{background:#141D28;border:1px solid #263343;
    padding:16px;border-radius:12px}
    h1{letter-spacing:-1.5px} [data-testid="stSidebar"]{border-right:1px solid #263343}
    </style>""",
        unsafe_allow_html=True,
    )
    root = workspace()
    st.caption("VISION LAB  /  VIDEO INTELLIGENCE")
    st.title("Unattended luggage watch")
    st.write("Review people, bags, and time apart. Follow the video, then explore the evidence.")
    with st.sidebar:
        st.markdown("## 🧳 Vision Lab")
        st.caption("Fixed-camera luggage monitoring")
        source_kind = st.radio("Video source", ["Public demo", "Upload a video"])
        upload = None
        source = None
        if source_kind == "Public demo":
            video_files = []
            for folder in [root / "data/test", root / "data/videos"]:
                if folder.exists():
                    video_files.extend(sorted(folder.glob("*.mp4")) + sorted(folder.glob("*.avi")))
            seen_stems = set()
            available_paths = []
            for vf in video_files:
                if vf.stem not in seen_stems:
                    seen_stems.add(vf.stem)
                    available_paths.append(vf)

            def sort_key(p):
                name = p.stem
                if "generate" in name or "cctv" in name:
                    return (0, 0, name)
                if name.startswith("aboda_video"):
                    try:
                        num = int(name.replace("aboda_video", ""))
                        return (1, num, name)
                    except ValueError:
                        return (1, 99, name)
                if name == "LeftBag":
                    return (2, 0, name)
                if name == "LeftBag_PickedUp":
                    return (2, 1, name)
                if name == "Synthetic_Timer_310s":
                    return (3, 0, name)
                return (4, 0, name)

            available_paths.sort(key=sort_key)
            scenarios = [p.stem for p in available_paths] or ["LeftBag", "LeftBag_PickedUp"]
            selected = st.selectbox("Scenario", scenarios)
            default_path = root / f"data/videos/{selected}.mp4"
            source = next((p for p in available_paths if p.stem == selected), default_path)
            if selected == "Synthetic_Timer_310s":
                st.caption("Synthetic repeated COCO photo · Timer verification only")
            elif "cctv" in selected.lower() or "generate" in selected.lower():
                st.caption("CCTV test video · Person with suitcase leaves it · Unattended event")
            elif "aboda" in selected.lower():
                st.caption(f"ABODA Benchmark ({selected}) · Real surveillance abandoned object")
            else:
                st.caption("CAVIAR project · CC BY-SA · INRIA lobby footage")
        else:
            upload = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv", "mpg"])
        st.divider()
        st.markdown("### Detection settings")
        models = sorted(
            (root / "models").glob("*.pt"), key=lambda p: (p.name != "yolo11n.pt", p.name)
        )
        model = st.selectbox("YOLO model", models, format_func=lambda p: p.name) if models else None
        confidence = st.slider("Detection confidence", 0.05, 0.9, 0.3, 0.05)
        tracker_choice = st.selectbox(
            "Tracking engine",
            ["ILP (Integer Linear Programming)", "ByteTrack (Standard MOT)"],
            help="ILP uses mathematical optimization to preserve bag IDs across class flips.",
        )
        tracker_type = "ilp" if "ILP" in tracker_choice else "bytetrack"
        alert_seconds = st.number_input(
            "Unattended threshold (seconds)", min_value=1, max_value=3600, value=300, step=5
        )
        st.caption(
            "Default: 5 minutes of video time. Short demo clips need a shorter test threshold."
        )
        with st.expander("Camera calibration & performance"):
            image_size = st.select_slider(
                "Inference resolution", [320, 480, 640, 960, 1280], value=640
            )
            stationary = st.number_input("Confirm stationary (seconds)", 1, 30, 3)
            proximity = st.slider("Nearby person radius / person height", 0.1, 2.0, 0.65, 0.05)
            movement = st.slider("Movement tolerance (% frame diagonal)", 0.2, 5.0, 1.5, 0.1)
            limit = st.number_input("Frame limit (0 = complete video)", 0, 100000, 0, step=100)
            st.caption("Optional monitoring zone, as percentages of the frame.")
            roi_x = st.slider("Horizontal zone", 0, 100, (0, 100))
            roi_y = st.slider("Vertical zone", 0, 100, (0, 100))
        analyze = st.button(
            "Analyze video",
            type="primary",
            width="stretch",
            disabled=model is None or (source_kind == "Upload a video" and upload is None),
        )
        st.divider()
        st.caption(
            "MLflow tracks parameters, metrics, model hashes, Git revision, and output artifacts."
        )
    if not models:
        st.info("Prepare the public videos and detector with: uv run vision prepare")
    if analyze:
        temporary = None
        try:
            if upload is not None:
                with tempfile.NamedTemporaryFile(
                    suffix=Path(upload.name).suffix, delete=False
                ) as stream:
                    stream.write(upload.getbuffer())
                    temporary = Path(stream.name)
                source = temporary
            if source is None or not source.is_file():
                raise FileNotFoundError("Demo video missing. Run: uv run vision prepare")
            bar = st.progress(0, text="Loading detector…")
            with st.spinner(
                "Detecting people and luggage, checking attendance, and recording the run…"
            ):
                output = process_video(
                    source,
                    InferenceConfig(
                        model=str(model),
                        confidence=confidence,
                        image_size=image_size,
                        tracker_type=tracker_type,
                        max_frames=limit or None,
                    ),
                    MonitorConfig(
                        alert_seconds=alert_seconds,
                        stationary_seconds=stationary,
                        proximity_ratio=proximity,
                        movement_fraction=movement / 100,
                        roi=(roi_x[0] / 100, roi_y[0] / 100, roi_x[1] / 100, roi_y[1] / 100),
                    ),
                    progress=lambda current, total: bar.progress(
                        min(current / max(total, 1), 1.0),
                        text=f"Analyzed {current:,} of {total:,} frames",
                    ),
                )
            st.session_state["output"] = str(output)
            bar.empty()
        except Exception as error:
            st.error(f"Analysis could not finish: {error}")
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
    runs = sorted(
        (root / "outputs").glob("*/COMPLETE"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if runs:
        options = [p.parent for p in runs]
        current = Path(st.session_state.get("output", str(options[0])))
        index = options.index(current) if current in options else 0
        selected_output = st.selectbox(
            "Completed analysis",
            options,
            index=index,
            format_func=lambda p: (
                f"{p.name} · " + json.loads((p / "summary.json").read_text())["source_name"]
            ),
        )
        render_result(selected_output)
    else:
        st.markdown("### Start with a video")
        if source and source.is_file():
            st.video(str(source))
        st.info(
            "Choose a demo or upload a station video, then select Analyze video. "
            "Luggage timers and alert events will appear directly below the annotated player."
        )
    st.divider()
    st.caption(
        "An alert flags a stationary bag without a detected person nearby. It does not identify "
        "an owner or determine danger. COCO weights recognize backpacks, handbags, and suitcases; "
        "other bags need labeled examples and fine tuning. No external notifications are sent."
    )


if __name__ == "__main__":
    main()
