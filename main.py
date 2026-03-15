import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plate Vision",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --border: #1e1e2e;
    --accent: #00ff9d;
    --accent2: #ff6b35;
    --text: #e8e8f0;
    --muted: #5a5a7a;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.5rem;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #00ff9d 0%, #00b4d8 50%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    color: var(--muted);
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.plate-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin: 0.5rem 0;
    font-family: 'Space Mono', monospace;
}

.plate-number {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.08em;
}

.plate-meta {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.35rem;
    letter-spacing: 0.05em;
}

.timestamp-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin: 0.3rem 0;
    display: flex;
    justify-content: space-between;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}

.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 4px !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.6rem 1.5rem !important;
}

.stButton > button:hover {
    background: #00e68a !important;
    transform: translateY(-1px);
    transition: all 0.15s ease;
}

.stFileUploader {
    border: 1px dashed var(--border) !important;
    border-radius: 8px;
    background: var(--surface) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: 8px;
    border: 1px solid var(--border);
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted) !important;
    padding: 0.75rem 1.5rem;
}

.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    background: var(--border) !important;
}

.metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}

.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
}

.metric-label {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.stProgress > div > div {
    background: var(--accent) !important;
}

div[data-testid="stAlert"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}
</style>
""",
    unsafe_allow_html=True,
)


# ─── Model Loading ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    plate_model = YOLO("models/numberplate.pt")
    char_model = YOLO("models/alphanum.pt")
    return plate_model, char_model


# ─── Core Logic ──────────────────────────────────────────────────────────────────
def detect_plate_region(plate_model, image_bgr):
    """Returns list of (cropped_plate, x1,y1,x2,y2, conf)."""
    results = plate_model(image_bgr, verbose=False)
    plates = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                plates.append((crop, x1, y1, x2, y2, conf))
    return plates


def read_plate_text(char_model, cropped_plate):
    """Returns plate string from cropped plate image."""
    upscaled = cv2.resize(
        cropped_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
    )
    results = char_model(upscaled, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1 = float(box.xyxy[0][0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = char_model.names[cls]
        detections.append((x1, label, conf))
    detections.sort(key=lambda x: x[0])
    plate_text = "".join(char for _, char, conf in detections if conf > 0.25)
    return plate_text


def annotate_image(image_bgr, plates_info):
    """Draw bounding boxes and text on image."""
    annotated = image_bgr.copy()
    for text, x1, y1, x2, y2, conf in plates_info:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 157), 2)
        label = f"{text} ({conf:.2f})" if text else f"Plate ({conf:.2f})"
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 157),
            2,
        )
    return annotated


def process_image(plate_model, char_model, image_bgr):
    plates_raw = detect_plate_region(plate_model, image_bgr)
    results = []
    for crop, x1, y1, x2, y2, conf in plates_raw:
        text = read_plate_text(char_model, crop)
        results.append((text, x1, y1, x2, y2, conf, crop))
    return results


# ─── Header ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">PLATE VISION</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Automated Number Plate Recognition · YOLO v11</div>',
    unsafe_allow_html=True,
)

# ─── Load Models ─────────────────────────────────────────────────────────────────
with st.spinner("Loading models..."):
    try:
        plate_model, char_model = load_models()
        st.success("Models loaded ✓", icon="✅")
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

st.divider()

# ─── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📷  IMAGE", "🎬  VIDEO"])


# ═══════════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE
# ═══════════════════════════════════════════════════════════════════════════════════
with tab1:
    uploaded_img = st.file_uploader(
        "Drop an image here",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_upload",
    )

    if uploaded_img:
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col_img, col_results = st.columns([1.2, 1], gap="large")

        with col_img:
            st.markdown(
                '<div class="section-label">Input Image</div>', unsafe_allow_html=True
            )
            st.image(
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True
            )

        with col_results:
            st.markdown(
                '<div class="section-label">Detection Results</div>',
                unsafe_allow_html=True,
            )

            with st.spinner("Running detection..."):
                results = process_image(plate_model, char_model, image_bgr)

            if not results:
                st.warning("No plates detected in this image.")
            else:
                st.markdown(
                    f"""
                <div class="metric-box" style="margin-bottom:1rem;">
                    <div class="metric-val">{len(results)}</div>
                    <div class="metric-label">Plate(s) Detected</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                annotated = annotate_image(
                    image_bgr,
                    [(t, x1, y1, x2, y2, c) for t, x1, y1, x2, y2, c, _ in results],
                )
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption="Annotated",
                    use_container_width=True,
                )

                for i, (text, x1, y1, x2, y2, conf, crop) in enumerate(results):
                    plate_display = text if text else "—"
                    st.markdown(
                        f"""
                    <div class="plate-card">
                        <div class="plate-number">{plate_display}</div>
                        <div class="plate-meta">Plate {i + 1} · Confidence {conf:.1%} · Coords ({x1},{y1})→({x2},{y2})</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    with st.expander(f"Plate {i + 1} crop"):
                        st.image(
                            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                            use_container_width=False,
                        )


# ═══════════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO
# ═══════════════════════════════════════════════════════════════════════════════════
with tab2:
    col_ctl1, col_ctl2, col_ctl3 = st.columns(3)
    with col_ctl1:
        throttle_fps = st.slider(
            "Process every N frames",
            min_value=1,
            max_value=60,
            value=15,
            help="Higher = faster but less granular timestamps",
        )
    with col_ctl2:
        conf_threshold = st.slider("Min plate confidence", 0.1, 1.0, 0.4, 0.05)
    with col_ctl3:
        dedup_seconds = st.slider(
            "Detection Interval window(s)",
            0,
            10,
            3,
            help="Ignore same plate if seen within N seconds",
        )

    uploaded_vid = st.file_uploader(
        "Drop a video here", type=["mp4", "avi", "mov", "mkv", "webm"], key="vid_upload"
    )

    if uploaded_vid:
        # Save to temp file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded_vid.name).suffix
        ) as tmp:
            tmp.write(uploaded_vid.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()

        duration_s = total_frames / video_fps
        frames_to_process = total_frames // throttle_fps

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.markdown(
            f'<div class="metric-box"><div class="metric-val">{total_frames:,}</div><div class="metric-label">Total Frames</div></div>',
            unsafe_allow_html=True,
        )
        col_m2.markdown(
            f'<div class="metric-box"><div class="metric-val">{duration_s:.1f}s</div><div class="metric-label">Duration</div></div>',
            unsafe_allow_html=True,
        )
        col_m3.markdown(
            f'<div class="metric-box"><div class="metric-val">{frames_to_process:,}</div><div class="metric-label">Frames to Process</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("")

        if st.button("▶  START ANALYSIS"):
            cap = cv2.VideoCapture(tmp_path)
            frame_idx = 0

            progress_bar = st.progress(0, text="Initialising...")
            status_text = st.empty()

            # Results storage: {plate_text: [(timestamp, conf)]}
            plate_timeline = {}  # plate_text -> list of timestamps
            last_seen = {}  # plate_text -> last timestamp (for dedup)

            col_preview, col_log = st.columns([1.2, 1], gap="large")

            with col_preview:
                st.markdown(
                    '<div class="section-label">Live Preview</div>',
                    unsafe_allow_html=True,
                )
                preview_slot = st.empty()

            with col_log:
                st.markdown(
                    '<div class="section-label">Detections Log</div>',
                    unsafe_allow_html=True,
                )
                log_slot = st.empty()

            start_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % throttle_fps == 0:
                    timestamp_s = frame_idx / video_fps
                    progress = frame_idx / max(total_frames, 1)
                    elapsed = time.time() - start_time
                    eta = (
                        (elapsed / max(progress, 0.001)) * (1 - progress)
                        if progress > 0
                        else 0
                    )

                    progress_bar.progress(
                        min(progress, 1.0),
                        text=f"Frame {frame_idx:,}/{total_frames:,} · {timestamp_s:.1f}s · ETA {eta:.0f}s",
                    )

                    # Detect plates
                    plates_raw = detect_plate_region(plate_model, frame)
                    annotated_frame = frame.copy()

                    for crop, x1, y1, x2, y2, conf in plates_raw:
                        if conf < conf_threshold:
                            continue

                        text = read_plate_text(char_model, crop)
                        if not text:
                            continue

                        # Dedup check
                        last_t = last_seen.get(text, -999)
                        if timestamp_s - last_t >= dedup_seconds:
                            last_seen[text] = timestamp_s
                            if text not in plate_timeline:
                                plate_timeline[text] = []
                            plate_timeline[text].append(
                                {"ts": timestamp_s, "conf": conf}
                            )

                        # Annotate frame
                        cv2.rectangle(
                            annotated_frame, (x1, y1), (x2, y2), (0, 255, 157), 2
                        )
                        cv2.putText(
                            annotated_frame,
                            text or "?",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 157),
                            2,
                        )

                    # Update preview (every 5 processed frames to reduce flicker)
                    preview_slot.image(
                        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                        caption=f"t = {timestamp_s:.2f}s",
                    )

                    # Update log
                    if plate_timeline:
                        log_html = ""
                        for plate, entries in sorted(plate_timeline.items()):
                            timestamps_str = ", ".join(
                                f"{e['ts']:.1f}s" for e in entries[-5:]
                            )
                            log_html += f"""
                            <div class="plate-card" style="margin-bottom:0.4rem;">
                                <div class="plate-number" style="font-size:1.2rem;">{plate}</div>
                                <div class="plate-meta">
                                    {len(entries)} sighting(s) · Last: {entries[-1]["ts"]:.1f}s<br>
                                    Timestamps: {timestamps_str}
                                </div>
                            </div>
                            """
                        log_slot.markdown(log_html, unsafe_allow_html=True)

                frame_idx += 1

            cap.release()
            os.unlink(tmp_path)
            progress_bar.progress(1.0, text="Analysis complete ✓")

            # ─── Final Summary ───────────────────────────────────────────────────
            st.divider()
            st.markdown(
                '<div class="section-label">Final Summary</div>', unsafe_allow_html=True
            )

            if not plate_timeline:
                st.warning("No plates detected in the video.")
            else:
                st.markdown(
                    f"""
                <div class="metric-box" style="margin-bottom:1.5rem; display:inline-block; min-width:200px;">
                    <div class="metric-val">{len(plate_timeline)}</div>
                    <div class="metric-label">Unique Plates Found</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Table-style output
                for plate, entries in sorted(
                    plate_timeline.items(), key=lambda x: x[1][0]["ts"]
                ):
                    first_seen = entries[0]["ts"]
                    last_seen_t = entries[-1]["ts"]
                    all_ts = [f"{e['ts']:.1f}s" for e in entries]
                    avg_conf = sum(e["conf"] for e in entries) / len(entries)

                    st.markdown(
                        f"""
                    <div class="plate-card">
                        <div class="plate-number">{plate}</div>
                        <div class="plate-meta">
                            First seen: {first_seen:.2f}s · Last seen: {last_seen_t:.2f}s ·
                            {len(entries)} sighting(s) · Avg conf: {avg_conf:.1%}
                        </div>
                        <div class="plate-meta" style="margin-top:0.4rem; color:#7a7a9a;">
                            All timestamps → {" · ".join(all_ts)}
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
