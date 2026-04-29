"""
app.py — ANPR Streamlit Web Interface
Uses the robust anpr.py detection module.

Author: Parth Singh
"""

import cv2
import streamlit as st
import numpy as np
from PIL import Image
import imutils

# Import our robust detection module
from anpr import detect_plate

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="ANPR · Plate Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------
st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Exo+2:wght@300;400;600&display=swap');

    :root {
        --bg-base:       #040b14;
        --bg-panel:      #071220;
        --bg-card:       #0a1929;
        --border-dim:    rgba(0, 230, 255, 0.12);
        --border-glow:   rgba(0, 230, 255, 0.45);
        --cyan:          #00e6ff;
        --cyan-dim:      rgba(0, 230, 255, 0.6);
        --amber:         #f5a623;
        --green:         #00ff88;
        --red:           #ff4757;
        --text-primary:  #e0f4ff;
        --text-secondary:#7fa8c0;
    }

    .stApp {
        background-color: var(--bg-base);
        background-image:
            linear-gradient(rgba(0,230,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,230,255,0.03) 1px, transparent 1px);
        background-size: 40px 40px;
        color: var(--text-primary);
        font-family: 'Exo 2', sans-serif;
    }

    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    /* ── Sidebar toggle buttons ── */
    [data-testid="collapsedControl"] {
        visibility: visible !important; display: flex !important;
        background: #071220 !important;
        border-right: 1px solid rgba(0,230,255,0.25) !important;
        border-radius: 0 4px 4px 0 !important; z-index: 9999 !important;
    }
    [data-testid="collapsedControl"] svg { fill:#00e6ff !important; stroke:#00e6ff !important; }
    [data-testid="stSidebarCollapseButton"] { visibility:visible !important; display:flex !important; }
    [data-testid="stSidebarCollapseButton"] button {
        background:rgba(0,230,255,0.06) !important; border:1px solid rgba(0,230,255,0.2) !important;
        border-radius:3px !important; color:#00e6ff !important;
    }
    [data-testid="stSidebarCollapseButton"] button:hover { background:rgba(0,230,255,0.15) !important; border-color:#00e6ff !important; }
    [data-testid="stSidebarCollapseButton"] svg { fill:#00e6ff !important; }

    /* ── Top Banner ── */
    .top-bar {
        display:flex; align-items:center; justify-content:space-between;
        padding:14px 28px; background:rgba(4,11,20,0.95);
        border-bottom:1px solid var(--border-dim); margin-bottom:32px; position:relative;
    }
    .top-bar::after {
        content:''; position:absolute; bottom:0; left:0; width:100%; height:2px;
        background:linear-gradient(90deg,transparent,var(--cyan),transparent);
        animation:scanline 3s ease-in-out infinite;
    }
    @keyframes scanline { 0%,100%{opacity:.3} 50%{opacity:1} }
    .brand { font-family:'Share Tech Mono',monospace; font-size:22px; color:var(--cyan); letter-spacing:4px; }
    .brand span { color:var(--amber); }
    .status-pill {
        font-family:'Share Tech Mono',monospace; font-size:11px; letter-spacing:2px; color:var(--green);
        border:1px solid rgba(0,255,136,0.35); padding:5px 14px; border-radius:2px;
        background:rgba(0,255,136,0.06); animation:blink 2.5s step-end infinite;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.5} }

    /* ── Method Badge ── */
    .method-badge {
        display:inline-block; font-family:'Share Tech Mono',monospace; font-size:11px;
        letter-spacing:2px; color:var(--amber); background:rgba(245,166,35,0.08);
        border:1px solid rgba(245,166,35,0.3); border-radius:2px; padding:4px 12px; margin-top:10px;
    }

    /* ── Section Header ── */
    .section-header {
        font-family:'Rajdhani',sans-serif; font-size:13px; font-weight:600;
        letter-spacing:4px; text-transform:uppercase; color:var(--text-secondary);
        margin-bottom:6px; display:flex; align-items:center; gap:8px;
    }
    .section-header::before { content:''; display:inline-block; width:20px; height:1px; background:var(--cyan); }

    /* ── HUD Card ── */
    .hud-card {
        background:var(--bg-card); border:1px solid var(--border-dim);
        border-radius:4px; padding:22px; position:relative; overflow:hidden; margin-bottom:4px;
    }
    .hud-card::before { content:''; position:absolute; top:0; left:0; width:60px; height:2px; background:var(--cyan); }
    .hud-card::after  { content:''; position:absolute; bottom:0; right:0; width:60px; height:2px; background:var(--amber); }
    .hud-card-title {
        font-family:'Share Tech Mono',monospace; font-size:12px; letter-spacing:3px;
        color:var(--text-secondary); text-transform:uppercase;
        margin-bottom:16px; padding-bottom:10px; border-bottom:1px solid var(--border-dim);
    }

    /* ── Upload Zone ── */
    .upload-zone {
        border:1px dashed rgba(0,230,255,0.3); border-radius:4px; padding:36px 20px;
        text-align:center; background:rgba(0,230,255,0.02); margin-bottom:16px; transition:all 0.3s;
    }
    .upload-zone:hover { border-color:var(--cyan); background:rgba(0,230,255,0.04); }
    .upload-icon  { font-size:40px; margin-bottom:8px; display:block; }
    .upload-label { font-family:'Exo 2',sans-serif; font-size:15px; color:var(--text-secondary); letter-spacing:1px; }
    .upload-sub   { font-family:'Share Tech Mono',monospace; font-size:11px; color:rgba(127,168,192,0.5); margin-top:5px; }

    /* ── Result Plate ── */
    .plate-display {
        background:linear-gradient(135deg,#0d1f33,#071220); border:1px solid var(--border-glow);
        border-radius:4px; padding:28px 20px; text-align:center;
        position:relative; overflow:hidden; margin-top:18px;
        box-shadow:0 0 30px rgba(0,230,255,0.08),inset 0 0 40px rgba(0,230,255,0.03);
    }
    .plate-label { font-family:'Share Tech Mono',monospace; font-size:10px; letter-spacing:4px; color:var(--text-secondary); text-transform:uppercase; margin-bottom:14px; }
    .plate-text  { font-family:'Share Tech Mono',monospace; font-size:44px; font-weight:700; color:var(--cyan); letter-spacing:8px; text-shadow:0 0 20px rgba(0,230,255,.5),0 0 60px rgba(0,230,255,.15); word-break:break-all; }
    .plate-corner{ position:absolute; width:12px; height:12px; border-color:var(--cyan); border-style:solid; }
    .plate-corner.tl{top:8px;left:8px;border-width:1px 0 0 1px}
    .plate-corner.tr{top:8px;right:8px;border-width:1px 1px 0 0}
    .plate-corner.bl{bottom:8px;left:8px;border-width:0 0 1px 1px}
    .plate-corner.br{bottom:8px;right:8px;border-width:0 1px 1px 0}

    /* ── Stats Row ── */
    .stats-row { display:flex; gap:12px; margin-top:18px; }
    .stat-chip  { flex:1; background:rgba(0,230,255,0.04); border:1px solid var(--border-dim); border-radius:3px; padding:12px 10px; text-align:center; }
    .stat-chip .val { font-family:'Share Tech Mono',monospace; font-size:18px; color:var(--cyan); display:block; }
    .stat-chip .key { font-size:10px; letter-spacing:2px; color:var(--text-secondary); text-transform:uppercase; margin-top:4px; display:block; }

    /* ── Alert Banners ── */
    .alert-success { background:rgba(0,255,136,0.06); border-left:3px solid var(--green); border-radius:2px; padding:14px 18px; font-family:'Share Tech Mono',monospace; font-size:13px; letter-spacing:1px; color:var(--green); margin-top:14px; }
    .alert-error   { background:rgba(255,71,87,0.06); border-left:3px solid var(--red); border-radius:2px; padding:14px 18px; font-family:'Share Tech Mono',monospace; font-size:13px; letter-spacing:1px; color:var(--red); margin-top:14px; }
    .alert-warn    { background:rgba(245,166,35,0.06); border-left:3px solid var(--amber); border-radius:2px; padding:14px 18px; font-family:'Share Tech Mono',monospace; font-size:13px; letter-spacing:1px; color:var(--amber); margin-top:14px; }

    /* ── Tips Box ── */
    .tips-box {
        background:rgba(245,166,35,0.04); border:1px solid rgba(245,166,35,0.15);
        border-radius:4px; padding:16px 20px; margin-top:16px;
    }
    .tips-box-title { font-family:'Share Tech Mono',monospace; font-size:11px; letter-spacing:2px; color:var(--amber); margin-bottom:10px; }
    .tips-box li    { font-family:'Exo 2',sans-serif; font-size:13px; color:var(--text-secondary); margin:5px 0; }

    /* ── HUD Divider ── */
    .hud-divider { height:1px; background:linear-gradient(90deg,transparent,var(--border-glow),transparent); margin:28px 0; }

    /* ── Developer Card ── */
    .dev-card {
        background:linear-gradient(135deg,#071a2e,#0a1929); border:1px solid rgba(0,230,255,0.2);
        border-radius:6px; padding:20px; position:relative; overflow:hidden; margin-top:10px;
    }
    .dev-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,var(--cyan),var(--amber)); }
    .dev-avatar { width:56px; height:56px; border-radius:50%; background:linear-gradient(135deg,rgba(0,230,255,0.1),rgba(245,166,35,0.1)); border:1.5px solid rgba(0,230,255,0.4); margin:0 auto 10px; text-align:center; font-size:24px; line-height:56px; }
    .dev-name { font-family:'Rajdhani',sans-serif; font-size:18px; font-weight:700; color:var(--text-primary); text-align:center; letter-spacing:2px; }
    .dev-role { font-family:'Share Tech Mono',monospace; font-size:10px; color:var(--cyan); text-align:center; letter-spacing:3px; text-transform:uppercase; margin-bottom:14px; opacity:0.8; }
    .dev-links { display:flex; flex-direction:column; gap:7px; }
    .dev-link  { display:flex; align-items:center; gap:10px; padding:9px 12px; background:rgba(0,230,255,0.04); border:1px solid rgba(0,230,255,0.1); border-radius:3px; text-decoration:none !important; transition:all 0.2s; }
    .dev-link:hover { background:rgba(0,230,255,0.1); border-color:rgba(0,230,255,0.4); transform:translateX(3px); }
    .dev-link-icon  { font-size:15px; min-width:20px; }
    .dev-link-label { font-family:'Share Tech Mono',monospace; font-size:10px; letter-spacing:1px; color:var(--text-secondary); }
    .dev-link-val   { font-family:'Exo 2',sans-serif; font-size:12px; color:var(--cyan); word-break:break-all; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] { background-color:var(--bg-panel) !important; border-right:1px solid var(--border-dim) !important; }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li { font-family:'Exo 2',sans-serif; color:var(--text-secondary); font-size:13px; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { font-family:'Rajdhani',sans-serif !important; color:var(--text-primary) !important; letter-spacing:2px !important; }

    .sidebar-badge { display:inline-block; font-family:'Share Tech Mono',monospace; font-size:10px; letter-spacing:2px; color:var(--green); background:rgba(0,255,136,0.07); border:1px solid rgba(0,255,136,0.25); border-radius:2px; padding:4px 10px; margin-bottom:14px; }
    .tech-tag      { display:inline-block; font-family:'Share Tech Mono',monospace; font-size:11px; color:var(--cyan-dim); background:rgba(0,230,255,0.06); border:1px solid var(--border-dim); border-radius:2px; padding:3px 8px; margin:3px 2px; }

    .stImage figcaption { font-family:'Share Tech Mono',monospace !important; font-size:11px !important; letter-spacing:2px !important; color:var(--text-secondary) !important; text-transform:uppercase !important; text-align:center !important; }
    h2,h3 { font-family:'Rajdhani',sans-serif !important; letter-spacing:2px !important; color:var(--text-primary) !important; }

    .stButton > button { background:rgba(0,230,255,0.07) !important; border:1px solid rgba(0,230,255,0.3) !important; color:#00e6ff !important; font-family:'Share Tech Mono',monospace !important; font-size:12px !important; letter-spacing:2px !important; border-radius:3px !important; transition:all 0.2s !important; }
    .stButton > button:hover { background:rgba(0,230,255,0.15) !important; border-color:#00e6ff !important; box-shadow:0 0 12px rgba(0,230,255,0.2) !important; }

    .stTabs [data-baseweb="tab-list"] { background:var(--bg-card) !important; border-bottom:1px solid var(--border-dim) !important; gap:4px; }
    .stTabs [data-baseweb="tab"]      { font-family:'Share Tech Mono',monospace !important; font-size:11px !important; letter-spacing:2px !important; color:var(--text-secondary) !important; background:transparent !important; border:none !important; padding:10px 18px !important; }
    .stTabs [aria-selected="true"]    { color:var(--cyan) !important; border-bottom:2px solid var(--cyan) !important; }

    /* ── Slider ── */
    .stSlider label { font-family:'Share Tech Mono',monospace !important; font-size:11px !important; letter-spacing:2px !important; color:var(--text-secondary) !important; text-transform:uppercase !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------
# TOP BAR
# -----------------------------------
st.markdown(
    """
    <div class='top-bar'>
        <div class='brand'>
            ANPR<span>·</span>SYS
            <span style='font-size:13px;color:#7fa8c0;letter-spacing:2px;'>
                &nbsp; PLATE INTELLIGENCE v3.0
            </span>
        </div>
        <div class='status-pill'>● SYSTEM ONLINE</div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------
# SESSION STATE
# -----------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------------
# SIDEBAR
# -----------------------------------
with st.sidebar:

    st.markdown(
        "<div style='font-family:Share Tech Mono,monospace;font-size:18px;"
        "color:#00e6ff;letter-spacing:4px;padding:8px 0 4px 0;'>"
        "CONTROL PANEL</div>",
        unsafe_allow_html=True
    )
    st.markdown("<div class='sidebar-badge'>● OPERATIONAL</div>", unsafe_allow_html=True)

    st.markdown("<div class='hud-divider'></div>", unsafe_allow_html=True)

    # ── Detection Settings ──
    st.markdown(
        "<div style='font-family:Rajdhani,sans-serif;font-size:13px;"
        "letter-spacing:3px;color:#7fa8c0;text-transform:uppercase;"
        "margin-bottom:12px;'>Detection Settings</div>",
        unsafe_allow_html=True
    )
    resize_width = st.slider("Image Width (px)", 400, 1200, 800, 100)
    show_pipeline = st.checkbox("Show pipeline stages", value=True)
    show_ocr_input = st.checkbox("Show OCR input image", value=True)

    st.markdown("<div class='hud-divider'></div>", unsafe_allow_html=True)

    # ── Strategy Info ──
    st.markdown(
        """
        <div style='font-family:Rajdhani,sans-serif;font-size:13px;
        letter-spacing:3px;color:#7fa8c0;text-transform:uppercase;margin-bottom:10px;'>
        Detection Strategies</div>
        """,
        unsafe_allow_html=True
    )
    for num, label, desc in [
        ("S1", "CLAHE + Auto-Canny",    "Best for well-lit images"),
        ("S2", "Adaptive Threshold",    "Handles shadows / uneven light"),
        ("S3", "Morphological Close",   "Low-res or noisy images"),
    ]:
        st.markdown(
            f"""
            <div style='display:flex;align-items:flex-start;gap:10px;
            padding:8px 0;border-bottom:1px solid rgba(0,230,255,0.06);'>
                <span style='font-family:Share Tech Mono,monospace;font-size:10px;
                color:#f5a623;min-width:24px;padding-top:1px;'>{num}</span>
                <div>
                    <div style='font-family:Share Tech Mono,monospace;font-size:11px;
                    color:#00e6ff;'>{label}</div>
                    <div style='font-family:Exo 2,sans-serif;font-size:11px;
                    color:#7fa8c0;margin-top:2px;'>{desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div class='hud-divider'></div>", unsafe_allow_html=True)

    # ── Tech Stack ──
    st.markdown(
        """
        <div style='font-family:Rajdhani,sans-serif;font-size:13px;
        letter-spacing:3px;color:#7fa8c0;text-transform:uppercase;margin-bottom:10px;'>
        Stack</div>
        <span class='tech-tag'>Python</span>
        <span class='tech-tag'>OpenCV</span>
        <span class='tech-tag'>Tesseract</span>
        <span class='tech-tag'>CLAHE</span>
        <span class='tech-tag'>NumPy</span>
        <span class='tech-tag'>Streamlit</span>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='hud-divider'></div>", unsafe_allow_html=True)

    # ── Developer Card ──
    st.markdown(
        """
        <div style='font-family:Rajdhani,sans-serif;font-size:13px;
        letter-spacing:3px;color:#7fa8c0;text-transform:uppercase;margin-bottom:4px;'>
        Developer</div>

        <div class='dev-card'>
            <div class='dev-avatar'>👨‍💻</div>
            <div class='dev-name'>Parth Singh</div>
            <div class='dev-role'>Engineer · AI Developer</div>
            <div class='dev-links'>
                <a class='dev-link' href='mailto:parth.si2007@gmail.com'>
                    <span class='dev-link-icon'>✉️</span>
                    <div>
                        <div class='dev-link-label'>EMAIL</div>
                        <div class='dev-link-val'>parth.si2007@gmail.com</div>
                    </div>
                </a>
                <a class='dev-link'
                   href='https://www.linkedin.com/in/parth-singh-70a081225/'
                   target='_blank'>
                    <span class='dev-link-icon'>🔗</span>
                    <div>
                        <div class='dev-link-label'>LINKEDIN</div>
                        <div class='dev-link-val'>parth-singh-70a081225</div>
                    </div>
                </a>
                <a class='dev-link'
                   href='https://github.com/Parth-Singhh'
                   target='_blank'>
                    <span class='dev-link-icon'>🐙</span>
                    <div>
                        <div class='dev-link-label'>GITHUB</div>
                        <div class='dev-link-val'>Parth-Singhh</div>
                    </div>
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='hud-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:Share Tech Mono,monospace;font-size:10px;"
        "letter-spacing:2px;color:rgba(127,168,192,0.35);text-align:center;'>"
        "SUPPORTED · JPG · JPEG · PNG</div>",
        unsafe_allow_html=True
    )

# -----------------------------------
# UPLOAD SECTION
# -----------------------------------
st.markdown(
    """
    <div class='section-header'>Input Feed</div>
    <div class='upload-zone'>
        <span class='upload-icon'>📡</span>
        <div class='upload-label'>Drop vehicle image or click to browse</div>
        <div class='upload-sub'>JPG · JPEG · PNG &nbsp;|&nbsp; MAX 200 MB</div>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

# -----------------------------------
# PROCESS IMAGE
# -----------------------------------
if uploaded_file is not None:

    with st.spinner("Analysing feed — running multi-strategy detection..."):

        # Load and resize
        pil_img = Image.open(uploaded_file)
        image   = np.array(pil_img)
        if image.shape[2] == 4:                          # RGBA → BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image   = imutils.resize(image, width=resize_width)
        h_img, w_img = image.shape[:2]

        # ── Run detection ──
        result = detect_plate(image)

        # Rebuild pipeline views for display (re-derived from same image)
        gray_raw  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_cl   = clahe_obj.apply(gray_raw)
        blurred   = cv2.bilateralFilter(gray_cl, 11, 17, 17)
        edged_vis = cv2.Canny(blurred,
                              int(max(0, (1 - 0.33) * float(np.median(blurred)))),
                              int(min(255, (1 + 0.33) * float(np.median(blurred)))))

    # ── Interactive Tabs ──
    tab1, tab2 = st.tabs(["  DETECTION  ", "  PREPROCESSING PIPELINE  "])

    # ─────────────────────────────────────
    # TAB 1 — DETECTION
    # ─────────────────────────────────────
    with tab1:
        col1, col2 = st.columns([1, 1], gap="large")

        # Original image
        with col1:
            st.markdown(
                "<div class='hud-card'><div class='hud-card-title'>📷 &nbsp; RAW VEHICLE FEED</div>",
                unsafe_allow_html=True
            )
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(
                f"""
                <div class='stats-row'>
                    <div class='stat-chip'><span class='val'>{w_img}px</span><span class='key'>Width</span></div>
                    <div class='stat-chip'><span class='val'>{h_img}px</span><span class='key'>Height</span></div>
                    <div class='stat-chip'><span class='val'>RGB</span><span class='key'>Mode</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Detection result
        with col2:
            st.markdown(
                "<div class='hud-card'><div class='hud-card-title'>🎯 &nbsp; DETECTION RESULT</div>",
                unsafe_allow_html=True
            )

            if not result["success"]:
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                    caption="NO PLATE REGION ISOLATED"
                )
                st.markdown(
                    "<div class='alert-error'>✕ &nbsp; ALL THREE STRATEGIES FAILED — SEE TIPS BELOW</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    """
                    <div class='tips-box'>
                        <div class='tips-box-title'>⚠ TROUBLESHOOTING</div>
                        <ul>
                            <li>Ensure the number plate is clearly visible and not at a sharp angle</li>
                            <li>Try a higher resolution image (above 600px wide)</li>
                            <li>Avoid heavy motion blur or glare on the plate</li>
                            <li>Crop the image closer to the vehicle if possible</li>
                            <li>Plates at &gt;30° angle require perspective correction (not yet supported)</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                plate_text    = result["plate_text"]
                char_count    = len(plate_text)
                x, y, w, h    = cv2.boundingRect(result["plate_contour"])
                detected_text = plate_text if plate_text else "PLATE FOUND — OCR EMPTY"

                # Show cropped region
                if show_ocr_input and result["cropped_ocr"] is not None:
                    st.image(result["cropped_ocr"], caption="OCR INPUT (PREPROCESSED)", use_container_width=True)
                elif result["cropped_raw"] is not None:
                    st.image(result["cropped_raw"], caption="ISOLATED PLATE REGION", use_container_width=True)

                # Strategy badge
                st.markdown(
                    f"<div class='method-badge'>⚙ STRATEGY: {result['method'].upper()}</div>",
                    unsafe_allow_html=True
                )

                # Plate display
                st.markdown(
                    f"""
                    <div class='plate-display'>
                        <div class='plate-corner tl'></div>
                        <div class='plate-corner tr'></div>
                        <div class='plate-corner bl'></div>
                        <div class='plate-corner br'></div>
                        <div class='plate-label'>DETECTED PLATE NUMBER</div>
                        <div class='plate-text'>{detected_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""
                    <div class='stats-row'>
                        <div class='stat-chip'><span class='val'>{char_count}</span><span class='key'>Characters</span></div>
                        <div class='stat-chip'><span class='val'>{w}×{h}</span><span class='key'>Plate (px)</span></div>
                        <div class='stat-chip'><span class='val'>PSM 7/8</span><span class='key'>OCR Mode</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if plate_text:
                    st.markdown(
                        "<div class='alert-success'>✔ &nbsp; PLATE DETECTED &amp; TEXT RECOGNISED SUCCESSFULLY</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<div class='alert-warn'>⚠ &nbsp; PLATE REGION FOUND BUT OCR RETURNED EMPTY — TRY A CLEARER IMAGE</div>",
                        unsafe_allow_html=True
                    )

                # Add to scan history
                st.session_state.history.append({
                    "file":     uploaded_file.name,
                    "plate":    detected_text,
                    "chars":    char_count,
                    "size":     f"{w}×{h}",
                    "method":   result["method"]
                })

            st.markdown("</div>", unsafe_allow_html=True)

        # Annotated output
        if result["success"]:
            st.markdown("<div class='hud-divider'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>Annotated Output Frame</div>", unsafe_allow_html=True)
            st.markdown("<div class='hud-card'>", unsafe_allow_html=True)
            st.image(
                cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption="BOUNDING CONTOUR + PLATE TEXT OVERLAID"
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────
    # TAB 2 — PREPROCESSING PIPELINE
    # ─────────────────────────────────────
    with tab2:
        if show_pipeline:
            st.markdown(
                "<div class='section-header'>Visual Pipeline Stages</div>",
                unsafe_allow_html=True
            )
            pc1, pc2, pc3, pc4 = st.columns(4, gap="medium")
            for col, img_data, title, caption in [
                (pc1, gray_raw,  "⬜ GRAYSCALE",        "CHANNEL REDUCTION"),
                (pc2, gray_cl,   "🔆 CLAHE EQUALISED",  "CONTRAST NORMALISED"),
                (pc3, blurred,   "〰️ BILATERAL FILTER", "NOISE REDUCTION"),
                (pc4, edged_vis, "⚡ AUTO-CANNY EDGES",  "EDGE DETECTION"),
            ]:
                with col:
                    st.markdown(
                        f"<div class='hud-card'><div class='hud-card-title'>{title}</div>",
                        unsafe_allow_html=True
                    )
                    st.image(img_data, use_container_width=True, caption=caption)
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='font-family:Share Tech Mono,monospace;font-size:12px;"
                "color:#7fa8c0;letter-spacing:2px;padding:30px;text-align:center;'>"
                "PIPELINE DISPLAY DISABLED — ENABLE IN SIDEBAR</div>",
                unsafe_allow_html=True
            )

# -----------------------------------
# SCAN HISTORY
# -----------------------------------
if st.session_state.history:

    st.markdown("<div class='hud-divider'></div>", unsafe_allow_html=True)

    h_col, b_col = st.columns([5, 1])
    with h_col:
        st.markdown("<div class='section-header'>Scan History</div>", unsafe_allow_html=True)
    with b_col:
        if st.button("CLEAR HISTORY"):
            st.session_state.history = []
            st.rerun()

    for i, rec in enumerate(reversed(st.session_state.history[-10:])):
        idx = len(st.session_state.history) - i
        method_short = rec.get("method", "—").split("+")[0].strip()[:12]
        st.markdown(
            f"""
            <div style='display:flex;align-items:center;justify-content:space-between;
            padding:10px 16px;background:rgba(0,230,255,0.03);
            border:1px solid rgba(0,230,255,0.08);border-radius:3px;margin-bottom:6px;'>
                <span style='font-family:Share Tech Mono,monospace;font-size:11px;color:#7fa8c0;min-width:30px;'>#{idx:02d}</span>
                <span style='font-family:Exo 2,sans-serif;font-size:12px;color:#7fa8c0;flex:1;padding:0 10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{rec['file']}</span>
                <span style='font-family:Share Tech Mono,monospace;font-size:18px;color:#00e6ff;letter-spacing:4px;flex:1;text-align:center;text-shadow:0 0 10px rgba(0,230,255,0.4);'>{rec['plate']}</span>
                <span style='font-family:Share Tech Mono,monospace;font-size:10px;color:#f5a623;padding:0 10px;white-space:nowrap;'>{method_short}</span>
                <span style='font-family:Share Tech Mono,monospace;font-size:11px;color:#7fa8c0;white-space:nowrap;'>{rec['size']} px</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("<div class='hud-divider'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align:center;padding:8px 0 22px;'>
        <span style='font-family:Share Tech Mono,monospace;font-size:11px;
        letter-spacing:3px;color:rgba(127,168,192,0.35);'>
        BUILT BY &nbsp;
        <span style='color:rgba(0,230,255,0.55);letter-spacing:3px;'>PARTH SINGH</span>
        &nbsp;·&nbsp; PYTHON &nbsp;·&nbsp; OPENCV &nbsp;·&nbsp;
        TESSERACT &nbsp;·&nbsp; STREAMLIT
        </span>
    </div>
    """,
    unsafe_allow_html=True
)