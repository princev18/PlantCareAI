
import os
import json
from pathlib import Path
from html import escape

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# =========================================================
# 1. CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_v2.keras"
LABELS_PATH = BASE_DIR / "models" / "class_names_v2.json"
DATASET_DIR = BASE_DIR / "dataset" / "PlantVillage" / "PlantVillage"
ASSETS_DIR = BASE_DIR / "assets"

IMAGE_SIZE = (160, 160)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

if "page_nav" not in st.session_state:
    st.session_state["page_nav"] = "Home"

# =========================================================
# 2. PROFESSIONAL GREEN THEME + NAVBAR
# =========================================================
st.markdown(
    """
    <style>
    :root {
        --green: #16803c;
        --dark-green: #105c2b;
        --border: #dceade;
        --text: #193b28;
        --muted: #617467;
    }

    .stApp {
        background: #f7faf7;
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(247,250,247,0.98);
    }

    .block-container {
        max-width: 1400px;
        padding: 1.3rem 2rem 2.5rem;
    }

    /* NAVBAR BRAND */
    .navbar-brand {
        color: #105c2b !important;
        font-size: 22px !important;
        font-weight: 800 !important;
        line-height: 1.5 !important;
        white-space: nowrap !important;
        padding: 8px 0 !important;
        margin: 0 !important;
    }

    div[data-testid="stHorizontalBlock"]:has(.navbar-brand) {
        align-items: center !important;
        min-height: 76px;
        padding: 8px 0;
        gap: 10px !important;
    }

    div[data-testid="stHorizontalBlock"]:has(.navbar-brand)
    div[data-testid="stButton"] {
        width: 100%;
    }

    div[data-testid="stHorizontalBlock"]:has(.navbar-brand)
    div[data-testid="stButton"] button {
        min-height: 44px !important;
        height: auto !important;
        padding: 10px 7px !important;
        line-height: 1.4 !important;
        white-space: normal !important;
        overflow: visible !important;
        font-size: 13px !important;
        font-weight: 650 !important;
        border-radius: 10px !important;
    }

    /* GENERAL BUTTONS */
    div.stButton > button {
        border: 1px solid #d5e8d9;
        background: #ffffff;
        color: #105c2b;
        border-radius: 10px;
        font-weight: 600;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        background: #e5f4e9;
        color: #105c2b;
        border-color: #16803c;
    }

    div.stButton > button[kind="primary"] {
        background: #16803c;
        border-color: #16803c;
        color: #ffffff;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #105c2b;
        color: #ffffff;
    }

    /* HERO BANNER */
    .hero {
        background: linear-gradient(120deg, #105c2b, #279b50);
        padding: 36px;
        border-radius: 20px;
        color: #ffffff;
        margin: 18px 0 24px;
    }

    .hero .eyebrow {
        color: #d9f6e0;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.4px;
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: 35px !important;
        line-height: 1.3 !important;
        margin: 12px 0 !important;
    }

    .hero p {
        color: #e8f7ec !important;
        font-size: 15px;
        line-height: 1.8;
        margin-bottom: 0;
    }

    /* SECTION HEADINGS */
    .section-title {
        color: #105c2b;
        font-size: 25px;
        font-weight: 800;
        line-height: 1.5;
        margin: 24px 0 6px;
    }

    .muted {
        color: #617467;
        font-size: 14px;
        line-height: 1.7;
    }

    /* METRIC CARDS */
    .metric-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 20px;
        min-height: 130px;
        height: 100%;
        box-sizing: border-box;
    }

    .metric-label {
        color: #617467;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.3px;
    }

    .metric-value {
        color: #105c2b;
        font-size: 26px;
        font-weight: 800;
        margin: 10px 0 5px;
        overflow-wrap: anywhere;
    }

    /* PLANT GALLERY CARDS */
    .plant-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 0 0 16px 16px;
        padding: 16px;
        min-height: 150px;
        height: auto;
        box-sizing: border-box;
        margin-top: -8px;
        overflow-wrap: anywhere;
    }

    .plant-card h3 {
        color: #105c2b;
        font-size: 20px;
        margin: 4px 0 8px;
    }

    .plant-placeholder {
        width: 100%;
        height: 210px;
        border-radius: 14px 14px 0 0;
        background: linear-gradient(135deg, #e6f5e9, #c8e9d0);
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 65px;
        box-sizing: border-box;
    }

    /* Make gallery images consistent and remove gaps */
    div[data-testid="stImage"]:has(img[data-testid="stImage"]) {
        width: 100%;
    }

    div[data-testid="stImage"] img {
        border-radius: 14px 14px 0 0;
        object-fit: cover;
    }

    /* GENERAL PANELS */
    .panel {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 12px;
        box-sizing: border-box;
        overflow-wrap: anywhere;
    }

    .panel h3 {
        color: #105c2b;
    }

    .result-name {
        color: #105c2b;
        font-size: 21px;
        font-weight: 750;
        line-height: 1.5;
        overflow-wrap: anywhere;
    }

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #9bcaaa;
        border-radius: 14px;
        padding: 12px;
    }

    .footer {
        text-align: center;
        color: #718075;
        font-size: 12px;
        padding: 20px 0 8px;
        line-height: 1.8;
    }

    hr {
        border-color: #dceade;
    }

    @media (max-width: 900px) {
        .block-container {
            padding: 1rem;
        }

        .hero {
            padding: 24px;
        }

        .hero h1 {
            font-size: 28px !important;
        }

        .navbar-brand {
            font-size: 18px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 3. MODEL AND DATASET HELPERS
# =========================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def get_dataset_counts(dataset_path, labels_path):
    root = Path(dataset_path)

    with open(labels_path, "r", encoding="utf-8") as file:
        labels = json.load(file)

    counts = {}

    for class_name in labels:
        folder = root / class_name
        count = 0

        if folder.is_dir():
            for current_root, _, files in os.walk(folder):
                count += sum(
                    Path(filename).suffix.lower() in IMAGE_EXTENSIONS
                    for filename in files
                )

        counts[class_name] = count

    return counts


def display_name(name):
    return name.replace("___", " - ").replace("__", " ").replace("_", " ")


def is_healthy(name):
    return "healthy" in name.lower()


def page_change(name):
    st.session_state["page_nav"] = name
    st.rerun()


class_names = []
dataset_counts = {}
total_images = 0

if LABELS_PATH.exists():
    try:
        class_names = load_class_names()

        if DATASET_DIR.is_dir():
            dataset_counts = get_dataset_counts(
                str(DATASET_DIR), str(LABELS_PATH)
            )
            total_images = sum(dataset_counts.values())
    except Exception as error:
        st.warning(f"Dataset overview load nahi hua: {error}")

# =========================================================
# 4. NAVIGATION
# =========================================================
active_page = st.session_state["page_nav"]

brand, home, detect, library, about = st.columns(
    [1.65, 0.75, 1.25, 1.25, 1.15],
    gap="small",
    vertical_alignment="center",
)

with brand:
    st.markdown(
        '<div class="navbar-brand">🌿 PlantCare AI</div>',
        unsafe_allow_html=True,
    )

navigation = [
    (home, "Home", "nav_home"),
    (detect, "Detect Disease", "nav_detect"),
    (library, "Disease Library", "nav_library"),
    (about, "About Project", "nav_about"),
]

for column, label, key in navigation:
    with column:
        if st.button(
            label,
            key=key,
            type="primary" if active_page == label else "secondary",
            use_container_width=True,
        ):
            page_change(label)

page = st.session_state["page_nav"]
st.divider()

# =========================================================
# 5. HOME DASHBOARD
# =========================================================
if page == "Home":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">SMART AGRICULTURE · POWERED BY AI</div>
            <h1>PlantCare AI Dashboard 🌱</h1>
            <p>
                Explore plant health with deep learning. Upload a leaf image,
                review model predictions and explore insights from the
                PlantVillage dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3, gap="medium")

    metrics = [
        (
            col1,
            "🌿 SUPPORTED CLASSES",
            str(len(class_names)) if class_names else "15",
            "Pepper, Potato and Tomato",
        ),
        (
            col2,
            "🖼️ DATASET IMAGES",
            f"{total_images:,}" if total_images else "Unavailable",
            "Counted from local dataset folders",
        ),
        (
            col3,
            "🎯 VALIDATION ACCURACY",
            "94.21%",
            "Best recorded validation result",
        ),
    ]

    for column, label, value, description in metrics:
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="muted">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------------- PLANT GALLERY ----------------
    st.markdown(
        '<div class="section-title">🌱 Explore Plant Types</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="muted">Browse the three plant groups supported by this model.</p>',
        unsafe_allow_html=True,
    )

    plants = [
        (
            "Tomato",
            "tomato.jpg",
            "🍅",
            "Explore tomato disease and healthy categories.",
        ),
        (
            "Potato",
            "potato.jpg",
            "🥔",
            "Explore early blight, late blight and healthy leaves.",
        ),
        (
            "Pepper",
            "pepper.jpg",
            "🫑",
            "Explore bacterial spot and healthy pepper leaves.",
        ),
    ]

    plant_cols = st.columns(3, gap="medium")

    for column, (name, filename, emoji, description) in zip(plant_cols, plants):
        image_path = ASSETS_DIR / filename

        with column:
            # IMPORTANT: No outer HTML div around st.image().
            # This prevents the large empty white image boxes.
            if image_path.exists():
                with Image.open(image_path) as source:
                    picture = ImageOps.fit(
                        source.convert("RGB"),
                        (600, 350),
                        method=Image.Resampling.LANCZOS,
                    )

                st.image(
                    picture,
                    use_container_width=True,
                    caption=None,
                )
            else:
                st.markdown(
                    f'<div class="plant-placeholder">{emoji}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div class="plant-card">
                    <h3>{emoji} {escape(name)}</h3>
                    <p class="muted">{escape(description)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                f"Explore {name}",
                key=f"explore_{name.lower()}",
                use_container_width=True,
            ):
                page_change("Disease Library")

    # ---------------- DATASET GRAPHS ----------------
    st.markdown(
        '<div class="section-title">📊 Dataset Analytics</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="muted">Both charts use image counts from the configured local dataset.</p>',
        unsafe_allow_html=True,
    )

    if not DATASET_DIR.is_dir():
        st.warning(f"Dataset folder not found: {DATASET_DIR}")

    elif not dataset_counts:
        st.warning("Dataset counts unavailable. Check your dataset and labels.")

    else:
        healthy_count = sum(
            count
            for name, count in dataset_counts.items()
            if is_healthy(name)
        )
        disease_count = total_images - healthy_count

        st.caption(
            f"Dataset scan: {total_images:,} images · "
            f"{len(dataset_counts)} model classes"
        )

        graph1, graph2 = st.columns(2, gap="large")

        with graph1:
            st.markdown("#### 🌿 Healthy vs Disease Categories")

            comparison_df = pd.DataFrame({
                "Category": ["Healthy", "Disease category"],
                "Images": [healthy_count, disease_count],
            })

            st.bar_chart(
                comparison_df,
                x="Category",
                y="Images",
                color="#16803c",
                height=340,
            )

            st.markdown(
                f"""
                <div class="panel">
                    <b>Healthy category:</b> {healthy_count:,} images
                    <br><br>
                    <b>Disease categories:</b> {disease_count:,} images
                </div>
                """,
                unsafe_allow_html=True,
            )

        with graph2:
            st.markdown("#### 📈 Images Per Class")

            class_df = pd.DataFrame([
                {
                    "Class": display_name(name),
                    "Images": count,
                }
                for name, count in dataset_counts.items()
            ]).sort_values("Images", ascending=False)

            st.bar_chart(
                class_df,
                x="Images",
                y="Class",
                horizontal=True,
                color="#39a96b",
                height=540,
            )

        with st.expander("View exact dataset counts"):
            st.dataframe(
                class_df,
                hide_index=True,
                use_container_width=True,
            )

    # ---------------- WORKFLOW ----------------
    st.markdown(
        '<div class="section-title">⚙️ How It Works</div>',
        unsafe_allow_html=True,
    )

    work1, work2, work3 = st.columns(3, gap="medium")

    workflow = [
        (work1, "📷 Upload", "Upload a clear image of a plant leaf."),
        (work2, "🧠 Analyze", "MobileNetV2 estimates class probabilities."),
        (work3, "📋 Review", "Review the predicted class and top three results."),
    ]

    for column, heading, description in workflow:
        with column:
            st.markdown(
                f"""
                <div class="panel">
                    <h3>{heading}</h3>
                    <p class="muted">{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.button(
        "🔬 Start Disease Detection",
        key="start_detection",
        type="primary",
        use_container_width=True,
    ):
        page_change("Detect Disease")

    st.caption(
        "94.21% is the recorded validation accuracy. "
        "Real-world performance may differ."
    )

# =========================================================
# 6. DISEASE DETECTION
# =========================================================
elif page == "Detect Disease":

    st.markdown(
        '<div class="section-title">🔬 Plant Disease Detection</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="muted">Upload a clear Pepper, Potato or Tomato leaf image.</p>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        key="leaf_upload",
    )

    if uploaded_file is None:
        st.info("Upload an image to see the prediction.")

    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            left, right = st.columns([1, 1.1], gap="large")

            with left:
                st.markdown("#### 📷 Uploaded Leaf")
                st.image(image, use_container_width=True)
                st.caption(f"Image dimensions: {image.width} × {image.height}")

            with right:
                st.markdown("#### 🧪 Prediction")

                if not MODEL_PATH.exists():
                    st.error(f"Model file missing: {MODEL_PATH}")

                elif not LABELS_PATH.exists():
                    st.error(f"Class label file missing: {LABELS_PATH}")

                else:
                    with st.spinner("Analyzing leaf image..."):
                        model = load_model()
                        labels = load_class_names()

                        resized = image.resize(IMAGE_SIZE)
                        array = np.asarray(resized, dtype=np.float32)
                        array = np.expand_dims(array, axis=0)

                        # Model already includes MobileNetV2 input rescaling.
                        probabilities = model.predict(array, verbose=0)[0]

                    if len(probabilities) != len(labels):
                        st.error("Model output and label count do not match.")

                    else:
                        top_indices = np.argsort(probabilities)[-3:][::-1]
                        best_index = int(top_indices[0])
                        best_name = labels[best_index]
                        confidence = float(probabilities[best_index]) * 100

                        if is_healthy(best_name):
                            st.success("Model predicts a healthy category.")
                        else:
                            st.warning("Model predicts a possible disease category.")

                        st.markdown(
                            f'<div class="result-name">{escape(display_name(best_name))}</div>',
                            unsafe_allow_html=True,
                        )

                        st.metric("Prediction score", f"{confidence:.2f}%")
                        st.progress(float(np.clip(confidence / 100, 0, 1)))

                        if is_healthy(best_name):
                            st.write(
                                "The model predicts a healthy-leaf category. "
                                "Inspect the whole plant for other symptoms too."
                            )
                        else:
                            st.write(
                                "Compare the predicted category with visible symptoms. "
                                "Consult an agricultural expert before treatment."
                            )

                    st.divider()
                    st.markdown("#### 📊 Top 3 Predictions")

                    if len(probabilities) == len(labels):
                        result_cols = st.columns(3, gap="medium")

                        for rank, index in enumerate(top_indices):
                            index = int(index)
                            score = float(probabilities[index]) * 100
                            label = display_name(labels[index])

                            with result_cols[rank]:
                                st.markdown(
                                    f"""
                                    <div class="panel">
                                        <div class="metric-label">RANK {rank + 1}</div>
                                        <div class="result-name">{escape(label)}</div>
                                        <div class="metric-value">{score:.2f}%</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.progress(float(np.clip(score / 100, 0, 1)))

                    st.caption(
                        "These are model estimates, not a confirmed agricultural diagnosis."
                    )

        except Exception as error:
            st.error(f"Image process nahi ho payi: {error}")
            st.info("JPG ya PNG format ki clear leaf image try karo.")

# =========================================================
# 7. DISEASE LIBRARY
# =========================================================
elif page == "Disease Library":

    st.markdown(
        '<div class="section-title">📚 Disease Library</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="muted">Search all categories supported by your trained model.</p>',
        unsafe_allow_html=True,
    )

    if not LABELS_PATH.exists():
        st.error("Class labels file not found.")

    else:
        labels = load_class_names()

        query = st.text_input(
            "Search plant or category",
            placeholder="Try Tomato, Potato, Pepper or healthy...",
        )

        filtered = [
            name for name in labels
            if query.lower() in display_name(name).lower()
        ]

        st.caption(f"{len(filtered)} matching categories")

        if not filtered:
            st.info("No matching category found.")

        for start in range(0, len(filtered), 2):
            row_cols = st.columns(2, gap="medium")

            for column, name in zip(row_cols, filtered[start:start + 2]):
                category = (
                    "Healthy category"
                    if is_healthy(name)
                    else "Disease category"
                )

                with column:
                    st.markdown(
                        f"""
                        <div class="panel">
                            <h3>{escape(display_name(name))}</h3>
                            <p class="muted"><b>{category}</b></p>
                            <p class="muted">
                                This is a category recognized by the model.
                                A prediction alone does not confirm a diagnosis.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# =========================================================
# 8. ABOUT PROJECT
# =========================================================
elif page == "About Project":

    st.markdown(
        '<div class="section-title">ℹ️ About PlantCare AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">COMPUTER VISION · DEEP LEARNING</div>
            <h1>AI for Plant Health 🌱</h1>
            <p>
                An educational deep learning project that classifies plant
                leaf images into supported health and disease categories.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Project Objective")
    st.write(
        "Demonstrate how computer vision and deep learning can assist "
        "with preliminary classification of plant leaf images."
    )

    st.markdown("### Technology Stack")

    tech1, tech2, tech3 = st.columns(3, gap="medium")

    for column, title, description in [
        (tech1, "🐍 Python", "Application programming language"),
        (tech2, "🧠 TensorFlow", "Deep learning framework"),
        (tech3, "🌐 Streamlit", "Interactive web application"),
    ]:
        with column:
            st.markdown(
                f"""
                <div class="panel">
                    <h3>{title}</h3>
                    <p class="muted">{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Model Details")
    st.write("- Architecture: MobileNetV2 transfer learning")
    st.write("- Input size: 160 × 160 RGB")
    st.write(f"- Supported classes: {len(class_names) if class_names else 15}")
    st.write("- Recorded best validation accuracy: 94.21%")

    if total_images:
        st.write(f"- Dataset images counted locally: {total_images:,}")

    st.warning(
        "Validation accuracy does not guarantee performance on new images. "
        "This project is educational and does not replace expert advice."
    )

# =========================================================
# 9. FOOTER
# =========================================================
st.divider()

st.markdown(
    """
    <div class="footer">
        🌿 PlantCare AI · MobileNetV2 · TensorFlow · Streamlit<br>
        Educational project · Verify AI predictions independently
    </div>
    """,
    unsafe_allow_html=True,
)
