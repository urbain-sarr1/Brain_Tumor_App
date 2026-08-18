"""
Interface Streamlit de détection, classification et segmentation
de tumeurs cérébrales à partir d'images IRM.

Architecture :
    Streamlit
        ↓
    API FastAPI /predict
        ↓
    YOLO11m-seg
        ↓
    Détection + classification + segmentation
        ↓
    Indicateurs géométriques
        ↓
    Résultats affichés dans Streamlit

IMPORTANT :
    Cette application est un outil d'aide à la décision.
    Elle ne constitue pas un diagnostic médical.
"""

import base64
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "https://brain-tumor-app-tesg.onrender.com"

MAX_FILE_SIZE_MB = 10

ALLOWED_TYPES = [
    "jpg",
    "jpeg",
    "png"
]


# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Détection des tumeurs cérébrales",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .warning-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        margin-bottom: 1.5rem;
    }

    .success-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin-bottom: 1rem;
    }

    .danger-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITRE
# ============================================================

st.markdown(
    '<div class="main-title">🧠 Détection des tumeurs cérébrales</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Détection, classification et segmentation automatique
        à partir d'images IRM.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# AVERTISSEMENT MÉDICAL
# ============================================================

st.markdown(
    """
    <div class="warning-box">
        <strong>⚠️ Avertissement médical</strong><br>
        Cette application est un outil d'aide à la décision.
        Elle ne remplace pas l'interprétation d'un professionnel
        de santé qualifié.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.write("**API FastAPI**")

    st.code(
        API_URL,
        language="text"
    )

    st.divider()

    st.write("**Modèle**")

    st.write("YOLO11m-seg")

    st.write(
        "Détection + classification + segmentation"
    )

    st.divider()

    st.write("**Formats acceptés**")

    st.write("JPEG / JPG / PNG")

    st.write(
        f"Taille maximale : {MAX_FILE_SIZE_MB} Mo"
    )

    st.divider()

    # --------------------------------------------------------
    # Vérification API
    # --------------------------------------------------------

    st.subheader("🔌 État de l'API")

    try:

        health_response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        if health_response.status_code == 200:

            health = health_response.json()

            if health.get("model_loaded"):

                st.success(
                    "API disponible\n\n"
                    "Modèle chargé"
                )

            else:

                st.warning(
                    "API disponible\n\n"
                    "Modèle non chargé"
                )

        else:

            st.error(
                f"API indisponible "
                f"({health_response.status_code})"
            )

    except requests.exceptions.RequestException:

        st.error(
            "Impossible de contacter l'API"
        )


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.subheader("📤 Importer une image IRM")

uploaded_file = st.file_uploader(
    "Sélectionnez une image IRM",
    type=ALLOWED_TYPES,
    help=(
        "Formats acceptés : JPG, JPEG et PNG. "
        f"Taille maximale : {MAX_FILE_SIZE_MB} Mo."
    )
)


# ============================================================
# SI IMAGE FOURNIE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Vérification taille
    # --------------------------------------------------------

    file_size_mb = (
        uploaded_file.size
        / (1024 * 1024)
    )

    if file_size_mb > MAX_FILE_SIZE_MB:

        st.error(
            f"❌ Image trop volumineuse : "
            f"{file_size_mb:.2f} Mo. "
            f"Maximum autorisé : "
            f"{MAX_FILE_SIZE_MB} Mo."
        )

        st.stop()

    # --------------------------------------------------------
    # Lecture image
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded_file
        )

        image.load()

    except Exception:

        st.error(
            "❌ Impossible de lire cette image."
        )

        st.stop()

    # --------------------------------------------------------
    # Affichage image originale
    # --------------------------------------------------------

    st.subheader("🖼️ Image IRM sélectionnée")

    col1, col2 = st.columns(
        [2, 1]
    )

    with col1:

        st.image(
            image,
            caption="Image IRM originale",
            use_column_width=True
        )

    with col2:

        st.info(
            f"""
            **Nom :** {uploaded_file.name}

            **Format :** {uploaded_file.type}

            **Taille :** {file_size_mb:.2f} Mo

            **Dimensions :**
            {image.width} × {image.height} px
            """
        )

    st.divider()

    # ========================================================
    # BOUTON ANALYSE
    # ========================================================

    if st.button(
        "🔍 Analyser l'image",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # Préparation du fichier
        # ----------------------------------------------------

        file_bytes = uploaded_file.getvalue()

        files = {
            "file": (
                uploaded_file.name,
                file_bytes,
                uploaded_file.type
            )
        }

        # ----------------------------------------------------
        # Appel API
        # ----------------------------------------------------

        with st.spinner(
            "🧠 Analyse de l'image par le modèle YOLO11m-seg..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=120
                )

            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ L'analyse a dépassé le délai d'attente."
                )

                st.stop()

            except requests.exceptions.ConnectionError:

                st.error(
                    "🔌 Impossible de contacter l'API FastAPI."
                )

                st.info(
                    f"URL utilisée : {API_URL}"
                )

                st.stop()

            except requests.exceptions.RequestException as exc:

                st.error(
                    f"❌ Erreur de communication avec l'API : {exc}"
                )

                st.stop()

        # ----------------------------------------------------
        # Vérification réponse
        # ----------------------------------------------------

        if response.status_code != 200:

            try:

                error_data = response.json()

                detail = error_data.get(
                    "detail",
                    "Erreur inconnue"
                )

            except Exception:

                detail = response.text

            st.error(
                f"❌ Erreur API "
                f"({response.status_code}) : {detail}"
            )

            st.stop()

        # ----------------------------------------------------
        # Lecture JSON
        # ----------------------------------------------------

        try:

            result = response.json()

        except Exception:

            st.error(
                "❌ La réponse de l'API n'est pas un JSON valide."
            )

            st.stop()

        # ====================================================
        # RÉSULTATS
        # ====================================================

        st.divider()

        st.header("📊 Résultats de l'analyse")

        # ----------------------------------------------------
        # Avertissement API
        # ----------------------------------------------------

        if result.get("avertissement"):

            st.warning(
                result["avertissement"]
            )

        # ----------------------------------------------------
        # AUCUNE TUMEUR
        # ----------------------------------------------------

        if not result.get(
            "tumeur_detectee",
            False
        ):

            st.success(
                "✅ Aucune tumeur détectée."
            )

            st.metric(
                "Nombre de tumeurs détectées",
                0
            )

            image_base64 = result.get(
                "image_annotee_base64"
            )

            if image_base64:

                try:

                    image_bytes = base64.b64decode(
                        image_base64
                    )

                    result_image = Image.open(
                        BytesIO(image_bytes)
                    )

                    st.image(
                        result_image,
                        caption="Résultat de l'analyse",
                        use_column_width=True
                    )

                except Exception:

                    st.warning(
                        "Impossible d'afficher l'image retournée par l'API."
                    )

            st.info(
                f"Temps de traitement : "
                f"{result.get('temps_traitement_s', 0)} s"
            )

            st.stop()

        # ====================================================
        # AU MOINS UNE TUMEUR
        # ====================================================

        nombre_tumeurs = result.get(
            "nombre_tumeurs",
            0
        )

        best = result.get(
            "meilleure_detection"
        )

        detections = result.get(
            "detections",
            []
        )

        # ----------------------------------------------------
        # Résumé
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="danger-box">
                <strong>⚠️ Tumeur(s) détectée(s)</strong><br>
                Le modèle a identifié {nombre_tumeurs}
                zone(s) correspondant à une tumeur.
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # INDICATEURS PRINCIPAUX
        # ----------------------------------------------------

        if best is not None:

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Classe",
                    best.get(
                        "classe",
                        "Inconnue"
                    )
                )

            with col2:

                confidence = (
                    best.get(
                        "confiance",
                        0
                    ) * 100
                )

                st.metric(
                    "Confiance",
                    f"{confidence:.1f}%"
                )

            with col3:

                st.metric(
                    "Zones détectées",
                    nombre_tumeurs
                )

            with col4:

                st.metric(
                    "Temps",
                    f"{result.get('temps_traitement_s', 0)} s"
                )

        # ====================================================
        # IMAGE ANNOTÉE
        # ====================================================

        st.subheader(
            "🧠 Détection et segmentation"
        )

        image_base64 = result.get(
            "image_annotee_base64"
        )

        if image_base64:

            try:

                image_bytes = base64.b64decode(
                    image_base64
                )

                annotated_image = Image.open(
                    BytesIO(image_bytes)
                )

                st.image(
                    annotated_image,
                    caption=(
                        "Résultat YOLO11m-seg : "
                        "détection et masque de segmentation"
                    ),
                    use_column_width=True
                )

            except Exception as exc:

                st.error(
                    f"Impossible d'afficher l'image annotée : {exc}"
                )

        # ====================================================
        # DÉTAILS DE LA MEILLEURE DÉTECTION
        # ====================================================

        if best is not None:

            st.subheader(
                "📐 Indicateurs géométriques"
            )

            indicators = best.get(
                "indicateurs",
                {}
            )

            if indicators:

                # --------------------------------------------
                # Centre / position
                # --------------------------------------------

                centre = indicators.get(
                    "centre",
                    {}
                )

                position = indicators.get(
                    "position_dans_image",
                    "inconnue"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Position",
                        position
                    )

                with col2:

                    st.metric(
                        "Centre X",
                        f"{centre.get('x', 0):.2f} px"
                    )

                with col3:

                    st.metric(
                        "Centre Y",
                        f"{centre.get('y', 0):.2f} px"
                    )

                # --------------------------------------------
                # Dimensions
                # --------------------------------------------

                dimensions = indicators.get(
                    "dimensions_px",
                    {}
                )

                st.markdown(
                    "#### Dimensions de la tumeur"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Largeur",
                        f"{dimensions.get('largeur', 0)} px"
                    )

                with col2:

                    st.metric(
                        "Hauteur",
                        f"{dimensions.get('hauteur', 0)} px"
                    )

                with col3:

                    st.metric(
                        "Ratio largeur / hauteur",
                        indicators.get(
                            "ratio_largeur_hauteur",
                            0
                        )
                    )

                # --------------------------------------------
                # Surface
                # --------------------------------------------

                st.markdown(
                    "#### Surface"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Surface du masque",
                        f"{indicators.get('surface_masque_px', 0):.2f} px²"
                    )

                with col2:

                    st.metric(
                        "Surface",
                        f"{indicators.get('surface_masque_mm2', 0):.2f} mm²"
                    )

                with col3:

                    st.metric(
                        "Occupation de l'image",
                        f"{indicators.get('occupation_image_pourcent', 0):.2f}%"
                    )

                # --------------------------------------------
                # Forme
                # --------------------------------------------

                st.markdown(
                    "#### Forme de la tumeur"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Périmètre",
                        f"{indicators.get('perimetre_px', 0):.2f} px"
                    )

                with col2:

                    st.metric(
                        "Diamètre maximal",
                        f"{indicators.get('diametre_max_px', 0):.2f} px"
                    )

                with col3:

                    st.metric(
                        "Circularité",
                        indicators.get(
                            "circularite",
                            0
                        )
                    )

                # --------------------------------------------
                # Localisation
                # --------------------------------------------

                st.markdown(
                    "#### Localisation"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Position dans l'image",
                        position
                    )

                with col2:

                    st.metric(
                        "Distance du centre",
                        f"{indicators.get('distance_centre_image_px', 0):.2f} px"
                    )

                # --------------------------------------------
                # Avertissement indicateurs
                # --------------------------------------------

                if indicators.get("avertissement"):

                    st.info(
                        indicators["avertissement"]
                    )

        # ====================================================
        # TOUTES LES DÉTECTIONS
        # ====================================================

        if len(detections) > 1:

            st.subheader(
                "🔎 Détail de toutes les détections"
            )

            for detection in detections:

                detection_id = detection.get(
                    "id",
                    "?"
                )

                classe = detection.get(
                    "classe",
                    "Inconnue"
                )

                confiance = (
                    detection.get(
                        "confiance",
                        0
                    ) * 100
                )

                with st.expander(
                    f"Détection #{detection_id} — "
                    f"{classe} — "
                    f"{confiance:.1f}%"
                ):

                    col1, col2 = st.columns(2)

                    with col1:

                        st.write(
                            "**Classe :**",
                            classe
                        )

                        st.write(
                            "**Confiance :**",
                            f"{confiance:.1f}%"
                        )

                    with col2:

                        bbox = detection.get(
                            "boite_englobante",
                            {}
                        )

                        st.write(
                            "**Bounding box :**"
                        )

                        st.json(
                            bbox
                        )

                    detection_indicators = detection.get(
                        "indicateurs",
                        {}
                    )

                    if detection_indicators:

                        st.write(
                            "**Indicateurs :**"
                        )

                        st.json(
                            detection_indicators
                        )

        # ====================================================
        # INFORMATIONS TECHNIQUES
        # ====================================================

        st.divider()

        with st.expander(
            "ℹ️ Informations techniques"
        ):

            st.write(
                "**Modèle :** YOLO11m-seg"
            )

            st.write(
                "**Architecture :** FastAPI + Streamlit"
            )

            st.write(
                f"**API :** {API_URL}"
            )

            st.write(
                f"**Seuil de confiance :** "
                f"{os.getenv('CONFIDENCE_THRESHOLD', '0.5')}"
            )

            st.write(
                "**Segmentation :** masque réel de la tumeur"
            )

            st.write(
                "**Indicateurs :** surface, périmètre, "
                "diamètre, centre, position, occupation, "
                "ratio largeur/hauteur et circularité"
            )

            st.write(
                "**Limite :** les mesures sont calculées "
                "sur une image 2D et ne permettent pas "
                "d'estimer le volume réel en 3D."
            )