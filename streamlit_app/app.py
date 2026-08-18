"""
Interface Streamlit - Brain Tumor App

L'interface ne charge PAS le modèle YOLO.
Elle communique uniquement avec l'API FastAPI.

API attendue :
    POST /predict
"""

import os
import base64

import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

PREDICT_URL = f"{API_URL}/predict"

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
    page_title="Brain Tumor App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SUPPRESSION DE LA SIDEBAR
# ============================================================

st.markdown(
    """
    <style>

    /* Supprime complètement la sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Supprime l'espace réservé à la sidebar */
    [data-testid="stSidebarCollapsedControl"] {
        display: none;
    }

    /* Réduit légèrement les marges */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* Style des informations */
    .info-box {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
    }

    .warning-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #fff3cd;
        border: 1px solid #ffe69c;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITRE
# ============================================================

st.title("🧠 Détection et segmentation des tumeurs cérébrales")

st.markdown(
    """
    Cette application permet d'analyser une image IRM cérébrale
    à l'aide d'un modèle d'intelligence artificielle spécialisé
    dans la détection, la classification et la segmentation
    des tumeurs.
    """
)


# ============================================================
# AVERTISSEMENT
# ============================================================

st.warning(
    "⚠️ Cet outil constitue une aide à l'analyse et ne remplace "
    "pas l'interprétation d'un professionnel de santé."
)


# ============================================================
# VÉRIFICATION API
# ============================================================

with st.expander("🔧 État du service", expanded=False):

    st.write(
        f"**API :** `{API_URL}`"
    )

    if st.button(
        "Tester la connexion à l'API",
        use_container_width=True
    ):

        try:

            response = requests.get(
                f"{API_URL}/health",
                timeout=10
            )

            if response.status_code == 200:

                health = response.json()

                if health.get("model_loaded"):

                    st.success(
                        "✅ API disponible et modèle chargé."
                    )

                else:

                    st.warning(
                        "⚠️ API disponible mais modèle non chargé."
                    )

            else:

                st.error(
                    f"❌ API inaccessible : "
                    f"HTTP {response.status_code}"
                )

        except requests.RequestException as exc:

            st.error(
                f"❌ Impossible de contacter l'API : {exc}"
            )


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.subheader("📤 Sélectionner une image IRM")

uploaded_file = st.file_uploader(
    "Choisissez une image à analyser",
    type=ALLOWED_TYPES,
    accept_multiple_files=False,
    help=(
        f"Formats acceptés : JPG, JPEG et PNG. "
        f"Taille maximale : {MAX_FILE_SIZE_MB} Mo."
    )
)


# ============================================================
# SI AUCUNE IMAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Sélectionnez une image IRM puis cliquez sur "
        "**Analyser l'image**."
    )

    st.stop()


# ============================================================
# CONTRÔLE TAILLE
# ============================================================

file_size_mb = (
    uploaded_file.size
    / (1024 * 1024)
)

if file_size_mb > MAX_FILE_SIZE_MB:

    st.error(
        f"❌ L'image fait {file_size_mb:.2f} Mo. "
        f"La taille maximale autorisée est de "
        f"{MAX_FILE_SIZE_MB} Mo."
    )

    st.stop()


# ============================================================
# AFFICHAGE IMAGE AVANT ANALYSE
# ============================================================

st.subheader("🖼️ Image sélectionnée")

col_image, col_info = st.columns(
    [2, 1]
)

with col_image:

    st.image(
        uploaded_file,
        caption="Image IRM sélectionnée",
        use_container_width=True
    )

with col_info:

    st.markdown(
        """
        <div class="info-box">

        <strong>Informations du fichier</strong>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        f"**Nom :** {uploaded_file.name}"
    )

    st.write(
        f"**Type :** {uploaded_file.type}"
    )

    st.write(
        f"**Taille :** {file_size_mb:.2f} Mo"
    )


# ============================================================
# BOUTON ANALYSE
# ============================================================

st.divider()

analyze = st.button(
    "🔍 Analyser l'image",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSE
# ============================================================

if analyze:

    # --------------------------------------------------------
    # Lecture des bytes
    # --------------------------------------------------------

    try:

        file_bytes = uploaded_file.getvalue()

    except Exception as exc:

        st.error(
            f"❌ Impossible de lire l'image : {exc}"
        )

        st.stop()


    # --------------------------------------------------------
    # Appel API
    # --------------------------------------------------------

    with st.spinner(
        "🧠 Analyse de l'image en cours..."
    ):

        try:

            response = requests.post(
                PREDICT_URL,
                files={
                    "file": (
                        uploaded_file.name,
                        file_bytes,
                        uploaded_file.type
                    )
                },
                timeout=120
            )

        except requests.Timeout:

            st.error(
                "⏱️ L'analyse a dépassé le délai d'attente. "
                "L'API est peut-être en cours de démarrage."
            )

            st.stop()

        except requests.ConnectionError:

            st.error(
                "❌ Impossible de contacter l'API FastAPI."
            )

            st.info(
                f"API utilisée : {PREDICT_URL}"
            )

            st.stop()

        except requests.RequestException as exc:

            st.error(
                f"❌ Erreur de communication avec l'API : {exc}"
            )

            st.stop()


    # --------------------------------------------------------
    # Gestion des erreurs HTTP
    # --------------------------------------------------------

    if response.status_code != 200:

        if response.status_code == 413:

            st.error(
                "❌ Image trop volumineuse."
            )

        elif response.status_code == 415:

            st.error(
                "❌ Format d'image non accepté."
            )

        elif response.status_code == 403:

            st.error(
                "❌ API refusée (HTTP 403)."
            )

            st.info(
                "Vérifiez l'URL API utilisée par Streamlit "
                "et la configuration du service Render."
            )

        elif response.status_code == 503:

            st.error(
                "❌ Le modèle IA n'est actuellement pas disponible."
            )

        elif response.status_code == 500:

            st.error(
                "❌ Erreur interne pendant l'analyse."
            )

        else:

            st.error(
                f"❌ Erreur API : HTTP {response.status_code}"
            )

        # Essaye de récupérer le détail fourni par FastAPI
        try:

            error_data = response.json()

            if isinstance(error_data, dict):

                detail = error_data.get("detail")

                if detail:
                    st.caption(
                        f"Détail : {detail}"
                    )

        except Exception:
            pass

        st.stop()


    # --------------------------------------------------------
    # Lecture JSON
    # --------------------------------------------------------

    try:

        result = response.json()

    except Exception:

        st.error(
            "❌ La réponse de l'API n'est pas un JSON valide."
        )

        st.stop()


    # ========================================================
    # RÉSULTATS
    # ========================================================

    st.divider()

    st.subheader("📊 Résultats de l'analyse")


    tumor_detected = result.get(
        "tumeur_detectee",
        False
    )

    number_tumors = result.get(
        "nombre_tumeurs",
        0
    )

    processing_time = result.get(
        "temps_traitement_s"
    )


    # ========================================================
    # RÉSULTAT PRINCIPAL
    # ========================================================

    if tumor_detected:

        st.error(
            f"🔴 Tumeur détectée — "
            f"{number_tumors} zone(s)"
        )

    else:

        st.success(
            "🟢 Aucune tumeur détectée"
        )


    # ========================================================
    # INFORMATIONS RAPIDES
    # ========================================================

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Tumeur détectée",
            "Oui" if tumor_detected else "Non"
        )

    with metric2:

        st.metric(
            "Zones détectées",
            number_tumors
        )

    with metric3:

        if processing_time is not None:

            st.metric(
                "Temps de traitement",
                f"{processing_time:.3f} s"
            )

        else:

            st.metric(
                "Temps de traitement",
                "N/A"
            )


    # ========================================================
    # IMAGE ANNOTÉE
    # ========================================================

    annotated_base64 = result.get(
        "image_annotee_base64"
    )

    if annotated_base64:

        st.subheader(
            "🎯 Résultat de la segmentation"
        )

        try:

            annotated_bytes = base64.b64decode(
                annotated_base64
            )

            st.image(
                annotated_bytes,
                caption="Image analysée et annotée",
                use_container_width=True
            )

            # Libération immédiate
            del annotated_bytes

        except Exception as exc:

            st.error(
                f"❌ Impossible d'afficher l'image annotée : {exc}"
            )


    # ========================================================
    # MEILLEURE DÉTECTION
    # ========================================================

    best_detection = result.get(
        "meilleure_detection"
    )

    if best_detection:

        st.subheader(
            "🏆 Détection principale"
        )

        classe = best_detection.get(
            "classe",
            "Inconnue"
        )

        confiance = best_detection.get(
            "confiance",
            0
        )

        mask_available = best_detection.get(
            "masque_disponible",
            False
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Classe",
                classe
            )

        with c2:

            st.metric(
                "Confiance",
                f"{confiance * 100:.1f}%"
            )

        with c3:

            st.metric(
                "Segmentation",
                "Disponible"
                if mask_available
                else "Non disponible"
            )


        # ----------------------------------------------------
        # INDICATEURS
        # ----------------------------------------------------

        indicators = best_detection.get(
            "indicateurs"
        )

        if indicators:

            st.subheader(
                "📐 Indicateurs géométriques"
            )

            dimensions = indicators.get(
                "dimensions_px",
                {}
            )

            centre = indicators.get(
                "centre",
                {}
            )


            # Ligne 1
            c1, c2, c3 = st.columns(3)

            with c1:

                if "surface_masque_px" in indicators:

                    st.metric(
                        "Surface du masque",
                        f"{indicators['surface_masque_px']:.2f} px²"
                    )

                elif "surface_bounding_box_px" in indicators:

                    st.metric(
                        "Surface",
                        f"{indicators['surface_bounding_box_px']:.2f} px²"
                    )

            with c2:

                if "surface_masque_mm2" in indicators:

                    st.metric(
                        "Surface",
                        f"{indicators['surface_masque_mm2']:.2f} mm²"
                    )

            with c3:

                if "occupation_image_pourcent" in indicators:

                    st.metric(
                        "Occupation de l'image",
                        f"{indicators['occupation_image_pourcent']:.2f}%"
                    )


            # Ligne 2
            c1, c2, c3 = st.columns(3)

            with c1:

                if "perimetre_px" in indicators:

                    st.metric(
                        "Périmètre",
                        f"{indicators['perimetre_px']:.2f} px"
                    )

            with c2:

                if "diametre_max_px" in indicators:

                    st.metric(
                        "Diamètre maximal",
                        f"{indicators['diametre_max_px']:.2f} px"
                    )

            with c3:

                if "circularite" in indicators:

                    st.metric(
                        "Circularité",
                        f"{indicators['circularite']:.3f}"
                    )


            # Ligne 3
            c1, c2, c3 = st.columns(3)

            with c1:

                if "ratio_largeur_hauteur" in indicators:

                    st.metric(
                        "Ratio largeur / hauteur",
                        indicators["ratio_largeur_hauteur"]
                    )

            with c2:

                if "position_dans_image" in indicators:

                    st.metric(
                        "Position",
                        indicators["position_dans_image"]
                    )

            with c3:

                if "distance_centre_image_px" in indicators:

                    st.metric(
                        "Distance centre image",
                        f"{indicators['distance_centre_image_px']:.2f} px"
                    )


            # Dimensions
            if dimensions:

                st.write(
                    "**Dimensions de la zone segmentée :** "
                    f"{dimensions.get('largeur', 0)} × "
                    f"{dimensions.get('hauteur', 0)} px"
                )


            # Centre
            if centre:

                st.write(
                    "**Centre de la tumeur :** "
                    f"X = {centre.get('x', 0):.2f}, "
                    f"Y = {centre.get('y', 0):.2f}"
                )


            # Avertissement segmentation
            warning = indicators.get(
                "avertissement"
            )

            if warning:

                st.info(
                    warning
                )


    # ========================================================
    # AUTRES DÉTECTIONS
    # ========================================================

    detections = result.get(
        "detections",
        []
    )

    if len(detections) > 1:

        st.subheader(
            "🔎 Détails des autres détections"
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

            confidence = detection.get(
                "confiance",
                0
            )

            mask_available = detection.get(
                "masque_disponible",
                False
            )

            with st.expander(
                f"Détection #{detection_id} — {classe}"
            ):

                st.write(
                    f"**Classe :** {classe}"
                )

                st.write(
                    f"**Confiance :** "
                    f"{confidence * 100:.1f}%"
                )

                st.write(
                    "**Masque :** "
                    + (
                        "Disponible"
                        if mask_available
                        else "Non disponible"
                    )
                )

                indicators = detection.get(
                    "indicateurs"
                )

                if indicators:

                    st.json(
                        indicators
                    )


    # ========================================================
    # AVERTISSEMENT FINAL
    # ========================================================

    st.markdown(
        """
        <div class="warning-box">

        ⚠️ <strong>Important :</strong>
        les résultats présentés sont issus d'un modèle
        d'intelligence artificielle et doivent être interprétés
        et validés par un professionnel de santé.

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # LIBÉRATION DES OBJETS TEMPORAIRES
    # ========================================================

    del file_bytes

    # La réponse JSON n'est plus nécessaire après affichage.
    # On ne la conserve pas dans st.session_state.