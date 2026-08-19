"""
Interface Streamlit - Brain Tumor App

Cette interface :
    - ne charge PAS le modèle YOLO ;
    - communique uniquement avec l'API FastAPI ;
    - affiche les résultats de détection ;
    - affiche les indicateurs géométriques 2D ;
    - fournit une explication de chaque indicateur.

API :
    GET  /health
    POST /predict
"""

# ============================================================
# IMPORTS
# ============================================================

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

PREDICT_URL = (
    f"{API_URL}/predict"
)

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
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    [data-testid="stSidebar"] {
        display: none !important;
    }

    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    section[data-testid="stSidebar"] {
        display: none !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

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

    .definition-box {
        padding: 0.8rem;
        border-radius: 8px;
        background-color: #f5f7fa;
        border: 1px solid #e1e5ea;
        margin-top: -0.3rem;
        margin-bottom: 1rem;
        font-size: 0.92rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITRE
# ============================================================

st.title(
    "🧠 Détection et segmentation des tumeurs cérébrales"
)

st.markdown(
    """
    Cette application analyse une image IRM cérébrale à l'aide
    d'un modèle d'intelligence artificielle spécialisé
    dans la détection, la classification et la segmentation
    des lésions tumorales.
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
# UPLOAD
# ============================================================
st.subheader(
    "  "
)
st.subheader(
    "  Sélectionner une image IRM"
)

uploaded_file = st.file_uploader(
    "Choisissez une image **IRM** à analyser puis cliquez sur **Analyser l'image**.",
    type=ALLOWED_TYPES,
    accept_multiple_files=False,
    help=(
        "Formats acceptés : JPG, JPEG et PNG. "
        f"Taille maximale : {MAX_FILE_SIZE_MB} Mo."
    )
)


# ============================================================
# AUCUNE IMAGE
# ============================================================

if uploaded_file is None:

    st.stop()


# ============================================================
# TAILLE
# ============================================================

file_size_mb = (
    uploaded_file.size /
    (1024 * 1024)
)

if file_size_mb > MAX_FILE_SIZE_MB:

    st.error(
        f"❌ L'image fait {file_size_mb:.2f} Mo. "
        f"La taille maximale est de "
        f"{MAX_FILE_SIZE_MB} Mo."
    )

    st.stop()


# ============================================================
# IMAGE
# ============================================================

st.subheader(
    "🖼️ Image sélectionnée"
)

col_image, col_info = st.columns(
    [2, 1]
)

with col_image:

    st.image(
        uploaded_file,
        caption="Image IRM sélectionnée",
        width="stretch"
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
# BOUTON
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

    # ========================================================
    # LECTURE
    # ========================================================

    try:

        file_bytes = uploaded_file.getvalue()

    except Exception as exc:

        st.error(
            f"❌ Impossible de lire l'image : {exc}"
        )

        st.stop()


    # ========================================================
    # API
    # ========================================================

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
                "⏱️ L'analyse a dépassé le délai d'attente."
            )

            st.info(
                "L'API Render est peut-être en cours "
                "de démarrage ou le modèle met trop "
                "de temps à répondre."
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


    # ========================================================
    # ERREURS HTTP
    # ========================================================

    if response.status_code != 200:

        status = response.status_code

        if status == 400:

            st.error(
                "❌ Image invalide ou corrompue."
            )

        elif status == 403:

            st.error(
                "❌ Accès à l'API refusé (HTTP 403)."
            )

        elif status == 413:

            st.error(
                "❌ Image trop volumineuse."
            )

        elif status == 415:

            st.error(
                "❌ Format d'image non accepté."
            )

        elif status == 500:

            st.error(
                "❌ Erreur interne pendant l'analyse."
            )

        elif status == 502:

            st.error(
                "❌ L'API a retourné une erreur 502."
            )

            st.info(
                "Le service FastAPI a probablement "
                "redémarré pendant l'inférence."
            )

        elif status == 503:

            st.error(
                "❌ Le modèle IA n'est actuellement "
                "pas disponible."
            )

        else:

            st.error(
                f"❌ Erreur API : HTTP {status}"
            )

        try:

            error_data = response.json()

            if isinstance(
                error_data,
                dict
            ):

                detail = error_data.get(
                    "detail"
                )

                if detail:

                    st.caption(
                        f"Détail : {detail}"
                    )

        except Exception:

            pass

        st.stop()


    # ========================================================
    # JSON
    # ========================================================

    try:

        result = response.json()

    except Exception:

        st.error(
            "❌ La réponse de l'API n'est pas "
            "un JSON valide."
        )

        st.stop()


    # ========================================================
    # RÉSULTATS
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Résultats de l'analyse"
    )

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

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Tumeur détectée",
            "Oui" if tumor_detected else "Non"
        )

    with c2:

        st.metric(
            "Zones détectées",
            number_tumors
        )

    with c3:

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
                width="stretch"
            )

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

        st.divider()

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
                "Type de tumeur",
                classe
            )

        with c2:

            try:

                confidence_percent = (
                    float(confiance) * 100
                )

                st.metric(
                    "Confiance du modèle",
                    f"{confidence_percent:.1f}%"
                )

            except Exception:

                st.metric(
                    "Confiance du modèle",
                    str(confiance)
                )

        with c3:

            st.metric(
                "Segmentation",
                (
                    "Disponible"
                    if mask_available
                    else "Non disponible"
                )
            )


        # ====================================================
        # INDICATEURS
        # ====================================================

        indicators = best_detection.get(
            "indicateurs"
        )

        if indicators:

            st.divider()

            st.subheader(
                "📐 Analyse quantitative de la tumeur"
            )



            # =================================================
            # TAILLE
            # =================================================

            st.markdown(
                "### 📏 Taille de la lésion"
            )

            # -------------------------------------------------
            # Surface
            # -------------------------------------------------

            surface = indicators.get(
                "surface_masque_px"
            )

            if surface is not None:

                st.metric(
                    "Surface du masque",
                    f"{surface:.2f} px²"
                )

                st.markdown(
                    """
                    <div class="definition-box">
                    <strong>Surface du masque :</strong>
                    représente l'aire de la région identifiée
                    comme tumorale par le modèle dans cette coupe
                    IRM. La valeur est exprimée en pixels carrés.
                    Elle correspond à une mesure 2D et ne représente
                    pas un volume tumoral.
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            c1, c2 = st.columns(2)

            with c1:

                perimeter = indicators.get(
                    "perimetre_px"
                )

                if perimeter is not None:

                    st.metric(
                        "Périmètre",
                        f"{perimeter:.2f} px"
                    )

                    st.caption(
                        "Longueur approximative du contour "
                        "de la région segmentée."
                    )

            with c2:

                diameter_max = indicators.get(
                    "diametre_max_px"
                )

                if diameter_max is not None:

                    st.metric(
                        "Diamètre maximal",
                        f"{diameter_max:.2f} px"
                    )

                    st.caption(
                        "Plus grande distance mesurée "
                        "entre deux points du contour."
                    )


            c1, c2 = st.columns(2)

            with c1:

                diameter_min = indicators.get(
                    "diametre_min_px"
                )

                if diameter_min is not None:

                    st.metric(
                        "Diamètre minimal approximatif",
                        f"{diameter_min:.2f} px"
                    )

                    st.caption(
                        "Plus petite dimension du rectangle "
                        "orienté entourant la lésion. "
                        "Il s'agit d'une approximation géométrique."
                    )

            with c2:

                ratio = indicators.get(
                    "ratio_largeur_hauteur"
                )

                if ratio is not None:

                    st.metric(
                        "Ratio largeur / hauteur",
                        f"{float(ratio):.3f}"
                    )

                    st.caption(
                        "Compare la largeur et la hauteur "
                        "de la boîte englobante de la lésion."
                    )


            # =================================================
            # AXES
            # =================================================

            st.markdown(
                "### 📐 Axes et dimensions"
            )

            c1, c2 = st.columns(2)

            with c1:

                major_axis = indicators.get(
                    "axe_majeur_px"
                )

                if major_axis is not None:

                    st.metric(
                        "Axe majeur",
                        f"{major_axis:.2f} px"
                    )

                    st.caption(
                        "Direction principale selon laquelle "
                        "la lésion est la plus étendue."
                    )

            with c2:

                minor_axis = indicators.get(
                    "axe_mineur_px"
                )

                if minor_axis is not None:

                    st.metric(
                        "Axe mineur",
                        f"{minor_axis:.2f} px"
                    )

                    st.caption(
                        "Dimension perpendiculaire à l'axe majeur."
                    )


            dimensions = indicators.get(
                "dimensions_px",
                {}
            )

            if dimensions:

                largeur = dimensions.get(
                    "largeur",
                    0
                )

                hauteur = dimensions.get(
                    "hauteur",
                    0
                )

                st.info(
                    f"📦 **Boîte englobante :** "
                    f"{largeur} × {hauteur} px"
                )


            # =================================================
            # FORME
            # =================================================

            st.markdown(
                "### 🔵 Caractéristiques de la forme"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                circularity = indicators.get(
                    "circularite"
                )

                if circularity is not None:

                    st.metric(
                        "Circularité",
                        f"{float(circularity):.3f}"
                    )

                    st.caption(
                        "Mesure à quel point la forme se rapproche "
                        "d'un cercle. Une valeur proche de 1 indique "
                        "une forme très circulaire."
                    )

            with c2:

                eccentricity = indicators.get(
                    "excentricite"
                )

                if eccentricity is not None:

                    st.metric(
                        "Excentricité",
                        f"{float(eccentricity):.3f}"
                    )

                    st.caption(
                        "Mesure l'allongement de la forme. "
                        "Une valeur proche de 0 indique une forme "
                        "plus circulaire ; une valeur proche de 1 "
                        "indique une forme plus allongée."
                    )

            with c3:

                solidity = indicators.get(
                    "solidite"
                )

                if solidity is not None:

                    st.metric(
                        "Solidité",
                        f"{float(solidity):.3f}"
                    )

                    st.caption(
                        "Rapport entre la surface de la lésion "
                        "et celle de son enveloppe convexe. "
                        "Une valeur proche de 1 indique une "
                        "forme relativement compacte."
                    )


            c1, c2, c3 = st.columns(3)

            with c1:

                extent = indicators.get(
                    "etendue"
                )

                if extent is not None:

                    st.metric(
                        "Étendue",
                        f"{float(extent):.3f}"
                    )

                    st.caption(
                        "Rapport entre la surface du masque "
                        "et la surface de sa boîte englobante."
                    )

            with c2:

                convexity = indicators.get(
                    "convexite"
                )

                if convexity is not None:

                    st.metric(
                        "Convexité",
                        f"{float(convexity):.3f}"
                    )

                    st.caption(
                        "Compare le contour réel avec son "
                        "enveloppe convexe. Une valeur proche "
                        "de 1 indique un contour plus proche "
                        "d'une forme convexe."
                    )

            with c3:

                orientation = indicators.get(
                    "orientation_degres"
                )

                if orientation is not None:

                    st.metric(
                        "Orientation",
                        f"{float(orientation):.1f}°"
                    )

                    st.caption(
                        "Angle de l'axe principal de la lésion "
                        "par rapport à l'axe horizontal de l'image."
                    )


            # =================================================
            # LOCALISATION
            # =================================================

            st.markdown(
                "### 📍 Localisation dans l'image"
            )

            centre = indicators.get(
                "centre",
                {}
            )

            position_normalisee = indicators.get(
                "position_normalisee",
                {}
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                if centre:

                    x = centre.get(
                        "x",
                        0
                    )

                    y = centre.get(
                        "y",
                        0
                    )

                    st.metric(
                        "Centre X",
                        f"{float(x):.1f} px"
                    )

                    st.caption(
                        "Coordonnée horizontale du centre "
                        "de la région segmentée."
                    )

            with c2:

                if centre:

                    st.metric(
                        "Centre Y",
                        f"{float(y):.1f} px"
                    )

                    st.caption(
                        "Coordonnée verticale du centre "
                        "de la région segmentée."
                    )

            with c3:

                position_label = indicators.get(
                    "position_dans_image"
                )

                if position_label:

                    st.metric(
                        "Position",
                        position_label
                    )

                    st.caption(
                        "Localisation qualitative de la lésion "
                        "dans la coupe analysée."
                    )


            c1, c2 = st.columns(2)

            with c1:

                x_percent = position_normalisee.get(
                    "x_pourcent"
                )

                if x_percent is not None:

                    st.metric(
                        "Position horizontale",
                        f"{float(x_percent):.1f}%"
                    )

                    st.caption(
                        "Position du centre exprimée en "
                        "pourcentage de la largeur de l'image."
                    )

            with c2:

                y_percent = position_normalisee.get(
                    "y_pourcent"
                )

                if y_percent is not None:

                    st.metric(
                        "Position verticale",
                        f"{float(y_percent):.1f}%"
                    )

                    st.caption(
                        "Position du centre exprimée en "
                        "pourcentage de la hauteur de l'image."
                    )


            distance = indicators.get(
                "distance_centre_image_px"
            )

            if distance is not None:

                st.metric(
                    "Distance au centre de l'image",
                    f"{float(distance):.2f} px"
                )

                st.caption(
                    "Distance entre le centre de la lésion "
                    "et le centre géométrique de l'image."
                )


            # =================================================
            # OCCUPATION
            # =================================================

            st.markdown(
                "### 📊 Occupation de l'image"
            )

            occupation = indicators.get(
                "occupation_image_pourcent"
            )

            if occupation is not None:

                st.metric(
                    "Occupation de l'image",
                    f"{float(occupation):.4f}%"
                )

                st.caption(
                    "Pourcentage de la surface totale de "
                    "l'image occupé par la région segmentée. "
                    "Ce n'est pas le pourcentage du cerveau "
                    "occupé par la tumeur."
                )



            # =================================================
            # AVERTISSEMENT API
            # =================================================

            api_warning = indicators.get(
                "avertissement"
            )

            if api_warning:

                st.warning(
                    api_warning
                )


    # ========================================================
    # AUTRES DÉTECTIONS
    # ========================================================

    detections = result.get(
        "detections",
        []
    )

    if len(detections) > 1:

        st.divider()

        st.subheader(
            "🔎 Autres détections"
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

            with st.expander(
                f"Détection #{detection_id} — {classe}"
            ):

                st.write(
                    f"**Classe :** {classe}"
                )

                st.write(
                    f"**Confiance :** "
                    f"{float(confidence) * 100:.1f}%"
                )

                st.write(
                    "**Segmentation :** "
                    "Disponible"
                    if detection.get(
                        "masque_disponible",
                        False
                    )
                    else
                    "**Segmentation :** Non disponible"
                )

                detection_indicators = detection.get(
                    "indicateurs"
                )

                if detection_indicators:

                    st.json(
                        detection_indicators
                    )


    # ========================================================
    # AVERTISSEMENT FINAL
    # ========================================================

    st.markdown(
        """
        <div class="warning-box">

        ⚠️ <strong>Important :</strong><br>

        Les résultats présentés sont issus d'un modèle
        d'intelligence artificielle et doivent être
        interprétés et validés par un professionnel de santé.

        Les indicateurs géométriques correspondent à des
        mesures réalisées sur une image IRM 2D. Ils décrivent
        la morphologie apparente de la région segmentée et
        ne permettent pas, à eux seuls, d'établir un diagnostic,
        d'évaluer la gravité d'une tumeur ou de déterminer
        son volume réel en 3D.

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # NETTOYAGE
    # ========================================================

    del file_bytes