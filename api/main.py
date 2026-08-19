"""
API FastAPI - Brain Tumor App

Modèle :
    YOLO11-seg

Fonctions :
    - Détection
    - Classification
    - Segmentation
    - Analyse géométrique 2D des tumeurs

IMPORTANT :
Les indicateurs géométriques sont calculés à partir du masque
de segmentation 2D retourné par YOLO.

Ils décrivent la forme, la taille et la localisation apparentes
de la lésion dans la coupe IRM analysée.

Ils ne constituent PAS un diagnostic médical.
Une image 2D ne permet notamment pas de calculer un véritable
volume tumoral 3D.
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import io
import gc
import time
import base64
import logging
from typing import Any

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image

from ultralytics import YOLO


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger("brain-tumor-api")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best.pt"
)

IMG_SIZE = 640

CONF_THRESHOLD = 0.25

MAX_FILE_SIZE_MB = 10

MAX_FILE_SIZE_BYTES = (
    MAX_FILE_SIZE_MB * 1024 * 1024
)


# ============================================================
# APPLICATION FASTAPI
# ============================================================

app = FastAPI(
    title="Brain Tumor API",
    description=(
        "API de détection, classification, segmentation "
        "et analyse géométrique 2D des tumeurs cérébrales."
    ),
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

logger.info("=" * 60)
logger.info("CHARGEMENT DU MODÈLE")
logger.info("=" * 60)

logger.info(
    f"Chemin : {MODEL_PATH}"
)

logger.info(
    f"Existe : {os.path.exists(MODEL_PATH)}"
)

if not os.path.exists(MODEL_PATH):

    logger.error(
        "❌ Fichier best.pt introuvable."
    )

    model = None

else:

    try:

        model = YOLO(
            MODEL_PATH,
            task="segment"
        )

        logger.info(
            "✅ Modèle chargé avec succès."
        )

        logger.info(
            f"Tâche : {model.task}"
        )

        logger.info(
            f"Classes : {model.names}"
        )

    except Exception as exc:

        logger.exception(
            "❌ Erreur lors du chargement du modèle."
        )

        model = None


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Convertit une valeur en float sans provoquer d'erreur.
    """

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def encode_image_base64(
    image: np.ndarray
) -> str:
    """
    Convertit une image OpenCV en Base64 JPEG.
    """

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            85
        ]
    )

    if not success:

        raise ValueError(
            "Impossible d'encoder l'image."
        )

    return base64.b64encode(
        encoded.tobytes()
    ).decode("utf-8")


# ============================================================
# ANALYSE DU POLYGONE
# ============================================================

def calculate_geometric_indicators(
    polygon: np.ndarray,
    image_width: int,
    image_height: int
) -> dict:
    """
    Calcule les indicateurs géométriques à partir du
    polygone du masque YOLO.

    Le calcul est effectué directement sur le contour
    afin d'éviter de créer inutilement un masque complet
    de la taille de l'image.
    """

    # --------------------------------------------------------
    # Sécurisation du polygone
    # --------------------------------------------------------

    polygon = np.asarray(
        polygon,
        dtype=np.float32
    )

    if polygon.ndim != 2 or polygon.shape[0] < 3:

        return {}

    contour = polygon.reshape(
        (-1, 1, 2)
    )

    # --------------------------------------------------------
    # Surface du masque
    # --------------------------------------------------------

    surface = cv2.contourArea(
        contour
    )

    # --------------------------------------------------------
    # Périmètre
    # --------------------------------------------------------

    perimeter = cv2.arcLength(
        contour,
        True
    )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    x, y, width, height = cv2.boundingRect(
        contour
    )

    bbox_area = (
        float(width) * float(height)
    )

    # --------------------------------------------------------
    # Centre du masque
    # --------------------------------------------------------

    moments = cv2.moments(
        contour
    )

    if moments["m00"] != 0:

        center_x = (
            moments["m10"] /
            moments["m00"]
        )

        center_y = (
            moments["m01"] /
            moments["m00"]
        )

    else:

        center_x = (
            x + width / 2
        )

        center_y = (
            y + height / 2
        )

    # --------------------------------------------------------
    # Occupation de l'image
    # --------------------------------------------------------

    image_area = (
        float(image_width) *
        float(image_height)
    )

    occupation = 0.0

    if image_area > 0:

        occupation = (
            surface /
            image_area
        ) * 100.0

    # --------------------------------------------------------
    # Ratio largeur / hauteur
    # --------------------------------------------------------

    ratio = 0.0

    if height > 0:

        ratio = (
            float(width) /
            float(height)
        )

    # --------------------------------------------------------
    # Circularité
    #
    # 1 ≈ cercle parfait
    # plus la valeur diminue,
    # plus la forme s'éloigne du cercle.
    # --------------------------------------------------------

    circularity = 0.0

    if perimeter > 0:

        circularity = (
            4.0 *
            np.pi *
            surface /
            (perimeter ** 2)
        )

    circularity = min(
        max(circularity, 0.0),
        1.0
    )

    # --------------------------------------------------------
    # Diamètre maximal
    # --------------------------------------------------------

    diameter_max = 0.0

    if len(polygon) >= 2:

        max_distance = 0.0

        for i in range(
            len(polygon)
        ):

            point_a = polygon[i]

            distances = np.linalg.norm(
                polygon[
                    i + 1:
                ] - point_a,
                axis=1
            )

            if len(distances) > 0:

                current_max = float(
                    np.max(distances)
                )

                if current_max > max_distance:
                    max_distance = current_max

        diameter_max = max_distance

    # --------------------------------------------------------
    # Enveloppe convexe
    # --------------------------------------------------------

    hull = cv2.convexHull(
        contour
    )

    hull_area = cv2.contourArea(
        hull
    )

    hull_perimeter = cv2.arcLength(
        hull,
        True
    )

    # --------------------------------------------------------
    # SOLIDITÉ
    #
    # Surface du masque / surface de l'enveloppe convexe.
    #
    # Une valeur proche de 1 indique une forme compacte.
    # --------------------------------------------------------

    solidity = 0.0

    if hull_area > 0:

        solidity = (
            surface /
            hull_area
        )

    # --------------------------------------------------------
    # ÉTENDUE
    #
    # Surface du masque /
    # surface de la bounding box.
    # --------------------------------------------------------

    extent = 0.0

    if bbox_area > 0:

        extent = (
            surface /
            bbox_area
        )

    # --------------------------------------------------------
    # CONVEXITÉ
    #
    # Périmètre de l'enveloppe convexe /
    # périmètre du contour.
    #
    # Plus la valeur est proche de 1,
    # plus le contour est proche d'une forme convexe.
    # --------------------------------------------------------

    convexity = 0.0

    if perimeter > 0:

        convexity = (
            hull_perimeter /
            perimeter
        )

    # --------------------------------------------------------
    # ANALYSE DE LA FORME PAR PCA
    #
    # Permet d'obtenir :
    # - axe majeur
    # - axe mineur
    # - excentricité
    # - orientation
    # --------------------------------------------------------

    major_axis = 0.0
    minor_axis = 0.0
    eccentricity = 0.0
    orientation = 0.0

    points = polygon.astype(
        np.float64
    )

    if len(points) >= 3:

        centered = (
            points -
            np.mean(
                points,
                axis=0
            )
        )

        covariance = np.cov(
            centered,
            rowvar=False
        )

        try:

            eigenvalues, eigenvectors = np.linalg.eigh(
                covariance
            )

            # Tri décroissant
            order = np.argsort(
                eigenvalues
            )[::-1]

            eigenvalues = eigenvalues[
                order
            ]

            eigenvectors = eigenvectors[
                :,
                order
            ]

            lambda_major = max(
                float(eigenvalues[0]),
                0.0
            )

            lambda_minor = max(
                float(eigenvalues[1]),
                0.0
            )

            # Approximation des axes
            major_axis = (
                4.0 *
                np.sqrt(
                    lambda_major
                )
            )

            minor_axis = (
                4.0 *
                np.sqrt(
                    lambda_minor
                )
            )

            if lambda_major > 0:

                eccentricity = np.sqrt(
                    max(
                        0.0,
                        1.0 -
                        (
                            lambda_minor /
                            lambda_major
                        )
                    )
                )

            vector = eigenvectors[
                :,
                0
            ]

            orientation = np.degrees(
                np.arctan2(
                    vector[1],
                    vector[0]
                )
            )

            # Normalisation de l'angle
            if orientation < 0:

                orientation += 180.0

        except Exception:

            pass

    # --------------------------------------------------------
    # DIAMÈTRE MINIMAL APPROXIMATIF
    #
    # On utilise le plus petit côté du rectangle
    # orienté entourant le contour.
    # --------------------------------------------------------

    diameter_min = 0.0

    try:

        rotated_rect = cv2.minAreaRect(
            contour
        )

        rect_width, rect_height = (
            rotated_rect[1]
        )

        diameter_min = min(
            float(rect_width),
            float(rect_height)
        )

    except Exception:

        diameter_min = minor_axis

    # --------------------------------------------------------
    # POSITION NORMALISÉE
    # --------------------------------------------------------

    position_x_percent = 0.0
    position_y_percent = 0.0

    if image_width > 0:

        position_x_percent = (
            center_x /
            image_width
        ) * 100.0

    if image_height > 0:

        position_y_percent = (
            center_y /
            image_height
        ) * 100.0

    # --------------------------------------------------------
    # DISTANCE AU CENTRE DE L'IMAGE
    # --------------------------------------------------------

    image_center_x = (
        image_width / 2.0
    )

    image_center_y = (
        image_height / 2.0
    )

    distance_center = float(
        np.sqrt(
            (
                center_x -
                image_center_x
            ) ** 2
            +
            (
                center_y -
                image_center_y
            ) ** 2
        )
    )

    # --------------------------------------------------------
    # POSITION QUALITATIVE
    # --------------------------------------------------------

    horizontal = "centre"

    vertical = "centre"

    if position_x_percent < 33.33:
        horizontal = "gauche"

    elif position_x_percent > 66.67:
        horizontal = "droite"

    if position_y_percent < 33.33:
        vertical = "supérieure"

    elif position_y_percent > 66.67:
        vertical = "inférieure"

    if vertical == "centre" and horizontal == "centre":

        position_label = "centre"

    else:

        position_label = (
            f"région {vertical} {horizontal}"
        )

    # --------------------------------------------------------
    # NOMBRE DE POINTS DU CONTOUR
    # --------------------------------------------------------

    contour_points = int(
        len(polygon)
    )

    # --------------------------------------------------------
    # RÉSULTAT
    # --------------------------------------------------------

    return {

        # ------------------------------
        # Taille
        # ------------------------------

        "surface_masque_px": round(
            float(surface),
            2
        ),

        "perimetre_px": round(
            float(perimeter),
            2
        ),

        "diametre_max_px": round(
            float(diameter_max),
            2
        ),

        "diametre_min_px": round(
            float(diameter_min),
            2
        ),

        "axe_majeur_px": round(
            float(major_axis),
            2
        ),

        "axe_mineur_px": round(
            float(minor_axis),
            2
        ),

        # ------------------------------
        # Forme
        # ------------------------------

        "circularite": round(
            float(circularity),
            4
        ),

        "excentricite": round(
            float(eccentricity),
            4
        ),

        "solidite": round(
            float(solidity),
            4
        ),

        "etendue": round(
            float(extent),
            4
        ),

        "convexite": round(
            float(convexity),
            4
        ),

        "orientation_degres": round(
            float(orientation),
            2
        ),

        # ------------------------------
        # Dimensions
        # ------------------------------

        "dimensions_px": {

            "largeur": int(width),

            "hauteur": int(height)
        },

        "ratio_largeur_hauteur": round(
            float(ratio),
            4
        ),

        # ------------------------------
        # Localisation
        # ------------------------------

        "centre": {

            "x": round(
                float(center_x),
                2
            ),

            "y": round(
                float(center_y),
                2
            )
        },

        "position_dans_image": (
            position_label
        ),

        "position_normalisee": {

            "x_pourcent": round(
                float(position_x_percent),
                2
            ),

            "y_pourcent": round(
                float(position_y_percent),
                2
            )
        },

        "distance_centre_image_px": round(
            float(distance_center),
            2
        ),

        "occupation_image_pourcent": round(
            float(occupation),
            4
        ),

        # ------------------------------
        # Informations techniques
        # ------------------------------

        "nombre_points_contour": contour_points,

        # ------------------------------
        # Limitation
        # ------------------------------

    }


# ============================================================
# ROUTE RACINE
# ============================================================

@app.get("/")
def root():

    return {
        "application": "Brain Tumor API",
        "status": "online",
        "model_loaded": model is not None,
        "model_task": (
            model.task
            if model is not None
            else None
        ),
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "model_loaded": (
            model is not None
        ),

        "model_exists": (
            os.path.exists(
                MODEL_PATH
            )
        ),

        "model_path": MODEL_PATH,

        "model_task": (
            model.task
            if model is not None
            else None
        )
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    start_time = time.time()

    # ========================================================
    # VÉRIFICATION MODÈLE
    # ========================================================

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Modèle IA non disponible."
        )

    # ========================================================
    # VÉRIFICATION FORMAT
    # ========================================================

    allowed_types = {
        "image/jpeg",
        "image/png"
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=415,
            detail=(
                "Format non supporté. "
                "Utilisez JPG, JPEG ou PNG."
            )
        )

    # ========================================================
    # LECTURE DU FICHIER
    # ========================================================

    try:

        file_bytes = await file.read()

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Impossible de lire le fichier : {exc}"
            )
        )

    # ========================================================
    # TAILLE
    # ========================================================

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:

        del file_bytes

        raise HTTPException(
            status_code=413,
            detail=(
                f"Fichier trop volumineux. "
                f"Maximum : {MAX_FILE_SIZE_MB} Mo."
            )
        )

    # ========================================================
    # OUVERTURE IMAGE
    # ========================================================

    try:

        image = Image.open(
            io.BytesIO(file_bytes)
        )

        image.verify()

        # Recréation après verify()
        image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")

    except Exception as exc:

        del file_bytes

        raise HTTPException(
            status_code=400,
            detail=(
                f"Image invalide ou corrompue : {exc}"
            )
        )

    # On n'a plus besoin du fichier brut
    del file_bytes

    # ========================================================
    # CONVERSION NUMPY
    # ========================================================

    try:

        image_np = np.asarray(
            image,
            dtype=np.uint8
        )

        image_height, image_width = (
            image_np.shape[:2]
        )

    except Exception as exc:

        del image

        gc.collect()

        raise HTTPException(
            status_code=400,
            detail=(
                f"Impossible de traiter l'image : {exc}"
            )
        )

    # ========================================================
    # INFÉRENCE YOLO
    # ========================================================

    try:

        logger.info(
            "Début de l'inférence YOLO..."
        )

        results = model.predict(
            source=image_np,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            device="cpu",
            verbose=False,
            save=False,
            show=False,
            stream=False,
            retina_masks=False
        )

        logger.info(
            "Inférence terminée."
        )

    except Exception as exc:

        del image_np
        del image

        gc.collect()

        logger.exception(
            "Erreur pendant l'inférence."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Erreur pendant l'inférence : {exc}"
            )
        )

    # ========================================================
    # RÉSULTAT
    # ========================================================

    if not results:

        del image_np
        del image

        gc.collect()

        raise HTTPException(
            status_code=500,
            detail="Aucun résultat retourné par le modèle."
        )

    result = results[0]

    detections = []

    # ========================================================
    # RÉCUPÉRATION DES DÉTECTIONS
    # ========================================================

    if (
        result.boxes is not None
        and
        len(result.boxes) > 0
        and
        result.masks is not None
    ):

        boxes = result.boxes

        # Polygones dans les coordonnées de l'image
        polygons = result.masks.xy

        for index in range(
            len(boxes)
        ):

            try:

                # ------------------------------------------------
                # Classe
                # ------------------------------------------------

                class_id = int(
                    boxes.cls[index].item()
                )

                if isinstance(
                    model.names,
                    dict
                ):

                    class_name = model.names.get(
                        class_id,
                        str(class_id)
                    )

                else:

                    class_name = model.names[
                        class_id
                    ]

                # ------------------------------------------------
                # Confiance
                # ------------------------------------------------

                confidence = float(
                    boxes.conf[index].item()
                )

                # ------------------------------------------------
                # Polygone
                # ------------------------------------------------

                if index >= len(polygons):

                    continue

                polygon = polygons[index]

                # ------------------------------------------------
                # Indicateurs
                # ------------------------------------------------

                indicators = (
                    calculate_geometric_indicators(
                        polygon,
                        image_width,
                        image_height
                    )
                )

                if not indicators:

                    continue

                # ------------------------------------------------
                # Détection
                # ------------------------------------------------

                detections.append({

                    "id": index + 1,

                    "classe": class_name,

                    "confiance": round(
                        confidence,
                        4
                    ),

                    "masque_disponible": True,

                    "indicateurs": indicators
                })

            except Exception as exc:

                logger.warning(
                    f"Erreur détection {index}: {exc}"
                )

    # ========================================================
    # NOMBRE DE TUMEURS
    # ========================================================

    number_tumors = len(
        detections
    )

    tumor_detected = (
        number_tumors > 0
    )

    # ========================================================
    # MEILLEURE DÉTECTION
    # ========================================================

    best_detection = None

    if detections:

        best_detection = max(
            detections,
            key=lambda item:
            item.get(
                "confiance",
                0
            )
        )

    # ========================================================
    # IMAGE ANNOTÉE
    # ========================================================

    annotated_base64 = None

    try:

        if tumor_detected:

            annotated_image = result.plot(
                img=image_np,
                boxes=True,
                labels=True,
                conf=True,
                masks=True
            )

        else:

            annotated_image = image_np.copy()

        annotated_base64 = (
            encode_image_base64(
                annotated_image
            )
        )

        del annotated_image

    except Exception as exc:

        logger.warning(
            f"Impossible de créer l'image annotée : {exc}"
        )

    # ========================================================
    # TEMPS
    # ========================================================

    processing_time = (
        time.time() -
        start_time
    )

    # ========================================================
    # NETTOYAGE MÉMOIRE
    # ========================================================

    del results
    del result
    del image_np
    del image

    gc.collect()

    # ========================================================
    # RÉPONSE
    # ========================================================

    return {

        "status": "success",

        "tumeur_detectee": tumor_detected,

        "nombre_tumeurs": number_tumors,

        "meilleure_detection": best_detection,

        "detections": detections,

        "image_annotee_base64": annotated_base64,

        "temps_traitement_s": round(
            processing_time,
            3
        ),

        "image": {

            "largeur_px": image_width,

            "hauteur_px": image_height
        },

        "modele": {

            "tache": "segment",

            "imgsz": IMG_SIZE,

            "seuil_confiance": CONF_THRESHOLD
        },

        "avertissement": (
            "Les résultats sont issus d'une analyse "
            "automatique par intelligence artificielle. "
            "Les indicateurs géométriques sont calculés "
            "sur une coupe IRM 2D et ne remplacent pas "
            "l'interprétation d'un professionnel de santé."
        )
    }