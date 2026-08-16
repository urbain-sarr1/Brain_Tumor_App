"""
API de détection, classification et segmentation de tumeurs cérébrales.

Modèle :
    models/best.pt

Le modèle YOLO utilisé est un modèle de SEGMENTATION.
Les indicateurs géométriques (surface, périmètre, diamètre,
centre, occupation) sont calculés à partir du masque réel de
la tumeur, et non plus d'une simple bounding box.
"""

import base64
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(BASE_DIR / "models" / "best.pt")
    )
)

MAX_FILE_SIZE_MB = 10
CONFIDENCE_THRESHOLD = 0.5

# Facteur de conversion pixel -> mm. À défaut de métadonnées DICOM
# (spacing réel), une valeur par défaut est utilisée. À ajuster si
# le spacing réel de l'IRM est disponible.
PIXEL_SPACING_MM = float(
    os.getenv(
        "PIXEL_SPACING_MM",
        "1.0"
    )
)

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
}

DISCLAIMER = (
    "Ce système est un outil d'aide à la décision et ne constitue "
    "pas un diagnostic médical. La validation finale doit être "
    "réalisée par un professionnel de santé."
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("brain-tumor-api")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Brain Tumor Segmentation API",
    description=(
        "API de détection, classification et segmentation des "
        "tumeurs cérébrales à partir d'images IRM."
    ),
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "ALLOWED_ORIGINS",
        "*"
    ).split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODÈLE
# ============================================================

model = None


def load_model():
    global model

    logger.info("=" * 60)
    logger.info("CHARGEMENT DU MODÈLE")
    logger.info("=" * 60)

    logger.info("Chemin : %s", MODEL_PATH)
    logger.info("Existe : %s", MODEL_PATH.is_file())

    if not MODEL_PATH.is_file():
        logger.error(
            "❌ Modèle introuvable : %s",
            MODEL_PATH
        )
        return

    try:
        model = YOLO(str(MODEL_PATH))

        logger.info("✅ Modèle chargé avec succès.")
        logger.info(
            "Tâche : %s",
            getattr(model, "task", "inconnue")
        )
        logger.info(
            "Classes : %s",
            getattr(model, "names", {})
        )

        if getattr(model, "task", None) != "segment":
            logger.warning(
                "⚠️ Le modèle chargé n'est pas un modèle de "
                "segmentation (task=%s) : les indicateurs basés "
                "sur le masque ne seront pas disponibles.",
                getattr(model, "task", "inconnue")
            )

    except Exception as exc:
        logger.exception(
            "❌ Erreur chargement modèle : %s",
            exc
        )
        model = None


# Chargement immédiat
load_model()


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_exists": MODEL_PATH.is_file(),
        "model_path": str(MODEL_PATH),
        "model_task": getattr(model, "task", None),
    }


# ============================================================
# METRICS
# ============================================================

@app.get("/metrics")
def metrics():

    model_loaded = (
        1 if model is not None else 0
    )

    content = (
        "# HELP brain_tumor_api_model_loaded "
        "Etat du modele YOLO.\n"
        "# TYPE brain_tumor_api_model_loaded gauge\n"
        f"brain_tumor_api_model_loaded "
        f"{model_loaded}\n"
    )

    return (
        content,
        200,
        {
            "Content-Type":
                "text/plain; version=0.0.4"
        }
    )


# ============================================================
# LECTURE IMAGE
# ============================================================

def read_image(raw_bytes):

    array = np.frombuffer(
        raw_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            "Image illisible."
        )

    return image


# ============================================================
# IMAGE → BASE64
# ============================================================

def image_to_base64(image):

    success, encoded = cv2.imencode(
        ".jpg",
        image
    )

    if not success:

        raise ValueError(
            "Impossible d'encoder l'image."
        )

    return base64.b64encode(
        encoded.tobytes()
    ).decode("utf-8")


# ============================================================
# POSITION
# ============================================================

def determine_position(
    center_x,
    center_y,
    image_width,
    image_height
):

    if (
        image_width <= 0
        or image_height <= 0
    ):
        return "inconnue"

    x_ratio = (
        center_x / image_width
    )

    y_ratio = (
        center_y / image_height
    )

    # Horizontal
    if x_ratio < 1 / 3:
        horizontal = "gauche"

    elif x_ratio < 2 / 3:
        horizontal = "centre"

    else:
        horizontal = "droite"

    # Vertical
    if y_ratio < 1 / 3:
        vertical = "haut"

    elif y_ratio < 2 / 3:
        vertical = "centre"

    else:
        vertical = "bas"

    # Centre
    if (
        horizontal == "centre"
        and vertical == "centre"
    ):
        return "centre"

    if horizontal == "centre":
        return vertical

    if vertical == "centre":
        return horizontal

    return (
        f"{vertical}-{horizontal}"
    )


# ============================================================
# REDIMENSIONNEMENT DU MASQUE
# ============================================================

def resize_mask_to_image(mask, image_width, image_height):
    """
    Le masque produit par Ultralytics est à la résolution interne
    du modèle (imgsz) : on le redimensionne à la taille réelle de
    l'image d'origine avant tout calcul d'indicateur.
    """

    mask_uint8 = (
        mask * 255
    ).astype("uint8")

    if (
        mask_uint8.shape[1] != image_width
        or mask_uint8.shape[0] != image_height
    ):

        mask_uint8 = cv2.resize(
            mask_uint8,
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST
        )

    return mask_uint8


# ============================================================
# INDICATEURS — À PARTIR DU MASQUE DE SEGMENTATION
# ============================================================

def compute_segmentation_indicators(
    mask,
    box,
    image_width,
    image_height,
    pixel_spacing_mm=PIXEL_SPACING_MM
):
    """
    Calcule l'ensemble des indicateurs géométriques réels à partir
    du masque de segmentation (et non plus de la bounding box).

    Limite importante : à partir d'une unique coupe IRM 2D, seuls
    des indicateurs de surface et de forme peuvent être calculés
    de façon fiable. Le volume 3D réel (qui nécessiterait une pile
    de coupes IRM successives) n'est PAS estimé ici.
    """

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Repli sur la bounding box si aucun contour n'est exploitable
    if not contours:
        return compute_bbox_fallback_indicators(
            box,
            image_width,
            image_height
        )

    largest = max(
        contours,
        key=cv2.contourArea
    )

    # Surface réelle du masque
    area_px = float(
        cv2.contourArea(largest)
    )

    area_mm2 = round(
        area_px * (pixel_spacing_mm ** 2),
        2
    )

    # Périmètre
    perimeter_px = float(
        cv2.arcLength(
            largest,
            closed=True
        )
    )

    perimeter_mm = round(
        perimeter_px * pixel_spacing_mm,
        2
    )

    # Diamètre maximal (plus grand axe du contour)
    (_, _), radius = cv2.minEnclosingCircle(largest)

    max_diameter_px = round(
        2 * radius,
        2
    )

    max_diameter_mm = round(
        2 * radius * pixel_spacing_mm,
        2
    )

    # Boîte englobante du masque (largeur / hauteur réelles)
    x, y, width, height = cv2.boundingRect(largest)

    aspect_ratio = (
        round(width / height, 3)
        if height > 0
        else 0
    )

    # Centroïde réel du masque (pas le centre de la bbox)
    moments = cv2.moments(largest)

    center_x = (
        moments["m10"] / moments["m00"]
        if moments["m00"]
        else x + width / 2
    )

    center_y = (
        moments["m01"] / moments["m00"]
        if moments["m00"]
        else y + height / 2
    )

    # Position dans l'image (basée sur le centroïde réel)
    position = determine_position(
        center_x,
        center_y,
        image_width,
        image_height
    )

    # Distance au centre de l'image
    image_center_x = image_width / 2
    image_center_y = image_height / 2

    distance_center = (
        (
            (center_x - image_center_x) ** 2
            +
            (center_y - image_center_y) ** 2
        ) ** 0.5
    )

    # Occupation de l'image (surface réelle, pas la bbox)
    image_area = image_width * image_height

    occupation = (
        area_px / image_area * 100
        if image_area > 0
        else 0
    )

    # Circularité : indicateur de régularité de la forme
    # (1 = cercle parfait, proche de 0 = forme très irrégulière)
    circularity = (
        round(
            (4 * np.pi * area_px) / (perimeter_px ** 2),
            3
        )
        if perimeter_px > 0
        else 0
    )

    return {

        "centre": {
            "x": round(center_x, 2),
            "y": round(center_y, 2)
        },

        "position_dans_image":
            position,

        "dimensions_px": {
            "largeur": width,
            "hauteur": height
        },

        "surface_masque_px":
            round(area_px, 2),

        "surface_masque_mm2":
            area_mm2,

        "perimetre_px":
            round(perimeter_px, 2),

        "perimetre_mm":
            perimeter_mm,

        "diametre_max_px":
            max_diameter_px,

        "diametre_max_mm":
            max_diameter_mm,

        "occupation_image_pourcent":
            round(occupation, 2),

        "ratio_largeur_hauteur":
            aspect_ratio,

        "circularite":
            circularity,

        "distance_centre_image_px":
            round(distance_center, 2),

        "avertissement": (
            "Indicateurs calculés à partir du masque de segmentation "
            "réel sur une unique coupe 2D : surface, périmètre et "
            "diamètre sont fiables sur cette coupe, mais le volume 3D "
            "n'est pas estimé (nécessiterait une pile de coupes IRM "
            "successives)."
        ),
    }


# ============================================================
# INDICATEURS — REPLI SUR LA BOUNDING BOX (si pas de masque)
# ============================================================

def compute_bbox_fallback_indicators(
    box,
    image_width,
    image_height
):
    """
    Utilisé uniquement si aucun masque n'est disponible pour une
    détection donnée (cas normalement rare avec un modèle de
    segmentation, mais gardé par sécurité).
    """

    x1, y1, x2, y2 = map(float, box)

    x1 = max(0, min(x1, image_width))
    x2 = max(0, min(x2, image_width))
    y1 = max(0, min(y1, image_height))
    y2 = max(0, min(y2, image_height))

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    bbox_area = width * height
    image_area = image_width * image_height

    occupation = (
        bbox_area / image_area * 100
        if image_area > 0
        else 0
    )

    aspect_ratio = (
        round(width / height, 3)
        if height > 0
        else 0
    )

    image_center_x = image_width / 2
    image_center_y = image_height / 2

    distance_center = (
        (
            (center_x - image_center_x) ** 2
            +
            (center_y - image_center_y) ** 2
        ) ** 0.5
    )

    return {

        "centre": {
            "x": round(center_x, 2),
            "y": round(center_y, 2)
        },

        "position_dans_image":
            determine_position(
                center_x,
                center_y,
                image_width,
                image_height
            ),

        "dimensions_px": {
            "largeur": round(width, 2),
            "hauteur": round(height, 2)
        },

        "surface_bounding_box_px":
            round(bbox_area, 2),

        "occupation_image_pourcent":
            round(occupation, 2),

        "ratio_largeur_hauteur":
            aspect_ratio,

        "distance_centre_image_px":
            round(distance_center, 2),

        "avertissement": (
            "Aucun masque de segmentation disponible pour cette "
            "détection : ces indicateurs sont calculés à partir de "
            "la bounding box et ne reflètent pas le contour réel "
            "de la tumeur."
        ),
    }


# ============================================================
# DESSIN
# ============================================================

def draw_detection(
    image,
    box,
    mask,
    class_name,
    confidence,
    color=(0, 0, 255)
):

    if mask is not None:

        overlay = image.copy()
        overlay[mask > 0] = color

        image = cv2.addWeighted(
            overlay,
            0.35,
            image,
            0.65,
            0
        )

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            image,
            contours,
            -1,
            color,
            2
        )

    x1, y1, x2, y2 = map(
        int,
        box
    )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        1
    )

    label = (
        f"{class_name} "
        f"{confidence * 100:.1f}%"
    )

    cv2.putText(
        image,
        label,
        (
            x1,
            max(
                25,
                y1 - 10
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA
    )

    return image


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    start_time = time.perf_counter()

    # ========================================================
    # 1. FORMAT
    # ========================================================

    if (
        file.content_type
        not in ALLOWED_CONTENT_TYPES
    ):

        raise HTTPException(
            status_code=415,
            detail=(
                "Fichier non conforme. "
                "Formats acceptés : "
                "JPEG et PNG."
            )
        )

    # ========================================================
    # 2. LECTURE
    # ========================================================

    raw_bytes = await file.read()

    if not raw_bytes:

        raise HTTPException(
            status_code=400,
            detail="Le fichier est vide."
        )

    # ========================================================
    # 3. TAILLE
    # ========================================================

    size_mb = (
        len(raw_bytes)
        / (1024 * 1024)
    )

    if size_mb > MAX_FILE_SIZE_MB:

        raise HTTPException(
            status_code=413,
            detail=(
                f"Fichier trop volumineux "
                f"({size_mb:.1f} Mo). "
                f"Taille maximale : "
                f"{MAX_FILE_SIZE_MB} Mo."
            )
        )

    # ========================================================
    # 4. DÉCODAGE
    # ========================================================

    try:

        image = read_image(
            raw_bytes
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail=(
                "Image non conforme "
                "ou fichier corrompu."
            )
        )

    # ========================================================
    # 5. MODÈLE
    # ========================================================

    if model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Le modèle d'IA n'est "
                "actuellement pas disponible."
            )
        )

    # ========================================================
    # 6. DIMENSIONS
    # ========================================================

    image_height, image_width = (
        image.shape[:2]
    )

    # ========================================================
    # 7. PRÉDICTION
    # ========================================================

    try:

        results = model.predict(
            source=image,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=640,
            verbose=False
        )

    except Exception:

        logger.exception(
            "Erreur pendant la prédiction."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Une erreur interne est "
                "survenue pendant l'analyse."
            )
        )

    result = results[0]

    # ========================================================
    # 8. AUCUNE TUMEUR
    # ========================================================

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):

        elapsed = (
            time.perf_counter()
            - start_time
        )

        logger.info(
            "Aucune tumeur détectée."
        )

        return {

            "avertissement":
                DISCLAIMER,

            "tumeur_detectee":
                False,

            "nombre_tumeurs":
                0,

            "detections":
                [],

            "image_dimensions": {
                "largeur":
                    image_width,
                "hauteur":
                    image_height
            },

            "image_annotee_base64":
                image_to_base64(
                    image
                ),

            "temps_traitement_s":
                round(
                    elapsed,
                    3
                )
        }

    # ========================================================
    # 9. DÉTECTIONS
    # ========================================================

    detections = []

    annotated = image.copy()

    has_masks = (
        result.masks is not None
        and len(result.masks.data) > 0
    )

    colors = [
        (0, 0, 255),
        (255, 128, 0),
        (0, 200, 100),
        (200, 0, 200),
    ]

    for index in range(
        len(result.boxes)
    ):

        # Bounding box
        box = (
            result
            .boxes
            .xyxy[index]
            .tolist()
        )

        # Classe
        class_id = int(
            result
            .boxes
            .cls[index]
        )

        class_name = (
            model.names.get(
                class_id,
                f"Classe {class_id}"
            )
        )

        # Confiance
        confidence = float(
            result
            .boxes
            .conf[index]
        )

        color = colors[
            index % len(colors)
        ]

        # ----------------------------------------------------
        # Masque de segmentation (si disponible pour cet index)
        # ----------------------------------------------------

        mask_resized = None
        indicators = None

        if (
            has_masks
            and index < len(result.masks.data)
        ):

            raw_mask = (
                result
                .masks
                .data[index]
                .cpu()
                .numpy()
            )

            mask_resized = resize_mask_to_image(
                raw_mask,
                image_width,
                image_height
            )

            indicators = compute_segmentation_indicators(
                mask_resized,
                box,
                image_width,
                image_height
            )

        else:

            indicators = compute_bbox_fallback_indicators(
                box,
                image_width,
                image_height
            )

        # Détection
        detection = {

            "id":
                index + 1,

            "classe":
                class_name,

            "classe_id":
                class_id,

            "confiance":
                round(
                    confidence,
                    3
                ),

            "boite_englobante": {
                "x1":
                    round(box[0], 2),
                "y1":
                    round(box[1], 2),
                "x2":
                    round(box[2], 2),
                "y2":
                    round(box[3], 2)
            },

            "masque_disponible":
                mask_resized is not None,

            "indicateurs":
                indicators
        }

        detections.append(
            detection
        )

        # Annotation
        annotated = draw_detection(
            annotated,
            box,
            mask_resized,
            class_name,
            confidence,
            color
        )

    # ========================================================
    # 10. MEILLEURE DÉTECTION
    # ========================================================

    best_detection = max(
        detections,
        key=lambda x:
            x["confiance"]
    )

    # ========================================================
    # 11. TEMPS
    # ========================================================

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # ========================================================
    # 12. LOG
    # ========================================================

    logger.info(
        "Détection : %d | "
        "classe=%s | "
        "confiance=%.3f | "
        "masque=%s | "
        "temps=%.3fs",
        len(detections),
        best_detection["classe"],
        best_detection["confiance"],
        best_detection["masque_disponible"],
        elapsed
    )

    # ========================================================
    # 13. RÉPONSE
    # ========================================================

    return {

        "avertissement":
            DISCLAIMER,

        "tumeur_detectee":
            True,

        "nombre_tumeurs":
            len(detections),

        "meilleure_detection":
            best_detection,

        "detections":
            detections,

        "image_dimensions": {
            "largeur":
                image_width,
            "hauteur":
                image_height
        },

        "image_annotee_base64":
            image_to_base64(
                annotated
            ),

        "temps_traitement_s":
            round(
                elapsed,
                3
            )
    }