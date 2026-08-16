"""
Fonctions utilitaires pour le post-traitement des prédictions d'un modèle
de segmentation YOLO (box + classe + masque en une seule inférence) :
- dessin des détections (boîtes + masques superposés)
- calcul des indicateurs géométriques réels à partir des masques
- encodage de l'image annotée en base64
"""

import base64
import io

import cv2
import numpy as np
from PIL import Image

# Facteur de conversion pixel -> mm. À défaut de métadonnées DICOM (spacing
# réel), une valeur par défaut est utilisée. À ajuster si le spacing réel
# de l'IRM est disponible, sous peine de fausser les indicateurs en mm.
DEFAULT_PIXEL_SPACING_MM = 1.0

INSTANCE_COLORS = [
    (44, 130, 201),   # bleu
    (201, 84, 44),    # orange
    (84, 201, 130),   # vert
    (170, 84, 201),   # violet
]


def image_to_base64(image_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        raise ValueError("Échec de l'encodage de l'image annotée")
    return base64.b64encode(buffer).decode("utf-8")


def read_upload_to_bgr(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def draw_instance(image_bgr: np.ndarray, box, mask, class_name: str, confidence: float,
                   color: tuple, alpha: float = 0.35) -> np.ndarray:
    """Dessine la boîte, le libellé et (si disponible) le masque d'une instance détectée."""
    annotated = image_bgr.copy()

    if mask is not None:
        overlay = annotated.copy()
        overlay[mask > 0] = color
        annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, color, 2)

    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
    label = f"{class_name} ({confidence * 100:.1f}%)"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
    cv2.putText(annotated, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def compute_mask_indicators(mask: np.ndarray, image_shape: tuple,
                             pixel_spacing_mm: float = DEFAULT_PIXEL_SPACING_MM) -> dict:
    """
    Calcule l'ensemble des indicateurs géométriques à partir du masque réel
    de segmentation d'une instance détectée.

    Limite importante : à partir d'une unique coupe IRM 2D, seuls des
    indicateurs de surface et de forme peuvent être calculés de façon
    fiable. Le volume 3D réel (qui nécessiterait une pile de coupes
    successives) n'est PAS estimé ici.
    """
    h_img, w_img = image_shape[:2]

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}

    largest = max(contours, key=cv2.contourArea)

    # ---- Surface réelle du masque (pas de la bounding box) ----
    area_px = float(cv2.contourArea(largest))
    area_mm2 = round(area_px * (pixel_spacing_mm ** 2), 2)

    # ---- Périmètre ----
    perimeter_px = float(cv2.arcLength(largest, closed=True))
    perimeter_mm = round(perimeter_px * pixel_spacing_mm, 2)

    # ---- Diamètre maximal (plus grand axe du contour) ----
    (_, _), radius = cv2.minEnclosingCircle(largest)
    max_diameter_mm = round(2 * radius * pixel_spacing_mm, 2)

    # ---- Boîte englobante du masque (utile pour largeur/hauteur/ratio) ----
    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = round(w / h, 2) if h > 0 else 0

    # ---- Centroïde réel du masque (pas le centre de la bbox) ----
    moments = cv2.moments(largest)
    cx = moments["m10"] / moments["m00"] if moments["m00"] else x + w / 2
    cy = moments["m01"] / moments["m00"] if moments["m00"] else y + h / 2

    # ---- Localisation indicative dans l'image ----
    horizontal = "hémisphère gauche" if cx < w_img / 2 else "hémisphère droit"
    vertical = "région supérieure" if cy < h_img / 2 else "région inférieure"
    position_dans_image = f"{horizontal}, {vertical}"

    # ---- Distance au centre de l'image ----
    img_cx, img_cy = w_img / 2, h_img / 2
    distance_centre_px = float(np.hypot(cx - img_cx, cy - img_cy))

    # ---- Occupation de l'image ----
    occupation_pct = round((area_px / (w_img * h_img)) * 100, 2)

    # ---- Circularité (indicateur de régularité de la forme, 1 = cercle parfait) ----
    circularity = round((4 * np.pi * area_px) / (perimeter_px ** 2), 2) if perimeter_px > 0 else 0

    return {
        "surface_masque_px": round(area_px, 1),
        "surface_masque_mm2": area_mm2,
        "perimetre_px": round(perimeter_px, 1),
        "perimetre_mm": perimeter_mm,
        "diametre_max_mm": max_diameter_mm,
        "dimensions_px": {"largeur": w, "hauteur": h},
        "ratio_largeur_hauteur": aspect_ratio,
        "circularite": circularity,
        "centre": {"x": round(cx, 1), "y": round(cy, 1)},
        "position_dans_image": position_dans_image,
        "distance_centre_image_px": round(distance_centre_px, 1),
        "occupation_image_pourcent": occupation_pct,
        "avertissement": (
            "Indicateurs calculés à partir du masque de segmentation réel sur "
            "une unique coupe 2D : surface, périmètre et diamètre sont fiables "
            "sur cette coupe, mais le volume 3D n'est pas estimé (nécessiterait "
            "une pile de coupes IRM successives)."
        ),
    }