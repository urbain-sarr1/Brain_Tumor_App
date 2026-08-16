"""
Tests des indicateurs de qualité de l'API Brain Tumor.

Indicateurs contrôlés :

1. Disponibilité
   - API accessible
   - modèle best.pt chargé

2. Temps de réponse
   - /health < 2 secondes
   - /predict < 2 secondes

3. Performance du modèle
   - Precision >= 85 %
   - Recall >= 85 %
   - mAP@0.5 >= 90 %
   - mAP@0.5:0.95 >= 70 %

4. Robustesse
   - formats non supportés
   - images corrompues
   - requêtes mal formées

5. Stabilité
   - plusieurs requêtes successives sans erreur 500

Les performances du modèle sont lues depuis :
models/metrics.json

Toutes les valeurs numériques mesurées (temps de réponse, métriques du
modèle, moyennes) sont consignées dans resultats_tests.txt.
"""

import io
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app


# ============================================================
# CONFIGURATION
# ============================================================

client = TestClient(app)

RESPONSE_TIME_THRESHOLD = 2.0

METRICS_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "metrics.json"
)

RESULTS_FILE = (
    Path(__file__).resolve().parent
    / "resultats_tests.txt"
)

# Seuils définis dans le cahier des charges
MIN_PRECISION = 0.85
MIN_RECALL = 0.85
MIN_MAP50 = 0.90
MIN_MAP50_95 = 0.70


# ============================================================
# COLLECTEUR DE MESURES
# ============================================================
# Rempli au fil des tests, puis écrit dans resultats_tests.txt
# à la fin de la session (voir save_test_results ci-dessous).

mesures = {
    "health_temps_s": [],
    "predict_temps_s": [],
    "predict_warmup_temps_s": None,
    "stabilite_temps_s": [],
    "performance_modele": {},
}


# ============================================================
# IMAGE DE TEST
# ============================================================

def make_test_image():
    """Crée une image JPEG valide pour tester l'API."""

    image = Image.new(
        "RGB",
        (640, 640),
        color=(120, 120, 120),
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
    )

    return buffer.getvalue()


# ============================================================
# RAPPORT
# ============================================================

@pytest.fixture(
    scope="session",
    autouse=True,
)
def save_test_results(request):

    yield

    session = request.session

    total = session.testscollected
    failed = session.testsfailed
    passed = total - failed

    with open(
        RESULTS_FILE,
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n"
            "========================================\n"
        )

        file.write(
            f"Exécution : "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        file.write(
            f"Tests exécutés : {total}\n"
        )

        file.write(
            f"Tests réussis : {passed}\n"
        )

        file.write(
            f"Tests échoués : {failed}\n"
        )

        file.write(
            f"Statut : "
            f"{'OK' if failed == 0 else 'ÉCHEC'}\n"
        )

        file.write(
            "----------------------------------------\n"
        )
        file.write("Mesures détaillées\n")
        file.write(
            "----------------------------------------\n"
        )

        # ---- Temps de réponse /health ----
        if mesures["health_temps_s"]:
            valeurs = mesures["health_temps_s"]
            file.write(
                f"/health  -> temps moyen : {statistics.mean(valeurs):.4f} s "
                f"| min : {min(valeurs):.4f} s "
                f"| max : {max(valeurs):.4f} s "
                f"(sur {len(valeurs)} requête(s))\n"
            )

        # ---- Temps de réponse /predict ----
        if mesures["predict_warmup_temps_s"] is not None:
            file.write(
                f"/predict (warm-up, non comptabilisé) : "
                f"{mesures['predict_warmup_temps_s']:.4f} s\n"
            )

        if mesures["predict_temps_s"]:
            valeurs = mesures["predict_temps_s"]
            file.write(
                f"/predict -> temps moyen : {statistics.mean(valeurs):.4f} s "
                f"| min : {min(valeurs):.4f} s "
                f"| max : {max(valeurs):.4f} s "
                f"(sur {len(valeurs)} requête(s))\n"
            )

        # ---- Stabilité (requêtes successives) ----
        if mesures["stabilite_temps_s"]:
            valeurs = mesures["stabilite_temps_s"]
            file.write(
                f"Stabilité ({len(valeurs)} requêtes successives) -> "
                f"temps moyen : {statistics.mean(valeurs):.4f} s "
                f"| min : {min(valeurs):.4f} s "
                f"| max : {max(valeurs):.4f} s "
                f"| écart-type : {statistics.pstdev(valeurs):.4f} s\n"
            )

        # ---- Performance du modèle ----
        if mesures["performance_modele"]:
            perf = mesures["performance_modele"]
            file.write("Performance du modèle (models/metrics.json) :\n")
            file.write(
                f"  Precision    : {perf.get('precision', 0):.1%} "
                f"(seuil requis : {MIN_PRECISION:.0%})\n"
            )
            file.write(
                f"  Recall       : {perf.get('recall', 0):.1%} "
                f"(seuil requis : {MIN_RECALL:.0%})\n"
            )
            file.write(
                f"  mAP@0.5      : {perf.get('map50', 0):.1%} "
                f"(seuil requis : {MIN_MAP50:.0%})\n"
            )
            file.write(
                f"  mAP@0.5:0.95 : {perf.get('map50_95', 0):.1%} "
                f"(seuil requis : {MIN_MAP50_95:.0%})\n"
            )

        file.write(
            "========================================\n"
        )


# ============================================================
# 1 — DISPONIBILITÉ
# ============================================================

class TestDisponibilite:

    def test_api_disponible(self):
        """
        Vérifie que l'API répond correctement
        à l'endpoint de supervision.
        """

        response = client.get("/health")

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "ok"


    def test_modele_charge(self):
        """
        Vérifie que le modèle best.pt utilisé
        pour la production est correctement chargé.
        """

        response = client.get("/health")

        assert response.status_code == 200

        data = response.json()

        assert data["model_loaded"] is True


# ============================================================
# 2 — TEMPS DE RÉPONSE
# ============================================================

class TestTempsDeReponse:

    N_REQUETES = 10

    def test_health_moins_de_2_secondes(self):
        """
        Vérifie que l'API répond à /health en moins de 2 secondes,
        sur la base de plusieurs requêtes successives.
        """

        for _ in range(self.N_REQUETES):

            start = time.perf_counter()

            response = client.get("/health")

            elapsed = time.perf_counter() - start

            mesures["health_temps_s"].append(elapsed)

            assert response.status_code == 200

            assert elapsed < RESPONSE_TIME_THRESHOLD



    def test_prediction_moins_de_2_secondes(self):
        """
        Vérifie que l'analyse d'une image respecte le seuil de 2 secondes,
        sur la base de plusieurs requêtes successives.

        La première prédiction est effectuée hors mesure afin d'exclure
        le temps de chauffe initial du modèle (chargement en mémoire,
        premières allocations GPU/CPU).
        """

        image = make_test_image()

        # --------------------------------------------------------
        # Warm-up : première prédiction non comptabilisée
        # --------------------------------------------------------

        warmup_start = time.perf_counter()

        warmup_response = client.post(
            "/predict",
            files={
                "file": (
                    "irm.jpg",
                    image,
                    "image/jpeg",
                )
            },
        )

        mesures["predict_warmup_temps_s"] = time.perf_counter() - warmup_start

        assert warmup_response.status_code == 200

        # --------------------------------------------------------
        # Mesures réelles sur plusieurs requêtes
        # --------------------------------------------------------

        for _ in range(self.N_REQUETES):

            start = time.perf_counter()

            response = client.post(
                "/predict",
                files={
                    "file": (
                        "irm.jpg",
                        image,
                        "image/jpeg",
                    )
                },
            )

            elapsed = time.perf_counter() - start

            mesures["predict_temps_s"].append(elapsed)

            assert response.status_code == 200

            assert elapsed < RESPONSE_TIME_THRESHOLD, (
                f"Temps de réponse trop élevé : "
                f"{elapsed:.3f}s "
                f"(seuil : {RESPONSE_TIME_THRESHOLD:.1f}s)"
            )



# ============================================================
# 3 — PERFORMANCE DU MODÈLE
# ============================================================

class TestPerformanceModele:

    @pytest.fixture
    def metrics(self):

        if not METRICS_PATH.exists():

            pytest.fail(
                "models/metrics.json est introuvable."
            )

        with open(
            METRICS_PATH,
            encoding="utf-8",
        ) as file:

            data = json.load(file)
            mesures["performance_modele"] = data
            return data


    def test_precision(self, metrics):

        assert (
            metrics["precision"]
            >= MIN_PRECISION
        )


    def test_recall(self, metrics):

        assert (
            metrics["recall"]
            >= MIN_RECALL
        )


    def test_map50(self, metrics):

        assert (
            metrics["map50"]
            >= MIN_MAP50
        )


    def test_map50_95(self, metrics):

        assert (
            metrics["map50_95"]
            >= MIN_MAP50_95
        )


# ============================================================
# 4 — ROBUSTESSE
# ============================================================

class TestRobustesse:

    def test_format_non_supporte(self):
        """
        Un fichier non image doit être rejeté
        avec HTTP 415.
        """

        response = client.post(
            "/predict",
            files={
                "file": (
                    "document.txt",
                    b"contenu texte",
                    "text/plain",
                )
            },
        )

        assert response.status_code == 415


    def test_image_corrompue(self):
        """
        Une image illisible doit être rejetée
        avec HTTP 400.
        """

        response = client.post(
            "/predict",
            files={
                "file": (
                    "image.jpg",
                    b"image-invalide",
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 400


    def test_requete_sans_fichier(self):
        """
        Une requête sans image doit être rejetée
        avec HTTP 422.
        """

        response = client.post("/predict")

        assert response.status_code == 422


# ============================================================
# 5 — STABILITÉ
# ============================================================

class TestStabilite:

    def test_requetes_successives(self):
        """
        Vérifie que l'API reste fonctionnelle
        après plusieurs requêtes successives.

        Ce test ne remplace pas un monitoring continu
        sur Render, mais constitue un contrôle automatisé
        de stabilité.
        """

        image = make_test_image()

        for _ in range(5):

            start = time.perf_counter()

            response = client.post(
                "/predict",
                files={
                    "file": (
                        "irm.jpg",
                        image,
                        "image/jpeg",
                    )
                },
            )

            elapsed = time.perf_counter() - start
            mesures["stabilite_temps_s"].append(elapsed)

            assert response.status_code == 200

            assert response.status_code != 500