# Détection et classification de tumeurs cérébrales — API + Interface

Solution d'aide à la décision pour la détection et la classification de
tumeurs cérébrales (Glioma, Meningioma, Pituitary) à partir d'images IRM,
conforme au cahier des charges technique et fonctionnel du projet.

**Ce système est un outil d'aide à la décision. Il ne remplace pas le
diagnostic d'un professionnel de santé qualifié.**

## Architecture

```
Utilisateur → Interface Streamlit → API FastAPI → Modèle YOLO (.pt)
```

- `api/` — service FastAPI exposant `/predict`, `/health`, `/metrics`
- `streamlit_app/` — interface web permettant l'upload d'image et
  l'affichage des résultats
- Chaque service est conteneurisé indépendamment (Docker) et déployé comme
  deux services Render distincts (voir `render.yaml`)

## Fonctionnement de l'endpoint `/predict`

1. Validation du fichier (format JPEG/PNG, taille ≤ 10 Mo)
2. Détection de la tumeur par le modèle YOLO (bounding box + classe +
   confiance)
3. Si un modèle de segmentation est configuré (`MODEL_SEG_PATH`) :
   génération d'un masque pixel par pixel et calcul d'indicateurs
   cliniques indicatifs (surface en mm², diamètre maximal, localisation
   indicative). **Le volume 3D n'est pas estimé**, une coupe IRM 2D seule
   ne permettant pas ce calcul.
4. Retour d'une réponse JSON incluant systématiquement le message
   d'avertissement rappelant la nature d'aide à la décision de l'outil.

## Lancer le projet en local

```bash
docker compose up --build
```

- API disponible sur `http://localhost:8000/docs`
- Interface Streamlit sur `http://localhost:8501`

Placez au préalable vos poids de modèle dans `api/models/best.pt` (et
`api/models/best-seg.pt` si vous disposez d'un modèle de segmentation).

## Déploiement sur Render

1. Connecter le dépôt GitHub à Render.
2. Render détecte automatiquement `render.yaml` et crée les deux services
   (`brain-tumor-api` et `brain-tumor-ui`).
3. Renseigner manuellement dans le dashboard Render :
   - `ALLOWED_ORIGINS` sur `brain-tumor-api` avec l'URL du service UI ;
   - `API_URL` sur `brain-tumor-ui` avec l'URL publique du service API.
4. Le endpoint `GET /health` est utilisé par Render comme health check
   pour le service API.

## CI/CD

Le pipeline GitHub Actions (`.github/workflows/ci-cd.yml`) exécute, à
chaque push sur `main` :
1. Analyse statique (flake8) et tests unitaires de l'API ;
2. Construction des images Docker (API et interface) ;
3. Déclenchement du déploiement Render via *deploy hooks* uniquement si
   les étapes précédentes réussissent ;
4. Vérification post-déploiement via l'endpoint `/health`.

Secrets GitHub requis : `RENDER_DEPLOY_HOOK_API`, `RENDER_DEPLOY_HOOK_UI`,
`API_HEALTH_URL`.

## Monitoring

- `GET /health` — supervision de la disponibilité du service et de l'état
  de chargement des modèles (utilisé par Render et par le pipeline CI/CD).
- `GET /metrics` — métriques au format Prometheus (nombre de requêtes,
  temps de réponse, taux d'erreur), exposées via
  `prometheus-fastapi-instrumentator`. Ces métriques peuvent être scrapées
  par un Prometheus externe ou visualisées via Grafana Cloud.
- Logs applicatifs (résultats de prédiction, erreurs) consultables
  directement dans le dashboard Render (onglet *Logs*).

## Limites connues

- Les indicateurs cliniques (surface, diamètre) nécessitent un modèle de
  segmentation dédié ; sans celui-ci, seule la détection (bounding box) est
  disponible.
- Le facteur de conversion pixel → mm (`DEFAULT_PIXEL_SPACING_MM`) est une
  valeur par défaut ; il doit être ajusté si les métadonnées réelles de
  l'IRM (spacing DICOM) sont disponibles, sous peine de fausser les
  indicateurs.
