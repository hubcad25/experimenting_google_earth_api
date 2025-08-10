# Algorithmes de Détection de Piscines

Chaque algorithme a son propre environnement virtuel pour éviter les conflits de dépendances.

## Structure

```
algorithms/
├── opencv_hsv/          # Détection HSV simple
│   ├── detector.py      # Code principal
│   ├── requirements.txt # Dépendances
│   ├── setup.sh        # Script d'installation
│   └── venv/           # Environnement virtuel (créé par setup.sh)
├── opencv_advanced/     # OpenCV avancé
├── cnn_simple/         # CNN 3 couches
├── resnet50/           # ResNet50 + heatmaps
└── clip_kmeans/        # CLIP + K-means
```

## Installation et Test

### 1. OpenCV HSV (Simple)
```bash
cd opencv_hsv
./setup.sh
source venv/bin/activate
python detector.py
```

### 2. Autres algorithmes
Pas encore implémentés.

## Usage depuis le Framework Principal

Le framework `test_all_algorithms.py` activera automatiquement le bon environnement pour chaque algorithme.