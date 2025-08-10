# Dataset de Validation

Ce dossier contient les annotations manuelles pour valider la qualité des algorithmes.

## Structure

```
validation_dataset/
├── satellite_image1.json    # Annotations pour image1
├── satellite_image2.json    # Annotations pour image2
└── validation_results.json  # Résultats d'évaluation des algos
```

## Format des annotations

Chaque fichier JSON contient :
- `image` : Nom du fichier image
- `image_size` : Dimensions [width, height]
- `pools` : Liste des piscines avec bounding boxes
- `total_pools` : Nombre total de piscines

## Utilisation

1. Utilisez `annotation_tool/pool_annotator.py` pour créer les annotations
2. Lancez `validation_evaluator.py` pour évaluer les algorithmes
3. Comparez les métriques de performance