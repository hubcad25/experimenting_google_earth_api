#!/bin/bash
# Setup script pour l'algorithme CNN Simple

echo "🔧 Configuration environnement CNN Simple"

# Créer et activer l'environnement virtuel
if [ ! -d "venv" ]; then
    echo "Création du venv..."
    python3 -m venv venv
fi

echo "Activation du venv..."
source venv/bin/activate

echo "Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Environnement CNN Simple prêt!"
echo "Pour l'activer: source code/algorithms/cnn_simple/venv/bin/activate"