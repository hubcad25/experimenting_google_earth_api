#!/usr/bin/env python3
"""
Script simple pour récupérer une image satellite via Google Maps Static API
"""

import os
import requests
from datetime import datetime
from dotenv import load_dotenv

# Charger les variables du fichier .env
load_dotenv()

def fetch_satellite_image(lat, lng, label, zoom=18, size="640x640", api_key=None):
    """
    Récupère une image satellite pour des coordonnées données avec nom personnalisé
    
    Args:
        lat (float): Latitude
        lng (float): Longitude
        label (str): Nom/label pour l'image  
        zoom (int): Niveau de zoom (1-20, 18+ pour détails maisons)
        size (str): Taille image "widthxheight" (max 640x640 gratuit)
        api_key (str): Clé API Google Maps
        
    Returns:
        bool: True si succès, False sinon
    """
    if not api_key:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("❌ Clé API manquante. Définissez GOOGLE_API_KEY")
            return False
    
    # URL Google Maps Static API
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    
    params = {
        'center': f"{lat},{lng}",
        'zoom': zoom,
        'size': size,
        'maptype': 'satellite',
        'key': api_key
    }
    
    try:
        print(f"🛰️  Récupération image satellite: {lat}, {lng} (zoom {zoom})")
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        # Nom fichier avec label personnalisé
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_z{zoom}_{timestamp}.jpg"
        filepath = os.path.join("../images", filename)
        
        # Créer dossier images si inexistant
        os.makedirs("../images", exist_ok=True)
        
        # Sauvegarder image
        with open(filepath, 'wb') as f:
            f.write(response.content)
            
        print(f"✅ Image sauvegardée: {filepath}")
        print(f"   Taille: {len(response.content)} bytes")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur requête: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    """Exemple d'utilisation"""
    
    # Coordonnées avec labels personnalisés
    test_locations = [
        (45.5780309414347, -73.1879815989526, "chapleau"),
        (45.53280102910986, -73.20984197686305, "bousquet"), 
        (45.58021382961247, -73.18089712864787, "gilbertdionne"),
        (45.55994802880481, -73.19360986021864, "sthilairerandom"),
        (45.58319990832722, -73.20429801010346, "beloeil"),
        (46.71361046403241, -71.20549487091733, "levis")
    ]
    
    print("🚀 Test récupération images satellites")
    print("-" * 40)
    
    for i, (lat, lng, label) in enumerate(test_locations, 1):
        print(f"\n📍 Location {i}/{len(test_locations)}: {label}")
        success = fetch_satellite_image(lat, lng, label, zoom=19)
        if not success:
            print(f"Échec pour la location {i}")

if __name__ == "__main__":
    main()