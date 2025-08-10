#!/usr/bin/env python3
"""
Outil d'annotation pour marquer manuellement les piscines sur images satellites
Cliquez pour dessiner les bounding boxes des piscines
"""

import cv2
import json
import os
from pathlib import Path
import argparse

class PoolAnnotator:
    def __init__(self, images_dir="images", output_dir="validation_dataset"):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # État de l'annotation
        self.current_image = None
        self.current_image_path = None
        self.image_display = None
        self.pools = []
        self.drawing = False
        self.start_point = None
        self.current_rect = None
        
        # Configuration d'affichage
        self.scale_factor = 1.0
        self.window_name = "Pool Annotator - Cliquez et glissez pour marquer les piscines"
        
    def load_existing_annotations(self, image_name):
        """Charge les annotations existantes pour une image"""
        json_path = self.output_dir / f"{Path(image_name).stem}.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    return data.get('pools', [])
            except:
                pass
        return []
    
    def save_annotations(self):
        """Sauvegarde les annotations actuelles"""
        if not self.current_image_path:
            return
            
        image_name = Path(self.current_image_path).name
        annotations = {
            "image": image_name,
            "image_size": [self.current_image.shape[1], self.current_image.shape[0]],
            "pools": self.pools,
            "annotated_by": "manual",
            "total_pools": len(self.pools)
        }
        
        json_path = self.output_dir / f"{Path(image_name).stem}.json"
        with open(json_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"✅ Sauvegardé {len(self.pools)} piscines dans {json_path}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback pour les événements de souris"""
        # Ajuster les coordonnées selon le scale
        actual_x = int(x / self.scale_factor)
        actual_y = int(y / self.scale_factor)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Commencer à dessiner une piscine
            self.drawing = True
            self.start_point = (actual_x, actual_y)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Mise à jour du rectangle pendant le dessin
            self.current_rect = (self.start_point[0], self.start_point[1], 
                               actual_x - self.start_point[0], actual_y - self.start_point[1])
            self.update_display()
            
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            # Finir de dessiner la piscine
            self.drawing = False
            if self.start_point:
                w = actual_x - self.start_point[0]
                h = actual_y - self.start_point[1]
                
                # Assurer que width et height sont positifs
                if w < 0:
                    self.start_point = (actual_x, self.start_point[1])
                    w = abs(w)
                if h < 0:
                    self.start_point = (self.start_point[0], actual_y)
                    h = abs(h)
                
                # Ajouter la piscine si assez grande
                if w > 5 and h > 5:
                    pool = {
                        "id": len(self.pools),
                        "bbox": [self.start_point[0], self.start_point[1], w, h],
                        "center": [self.start_point[0] + w//2, self.start_point[1] + h//2],
                        "type": "residential"  # Type par défaut
                    }
                    self.pools.append(pool)
                    print(f"➕ Piscine {len(self.pools)} ajoutée: {pool['bbox']}")
                
                self.current_rect = None
                self.update_display()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Clic droit : supprimer la piscine la plus proche
            self.remove_closest_pool(actual_x, actual_y)
    
    def remove_closest_pool(self, x, y):
        """Supprime la piscine la plus proche du clic"""
        if not self.pools:
            return
            
        closest_pool = None
        min_distance = float('inf')
        closest_index = -1
        
        for i, pool in enumerate(self.pools):
            center_x, center_y = pool["center"]
            distance = ((x - center_x)**2 + (y - center_y)**2)**0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_pool = pool
                closest_index = i
        
        if closest_pool and min_distance < 50:  # Seuil de distance
            removed_pool = self.pools.pop(closest_index)
            print(f"➖ Piscine supprimée: {removed_pool['bbox']}")
            
            # Réindexer les IDs
            for i, pool in enumerate(self.pools):
                pool["id"] = i
                
            self.update_display()
    
    def update_display(self):
        """Met à jour l'affichage avec les annotations"""
        if self.current_image is None:
            return
            
        display_img = self.current_image.copy()
        
        # Dessiner les piscines existantes
        for pool in self.pools:
            x, y, w, h = pool["bbox"]
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_img, f"Pool {pool['id']}", 
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Dessiner le rectangle en cours de création
        if self.current_rect:
            x, y, w, h = self.current_rect
            if w > 0 and h > 0:
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Redimensionner pour l'affichage si nécessaire
        if self.scale_factor != 1.0:
            new_width = int(display_img.shape[1] * self.scale_factor)
            new_height = int(display_img.shape[0] * self.scale_factor)
            display_img = cv2.resize(display_img, (new_width, new_height))
        
        self.image_display = display_img
        cv2.imshow(self.window_name, display_img)
    
    def annotate_image(self, image_path):
        """Annote une image spécifique"""
        if not Path(image_path).exists():
            print(f"❌ Image non trouvée: {image_path}")
            return False
        
        # Charger l'image
        self.current_image = cv2.imread(str(image_path))
        self.current_image_path = image_path
        
        if self.current_image is None:
            print(f"❌ Impossible de charger: {image_path}")
            return False
        
        # Charger annotations existantes
        self.pools = self.load_existing_annotations(Path(image_path).name)
        
        # Calculer le facteur d'échelle pour l'affichage
        height, width = self.current_image.shape[:2]
        max_display_size = 800
        
        if width > max_display_size or height > max_display_size:
            self.scale_factor = min(max_display_size / width, max_display_size / height)
        else:
            self.scale_factor = 1.0
        
        print(f"\n🖼️  Annotation de: {Path(image_path).name}")
        print(f"   Taille: {width}x{height}, Scale: {self.scale_factor:.2f}")
        print(f"   Piscines existantes: {len(self.pools)}")
        print("\n📝 Instructions:")
        print("   - Clic gauche + glisser: Dessiner une piscine")
        print("   - Clic droit: Supprimer la piscine la plus proche")
        print("   - 's': Sauvegarder")
        print("   - 'n': Image suivante")
        print("   - 'q': Quitter")
        
        # Configurer la fenêtre
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quitter
                self.save_annotations()
                return False
            elif key == ord('s'):  # Sauvegarder
                self.save_annotations()
            elif key == ord('n'):  # Image suivante
                self.save_annotations()
                return True
            elif key == 27:  # ESC
                self.save_annotations()
                return False
        
        return True
    
    def annotate_all_images(self):
        """Annote toutes les images du dossier"""
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(self.images_dir.glob(ext))
        
        if not image_files:
            print(f"❌ Aucune image trouvée dans {self.images_dir}")
            return
        
        print(f"📸 {len(image_files)} images trouvées")
        
        for i, image_path in enumerate(image_files):
            print(f"\n--- Image {i+1}/{len(image_files)} ---")
            
            continue_annotation = self.annotate_image(image_path)
            if not continue_annotation:
                break
        
        cv2.destroyAllWindows()
        print(f"\n✅ Annotation terminée! Fichiers sauvés dans {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Outil d'annotation de piscines")
    parser.add_argument("--images", default="../images", help="Dossier des images")
    parser.add_argument("--output", default="../validation_dataset", help="Dossier de sortie")
    parser.add_argument("--image", help="Annoter une image spécifique")
    
    args = parser.parse_args()
    
    annotator = PoolAnnotator(args.images, args.output)
    
    if args.image:
        annotator.annotate_image(args.image)
    else:
        annotator.annotate_all_images()

if __name__ == "__main__":
    main()