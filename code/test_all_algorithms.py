#!/usr/bin/env python3
"""
Framework pour tester tous les algorithmes de détection de piscines
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

class PoolDetectionTester:
    def __init__(self, images_dir="images", results_dir="results"):
        self.images_dir = Path(images_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.algorithms = {
            "opencv_hsv": {"dir": "code/algorithms/opencv_hsv", "file": "detector.py", "status": "ready"},
            "opencv_advanced": {"dir": "code/algorithms/opencv_advanced", "file": "detector.py", "status": "ready"},
            "cnn_simple": {"dir": "code/algorithms/cnn_simple", "file": "detector.py", "status": "ready"},
            "resnet50": {"dir": "code/algorithms/resnet50", "file": "detector.py", "status": "todo"},
            "clip_kmeans": {"dir": "code/algorithms/clip_kmeans", "file": "detector.py", "status": "todo"}
        }
    
    def list_test_images(self):
        """Liste toutes les images disponibles pour les tests"""
        if not self.images_dir.exists():
            print("❌ Aucun dossier d'images trouvé")
            return []
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(self.images_dir.glob(ext))
        
        print(f"📸 {len(image_files)} images trouvées:")
        for img in image_files:
            print(f"   - {img.name}")
        
        return image_files
    
    def test_algorithm(self, algo_name, image_path):
        """Teste un algorithme sur une image"""
        if algo_name not in self.algorithms:
            print(f"❌ Algorithme {algo_name} non trouvé")
            return None
        
        algo_dir = Path(self.algorithms[algo_name]["dir"])
        algo_file = algo_dir / self.algorithms[algo_name]["file"]
        if not algo_file.exists():
            print(f"⚠️  {algo_name}: {algo_file} pas encore implémenté")
            return None
        
        print(f"🔍 Test {algo_name} sur {image_path.name}")
        
        start_time = time.time()
        try:
            # Test spécifique pour OpenCV HSV
            if algo_name == "opencv_hsv":
                import sys
                sys.path.append(str(algo_dir))
                from detector import OpenCVHSVPoolDetector
                detector = OpenCVHSVPoolDetector()
                result = detector.detect_pools(str(image_path))
                result["processing_time"] = time.time() - start_time
                
                # Sauvegarder visualisation si des piscines détectées
                if result["success"] and result["pools_detected"] > 0:
                    vis_path = self.results_dir / f"opencv_hsv_{image_path.stem}.jpg"
                    detector.save_visualization(str(image_path), result, str(vis_path))
                
                return result
            
            # Test spécifique pour OpenCV Advanced
            elif algo_name == "opencv_advanced":
                import sys
                import importlib
                if str(algo_dir) not in sys.path:
                    sys.path.insert(0, str(algo_dir))
                
                # Force reload pour éviter les conflits d'import
                if 'detector' in sys.modules:
                    importlib.reload(sys.modules['detector'])
                
                from detector import OpenCVAdvancedPoolDetector
                detector = OpenCVAdvancedPoolDetector()
                result = detector.detect_pools(str(image_path))
                result["processing_time"] = time.time() - start_time
                
                # Sauvegarder visualisation si des piscines détectées
                if result["success"] and result["pools_detected"] > 0:
                    vis_path = self.results_dir / f"opencv_advanced_{image_path.stem}.jpg"
                    detector.save_visualization(str(image_path), result, str(vis_path))
                
                return result
            
            # Test spécifique pour CNN Simple
            elif algo_name == "cnn_simple":
                import sys
                import subprocess
                import tempfile
                
                # Activer l'environnement virtuel et lancer la détection
                venv_python = Path(algo_dir) / "venv/bin/python"
                if not venv_python.exists():
                    result = {
                        "algorithm": algo_name,
                        "image": str(image_path),
                        "error": "Environnement virtuel CNN non trouvé. Lancez d'abord ./setup.sh",
                        "processing_time": time.time() - start_time,
                        "status": "error"
                    }
                    return result
                
                # Exécuter le CNN avec l'environnement virtuel
                cmd = [str(venv_python), str(algo_dir / "detector.py"), "--image", str(image_path)]
                
                try:
                    # Capturer la sortie pour extraire les résultats
                    result_process = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=60,
                        cwd="."  # Rester dans le répertoire courant (root du projet)
                    )
                    
                    if result_process.returncode == 0:
                        # Parser la sortie pour extraire les informations
                        output = result_process.stdout
                        has_pool = "Piscine détectée: Oui" in output
                        confidence_line = [line for line in output.split('\n') if 'Confiance:' in line]
                        confidence = 0.0
                        
                        if confidence_line:
                            try:
                                conf_text = confidence_line[0].split('Confiance:')[1].strip()
                                confidence = float(conf_text.replace('%', '')) / 100
                            except:
                                confidence = 0.0
                        
                        # Créer une bbox factice pour compatibilité
                        pools = []
                        if has_pool:
                            # CNN simple ne fournit pas de coordonnées précises, on crée une bbox centrée
                            pools = [{
                                "bbox": [200, 200, 400, 300],  # x, y, w, h factice
                                "confidence": confidence,
                                "type": "pool"
                            }]
                        
                        result = {
                            "algorithm": algo_name,
                            "image_path": str(image_path),
                            "pools": pools,
                            "pools_detected": len(pools),
                            "confidence": confidence,
                            "has_pool": has_pool,
                            "processing_time": time.time() - start_time,
                            "success": True,
                            "raw_output": output
                        }
                    else:
                        result = {
                            "algorithm": algo_name,
                            "image": str(image_path),
                            "error": f"Erreur CNN: {result_process.stderr}",
                            "processing_time": time.time() - start_time,
                            "status": "error"
                        }
                        
                except subprocess.TimeoutExpired:
                    result = {
                        "algorithm": algo_name,
                        "image": str(image_path),
                        "error": "Timeout CNN (>60s)",
                        "processing_time": time.time() - start_time,
                        "status": "timeout"
                    }
                except Exception as e:
                    result = {
                        "algorithm": algo_name,
                        "image": str(image_path),
                        "error": f"Erreur subprocess CNN: {str(e)}",
                        "processing_time": time.time() - start_time,
                        "status": "error"
                    }
                
                return result
            else:
                # Placeholder pour autres algorithmes
                result = {
                    "algorithm": algo_name,
                    "image": str(image_path),
                    "pools_detected": 0,
                    "processing_time": time.time() - start_time,
                    "status": "not_implemented"
                }
        except Exception as e:
            result = {
                "algorithm": algo_name,
                "image": str(image_path),
                "error": str(e),
                "processing_time": time.time() - start_time,
                "status": "error"
            }
        
        return result
    
    def run_all_tests(self):
        """Lance tous les tests sur toutes les images"""
        images = self.list_test_images()
        if not images:
            print("❌ Aucune image à tester")
            return
        
        all_results = []
        
        print(f"\n🚀 Lancement des tests sur {len(images)} images")
        print("=" * 50)
        
        for image_path in images:
            print(f"\n📍 Test image: {image_path.name}")
            
            for algo_name in self.algorithms:
                result = self.test_algorithm(algo_name, image_path)
                if result:
                    all_results.append(result)
        
        # Sauvegarde des résultats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ Résultats sauvegardés: {results_file}")
        return all_results
    
    def show_algorithm_status(self):
        """Affiche le statut de chaque algorithme"""
        print("📋 Statut des algorithmes:")
        print("-" * 30)
        
        for name, info in self.algorithms.items():
            algo_file = Path(info["dir"]) / info["file"]
            exists = "✅" if algo_file.exists() else "❌"
            print(f"{exists} {name:15} - {info['dir']}/{info['file']}")

def main():
    tester = PoolDetectionTester()
    
    print("🏊 Framework de Test - Détection de Piscines")
    print("=" * 45)
    
    # Afficher le statut
    tester.show_algorithm_status()
    
    # Lister les images
    print("\n")
    tester.list_test_images()
    
    # Option pour lancer les tests
    print(f"\n💡 Pour lancer tous les tests: tester.run_all_tests()")

if __name__ == "__main__":
    main()