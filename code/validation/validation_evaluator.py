#!/usr/bin/env python3
"""
Évaluateur pour comparer les performances des algorithmes de détection de piscines
Charge les annotations manuelles et les résultats d'algorithmes pour calculer les métriques
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

class ValidationEvaluator:
    def __init__(self, validation_dir="../../validation_dataset", results_dir="../../results"):
        self.validation_dir = Path(validation_dir)
        self.results_dir = Path(results_dir)
        
    def load_manual_annotations(self) -> Dict[str, Dict]:
        """Charge toutes les annotations manuelles"""
        annotations = {}
        
        for json_file in self.validation_dir.glob("*.json"):
            if json_file.name == "validation_results.json":
                continue  # Skip les résultats d'évaluation
                
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    image_name = data["image"]
                    annotations[image_name] = data
            except Exception as e:
                print(f"⚠️  Erreur lecture {json_file}: {e}")
        
        return annotations
    
    def load_algorithm_results(self) -> Dict[str, List[Dict]]:
        """Charge tous les résultats d'algorithmes"""
        algorithm_results = {}
        
        for json_file in self.results_dir.glob("test_results_*.json"):
            try:
                with open(json_file, 'r') as f:
                    results = json.load(f)
                    
                    for result in results:
                        if not result.get("success", False):
                            continue
                            
                        algorithm = result["algorithm"]
                        image_path = result["image_path"]
                        image_name = Path(image_path).name
                        
                        if algorithm not in algorithm_results:
                            algorithm_results[algorithm] = {}
                        
                        algorithm_results[algorithm][image_name] = result
                        
            except Exception as e:
                print(f"⚠️  Erreur lecture {json_file}: {e}")
        
        return algorithm_results
    
    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calcule l'Intersection over Union entre deux bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Coordonnées des coins
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_max, y2_max = x2 + w2, y2 + h2
        
        # Intersection
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def match_detections(self, ground_truth: List[Dict], predictions: List[Dict], 
                        iou_threshold: float = 0.1) -> Tuple[List[Tuple], List[int], List[int]]:
        """
        Match les prédictions avec la ground truth
        Returns: (matches, unmatched_gt_indices, unmatched_pred_indices)
        """
        matches = []
        used_gt = set()
        used_pred = set()
        
        # Calcul de toutes les IoU
        iou_matrix = []
        for i, gt_pool in enumerate(ground_truth):
            iou_row = []
            for j, pred_pool in enumerate(predictions):
                iou = self.calculate_iou(gt_pool["bbox"], pred_pool["bbox"])
                iou_row.append(iou)
            iou_matrix.append(iou_row)
        
        # Matching glouton (prendre les meilleurs IoU d'abord)
        while True:
            best_iou = 0
            best_match = None
            
            for i in range(len(ground_truth)):
                if i in used_gt:
                    continue
                for j in range(len(predictions)):
                    if j in used_pred:
                        continue
                    
                    if iou_matrix[i][j] > best_iou and iou_matrix[i][j] >= iou_threshold:
                        best_iou = iou_matrix[i][j]
                        best_match = (i, j, best_iou)
            
            if best_match is None:
                break
                
            i, j, iou = best_match
            matches.append((i, j, iou))
            used_gt.add(i)
            used_pred.add(j)
        
        # Unmatched indices
        unmatched_gt = [i for i in range(len(ground_truth)) if i not in used_gt]
        unmatched_pred = [j for j in range(len(predictions)) if j not in used_pred]
        
        return matches, unmatched_gt, unmatched_pred
    
    def calculate_metrics(self, matches: List[Tuple], num_ground_truth: int, 
                         num_predictions: int) -> Dict[str, float]:
        """Calcule les métriques de performance"""
        true_positives = len(matches)
        false_positives = num_predictions - true_positives
        false_negatives = num_ground_truth - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
    
    def match_image_names(self, manual_image_name: str, algorithm_results: Dict[str, Dict]) -> str:
        """Match manual annotation image name with algorithm result image name"""
        # Extract label from validation JSON filename (e.g., chapleau.json -> chapleau)
        if manual_image_name.startswith("satellite_"):
            # Map coordinates to labels
            coord_to_label = {
                "45.5780309414347_-73.1879815989526": "chapleau",
                "45.53280102910986_-73.20984197686305": "bousquet",
                "45.58021382961247_-73.18089712864787": "gilbertdionne", 
                "45.55994802880481_-73.19360986021864": "sthilairerandom",
                "45.58319990832722_-73.20429801010346": "beloeil",
                "46.71361046403241_-71.20549487091733": "levis"
            }
            
            for coord_pattern, label in coord_to_label.items():
                if coord_pattern in manual_image_name:
                    # Find algorithm result with this label
                    for algo_image_name in algorithm_results.keys():
                        if algo_image_name.startswith(label + "_"):
                            return algo_image_name
        
        # Direct match attempt
        if manual_image_name in algorithm_results:
            return manual_image_name
            
        return None

    def evaluate_algorithm(self, algorithm_name: str, algorithm_results: Dict[str, Dict],
                          manual_annotations: Dict[str, Dict], iou_threshold: float = 0.1) -> Dict:
        """Évalue un algorithme spécifique"""
        
        total_metrics = {
            "true_positives": 0,
            "false_positives": 0, 
            "false_negatives": 0
        }
        
        image_results = {}
        
        for image_name, manual_data in manual_annotations.items():
            manual_image_name = manual_data["image"]
            matched_algo_image = self.match_image_names(manual_image_name, algorithm_results)
            
            if not matched_algo_image:
                print(f"⚠️  {algorithm_name}: Pas de résultats pour {manual_image_name}")
                continue
            
            ground_truth = manual_data["pools"]
            predictions = algorithm_results[matched_algo_image]["pools"]
            
            matches, unmatched_gt, unmatched_pred = self.match_detections(
                ground_truth, predictions, iou_threshold
            )
            
            metrics = self.calculate_metrics(matches, len(ground_truth), len(predictions))
            
            # Accumulation des métriques totales
            total_metrics["true_positives"] += metrics["true_positives"]
            total_metrics["false_positives"] += metrics["false_positives"]
            total_metrics["false_negatives"] += metrics["false_negatives"]
            
            # Détails par image
            image_results[image_name] = {
                "ground_truth_pools": len(ground_truth),
                "predicted_pools": len(predictions),
                "matches": len(matches),
                "metrics": metrics,
                "matched_pairs": matches,
                "avg_iou": sum(match[2] for match in matches) / len(matches) if matches else 0.0
            }
        
        # Calcul des métriques globales
        tp = total_metrics["true_positives"]
        fp = total_metrics["false_positives"]
        fn = total_metrics["false_negatives"]
        
        global_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        global_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0
        
        return {
            "algorithm": algorithm_name,
            "iou_threshold": iou_threshold,
            "global_metrics": {
                "precision": global_precision,
                "recall": global_recall,
                "f1_score": global_f1,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn
            },
            "per_image_results": image_results,
            "total_images_evaluated": len(image_results)
        }
    
    def run_evaluation(self, iou_threshold: float = 0.1) -> Dict:
        """Lance l'évaluation complète de tous les algorithmes"""
        
        print(f"🔍 Évaluation des algorithmes (IoU seuil: {iou_threshold})")
        print("=" * 60)
        
        # Charger les données
        manual_annotations = self.load_manual_annotations()
        algorithm_results = self.load_algorithm_results()
        
        if not manual_annotations:
            print("❌ Aucune annotation manuelle trouvée!")
            return {}
        
        if not algorithm_results:
            print("❌ Aucun résultat d'algorithme trouvé!")
            return {}
        
        print(f"📋 {len(manual_annotations)} images annotées manuellement")
        print(f"🤖 {len(algorithm_results)} algorithmes trouvés: {list(algorithm_results.keys())}")
        
        # Évaluer chaque algorithme
        evaluation_results = {}
        
        for algorithm_name, algo_results in algorithm_results.items():
            print(f"\n🔬 Évaluation de {algorithm_name}...")
            
            evaluation_results[algorithm_name] = self.evaluate_algorithm(
                algorithm_name, algo_results, manual_annotations, iou_threshold
            )
        
        # Sauvegarde des résultats
        output_path = self.validation_dir / "validation_results.json"
        with open(output_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"\n✅ Résultats sauvés dans {output_path}")
        return evaluation_results
    
    def print_summary(self, evaluation_results: Dict):
        """Affiche un résumé des performances"""
        
        print("\n📊 RÉSUMÉ DES PERFORMANCES")
        print("=" * 50)
        
        algorithms = []
        for algo_name, results in evaluation_results.items():
            metrics = results["global_metrics"]
            algorithms.append((
                algo_name,
                metrics["precision"],
                metrics["recall"], 
                metrics["f1_score"],
                results["total_images_evaluated"]
            ))
        
        # Tri par F1-score décroissant
        algorithms.sort(key=lambda x: x[3], reverse=True)
        
        print(f"{'Algorithme':<15} | {'Précision':<9} | {'Rappel':<7} | {'F1-Score':<8} | {'Images'}")
        print("-" * 60)
        
        for algo, precision, recall, f1, images in algorithms:
            print(f"{algo:<15} | {precision:8.3f} | {recall:6.3f} | {f1:7.3f} | {images:6d}")
        
        # Meilleur algorithme
        if algorithms:
            best_algo, best_p, best_r, best_f1, _ = algorithms[0]
            print(f"\n🏆 Meilleur algorithme: {best_algo} (F1: {best_f1:.3f})")

def main():
    parser = argparse.ArgumentParser(description="Évaluateur de validation pour algorithmes de piscines")
    parser.add_argument("--iou", type=float, default=0.1, help="Seuil IoU pour matching (défaut: 0.1)")
    parser.add_argument("--validation-dir", default="../../validation_dataset", help="Dossier des annotations manuelles")
    parser.add_argument("--results-dir", default="../../results", help="Dossier des résultats d'algorithmes")
    
    args = parser.parse_args()
    
    evaluator = ValidationEvaluator(args.validation_dir, args.results_dir)
    results = evaluator.run_evaluation(args.iou)
    
    if results:
        evaluator.print_summary(results)

if __name__ == "__main__":
    main()