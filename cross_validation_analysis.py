import json
import re
import random
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
import sys
sys.path.append('.')
from intent_ontologist import IntentOntologist
from models.intent_models import IntentCategory

class CrossValidationAnalyzer:
    def __init__(self, conversations_file: str = "data/conversations.txt", n_folds: int = 3):
        self.conversations_file = conversations_file
        self.n_folds = n_folds
        self.conversations = []
        
    def load_conversations(self) -> List[Dict]:
        """Load conversations from file"""
        try:
            with open(self.conversations_file, 'r', encoding='utf-8') as file:
                content = file.read()
            
            conversations = re.split(r'(?="Agent: Thank you for calling)', content)
            conversations = [conv.strip().strip('"') for conv in conversations if conv.strip()]
            
            parsed_conversations = []
            for i, conversation in enumerate(conversations):
                if conversation.strip():
                    parsed_conversations.append({
                        'id': f'conv_{i+1:03d}',
                        'conversation': conversation
                    })
            
            print(f"Loaded {len(parsed_conversations)} conversations")
            return parsed_conversations
            
        except FileNotFoundError:
            print(f"Error: Could not find file {self.conversations_file}")
            return []

    def create_cross_validation_splits(self, conversations: List[Dict]) -> List[List[Dict]]:
        """Create k-fold splits with reproducible randomization"""
        random.seed(42)
        shuffled_conversations = conversations.copy()
        random.shuffle(shuffled_conversations)
        
        fold_size = len(shuffled_conversations) // self.n_folds
        splits = []
        
        for i in range(self.n_folds):
            start_idx = i * fold_size
            if i == self.n_folds - 1:  # Last fold gets remainder
                end_idx = len(shuffled_conversations)
            else:
                end_idx = (i + 1) * fold_size
            
            splits.append(shuffled_conversations[start_idx:end_idx])
        
        print(f"Created {self.n_folds} folds with sizes: {[len(split) for split in splits]}")
        return splits

    def run_intent_discovery_on_split(self, split_conversations: List[Dict], split_id: int) -> List[IntentCategory]:
        """Run intent discovery on a single split"""
        print(f"Running intent discovery on split {split_id + 1}...")
        
        # Create temporary file for this split
        temp_file = f"temp_split_{split_id}.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            for conv in split_conversations:
                f.write(f'"{conv["conversation"]}"\n')
        
        # Run intent discovery
        ontologist = IntentOntologist(temp_file)
        ontologist.build_ontology()
        
        # Clean up temp file
        import os
        os.remove(temp_file)
        
        print(f"Split {split_id + 1}: Found {len(ontologist.ontology.categories)} categories")
        return ontologist.ontology.categories

    def calculate_intent_stability_score(self, all_split_categories: List[List[IntentCategory]]) -> Dict:
        """Calculate intent stability across splits"""
        # Get all unique intent names across splits
        all_intent_names = set()
        split_intent_sets = []
        
        for categories in all_split_categories:
            intent_names = {cat.name for cat in categories}
            all_intent_names.update(intent_names)
            split_intent_sets.append(intent_names)
        
        # Find common intents (appear in all splits)
        common_intents = set.intersection(*split_intent_sets) if split_intent_sets else set()
        
        # Calculate stability metrics
        avg_intents_per_split = np.mean([len(intent_set) for intent_set in split_intent_sets])
        stability_score = len(common_intents) / avg_intents_per_split if avg_intents_per_split > 0 else 0
        
        # Intent frequency across splits
        intent_frequency = Counter()
        for intent_set in split_intent_sets:
            for intent in intent_set:
                intent_frequency[intent] += 1
        
        return {
            'total_unique_intents': len(all_intent_names),
            'common_intents': list(common_intents),
            'stability_score': stability_score,
            'avg_intents_per_split': avg_intents_per_split,
            'intent_frequency': dict(intent_frequency),
            'split_intent_counts': [len(intent_set) for intent_set in split_intent_sets]
        }

    def calculate_volume_correlation(self, all_split_categories: List[List[IntentCategory]]) -> Dict:
        """Calculate Spearman correlation of intent rankings by volume"""
        # Get volume rankings for each split
        split_rankings = []
        all_intents = set()
        
        for categories in all_split_categories:
            # Sort by volume (descending) and create ranking
            sorted_categories = sorted(categories, key=lambda x: x.volume(), reverse=True)
            ranking = {cat.name: idx for idx, cat in enumerate(sorted_categories)}
            split_rankings.append(ranking)
            all_intents.update(ranking.keys())
        
        if len(split_rankings) < 2:
            return {'correlations': [], 'avg_correlation': 0.0}
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(split_rankings)):
            for j in range(i + 1, len(split_rankings)):
                ranking1 = split_rankings[i]
                ranking2 = split_rankings[j]
                
                # Get common intents between the two splits
                common_intents = set(ranking1.keys()) & set(ranking2.keys())
                
                if len(common_intents) > 1:
                    ranks1 = [ranking1[intent] for intent in common_intents]
                    ranks2 = [ranking2[intent] for intent in common_intents]
                    
                    correlation, p_value = spearmanr(ranks1, ranks2)
                    correlations.append({
                        'split_pair': (i + 1, j + 1),
                        'correlation': correlation if not np.isnan(correlation) else 0.0,
                        'p_value': p_value if not np.isnan(p_value) else 1.0,
                        'common_intents_count': len(common_intents)
                    })
        
        avg_correlation = np.mean([c['correlation'] for c in correlations]) if correlations else 0.0
        
        return {
            'correlations': correlations,
            'avg_correlation': avg_correlation
        }

    def analyze_intent_name_consistency(self, all_split_categories: List[List[IntentCategory]]) -> Dict:
        """Analyze how consistently similar customer needs get same names"""
        # This is a simplified version - in practice you'd need semantic similarity
        # For now, we'll look at description similarity and naming patterns
        
        consistency_analysis = {}
        all_categories = []
        
        for split_id, categories in enumerate(all_split_categories):
            for cat in categories:
                all_categories.append({
                    'split_id': split_id,
                    'name': cat.name,
                    'description': cat.description,
                    'volume': cat.volume()
                })
        
        # Group by similar names (basic string matching)
        name_groups = {}
        for cat in all_categories:
            # Simple grouping by name similarity
            name_words = set(cat['name'].split('_'))
            
            found_group = False
            for group_key in name_groups:
                group_words = set(group_key.split('_'))
                # If they share significant word overlap, group them
                if len(name_words & group_words) >= 2:
                    name_groups[group_key].append(cat)
                    found_group = True
                    break
            
            if not found_group:
                name_groups[cat['name']] = [cat]
        
        # Analyze consistency within groups
        consistent_groups = 0
        total_groups = len(name_groups)
        
        for group_name, group_cats in name_groups.items():
            if len(group_cats) > 1:
                unique_names = len(set(cat['name'] for cat in group_cats))
                if unique_names == 1:  # All have same name
                    consistent_groups += 1
        
        consistency_rate = consistent_groups / total_groups if total_groups > 0 else 0
        
        return {
            'name_groups': {k: [cat['name'] for cat in v] for k, v in name_groups.items()},
            'consistency_rate': consistency_rate,
            'total_groups': total_groups,
            'consistent_groups': consistent_groups
        }

    def generate_cross_validation_report(self, stability_analysis: Dict, volume_analysis: Dict, consistency_analysis: Dict) -> str:
        """Generate comprehensive cross-validation report"""
        lines = []
        lines.append("Cross-Validation Analysis Report")
        lines.append("=" * 50)
        
        # Intent Stability
        lines.append(f"\nIntent Stability Analysis")
        lines.append("-" * 30)
        lines.append(f"Total unique intents across all splits: {stability_analysis['total_unique_intents']}")
        lines.append(f"Common intents (appear in all splits): {len(stability_analysis['common_intents'])}")
        lines.append(f"Stability score: {stability_analysis['stability_score']:.3f}")
        lines.append(f"Average intents per split: {stability_analysis['avg_intents_per_split']:.1f}")
        lines.append(f"Split intent counts: {stability_analysis['split_intent_counts']}")
        
        if stability_analysis['common_intents']:
            lines.append(f"\nStable intents (all splits):")
            for intent in sorted(stability_analysis['common_intents']):
                lines.append(f"  - {intent}")
        
        lines.append(f"\nIntent frequency across splits:")
        for intent, freq in sorted(stability_analysis['intent_frequency'].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {intent}: {freq}/{self.n_folds} splits")
        
        # Volume Correlation
        lines.append(f"\nVolume Correlation Analysis")
        lines.append("-" * 30)
        lines.append(f"Average Spearman correlation: {volume_analysis['avg_correlation']:.3f}")
        
        if volume_analysis['correlations']:
            lines.append(f"\nPairwise correlations:")
            for corr in volume_analysis['correlations']:
                lines.append(f"  Splits {corr['split_pair'][0]}-{corr['split_pair'][1]}: r={corr['correlation']:.3f} (p={corr['p_value']:.3f})")
        
        # Name Consistency
        lines.append(f"\nIntent Name Consistency Analysis")
        lines.append("-" * 35)
        lines.append(f"Name consistency rate: {consistency_analysis['consistency_rate']:.3f}")
        lines.append(f"Consistent groups: {consistency_analysis['consistent_groups']}/{consistency_analysis['total_groups']}")
        
        # Overall Assessment
        lines.append(f"\nOverall Stability Assessment")
        lines.append("-" * 30)
        
        # Define thresholds
        stability_threshold = 0.7
        correlation_threshold = 0.6
        consistency_threshold = 0.8
        
        stability_pass = stability_analysis['stability_score'] >= stability_threshold
        correlation_pass = volume_analysis['avg_correlation'] >= correlation_threshold
        consistency_pass = consistency_analysis['consistency_rate'] >= consistency_threshold
        
        lines.append(f"Stability score: {'PASS' if stability_pass else 'FAIL'} (≥{stability_threshold})")
        lines.append(f"Volume correlation: {'PASS' if correlation_pass else 'FAIL'} (≥{correlation_threshold})")
        lines.append(f"Name consistency: {'PASS' if consistency_pass else 'FAIL'} (≥{consistency_threshold})")
        
        overall_pass = stability_pass and correlation_pass and consistency_pass
        lines.append(f"\nOverall: {'STABLE ONTOLOGY' if overall_pass else 'UNSTABLE ONTOLOGY'}")
        
        return "\n".join(lines)

    def run_cross_validation_analysis(self):
        """Run complete cross-validation analysis"""
        print("Starting cross-validation analysis...")
        
        # Load conversations
        conversations = self.load_conversations()
        if not conversations:
            return
        
        # Create splits
        splits = self.create_cross_validation_splits(conversations)
        
        # Run intent discovery on each split
        all_split_categories = []
        for i, split in enumerate(splits):
            categories = self.run_intent_discovery_on_split(split, i)
            all_split_categories.append(categories)
        
        # Calculate stability metrics
        stability_analysis = self.calculate_intent_stability_score(all_split_categories)
        volume_analysis = self.calculate_volume_correlation(all_split_categories)
        consistency_analysis = self.analyze_intent_name_consistency(all_split_categories)
        
        # Generate report
        report = self.generate_cross_validation_report(stability_analysis, volume_analysis, consistency_analysis)
        print("\n" + report)
        
        # Save report
        with open("cross_validation_report.txt", "w") as f:
            f.write(report)
        print(f"\nCross-validation report saved to cross_validation_report.txt")
        
        return {
            'stability_analysis': stability_analysis,
            'volume_analysis': volume_analysis,
            'consistency_analysis': consistency_analysis
        }

if __name__ == "__main__":
    analyzer = CrossValidationAnalyzer()
    results = analyzer.run_cross_validation_analysis()