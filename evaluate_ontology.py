import json
import re
from typing import List, Dict, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import sys
sys.path.append('.')
from intent_ontologist import IntentOntologist

class OntologyEvaluator:
    def __init__(self, ontology_file: str = "intent_ontology.json", conversations_file: str = "data/conversations.txt"):
        self.ontology_file = ontology_file
        self.conversations_file = conversations_file
        self.ontologist = IntentOntologist(conversations_file)
        self.conversations = []
        self.classifications = []
        
    def load_ontology(self):
        """Load existing ontology"""
        try:
            with open(self.ontology_file, 'r') as f:
                ontology_data = json.load(f)
            
            from models.intent_models import IntentCategory
            for category_data in ontology_data['categories']:
                category = IntentCategory(
                    name=category_data['name'],
                    description=category_data['description'],
                    examples=category_data['examples']
                )
                self.ontologist.ontology.add_category(category)
            
            print(f"Loaded ontology with {len(ontology_data['categories'])} categories")
            return True
        except Exception as e:
            print(f"Error loading ontology: {e}")
            return False
    
    def load_conversations(self):
        """Load conversations from file"""
        try:
            with open(self.conversations_file, 'r', encoding='utf-8') as file:
                content = file.read()
            
            conversations = re.split(r'(?="Agent: Thank you for calling)', content)
            conversations = [conv.strip().strip('"') for conv in conversations if conv.strip()]
            
            self.conversations = []
            for i, conversation in enumerate(conversations):
                if conversation.strip():
                    self.conversations.append({
                        'id': f'conv_{i+1:03d}',
                        'text': conversation
                    })
            
            print(f"Loaded {len(self.conversations)} conversations")
            return True
        except Exception as e:
            print(f"Error loading conversations: {e}")
            return False
    
    def classify_all_conversations(self):
        """Classify all conversations using existing ontology"""
        print("Classifying conversations...")
        
        self.classifications = []
        self.category_scores = []
        for i, conv in enumerate(self.conversations):
            if i % 25 == 0:
                print(f"Processed {i}/{len(self.conversations)} conversations")
            
            classification, scores = self.ontologist.classify_intent_with_scores(conv['id'], conv['text'])
            self.classifications.append(classification)
            self.category_scores.append(scores)
        
        print(f"Classified {len(self.classifications)} conversations")
    
    def calculate_basic_metrics(self) -> Dict:
        """Calculate basic evaluation metrics"""
        if not self.classifications:
            return {}
        
        confidence_scores = [c.confidence for c in self.classifications]
        reuse_count = sum(1 for c in self.classifications if 'use_existing' in str(c.action).lower())
        low_confidence_count = sum(1 for score in confidence_scores if score < 0.6)
        
        metrics = {
            'total_conversations': len(self.classifications),
            'avg_confidence': sum(confidence_scores) / len(confidence_scores),
            'reuse_rate': reuse_count / len(self.classifications),
            'low_confidence_rate': low_confidence_count / len(self.classifications),
            'total_categories': len(self.ontologist.ontology.categories),
            'intent_distribution': Counter(c.intent_name for c in self.classifications)
        }
        
        return metrics
    
    def check_cluster_consistency(self) -> Dict:
        """Check if semantically similar conversations get same classification"""
        print("Analyzing cluster consistency...")
        
        # Create clusters using same method as EDA
        conversation_texts = [conv['text'] for conv in self.conversations]
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        X = vectorizer.fit_transform(conversation_texts)
        
        # Use 5 clusters like in EDA
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Analyze consistency within each cluster
        cluster_analysis = {}
        total_inconsistencies = 0
        
        for cluster_id in range(5):
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            cluster_classifications = [self.classifications[i].intent_name for i in cluster_indices]
            
            # Calculate consistency
            intent_counts = Counter(cluster_classifications)
            most_common_intent, most_common_count = intent_counts.most_common(1)[0]
            consistency_rate = most_common_count / len(cluster_classifications)
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster_indices),
                'intents_found': list(intent_counts.keys()),
                'consistency_rate': consistency_rate,
                'dominant_intent': most_common_intent
            }
            
            if consistency_rate < 0.8:  # Flag clusters with low consistency
                total_inconsistencies += 1
        
        return {
            'cluster_analysis': cluster_analysis,
            'total_inconsistent_clusters': total_inconsistencies,
            'avg_consistency': np.mean([c['consistency_rate'] for c in cluster_analysis.values()])
        }
    
    def calculate_classification_margins(self) -> Dict:
        """Calculate real margins between top classifications"""
        print("Calculating classification margins...")
        
        uncertain_classifications = []
        margin_scores = []
        
        for i, scores in enumerate(self.category_scores):
            if not scores:  # Skip if no scores available
                continue
                
            # Calculate real margin
            sorted_scores = sorted(scores.values(), reverse=True)
            if len(sorted_scores) >= 2:
                top_score = sorted_scores[0]
                second_score = sorted_scores[1]
                margin = top_score - second_score
            else:
                margin = sorted_scores[0] if sorted_scores else 0
            
            margin_scores.append(margin)
            
            # Flag low margin cases
            if margin < 0.3:
                uncertain_classifications.append({
                    'conversation_id': self.conversations[i]['id'],
                    'intent': self.classifications[i].intent_name,
                    'confidence': self.classifications[i].confidence,
                    'margin': margin,
                    'top_scores': dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]),
                    'snippet': self.conversations[i]['text'][:150]
                })
        
        return {
            'avg_margin': np.mean(margin_scores) if margin_scores else 0,
            'low_margin_count': len(uncertain_classifications),
            'low_margin_rate': len(uncertain_classifications) / len(self.classifications) if self.classifications else 0,
            'uncertain_examples': uncertain_classifications[:5]  # Show top 5
        }
    
    def generate_report(self, basic_metrics: Dict, cluster_analysis: Dict, margin_analysis: Dict) -> str:
        """Generate comprehensive evaluation report"""
        lines = []
        lines.append("Ontology Evaluation Report")
        lines.append("=" * 50)
        
        # Basic metrics
        lines.append(f"\nBasic Metrics")
        lines.append("-" * 20)
        lines.append(f"Total conversations: {basic_metrics['total_conversations']}")
        lines.append(f"Average confidence: {basic_metrics['avg_confidence']:.3f}")
        lines.append(f"Category reuse rate: {basic_metrics['reuse_rate']:.3f}")
        lines.append(f"Low confidence rate: {basic_metrics['low_confidence_rate']:.3f}")
        lines.append(f"Total categories: {basic_metrics['total_categories']}")
        
        # Threshold assessment
        passes_thresholds = (
            basic_metrics['avg_confidence'] >= 0.9 and
            basic_metrics['reuse_rate'] >= 0.9 and
            basic_metrics['low_confidence_rate'] <= 0.1
        )
        status = "Passed thresholds" if passes_thresholds else "Failed thresholds"
        lines.append(f"Threshold assessment: {status}")
        
        # Intent distribution
        lines.append(f"\nIntent Distribution")
        lines.append("-" * 20)
        for intent, count in basic_metrics['intent_distribution'].most_common():
            lines.append(f"{intent}: {count}")
        
        # Cluster consistency
        lines.append(f"\nCluster Consistency Analysis")
        lines.append("-" * 30)
        lines.append(f"Average consistency rate: {cluster_analysis['avg_consistency']:.3f}")
        lines.append(f"Inconsistent clusters: {cluster_analysis['total_inconsistent_clusters']}/5")
        
        for cluster_id, data in cluster_analysis['cluster_analysis'].items():
            if data['consistency_rate'] < 0.8:
                lines.append(f"Cluster {cluster_id}: {data['consistency_rate']:.3f} consistency")
                lines.append(f"  Dominant intent: {data['dominant_intent']}")
                lines.append(f"  Mixed intents: {data['intents_found']}")
        
        # Margin analysis
        lines.append(f"\nClassification Margin Analysis")
        lines.append("-" * 35)
        lines.append(f"Average margin: {margin_analysis['avg_margin']:.3f}")
        lines.append(f"Low margin rate: {margin_analysis['low_margin_rate']:.3f}")
        
        if margin_analysis['uncertain_examples']:
            lines.append(f"\nUncertain classifications:")
            for example in margin_analysis['uncertain_examples']:
                lines.append(f"  {example['conversation_id']}: {example['intent']} (margin: {example['margin']:.3f})")
        
        return "\n".join(lines)
    
    def run_evaluation(self):
        """Run complete evaluation"""
        if not self.load_ontology():
            return
        
        if not self.load_conversations():
            return
        
        self.classify_all_conversations()
        
        basic_metrics = self.calculate_basic_metrics()
        cluster_analysis = self.check_cluster_consistency()
        margin_analysis = self.calculate_classification_margins()
        
        report = self.generate_report(basic_metrics, cluster_analysis, margin_analysis)
        print("\n" + report)
        
        # Save report
        with open("evaluation_report.txt", "w") as f:
            f.write(report)
        print(f"\nReport saved to evaluation_report.txt")

if __name__ == "__main__":
    evaluator = OntologyEvaluator()
    evaluator.run_evaluation()