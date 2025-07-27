# Using an LLM to loop through conversations and iteratively add more customer intents to the ontology

import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict
from models.intent_models import IntentClassification, IntentOntology, IntentCategory, IntentAction

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class IntentOntologist:
    def __init__(self, conversations_file: str = "data/conversations.txt"):
        self.ontology = IntentOntology()
        self.conversations_file = conversations_file

    def load_conversations_from_file(self) -> List[Dict]:
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

    def classify_intent_with_scores(self, conversation_id: str, conversation: str) -> tuple:
        """Classify intent and return all category scores for margin analysis"""
        existing_categories = self.ontology.get_category_names()
        
        if not existing_categories:
            # No existing categories, use regular classification
            classification = self.classify_intent(conversation_id, conversation)
            return classification, {}
        
        # Get scores for all existing categories
        category_scores = {}
        for category in existing_categories:
            score_prompt = f"""Rate how well this conversation fits the category "{category}" on a scale of 0.0 to 1.0.
            
Category: {category}
Description: {self.ontology.get_category(category).description if self.ontology.get_category(category) else ""}

Conversation: {conversation[:500]}

Return only a number between 0.0 and 1.0"""
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": score_prompt}],
                    max_tokens=10,
                    temperature=0.1
                )
                score = float(response.choices[0].message.content.strip())
                category_scores[category] = max(0.0, min(1.0, score))  # Clamp to 0-1
            except:
                category_scores[category] = 0.0
        
        # Get regular classification
        classification = self.classify_intent(conversation_id, conversation)
        
        return classification, category_scores
    
    def classify_intent(self, conversation_id: str, conversation: str) -> IntentClassification:
        """Classify customer intent based on their fundamental need"""
        existing_categories = self.ontology.get_category_names()
        
        system_prompt = f"""You are analyzing customer support conversations to understand what customers fundamentally need.

Focus on the CUSTOMER'S UNDERLYING NEED/PROBLEM, not solutions.

Examples of good intent categories:
- account_access_issues: Customer cannot access their account  
- product_quality_concerns: Customer received defective/damaged item
- order_status_uncertainty: Customer needs information about their purchase
- billing_discrepancies: Customer questions charges or payments
- delivery_problems: Issues with shipping/receiving orders

EXISTING CATEGORIES: {existing_categories if existing_categories else "None yet"}

Rules:
- Focus on what the customer fundamentally needs
- Use existing categories if the need matches
- Create new categories for genuinely different customer needs
- Category names: lowercase_with_underscores
- Be specific enough to be actionable, but broad enough to be reusable"""

        # Define the JSON schema for structured output
        response_schema = {
            "type": "object",
            "properties": {
                "conversation_id": {"type": "string"},
                "intent_name": {"type": "string"},
                "intent_description": {"type": "string"},
                "action": {"type": "string", "enum": ["use_existing", "create_new"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reasoning": {"type": "string"}
            },
            "required": ["conversation_id", "intent_name", "intent_description", "action", "confidence", "reasoning"],
            "additionalProperties": False
        }

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"CONVERSATION:\n{conversation}\n\nClassify the customer's fundamental need for conversation_id: {conversation_id}"}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "intent_classification",
                        "schema": response_schema
                    }
                },
                max_tokens=300,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return IntentClassification(**result)
            
        except Exception as error:
            print(f"Error: {error}")
            return IntentClassification(
                conversation_id=conversation_id,
                intent_name="classification_error",
                intent_description="Failed to classify customer need",
                action=IntentAction.CREATE_NEW,
                confidence=0.1,
                reasoning=f"Error: {error}"
            )

    def process_classification(self, classification: IntentClassification, conversation: str) -> None:
        """Update ontology with classification result"""
        if classification.action == IntentAction.CREATE_NEW:
            new_category = IntentCategory(
                name=classification.intent_name,
                description=classification.intent_description,
                examples=[conversation[:200] + "..."]
            )
            self.ontology.add_category(new_category)
            print(f"Created: {classification.intent_name}")
            
        elif classification.action == IntentAction.USE_EXISTING:
            existing_category = self.ontology.get_category(classification.intent_name)
            if existing_category:
                example = conversation[:200] + "..."
                if example not in existing_category.examples:
                    existing_category.examples.append(example)
                print(f"Added to: {classification.intent_name}")
            else:
                # Fallback: create new category
                classification.action = IntentAction.CREATE_NEW
                self.process_classification(classification, conversation)

    def build_ontology(self) -> List[IntentClassification]:
        """Main process: load conversations and build ontology with evaluation tracking"""
        conversations = self.load_conversations_from_file()
        if not conversations:
            return []
        
        results = []
        confidence_scores = []
        reuse_count = 0
        
        print(f"\nProcessing {len(conversations)} conversations...")
        print("-" * 50)
        
        for i, conv in enumerate(conversations, 1):
            print(f"[{i}/{len(conversations)}] {conv['id']}: ", end="")
            
            classification = self.classify_intent(conv['id'], conv['conversation'])
            self.process_classification(classification, conv['conversation'])
            results.append(classification)
            
            # Track evaluation metrics
            confidence_scores.append(classification.confidence)
            if classification.action == IntentAction.USE_EXISTING:
                reuse_count += 1
            
            # Print evaluation every 100 conversations
            if i % 100 == 0:
                avg_confidence = sum(confidence_scores[-100:]) / 100
                reuse_rate = reuse_count / i
                print(f"\nAfter {i} conversations: avg_confidence={avg_confidence:.3f}, reuse_rate={reuse_rate:.3f}")
                print("-" * 50)
        
        # Final evaluation metrics
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        reuse_rate = reuse_count / len(conversations)
        low_confidence_count = sum(1 for score in confidence_scores if score < 0.6)
        low_confidence_rate = low_confidence_count / len(conversations)
        
        print(f"\nFinal Evaluation Metrics")
        print("-" * 30)
        print(f"Total conversations processed: {len(conversations)}")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Category reuse rate: {reuse_rate:.3f}")
        print(f"Low confidence rate (<0.6): {low_confidence_rate:.3f}")
        print(f"Total categories created: {len(self.ontology.categories)}")
        
        # Passes Thresholds
        thresholds = (
            avg_confidence >= 0.9 and 
            reuse_rate >= 0.9 and 
            low_confidence_rate <= 0.1
        )
        
        status = "Passed thresholds" if thresholds else "failed thresholds"
        print(status)
        
        return results

    def analyze_routing_potential(self) -> Dict:
        """Analyze ontology to show routing distribution"""
        if not self.ontology.categories:
            return {}
        
        routing_analysis = {}
        for category in self.ontology.categories:
            volume = category.volume()
            
            # Categorize by volume for routing insights
            if volume >= 10:
                routing_type = "HIGH_VOLUME"
            elif volume >= 5:
                routing_type = "MEDIUM_VOLUME"
            else:
                routing_type = "LOW_VOLUME"
            
            routing_analysis[category.name] = {
                'volume': volume,
                'description': category.description,
                'routing_type': routing_type
            }
        
        return routing_analysis

    def print_summary(self) -> None:
        """Print ontology summary and agent recommendations"""
        print(f"\n{'='*60}")
        print("CUSTOMER INTENT ONTOLOGY SUMMARY")
        print(f"{'='*60}")
        
        if not self.ontology.categories:
            print("No categories found.")
            return
        
        # Sort by volume (most common first)
        sorted_categories = sorted(self.ontology.categories, key=lambda x: x.volume(), reverse=True)
        
        print(f"\nFound {len(sorted_categories)} customer intent categories:")
        print("-" * 60)
        
        for cat in sorted_categories:
            print(f"{cat.name}: {cat.volume()} conversations")
            print(f"  Description: {cat.description}")
            print()
        
        # Routing analysis
        routing_analysis = self.analyze_routing_potential()
        high_volume = [name for name, data in routing_analysis.items() if data['routing_type'] == 'HIGH_VOLUME']
        medium_volume = [name for name, data in routing_analysis.items() if data['routing_type'] == 'MEDIUM_VOLUME']
        low_volume = [name for name, data in routing_analysis.items() if data['routing_type'] == 'LOW_VOLUME']
        
        print("ROUTING DISTRIBUTION")
        print("-" * 60)
        
        if high_volume:
            print(f"HIGH VOLUME INTENTS ({len(high_volume)}):")
            for name in high_volume:
                volume = routing_analysis[name]['volume']
                print(f"  - {name}: {volume} conversations")
        
        if medium_volume:
            print(f"\nMEDIUM VOLUME INTENTS ({len(medium_volume)}):")
            for name in medium_volume:
                volume = routing_analysis[name]['volume']
                print(f"  - {name}: {volume} conversations")
        
        if low_volume:
            print(f"\nLOW VOLUME INTENTS ({len(low_volume)}):")
            for name in low_volume:
                volume = routing_analysis[name]['volume']
                print(f"  - {name}: {volume} conversations")

    def save_ontology(self, filename: str = "intent_ontology.json") -> None:
        """Save ontology to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.ontology.model_dump(), f, indent=2)
        print(f"\nSaved ontology to {filename}")

# Run the analysis
if __name__ == "__main__":
    ontologist = IntentOntologist("data/conversations.txt")
    results = ontologist.build_ontology()
    ontologist.print_summary()
    ontologist.save_ontology()
    
    print(f"\nProcessed {len(results)} conversations into {len(ontologist.ontology.categories)} intent categories.")