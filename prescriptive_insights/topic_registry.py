"""
Topic Registry for Prescriptive Insights

Defines topic categories, keywords, and embedding prompts for organizing
and retrieving transcript chunks by topic.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class Topic:
    """Defines a topic category with associated metadata."""
    id: str
    name: str
    description: str
    keywords: List[str]
    entities: List[str]  # Entity types to filter on
    embedding_prompt: str  # Prompt for generating topic embeddings
    
    def matches_keywords(self, text: str) -> bool:
        """Check if text contains any of the topic keywords."""
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in self.keywords)
    
    def matches_entities(self, entities: List[Dict[str, Any]]) -> bool:
        """Check if any entities match the topic entity types."""
        entity_labels = [entity.get("label", "").lower() for entity in entities]
        return any(entity_type.lower() in entity_labels for entity_type in self.entities)

# Define the main topic categories
TOPICS = {
    "start_plan": Topic(
        id="start_plan",
        name="Start Plan",
        description="Getting started with keto, initial steps, and planning",
        keywords=[
            "begin", "start", "getting started", "first steps", "how to start",
            "beginner", "introduction", "basics", "foundation", "initial",
            "plan", "planning", "prepare", "preparation", "guide", "tutorial"
        ],
        entities=["CARDINAL", "ORDINAL", "DATE"],  # Numbers and sequences often in plans
        embedding_prompt="Getting started with ketogenic diet, initial steps, planning, and preparation for beginners"
    ),
    
    "supplements": Topic(
        id="supplements",
        name="Supplements",
        description="Nutritional supplements, vitamins, minerals for keto",
        keywords=[
            "supplement", "vitamin", "mineral", "mct", "exogenous ketones",
            "electrolytes", "magnesium", "potassium", "sodium", "calcium",
            "omega", "fish oil", "vitamin d", "b vitamins", "zinc", "selenium"
        ],
        entities=["PRODUCT", "ORG"],  # Product names and organizations
        embedding_prompt="Nutritional supplements, vitamins, minerals, MCT oil, electrolytes for ketogenic diet"
    ),
    
    "holistic_view": Topic(
        id="holistic_view",
        name="Holistic View",
        description="Overall health perspective, long-term effects, comprehensive approach",
        keywords=[
            "holistic", "overall", "comprehensive", "long-term", "sustainable",
            "lifestyle", "health", "wellness", "balance", "whole body",
            "systemic", "complete", "total", "entire", "full picture"
        ],
        entities=["PERSON", "ORG"],  # People and organizations discussing holistic approaches
        embedding_prompt="Holistic health approach, comprehensive wellness, long-term sustainable lifestyle changes"
    ),
    
    "metabolic_health": Topic(
        id="metabolic_health",
        name="Metabolic Health",
        description="Metabolism, insulin resistance, blood sugar, metabolic markers",
        keywords=[
            "metabolism", "metabolic", "insulin", "blood sugar", "glucose",
            "insulin resistance", "diabetes", "prediabetes", "a1c", "fasting",
            "metabolic syndrome", "energy", "mitochondria", "fat burning"
        ],
        entities=["DISEASE", "DISORDER", "ORG"],  # Medical conditions and organizations
        embedding_prompt="Metabolic health, insulin resistance, blood sugar management, diabetes prevention"
    ),
    
    "weight_management": Topic(
        id="weight_management",
        name="Weight Management",
        description="Weight loss, weight gain, body composition, BMI",
        keywords=[
            "weight", "weight loss", "fat loss", "obesity", "overweight",
            "bmi", "body mass index", "body composition", "lean mass",
            "adipose", "visceral fat", "subcutaneous fat", "waist", "measurements"
        ],
        entities=["QUANTITY", "CARDINAL", "PERCENT"],  # Numbers and measurements
        embedding_prompt="Weight management, fat loss, body composition, BMI, obesity treatment"
    ),
    
    "intermittent_fasting": Topic(
        id="intermittent_fasting",
        name="Intermittent Fasting",
        description="Fasting protocols, time-restricted eating, fasting benefits",
        keywords=[
            "fasting", "intermittent fasting", "time-restricted eating", "if",
            "16:8", "18:6", "omad", "one meal a day", "extended fast",
            "autophagy", "fasted state", "fasting window", "eating window"
        ],
        entities=["TIME", "DATE", "CARDINAL"],  # Time references and numbers
        embedding_prompt="Intermittent fasting protocols, time-restricted eating, autophagy, fasting benefits"
    ),
    
    "keto_foods": Topic(
        id="keto_foods",
        name="Keto Foods",
        description="Keto-friendly foods, recipes, meal planning",
        keywords=[
            "keto food", "low carb", "carbohydrates", "macros", "fat",
            "protein", "vegetables", "dairy", "nuts", "seeds", "avocado",
            "olive oil", "butter", "eggs", "meat", "fish", "recipe"
        ],
        entities=["FOOD", "PRODUCT", "ORG"],  # Food items and brands
        embedding_prompt="Keto-friendly foods, low carbohydrate diet, meal planning, recipes"
    ),
    
    "exercise_fitness": Topic(
        id="exercise_fitness",
        name="Exercise & Fitness",
        description="Physical activity, exercise, fitness on keto",
        keywords=[
            "exercise", "workout", "fitness", "training", "cardio",
            "strength", "muscle", "gym", "physical activity", "movement",
            "performance", "endurance", "recovery", "adaptation"
        ],
        entities=["EVENT", "FAC", "ORG"],  # Events, facilities, organizations
        embedding_prompt="Exercise and fitness on ketogenic diet, workout performance, muscle adaptation"
    )
}

class TopicRegistry:
    """Registry for managing topic definitions and matching."""
    
    def __init__(self, topics: Dict[str, Topic] = None):
        """Initialize with custom topics or use default topics."""
        self.topics = topics or TOPICS
        
    def get_topic(self, topic_id: str) -> Topic:
        """Get a topic by ID."""
        if topic_id not in self.topics:
            raise ValueError(f"Unknown topic ID: {topic_id}")
        return self.topics[topic_id]
    
    def get_all_topics(self) -> Dict[str, Topic]:
        """Get all topics."""
        return self.topics
    
    def find_matching_topics(self, text: str, entities: List[Dict[str, Any]] = None) -> List[str]:
        """Find all topics that match the given text and entities."""
        matching_topics = []
        entities = entities or []
        
        for topic_id, topic in self.topics.items():
            if topic.matches_keywords(text):
                matching_topics.append(topic_id)
            elif topic.matches_entities(entities):
                matching_topics.append(topic_id)
                
        return matching_topics
    
    def get_topic_keywords(self, topic_id: str) -> List[str]:
        """Get keywords for a specific topic."""
        return self.get_topic(topic_id).keywords
    
    def get_topic_embedding_prompt(self, topic_id: str) -> str:
        """Get the embedding prompt for a specific topic."""
        return self.get_topic(topic_id).embedding_prompt