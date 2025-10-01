"""
Query Analysis Module for KG-Enhanced RAG System.

This module analyzes user queries to extract entities and classify intent,
providing structured understanding for graph-enhanced retrieval.
"""

import os
import json
import time
from typing import Dict, List, Any, Tuple
from pathlib import Path
import sys
from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(project_root / ".env")

# OpenAI API for query analysis
import openai
from openai import OpenAI


class QueryAnalyzer:
    """
    Analyzes user queries to extract entities and classify intent for KG-enhanced retrieval.
    
    This class provides structured understanding of user queries by:
    1. Extracting entities (persons, organizations, locations, events, topics)
    2. Classifying query intent (factual, relational, temporal, comparative, multi_hop)
    3. Assessing complexity and graph traversal requirements
    """
    
    # Intent classification definitions
    INTENT_TYPES = {
        'factual': 'Simple fact lookup about specific entities or topics',
        'relational': 'Asks about relationships between entities',
        'temporal': 'Time-based queries about when events occurred',
        'comparative': 'Comparing different entities or concepts',
        'multi_hop': 'Requires following multiple relationships or complex reasoning'
    }
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1):
        """
        Initialize the QueryAnalyzer.
        
        Args:
            model_name: OpenAI model for entity extraction and intent analysis
            temperature: Temperature for analysis (lower = more deterministic)
        """
        # OpenAI settings (using faster model for demo)
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI()
        
        # Query result caching for demo performance
        self.query_cache = {}
        
        # Statistics tracking
        self.stats = {
            'queries_analyzed': 0,
            'entities_extracted': 0,
            'intent_classifications': {},
            'total_analysis_time': 0.0,
            'average_analysis_time': 0.0,
            'cache_hits': 0
        }
        
        print(f"🔍 QueryAnalyzer initialized:")
        print(f"   Model: {self.model_name} (optimized for demo)")
        print(f"   Temperature: {self.temperature}")
        print(f"   Caching: Enabled for demo performance")
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Complete query analysis including entity extraction and intent classification.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary containing structured query understanding
        """
        start_time = time.time()
        
        try:
            print(f"\n🔍 Analyzing query: '{query}'")
            
            # Check cache first for demo performance
            query_key = query.strip().lower()
            if query_key in self.query_cache:
                print("⚡ Cache hit! Using cached analysis")
                self.stats['cache_hits'] += 1
                cached_result = self.query_cache[query_key].copy()
                cached_result['analysis_time'] = time.time() - start_time  # Update timing
                return cached_result
            
            # Validate input query
            if not query or not query.strip():
                print("⚠️  Empty or whitespace-only query provided")
                return {
                    'query': query,
                    'extracted_entities': {'persons': [], 'organizations': [], 'locations': [], 'events': [], 'topics': []},
                    'intent_type': 'FACTUAL',
                    'intent_confidence': 0.0,
                    'intent_reasoning': 'Empty query provided',
                    'complexity_score': 1,
                    'requires_graph_traversal': False,
                    'analysis_reasoning': 'Empty query cannot be analyzed',
                    'analysis_time': time.time() - start_time,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
            
            # Step 1: Extract entities from query
            print("📝 Extracting entities...")
            entities = self._extract_entities(query)
            
            # Step 2: Classify query intent
            print("🎯 Classifying intent...")
            intent_analysis = self._classify_intent(query, entities)
            
            # Step 3: Determine complexity and graph requirements
            print("⚖️ Assessing complexity...")
            complexity_analysis = self._assess_complexity(query, entities, intent_analysis)
            
            # Step 4: Compile results
            analysis_time = time.time() - start_time
            
            result = {
                'query': query,
                'extracted_entities': entities,
                'intent_type': intent_analysis['intent'],
                'intent_confidence': intent_analysis['confidence'],
                'intent_reasoning': intent_analysis['reasoning'],
                'complexity_score': complexity_analysis['score'],
                'requires_graph_traversal': complexity_analysis['requires_graph'],
                'analysis_reasoning': complexity_analysis['reasoning'],
                'analysis_time': analysis_time,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Cache the result for demo performance (avoid repeated API calls)
            self.query_cache[query_key] = result.copy()
            
            # Update statistics
            self._update_stats(result)
            
            print(f"✅ Query analysis complete in {analysis_time:.2f}s")
            print(f"   Entities found: {len(entities['persons']) + len(entities['organizations']) + len(entities['locations']) + len(entities['events']) + len(entities['topics'])}")
            print(f"   Intent: {result['intent_type']} (confidence: {result['intent_confidence']:.2f})")
            print(f"   Complexity: {result['complexity_score']}/5")
            print(f"   Requires graph: {result['requires_graph_traversal']}")
            
            return result
            
        except Exception as e:
            error_time = time.time() - start_time
            print(f"❌ Error analyzing query: {e}")
            
            return {
                'query': query,
                'extracted_entities': {'persons': [], 'organizations': [], 'locations': [], 'events': [], 'topics': []},
                'intent_type': 'factual',
                'intent_confidence': 0.0,
                'intent_reasoning': f"Error during analysis: {e}",
                'complexity_score': 1,
                'requires_graph_traversal': False,
                'analysis_reasoning': f"Analysis failed: {e}",
                'analysis_time': error_time,
                'error': str(e),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query using the same approach as document processing."""
        
        entity_extraction_prompt = f"""
You are an expert entity extraction system. Extract entities from the following query text and categorize them precisely.

Query: "{query}"

Extract entities in these categories:
1. PERSONS: Individual people (politicians, athletes, celebrities, business leaders, etc.)
2. ORGANIZATIONS: Companies, institutions, government bodies, sports teams, etc.  
3. LOCATIONS: Countries, cities, regions, specific places, venues, etc.
4. EVENTS: Specific events, meetings, competitions, incidents, etc.
5. TOPICS: Abstract concepts, subjects, themes, technologies, policies, etc.

Rules:
- Only extract entities explicitly mentioned in the query
- Use the exact form mentioned in the query
- Don't infer entities not directly stated
- For ambiguous terms, choose the most specific category
- Return empty lists for categories with no entities

Respond with a JSON object in this exact format:
{{
    "persons": ["name1", "name2"],
    "organizations": ["org1", "org2"], 
    "locations": ["location1", "location2"],
    "events": ["event1", "event2"],
    "topics": ["topic1", "topic2"]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": entity_extraction_prompt}
                ],
                temperature=self.temperature,
                max_tokens=500
            )
            
            entities_text = response.choices[0].message.content.strip()

            
            # Strip markdown formatting if present
            if entities_text.startswith('```json'):
                entities_text = entities_text[7:]  # Remove ```json
            if entities_text.endswith('```'):
                entities_text = entities_text[:-3]  # Remove ```
            entities_text = entities_text.strip()
            
            # Parse JSON response
            entities = json.loads(entities_text)
            
            # Validate structure
            expected_keys = ['persons', 'organizations', 'locations', 'events', 'topics']
            for key in expected_keys:
                if key not in entities:
                    entities[key] = []
                elif not isinstance(entities[key], list):
                    entities[key] = []
            
            return entities
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse entity extraction JSON: {e}")
            return {'persons': [], 'organizations': [], 'locations': [], 'events': [], 'topics': []}
        except Exception as e:
            print(f"❌ Error during entity extraction: {e}")
            return {'persons': [], 'organizations': [], 'locations': [], 'events': [], 'topics': []}
    
    def _classify_intent(self, query: str, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Classify the intent of the query."""
        
        entity_summary = []
        for category, entity_list in entities.items():
            if entity_list:
                entity_summary.append(f"{category}: {', '.join(entity_list)}")
        entity_context = "; ".join(entity_summary) if entity_summary else "No specific entities found"
        
        intent_classification_prompt = f"""
You are an expert at understanding query intent. Analyze this query and classify its intent type.

Query: "{query}"
Extracted Entities: {entity_context}

Intent Types:
1. FACTUAL: Simple fact lookup about specific entities or topics (e.g., "What is Microsoft?", "Who is Tim Henman?")
2. RELATIONAL: Asks about relationships between entities (e.g., "How are Microsoft and Google related?", "What connection does Tim Henman have with tennis?")
3. TEMPORAL: Time-based queries about when events occurred (e.g., "When did Microsoft acquire Sybari?", "What happened in 2004?")
4. COMPARATIVE: Comparing different entities or concepts (e.g., "Compare Microsoft and Google", "Which tennis player performed better?")
5. MULTI_HOP: Requires following multiple relationships or complex reasoning (e.g., "What technology companies mentioned in entertainment news have connections to sports events?")

Analyze the query structure, entities mentioned, and question type to determine intent.

Respond with JSON in this exact format:
{{
    "intent": "intent_type",
    "confidence": 0.85,
    "reasoning": "Brief explanation of why this intent was chosen"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": intent_classification_prompt}
                ],
                temperature=self.temperature,
                max_tokens=200
            )
            
            intent_text = response.choices[0].message.content.strip()

            
            # Strip markdown formatting if present
            if intent_text.startswith('```json'):
                intent_text = intent_text[7:]  # Remove ```json
            if intent_text.endswith('```'):
                intent_text = intent_text[:-3]  # Remove ```
            intent_text = intent_text.strip()
            
            intent_data = json.loads(intent_text)
            
            # Validate intent type
            if intent_data.get('intent', '').lower() not in self.INTENT_TYPES:
                intent_data['intent'] = 'factual'
                intent_data['confidence'] = 0.5
                intent_data['reasoning'] = 'Unknown intent, defaulted to factual'
            
            return intent_data
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"❌ Error during intent classification: {e}")
            return {
                'intent': 'factual',
                'confidence': 0.5,
                'reasoning': f'Classification failed: {e}'
            }
    
    def _assess_complexity(self, query: str, entities: Dict[str, List[str]], intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess query complexity and graph traversal requirements."""
        
        # Count total entities
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        
        # Base complexity scoring
        complexity_score = 1
        requires_graph = False
        reasoning_factors = []
        
        # Entity-based complexity
        if total_entities == 0:
            complexity_score = 1
            reasoning_factors.append("No specific entities (complexity +0)")
        elif total_entities == 1:
            complexity_score = 2
            reasoning_factors.append("Single entity (complexity +1)")
        elif total_entities <= 3:
            complexity_score = 3
            requires_graph = True
            reasoning_factors.append("Multiple entities (complexity +2, graph needed)")
        else:
            complexity_score = 4
            requires_graph = True
            reasoning_factors.append("Many entities (complexity +3, graph needed)")
        
        # Intent-based complexity
        intent = intent_analysis.get('intent', 'factual')
        if intent == 'factual':
            reasoning_factors.append("Factual intent (complexity +0)")
        elif intent == 'relational':
            complexity_score += 2
            requires_graph = True
            reasoning_factors.append("Relational intent (complexity +2, graph needed)")
        elif intent == 'temporal':
            complexity_score += 1
            reasoning_factors.append("Temporal intent (complexity +1)")
        elif intent == 'comparative':
            complexity_score += 2
            requires_graph = True
            reasoning_factors.append("Comparative intent (complexity +2, graph needed)")
        elif intent == 'multi_hop':
            complexity_score += 3
            requires_graph = True
            reasoning_factors.append("Multi-hop intent (complexity +3, graph needed)")
        
        # Query structure complexity
        query_lower = query.lower()
        complex_words = ['relationship', 'connection', 'related', 'between', 'compare', 'versus', 'vs', 'difference', 'similar', 'both', 'all', 'multiple']
        if any(word in query_lower for word in complex_words):
            complexity_score += 1
            requires_graph = True
            reasoning_factors.append("Complex query language (complexity +1, graph needed)")
        
        # Question words that suggest relationships
        relation_words = ['how', 'why', 'what connection', 'which relationship', 'who worked with']
        if any(phrase in query_lower for phrase in relation_words):
            requires_graph = True
            reasoning_factors.append("Relationship question detected (graph needed)")
        
        # Cap complexity at 5
        complexity_score = min(5, complexity_score)
        
        return {
            'score': complexity_score,
            'requires_graph': requires_graph,
            'reasoning': "; ".join(reasoning_factors)
        }
    
    def _update_stats(self, result: Dict[str, Any]) -> None:
        """Update analysis statistics."""
        self.stats['queries_analyzed'] += 1
        
        # Count entities
        entities = result['extracted_entities']
        entity_count = sum(len(entity_list) for entity_list in entities.values())
        self.stats['entities_extracted'] += entity_count
        
        # Track intent distribution
        intent = result['intent_type']
        if intent not in self.stats['intent_classifications']:
            self.stats['intent_classifications'][intent] = 0
        self.stats['intent_classifications'][intent] += 1
        
        # Track analysis time
        self.stats['total_analysis_time'] += result['analysis_time']
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive analysis statistics."""
        if self.stats['queries_analyzed'] == 0:
            return {'message': 'No queries analyzed yet'}
        
        avg_analysis_time = self.stats['total_analysis_time'] / self.stats['queries_analyzed']
        avg_entities_per_query = self.stats['entities_extracted'] / self.stats['queries_analyzed']
        
        return {
            'total_queries_analyzed': self.stats['queries_analyzed'],
            'total_entities_extracted': self.stats['entities_extracted'],
            'average_entities_per_query': avg_entities_per_query,
            'average_analysis_time': avg_analysis_time,
            'intent_distribution': self.stats['intent_classifications'],
            'total_analysis_time': self.stats['total_analysis_time']
        }
    
    def print_statistics(self) -> None:
        """Print comprehensive analysis statistics."""
        stats = self.get_analysis_statistics()
        
        if 'message' in stats:
            print(f"\n📊 {stats['message']}")
            return
        
        print(f"\n📊 QUERY ANALYSIS STATISTICS")
        print(f"=" * 50)
        print(f"📝 Total queries analyzed: {stats['total_queries_analyzed']}")
        print(f"🏷️  Total entities extracted: {stats['total_entities_extracted']}")
        print(f"📈 Average entities per query: {stats['average_entities_per_query']:.1f}")
        print(f"⏱️  Average analysis time: {stats['average_analysis_time']:.2f}s")
        
        print(f"\n🎯 Intent Distribution:")
        for intent, count in stats['intent_distribution'].items():
            percentage = (count / stats['total_queries_analyzed']) * 100
            print(f"   {intent.upper()}: {count} ({percentage:.1f}%)")
    
    def batch_analyze(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple queries in batch."""
        print(f"\n📋 Batch analyzing {len(queries)} queries...")
        
        results = []
        for i, query in enumerate(queries, 1):
            print(f"\n--- Query {i}/{len(queries)} ---")
            result = self.analyze_query(query)
            results.append(result)
            
            # Brief pause between queries to avoid rate limits
            if i < len(queries):
                time.sleep(0.5)
        
        print(f"\n✅ Batch analysis complete: {len(results)} queries processed")
        return results


# Simple test function for key scenarios
def test_key_scenarios() -> None:
    """Test the QueryAnalyzer with key scenarios to validate functionality."""
    print("🧪 TESTING KEY SCENARIOS")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = QueryAnalyzer()
    
    # Test queries representing different scenarios
    test_queries = [
        "What is Microsoft?",  # Simple factual
        "How are Microsoft and Google related?",  # Relational
        "What tennis players won tournaments in 2004?",  # Temporal with entities
        "Compare Tim Henman and Andy Murray",  # Comparative
    ]
    
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*15} TEST {i}/4 {'='*15}")
        print(f"Query: {query}")
        
        result = analyzer.analyze_query(query)
        
        # Extract entity count
        total_entities = sum(len(entities) for entities in result['extracted_entities'].values())
        
        print(f"\n📋 RESULT:")
        print(f"   Entities: {total_entities} → {result['extracted_entities']}")
        print(f"   Intent: {result['intent_type']} (confidence: {result['intent_confidence']:.2f})")
        print(f"   Complexity: {result['complexity_score']}/5")
        print(f"   Requires Graph: {result['requires_graph_traversal']}")
        print(f"   Reasoning: {result['intent_reasoning']}")
        
        if result.get('error'):
            print(f"   ❌ Error: {result['error']}")
        
        results.append(result)
        
        # Pause between queries
        if i < len(test_queries):
            time.sleep(1)
    
    # Print summary
    analyzer.print_statistics()
    
    return results


# Test function for the QueryAnalyzer
def test_query_analyzer() -> None:
    """Test the QueryAnalyzer with sample queries covering different intents and complexity levels."""
    print("🧪 TESTING QUERY ANALYZER")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = QueryAnalyzer()
    
    # Test queries covering different intents and complexities
    test_queries = [
        # Factual queries (simple)
        "What is Microsoft?",
        "Who is Tim Henman?",
        
        # Relational queries (medium complexity)
        "What is the relationship between Microsoft and technology?",
        "How is Tim Henman connected to tennis?",
        
        # Temporal queries (medium complexity)
        "When did Microsoft acquire Sybari Software?",
        "What tennis events happened in 2004?",
        
        # Comparative queries (high complexity)
        "Compare Microsoft and Google's business strategies",
        "Which tennis player performed better, Tim Henman or Andy Murray?",
        
        # Multi-hop queries (highest complexity)
        "What technology companies mentioned in the news have connections to sports events?",
        "How do political developments affect business and entertainment news?"
    ]
    
    print(f"📋 Testing with {len(test_queries)} queries across all intent types...\n")
    
    # Analyze each query
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*15} TEST {i}/{len(test_queries)} {'='*15}")
        result = analyzer.analyze_query(query)
        results.append(result)
        
        # Brief pause between queries
        if i < len(test_queries):
            time.sleep(1)
    
    # Print summary statistics
    analyzer.print_statistics()
    
    # Print detailed results summary
    print(f"\n📋 DETAILED RESULTS SUMMARY")
    print(f"=" * 60)
    
    for i, result in enumerate(results, 1):
        total_entities = sum(len(entities) for entities in result['extracted_entities'].values())
        
        print(f"\n{i:2d}. {result['query'][:50]}{'...' if len(result['query']) > 50 else ''}")
        print(f"    Intent: {result['intent_type'].upper()} ({result['intent_confidence']:.2f})")
        print(f"    Entities: {total_entities} | Complexity: {result['complexity_score']}/5 | Graph: {result['requires_graph_traversal']}")
        
        if result.get('error'):
            print(f"    ❌ Error: {result['error']}")
    
    print(f"\n✅ QueryAnalyzer testing complete!")
    
    return results


if __name__ == "__main__":
    test_key_scenarios()