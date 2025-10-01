#!/usr/bin/env python3
"""
Phase 6 - Script 6: ResponseGenerator

This module enhances KG-Enhanced RAG responses with detailed reasoning paths,
confidence indicators, visual representations, and comprehensive explanations
of the graph-enhanced retrieval and generation process.
"""

import sys
import os
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Add the source directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class ConfidenceLevel(Enum):
    """Confidence level categories."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class ReasoningStep:
    """Individual step in the reasoning path."""
    step_number: int
    description: str
    details: str
    confidence: float
    entities_involved: List[str]
    data_source: str  # 'query_analysis', 'graph_traversal', 'retrieval', 'generation'


@dataclass
class ConfidenceIndicators:
    """Comprehensive confidence metrics."""
    overall_confidence: float
    source_confidence: float
    graph_confidence: float
    entity_confidence: float
    retrieval_confidence: float
    generation_confidence: float
    confidence_level: ConfidenceLevel
    confidence_factors: Dict[str, Any]


@dataclass
class VisualPath:
    """Visual representation of the reasoning path."""
    text_representation: str
    ascii_graph: str
    json_structure: Dict[str, Any]
    entity_chain: List[str]
    relationship_chain: List[str]


class ResponseGenerator:
    """
    Generates enhanced responses with detailed reasoning paths, confidence indicators,
    and visual representations of the graph-enhanced retrieval process.
    """
    
    def __init__(self):
        """Initialize the ResponseGenerator."""
        print("🎨 Initializing ResponseGenerator...")
        
        # Configuration
        self.config = {
            'max_reasoning_steps': 10,
            'confidence_threshold': 0.7,
            'include_visual_paths': True,
            'detailed_explanations': True,
            'max_entity_chain_length': 8
        }
        
        # Confidence weights for overall calculation
        self.confidence_weights = {
            'source_confidence': 0.25,
            'graph_confidence': 0.25,
            'entity_confidence': 0.20,
            'retrieval_confidence': 0.15,
            'generation_confidence': 0.15
        }
        
        print(f"✅ ResponseGenerator initialized successfully")
        print(f"   Max reasoning steps: {self.config['max_reasoning_steps']}")
        print(f"   Confidence threshold: {self.config['confidence_threshold']}")
        print(f"   Visual paths enabled: {self.config['include_visual_paths']}")
    
    def enhance_response(self, kg_rag_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance a KG-Enhanced RAG result with detailed reasoning and explanations.
        
        Args:
            kg_rag_result: Result from KG-Enhanced RAG system
            
        Returns:
            Enhanced response with reasoning paths and confidence indicators
        """
        print(f"\n🎨 ENHANCING RESPONSE WITH REASONING PATHS")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Extract and analyze the reasoning components
            print("📋 Step 1: Analyzing KG-RAG Components...")
            reasoning_components = self._extract_reasoning_components(kg_rag_result)
            
            # Step 2: Generate step-by-step reasoning path
            print("🔍 Step 2: Generating Reasoning Path...")
            reasoning_path = self._generate_reasoning_path(reasoning_components)
            
            # Step 3: Calculate comprehensive confidence indicators
            print("📊 Step 3: Calculating Confidence Indicators...")
            confidence_indicators = self._calculate_confidence_indicators(reasoning_components)
            
            # Step 4: Create visual path representations
            print("🎯 Step 4: Creating Visual Path Representations...")
            visual_path = self._create_visual_path(reasoning_components) if self.config['include_visual_paths'] else None
            
            # Step 5: Format enhanced answer with citations
            print("✨ Step 5: Formatting Enhanced Answer...")
            enhanced_answer = self._format_enhanced_answer(kg_rag_result, confidence_indicators)
            
            # Step 6: Create comprehensive enhanced response
            enhanced_response = self._create_enhanced_response(
                kg_rag_result=kg_rag_result,
                reasoning_path=reasoning_path,
                confidence_indicators=confidence_indicators,
                visual_path=visual_path,
                enhanced_answer=enhanced_answer
            )
            
            processing_time = time.time() - start_time
            enhanced_response['processing_metadata'] = {
                'enhancement_time': processing_time,
                'reasoning_steps_generated': len(reasoning_path),
                'confidence_calculated': True,
                'visual_path_created': visual_path is not None
            }
            
            print(f"✅ Response enhancement complete in {processing_time:.2f}s")
            print(f"   Reasoning steps: {len(reasoning_path)}")
            print(f"   Overall confidence: {confidence_indicators.overall_confidence:.2f}")
            print(f"   Confidence level: {confidence_indicators.confidence_level.value}")
            
            return enhanced_response
            
        except Exception as e:
            print(f"❌ Error enhancing response: {str(e)}")
            
            # Return original response with error indication
            return {
                **kg_rag_result,
                'enhancement_error': str(e),
                'reasoning_path': [],
                'confidence_indicators': None,
                'visual_path': None,
                'enhanced_answer': kg_rag_result.get('answer', ''),
                'processing_metadata': {
                    'enhancement_time': time.time() - start_time,
                    'error': str(e)
                }
            }
    
    def _extract_reasoning_components(self, kg_rag_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract components needed for reasoning path generation."""
        
        components = {
            'original_query': kg_rag_result.get('question', ''),
            'answer': kg_rag_result.get('answer', ''),
            'sources': kg_rag_result.get('sources', []),
            'response_time': kg_rag_result.get('response_time', 0.0),
            'token_usage': kg_rag_result.get('token_usage', {}),
            'graph_enhancement': kg_rag_result.get('graph_enhancement', {}),
            'explanation': kg_rag_result.get('explanation', {}),
            'metadata': kg_rag_result.get('metadata', {})
        }
        
        # Extract graph-specific data
        explanation = components['explanation']
        if explanation and 'steps' in explanation:
            components['process_steps'] = explanation['steps']
        else:
            components['process_steps'] = []
        
        # Extract entity and graph information
        graph_enhancement = components['graph_enhancement']
        components['entities_extracted'] = graph_enhancement.get('entities_extracted', 0)
        components['entities_related'] = graph_enhancement.get('entities_related', 0)
        components['articles_connected'] = graph_enhancement.get('articles_connected', 0)
        components['enhancement_applied'] = graph_enhancement.get('enhancement_applied', False)
        
        return components
    
    def _generate_reasoning_path(self, components: Dict[str, Any]) -> List[ReasoningStep]:
        """Generate step-by-step reasoning path from components."""
        
        reasoning_steps = []
        step_number = 1
        
        # Step 1: Query Analysis and Entity Extraction
        entities_extracted = components['entities_extracted']
        if entities_extracted > 0:
            step = ReasoningStep(
                step_number=step_number,
                description=f"Found {entities_extracted} entities in query",
                details=f"Analyzed the query '{components['original_query'][:100]}...' and extracted {entities_extracted} key entities for graph exploration.",
                confidence=self._calculate_entity_extraction_confidence(entities_extracted),
                entities_involved=[],  # Will be populated from actual data if available
                data_source='query_analysis'
            )
            reasoning_steps.append(step)
            step_number += 1
        
        # Step 2: Graph Traversal and Related Entity Discovery
        entities_related = components['entities_related']
        if entities_related > 0:
            step = ReasoningStep(
                step_number=step_number,
                description=f"Traversed graph to find {entities_related} related entities",
                details=f"Performed knowledge graph traversal to discover {entities_related} related entities, expanding the search context beyond the original query terms.",
                confidence=self._calculate_graph_traversal_confidence(entities_related, components['articles_connected']),
                entities_involved=[],
                data_source='graph_traversal'
            )
            reasoning_steps.append(step)
            step_number += 1
        
        # Step 3: Search Query Enhancement
        if components['enhancement_applied']:
            step = ReasoningStep(
                step_number=step_number,
                description="Enhanced search query with graph insights",
                details="Incorporated discovered entities and relationships into the search query to improve retrieval of relevant documents.",
                confidence=0.85,  # Fixed high confidence for successful enhancement
                entities_involved=[],
                data_source='query_enhancement'
            )
            reasoning_steps.append(step)
            step_number += 1
        
        # Step 4: Document Retrieval
        sources_count = len(components['sources'])
        if sources_count > 0:
            step = ReasoningStep(
                step_number=step_number,
                description=f"Retrieved {sources_count} relevant documents",
                details=f"Used the enhanced query to retrieve {sources_count} relevant documents from the knowledge base, ranked by relevance to the expanded query context.",
                confidence=self._calculate_retrieval_confidence(sources_count, components['sources']),
                entities_involved=[],
                data_source='retrieval'
            )
            reasoning_steps.append(step)
            step_number += 1
        
        # Step 5: Answer Generation
        answer_length = len(components['answer'])
        token_usage = components['token_usage']
        if answer_length > 0:
            step = ReasoningStep(
                step_number=step_number,
                description="Generated comprehensive answer using graph-aware prompts",
                details=f"Synthesized information from retrieved sources using knowledge graph context to generate a {answer_length}-character answer with proper source attribution.",
                confidence=self._calculate_generation_confidence(answer_length, token_usage),
                entities_involved=[],
                data_source='generation'
            )
            reasoning_steps.append(step)
            step_number += 1
        
        return reasoning_steps
    
    def _calculate_confidence_indicators(self, components: Dict[str, Any]) -> ConfidenceIndicators:
        """Calculate comprehensive confidence indicators."""
        
        # Source-based confidence
        sources = components['sources']
        source_confidence = self._calculate_source_confidence(sources)
        
        # Graph-based confidence
        graph_confidence = self._calculate_graph_confidence(
            components['entities_extracted'],
            components['entities_related'],
            components['articles_connected']
        )
        
        # Entity extraction confidence
        entity_confidence = self._calculate_entity_extraction_confidence(
            components['entities_extracted']
        )
        
        # Retrieval confidence
        retrieval_confidence = self._calculate_retrieval_confidence(
            len(sources), sources
        )
        
        # Generation confidence
        generation_confidence = self._calculate_generation_confidence(
            len(components['answer']), components['token_usage']
        )
        
        # Calculate weighted overall confidence
        overall_confidence = (
            source_confidence * self.confidence_weights['source_confidence'] +
            graph_confidence * self.confidence_weights['graph_confidence'] +
            entity_confidence * self.confidence_weights['entity_confidence'] +
            retrieval_confidence * self.confidence_weights['retrieval_confidence'] +
            generation_confidence * self.confidence_weights['generation_confidence']
        )
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(overall_confidence)
        
        # Detailed confidence factors
        confidence_factors = {
            'source_factors': {
                'source_count': len(sources),
                'average_score': sum(s.get('score', 0) for s in sources) / len(sources) if sources else 0,
                'diversity': len(set(s.get('article_id', '') for s in sources))
            },
            'graph_factors': {
                'entities_extracted': components['entities_extracted'],
                'entities_related': components['entities_related'],
                'articles_connected': components['articles_connected'],
                'enhancement_applied': components['enhancement_applied']
            },
            'generation_factors': {
                'answer_length': len(components['answer']),
                'token_efficiency': components['token_usage'].get('total_tokens', 0) / max(len(components['answer']), 1) if components['answer'] else 0,
                'response_time': components['response_time']
            }
        }
        
        return ConfidenceIndicators(
            overall_confidence=overall_confidence,
            source_confidence=source_confidence,
            graph_confidence=graph_confidence,
            entity_confidence=entity_confidence,
            retrieval_confidence=retrieval_confidence,
            generation_confidence=generation_confidence,
            confidence_level=confidence_level,
            confidence_factors=confidence_factors
        )
    
    def _create_visual_path(self, components: Dict[str, Any]) -> VisualPath:
        """Create visual representation of the reasoning path."""
        
        # Create text representation
        text_parts = []
        text_parts.append("Query")
        
        if components['entities_extracted'] > 0:
            text_parts.extend(["→", "Entity Extraction"])
        
        if components['entities_related'] > 0:
            text_parts.extend(["→", "Graph Traversal"])
        
        if components['enhancement_applied']:
            text_parts.extend(["→", "Query Enhancement"])
        
        if len(components['sources']) > 0:
            text_parts.extend(["→", "Document Retrieval"])
        
        text_parts.extend(["→", "Answer Generation"])
        
        text_representation = " ".join(text_parts)
        
        # Create ASCII graph representation
        ascii_graph = self._create_ascii_graph(components)
        
        # Create JSON structure for programmatic use
        json_structure = {
            'nodes': [
                {'id': 'query', 'label': 'Original Query', 'type': 'input'},
                {'id': 'entities', 'label': f"Entities ({components['entities_extracted']})", 'type': 'extraction'},
                {'id': 'graph', 'label': f"Related Entities ({components['entities_related']})", 'type': 'traversal'},
                {'id': 'enhancement', 'label': 'Enhanced Query', 'type': 'enhancement'},
                {'id': 'retrieval', 'label': f"Documents ({len(components['sources'])})", 'type': 'retrieval'},
                {'id': 'answer', 'label': 'Final Answer', 'type': 'output'}
            ],
            'edges': [
                {'from': 'query', 'to': 'entities', 'label': 'extract'},
                {'from': 'entities', 'to': 'graph', 'label': 'traverse'},
                {'from': 'graph', 'to': 'enhancement', 'label': 'enhance'},
                {'from': 'enhancement', 'to': 'retrieval', 'label': 'search'},
                {'from': 'retrieval', 'to': 'answer', 'label': 'generate'}
            ]
        }
        
        # Create entity and relationship chains (simplified)
        entity_chain = ['query_entities', 'related_entities', 'retrieved_documents', 'generated_answer']
        relationship_chain = ['extracts', 'discovers', 'retrieves', 'synthesizes']
        
        return VisualPath(
            text_representation=text_representation,
            ascii_graph=ascii_graph,
            json_structure=json_structure,
            entity_chain=entity_chain,
            relationship_chain=relationship_chain
        )
    
    def _create_ascii_graph(self, components: Dict[str, Any]) -> str:
        """Create ASCII art representation of the process flow."""
        
        lines = []
        lines.append("┌─────────────┐")
        lines.append("│ User Query  │")
        lines.append("└──────┬──────┘")
        lines.append("       │")
        lines.append("       ▼")
        lines.append("┌─────────────┐")
        lines.append("│   Analyze   │")
        lines.append(f"│ {components['entities_extracted']} entities  │")
        lines.append("└──────┬──────┘")
        
        if components['entities_related'] > 0:
            lines.append("       │")
            lines.append("       ▼")
            lines.append("┌─────────────┐")
            lines.append("│  Traverse   │")
            lines.append(f"│ {components['entities_related']} related   │")
            lines.append("└──────┬──────┘")
        
        lines.append("       │")
        lines.append("       ▼")
        lines.append("┌─────────────┐")
        lines.append("│  Retrieve   │")
        lines.append(f"│ {len(components['sources'])} documents │")
        lines.append("└──────┬──────┘")
        lines.append("       │")
        lines.append("       ▼")
        lines.append("┌─────────────┐")
        lines.append("│   Answer    │")
        lines.append("└─────────────┘")
        
        return "\n".join(lines)
    
    def _format_enhanced_answer(self, kg_rag_result: Dict[str, Any], confidence_indicators: ConfidenceIndicators) -> str:
        """Format the answer with enhanced citations and confidence context."""
        
        original_answer = kg_rag_result.get('answer', '')
        
        # Add confidence context if confidence is moderate or below
        confidence_context = ""
        if confidence_indicators.confidence_level in [ConfidenceLevel.MODERATE, ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]:
            confidence_context = f"\n\n*Note: This answer has {confidence_indicators.confidence_level.value.replace('_', ' ')} confidence ({confidence_indicators.overall_confidence:.2f}) based on available sources and graph connections.*"
        
        # Enhanced citation formatting (basic implementation)
        enhanced_answer = original_answer + confidence_context
        
        return enhanced_answer
    
    def _create_enhanced_response(self, kg_rag_result: Dict[str, Any], reasoning_path: List[ReasoningStep], 
                                confidence_indicators: ConfidenceIndicators, visual_path: Optional[VisualPath],
                                enhanced_answer: str) -> Dict[str, Any]:
        """Create the final enhanced response structure."""
        
        # Convert reasoning steps to dictionary format
        reasoning_path_dict = {
            'steps': [
                {
                    'step': step.step_number,
                    'description': step.description,
                    'details': step.details,
                    'confidence': step.confidence,
                    'entities_involved': step.entities_involved,
                    'data_source': step.data_source
                }
                for step in reasoning_path
            ],
            'total_steps': len(reasoning_path),
            'overall_path_confidence': sum(step.confidence for step in reasoning_path) / len(reasoning_path) if reasoning_path else 0.0
        }
        
        # Convert confidence indicators to dictionary
        confidence_dict = {
            'overall_confidence': confidence_indicators.overall_confidence,
            'confidence_level': confidence_indicators.confidence_level.value,
            'detailed_confidence': {
                'source_confidence': confidence_indicators.source_confidence,
                'graph_confidence': confidence_indicators.graph_confidence,
                'entity_confidence': confidence_indicators.entity_confidence,
                'retrieval_confidence': confidence_indicators.retrieval_confidence,
                'generation_confidence': confidence_indicators.generation_confidence
            },
            'confidence_factors': confidence_indicators.confidence_factors
        }
        
        # Convert visual path to dictionary
        visual_path_dict = None
        if visual_path:
            visual_path_dict = {
                'text_representation': visual_path.text_representation,
                'ascii_graph': visual_path.ascii_graph,
                'json_structure': visual_path.json_structure,
                'entity_chain': visual_path.entity_chain,
                'relationship_chain': visual_path.relationship_chain
            }
        
        # Create enhanced response
        enhanced_response = {
            # Original KG-RAG result components
            **kg_rag_result,
            
            # Enhanced components
            'enhanced_answer': enhanced_answer,
            'reasoning_path': reasoning_path_dict,
            'confidence_indicators': confidence_dict,
            'visual_path': visual_path_dict,
            
            # Response enhancement metadata
            'enhancement_applied': True,
            'enhancement_version': '1.0'
        }
        
        return enhanced_response
    
    # Confidence calculation helper methods
    def _calculate_source_confidence(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate confidence based on source quality and quantity."""
        if not sources:
            return 0.0
        
        # Base confidence from count (diminishing returns)
        count_confidence = min(len(sources) / 8.0, 1.0)  # Max confidence at 8 sources
        
        # Average relevance score
        scores = [s.get('score', 0.0) for s in sources]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Source diversity (different articles)
        unique_articles = len(set(s.get('article_id', '') for s in sources))
        diversity_factor = min(unique_articles / len(sources), 1.0) if len(sources) > 0 else 0.0
        
        return (count_confidence * 0.4 + avg_score * 0.4 + diversity_factor * 0.2)
    
    def _calculate_graph_confidence(self, entities_extracted: int, entities_related: int, articles_connected: int) -> float:
        """Calculate confidence based on graph enhancement effectiveness."""
        if entities_extracted == 0:
            return 0.1  # Minimal confidence without entity extraction
        
        extraction_factor = min(entities_extracted / 3.0, 1.0)  # Max confidence at 3+ entities
        
        if entities_related == 0:
            return extraction_factor * 0.5  # Reduced confidence without graph traversal
        
        traversal_factor = min(entities_related / 10.0, 1.0)  # Max confidence at 10+ related entities
        connection_factor = min(articles_connected / 20.0, 1.0)  # Max confidence at 20+ connected articles
        
        return extraction_factor * 0.4 + traversal_factor * 0.3 + connection_factor * 0.3
    
    def _calculate_entity_extraction_confidence(self, entities_extracted: int) -> float:
        """Calculate confidence for entity extraction step."""
        return min(entities_extracted / 3.0, 1.0)  # Max confidence at 3+ entities
    
    def _calculate_graph_traversal_confidence(self, entities_related: int, articles_connected: int) -> float:
        """Calculate confidence for graph traversal step."""
        if entities_related == 0:
            return 0.1
        
        related_factor = min(entities_related / 8.0, 1.0)
        connection_factor = min(articles_connected / 15.0, 1.0)
        
        return (related_factor * 0.6 + connection_factor * 0.4)
    
    def _calculate_retrieval_confidence(self, source_count: int, sources: List[Dict[str, Any]]) -> float:
        """Calculate confidence for retrieval step."""
        return self._calculate_source_confidence(sources)
    
    def _calculate_generation_confidence(self, answer_length: int, token_usage: Dict[str, Any]) -> float:
        """Calculate confidence for answer generation step."""
        if answer_length == 0:
            return 0.0
        
        # Length factor (reasonable answer length)
        length_factor = min(answer_length / 500.0, 1.0)  # Max confidence at 500+ chars
        
        # Token efficiency (tokens per character)
        total_tokens = token_usage.get('total_tokens', answer_length)
        efficiency = min(answer_length / max(total_tokens, 1), 1.0)
        
        return (length_factor * 0.7 + efficiency * 0.3)
    
    def _determine_confidence_level(self, overall_confidence: float) -> ConfidenceLevel:
        """Determine confidence level category from numerical confidence."""
        if overall_confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif overall_confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif overall_confidence >= 0.6:
            return ConfidenceLevel.MODERATE
        elif overall_confidence >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


# Test function for single response enhancement
def test_single_response_enhancement() -> Dict[str, Any]:
    """Test ResponseGenerator with a sample KG-Enhanced RAG result."""
    
    print("🧪 TESTING RESPONSE GENERATOR")
    print("=" * 60)
    
    # Initialize ResponseGenerator
    generator = ResponseGenerator()
    
    # Create mock KG-Enhanced RAG result for testing
    mock_kg_rag_result = {
        'question': 'What is Microsoft\'s role in technology?',
        'answer': 'Microsoft is a major technology company that develops software, cloud services, and hardware products. The company plays a significant role in enterprise software with products like Windows, Office, and Azure cloud platform. [Source 1][Source 2]',
        'sources': [
            {
                'source_num': 1,
                'content': 'Microsoft announced new Azure capabilities for enterprise customers...',
                'score': 0.85,
                'article_id': 'bbc_tech_01',
                'metadata': {'category': 'technology'}
            },
            {
                'source_num': 2,
                'content': 'The company\'s Windows operating system continues to dominate...',
                'score': 0.78,
                'article_id': 'bbc_tech_02',
                'metadata': {'category': 'technology'}
            }
        ],
        'response_time': 15.5,
        'token_usage': {'total_tokens': 1200, 'prompt_tokens': 800, 'completion_tokens': 400},
        'graph_enhancement': {
            'enhancement_applied': True,
            'entities_extracted': 2,
            'entities_related': 8,
            'articles_connected': 15,
            'query_enhanced': True
        },
        'explanation': {
            'process_overview': 'KG-Enhanced RAG combines knowledge graph insights with vector retrieval.',
            'steps': [
                {'step': 1, 'name': 'Query Analysis & Entity Extraction', 'description': 'Extracted 2 entities'},
                {'step': 2, 'name': 'Knowledge Graph Traversal', 'description': 'Found 8 related entities'},
                {'step': 3, 'name': 'Graph-Informed Retrieval', 'description': 'Enhanced query retrieval'},
                {'step': 4, 'name': 'Answer Generation', 'description': 'Generated graph-aware answer'}
            ]
        },
        'metadata': {
            'model': 'gpt-4-turbo-preview',
            'sources_retrieved': 2,
            'retrieval_method': 'kg_enhanced_hybrid'
        }
    }
    
    print(f"\n📋 Testing with mock KG-Enhanced RAG result...")
    print(f"   Original question: '{mock_kg_rag_result['question']}'")
    print(f"   Graph enhancement: {mock_kg_rag_result['graph_enhancement']['enhancement_applied']}")
    print(f"   Sources: {len(mock_kg_rag_result['sources'])}")
    
    # Enhance the response
    enhanced_result = generator.enhance_response(mock_kg_rag_result)
    
    # Display results
    print(f"\n📊 ENHANCEMENT RESULTS:")
    print(f"   Enhancement applied: {enhanced_result.get('enhancement_applied', False)}")
    print(f"   Processing time: {enhanced_result.get('processing_metadata', {}).get('enhancement_time', 0):.2f}s")
    print(f"   Reasoning steps: {enhanced_result.get('processing_metadata', {}).get('reasoning_steps_generated', 0)}")
    
    # Show reasoning path
    reasoning_path = enhanced_result.get('reasoning_path', {})
    if reasoning_path and 'steps' in reasoning_path:
        print(f"\n🔍 REASONING PATH:")
        for step in reasoning_path['steps']:
            print(f"   {step['step']}. {step['description']}")
            print(f"      Confidence: {step['confidence']:.2f}")
    
    # Show confidence indicators
    confidence = enhanced_result.get('confidence_indicators', {})
    if confidence:
        print(f"\n📊 CONFIDENCE INDICATORS:")
        print(f"   Overall confidence: {confidence.get('overall_confidence', 0):.2f}")
        print(f"   Confidence level: {confidence.get('confidence_level', 'unknown')}")
        
        detailed = confidence.get('detailed_confidence', {})
        if detailed:
            print(f"   Source confidence: {detailed.get('source_confidence', 0):.2f}")
            print(f"   Graph confidence: {detailed.get('graph_confidence', 0):.2f}")
            print(f"   Entity confidence: {detailed.get('entity_confidence', 0):.2f}")
    
    # Show visual path
    visual_path = enhanced_result.get('visual_path', {})
    if visual_path:
        print(f"\n🎯 VISUAL PATH:")
        print(f"   Text representation: {visual_path.get('text_representation', 'N/A')}")
        print(f"   ASCII graph:")
        ascii_graph = visual_path.get('ascii_graph', '')
        if ascii_graph:
            for line in ascii_graph.split('\n'):
                print(f"      {line}")
    
    print(f"\n✅ Response enhancement test complete!")
    
    return enhanced_result


if __name__ == "__main__":
    # Test with single mock result
    result = test_single_response_enhancement()