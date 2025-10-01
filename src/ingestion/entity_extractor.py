"""
Entity Extraction Module

Extracts structured entities from articles using OpenAI's API with JSON mode.
Identifies persons, organizations, locations, events, and topics for knowledge graph construction.
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

try:
    from openai import OpenAI
except ImportError:
    print("⚠️ OpenAI library not available. Please install: pip install openai")
    OpenAI = None

from ..config import config


@dataclass
class EntityExtractionResult:
    """Result of entity extraction for a single article."""
    article_id: str
    persons: List[Dict[str, Any]]
    organizations: List[Dict[str, Any]]
    locations: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    topics: List[str]
    extraction_successful: bool
    error_message: Optional[str] = None


class EntityExtractor:
    """
    Extracts structured entities from article text using OpenAI's GPT API.
    
    Uses structured JSON output to identify persons, organizations, locations,
    events, and topics for knowledge graph construction.
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize the entity extractor.
        
        Args:
            api_key: OpenAI API key (defaults to config)
            model: Model to use (defaults to config.CHAT_MODEL)
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.CHAT_MODEL
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        if OpenAI is None:
            raise ImportError("OpenAI library not available. Please install: pip install openai")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Extraction prompt template
        self.extraction_prompt = """
You are an expert information extraction system. Analyze the following news article and extract structured entities.

Extract the following types of entities with their attributes:

1. PERSONS: People mentioned in the article
   - name: Full name of the person
   - role: Their role, job title, or relationship to the story

2. ORGANIZATIONS: Companies, institutions, government bodies, etc.
   - name: Full name of the organization
   - type: Type of organization (company, government, institution, etc.)

3. LOCATIONS: Places mentioned in the article
   - name: Name of the location
   - type: Type of location (city, country, region, etc.)

4. EVENTS: Specific events, incidents, or occurrences
   - name: Name or description of the event
   - date: Date if mentioned (or "unknown" if not specified)

5. TOPICS: Key themes, subjects, or topics covered in the article (list of strings)

Return your response as a JSON object with exactly this structure:
{
    "persons": [{"name": "string", "role": "string"}],
    "organizations": [{"name": "string", "type": "string"}],
    "locations": [{"name": "string", "type": "string"}],
    "events": [{"name": "string", "date": "string"}],
    "topics": ["string1", "string2", "string3"]
}

Article to analyze:
"""
    
    def extract_entities_from_text(self, text: str, article_id: str) -> EntityExtractionResult:
        """
        Extract entities from a single article text.
        
        Args:
            text: Article text content
            article_id: Unique identifier for the article
            
        Returns:
            EntityExtractionResult with extracted entities or error information
        """
        try:
            # Prepare the full prompt
            full_prompt = self.extraction_prompt + f"\n\n{text}"
            
            # Call OpenAI API with JSON mode
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise entity extraction system. Always respond with valid JSON."
                    },
                    {
                        "role": "user", 
                        "content": full_prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0,  # Consistent extraction
                max_tokens=1500
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content
            entities_data = json.loads(content)
            
            # Validate structure and provide defaults
            result = EntityExtractionResult(
                article_id=article_id,
                persons=entities_data.get('persons', []),
                organizations=entities_data.get('organizations', []),
                locations=entities_data.get('locations', []),
                events=entities_data.get('events', []),
                topics=entities_data.get('topics', []),
                extraction_successful=True
            )
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON parsing error for {article_id}: {e}")
            return EntityExtractionResult(
                article_id=article_id,
                persons=[], organizations=[], locations=[], events=[], topics=[],
                extraction_successful=False,
                error_message=f"JSON parsing error: {e}"
            )
            
        except Exception as e:
            print(f"   ❌ API error for {article_id}: {e}")
            return EntityExtractionResult(
                article_id=article_id,
                persons=[], organizations=[], locations=[], events=[], topics=[],
                extraction_successful=False,
                error_message=f"API error: {e}"
            )
    
    def extract_entities_from_articles(self, articles: List[Dict[str, Any]]) -> List[EntityExtractionResult]:
        """
        Extract entities from multiple articles with batch processing.
        
        Args:
            articles: List of article dictionaries with 'text' and 'article_id'
            
        Returns:
            List of EntityExtractionResult objects
        """
        print(f"🔍 Extracting entities from {len(articles)} articles...")
        print(f"   Using model: {self.model}")
        print(f"   Temperature: 0 (consistent extraction)")
        
        results = []
        successful_extractions = 0
        failed_extractions = 0
        
        for i, article in enumerate(articles):
            # Get article text and ID
            text = article.get('text', '')
            article_id = article.get('article_id', f'article_{i}')
            
            if not text:
                print(f"   ⚠️ Skipping article {article_id} - no text content")
                results.append(EntityExtractionResult(
                    article_id=article_id,
                    persons=[], organizations=[], locations=[], events=[], topics=[],
                    extraction_successful=False,
                    error_message="No text content"
                ))
                failed_extractions += 1
                continue
            
            # Extract entities
            result = self.extract_entities_from_text(text, article_id)
            results.append(result)
            
            if result.extraction_successful:
                successful_extractions += 1
                # Count extracted entities
                entity_count = (len(result.persons) + len(result.organizations) + 
                              len(result.locations) + len(result.events) + len(result.topics))
                print(f"   ✅ {article_id}: {entity_count} entities extracted")
            else:
                failed_extractions += 1
            
            # Progress reporting every 5 articles
            if (i + 1) % 5 == 0 or (i + 1) == len(articles):
                print(f"   Progress: {i + 1}/{len(articles)} articles processed")
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
        
        # Summary statistics
        print(f"\n📊 Entity Extraction Summary:")
        print(f"   Successful extractions: {successful_extractions}/{len(articles)}")
        print(f"   Failed extractions: {failed_extractions}")
        
        if successful_extractions > 0:
            # Calculate entity statistics
            total_persons = sum(len(r.persons) for r in results if r.extraction_successful)
            total_orgs = sum(len(r.organizations) for r in results if r.extraction_successful)
            total_locations = sum(len(r.locations) for r in results if r.extraction_successful)
            total_events = sum(len(r.events) for r in results if r.extraction_successful)
            total_topics = sum(len(r.topics) for r in results if r.extraction_successful)
            
            print(f"   Total entities extracted:")
            print(f"     Persons: {total_persons}")
            print(f"     Organizations: {total_orgs}")
            print(f"     Locations: {total_locations}")
            print(f"     Events: {total_events}")
            print(f"     Topics: {total_topics}")
        
        return results
    
    def convert_results_to_json(self, results: List[EntityExtractionResult]) -> Dict[str, Any]:
        """
        Convert extraction results to JSON-serializable format.
        
        Args:
            results: List of EntityExtractionResult objects
            
        Returns:
            Dictionary with extraction results and metadata
        """
        json_results = {
            'extraction_metadata': {
                'total_articles': len(results),
                'successful_extractions': sum(1 for r in results if r.extraction_successful),
                'failed_extractions': sum(1 for r in results if not r.extraction_successful),
                'model_used': self.model,
                'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'extractions': []
        }
        
        for result in results:
            extraction_data = {
                'article_id': result.article_id,
                'extraction_successful': result.extraction_successful,
                'entities': {
                    'persons': result.persons,
                    'organizations': result.organizations,
                    'locations': result.locations,
                    'events': result.events,
                    'topics': result.topics
                }
            }
            
            if not result.extraction_successful:
                extraction_data['error_message'] = result.error_message
            
            json_results['extractions'].append(extraction_data)
        
        return json_results


def main():
    """Demo function for entity extraction."""
    print("🔍 Entity Extractor Demo")
    print("=" * 40)
    
    # Sample article for testing
    sample_article = {
        'article_id': 'demo_entity_extraction',
        'text': """
        Apple Inc. CEO Tim Cook announced today that the company will be investing 
        $1 billion in a new manufacturing facility in Austin, Texas. The announcement 
        was made during a press conference at Apple's headquarters in Cupertino, California.
        
        Cook stated that the Austin facility will create over 5,000 new jobs and will 
        focus on producing the company's Mac Pro computers. Texas Governor Greg Abbott 
        welcomed the investment, calling it a major win for the state's technology sector.
        
        The facility is expected to be operational by late 2024, coinciding with Apple's 
        plans to increase domestic manufacturing capabilities. This investment is part of 
        Apple's broader commitment to American manufacturing and job creation.
        """
    }
    
    try:
        extractor = EntityExtractor()
        
        print("Sample article:")
        print(f"ID: {sample_article['article_id']}")
        print(f"Text: {sample_article['text'][:150]}...")
        
        result = extractor.extract_entities_from_text(
            sample_article['text'], 
            sample_article['article_id']
        )
        
        if result.extraction_successful:
            print(f"\n✅ Extraction successful!")
            print(f"Persons: {len(result.persons)}")
            for person in result.persons:
                print(f"  - {person['name']} ({person['role']})")
            
            print(f"Organizations: {len(result.organizations)}")
            for org in result.organizations:
                print(f"  - {org['name']} ({org['type']})")
            
            print(f"Locations: {len(result.locations)}")
            for loc in result.locations:
                print(f"  - {loc['name']} ({loc['type']})")
            
            print(f"Events: {len(result.events)}")
            for event in result.events:
                print(f"  - {event['name']} ({event['date']})")
            
            print(f"Topics: {len(result.topics)}")
            for topic in result.topics:
                print(f"  - {topic}")
        else:
            print(f"❌ Extraction failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure to set OPENAI_API_KEY environment variable")


if __name__ == "__main__":
    main()