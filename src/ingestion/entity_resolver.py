"""
Entity Resolution Module

Normalizes and deduplicates entities extracted from articles.
Resolves common variations and maintains entity-to-article mappings.
"""

from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
import re


class EntityResolver:
    """
    Resolves and normalizes extracted entities for knowledge graph construction.
    
    Handles deduplication, normalization, and aggregation of entity mentions
    across multiple articles.
    """
    
    def __init__(self):
        """Initialize the entity resolver with normalization mappings."""
        
        # Location normalization mappings
        self.location_mappings = {
            'uk': 'United Kingdom',
            'britain': 'United Kingdom', 
            'great britain': 'United Kingdom',
            'england': 'United Kingdom',
            'us': 'United States',
            'usa': 'United States',
            'america': 'United States',
            'united states of america': 'United States',
            'nyc': 'New York City',
            'ny': 'New York',
            'la': 'Los Angeles',
            'sf': 'San Francisco',
            'dc': 'Washington D.C.',
            'washington dc': 'Washington D.C.',
            'eu': 'European Union',
            'uae': 'United Arab Emirates',
        }
        
        # Organization normalization mappings
        self.organization_mappings = {
            'apple inc': 'Apple',
            'apple inc.': 'Apple',
            'apple computer': 'Apple',
            'microsoft corp': 'Microsoft',
            'microsoft corp.': 'Microsoft',
            'microsoft corporation': 'Microsoft',
            'google inc': 'Google',
            'google inc.': 'Google',
            'alphabet inc': 'Google',
            'amazon.com': 'Amazon',
            'amazon inc': 'Amazon',
            'facebook inc': 'Meta',
            'meta platforms': 'Meta',
            'ibm corp': 'IBM',
            'general motors': 'GM',
            'jpmorgan chase': 'JPMorgan',
            'goldman sachs': 'Goldman Sachs',
            'morgan stanley': 'Morgan Stanley',
            'coca cola': 'Coca-Cola',
            'pepsico': 'PepsiCo',
            'mcdonalds': 'McDonald\'s',
            'walmart inc': 'Walmart',
            'bbc news': 'BBC',
            'cnn news': 'CNN',
            'fox news': 'Fox News',
            'new york times': 'The New York Times',
            'wall street journal': 'The Wall Street Journal',
        }
        
        # Person name normalization patterns
        self.title_patterns = re.compile(r'\b(mr|mrs|ms|dr|prof|president|ceo|cto|cfo|chairman|senator|governor|minister|prime minister|king|queen|prince|princess)\b\.?\s*', re.IGNORECASE)
        
        # Common role mappings for consistency
        self.role_mappings = {
            'ceo': 'CEO',
            'chief executive officer': 'CEO',
            'chief executive': 'CEO',
            'cto': 'CTO',
            'chief technology officer': 'CTO',
            'cfo': 'CFO',
            'chief financial officer': 'CFO',
            'president': 'President',
            'chairman': 'Chairman',
            'chairwoman': 'Chairman',
            'prime minister': 'Prime Minister',
            'senator': 'Senator',
            'governor': 'Governor',
            'minister': 'Minister',
        }
    
    def normalize_string(self, text: str) -> str:
        """
        Normalize a string for comparison (lowercase, stripped).
        
        Args:
            text: Input string to normalize
            
        Returns:
            Normalized string for comparison
        """
        if not text:
            return ""
        return text.strip().lower()
    
    def title_case_string(self, text: str) -> str:
        """
        Convert string to proper title case for output.
        
        Args:
            text: Input string
            
        Returns:
            Title-cased string
        """
        if not text:
            return ""
        
        # Handle special cases and abbreviations
        words = text.split()
        result = []
        
        for word in words:
            # Preserve known abbreviations in uppercase
            if word.upper() in ['USA', 'UK', 'EU', 'UAE', 'NYC', 'LA', 'DC', 'CEO', 'CTO', 'CFO', 'AI', 'IT', 'HR']:
                result.append(word.upper())
            elif word.lower() in ['and', 'or', 'of', 'the', 'in', 'on', 'at', 'to', 'for', 'with']:
                # Keep articles and prepositions lowercase (except if first word)
                if len(result) == 0:
                    result.append(word.capitalize())
                else:
                    result.append(word.lower())
            else:
                result.append(word.capitalize())
        
        return ' '.join(result)
    
    def normalize_person_name(self, name: str) -> str:
        """
        Normalize person name by removing titles and standardizing format.
        
        Args:
            name: Person name to normalize
            
        Returns:
            Normalized person name
        """
        if not name:
            return ""
        
        # Remove titles and clean up
        cleaned = self.title_patterns.sub('', name).strip()
        
        # Title case the result
        return self.title_case_string(cleaned)
    
    def normalize_role(self, role: str) -> str:
        """
        Normalize person role/title.
        
        Args:
            role: Role to normalize
            
        Returns:
            Normalized role
        """
        if not role:
            return ""
        
        normalized_lower = self.normalize_string(role)
        
        # Check for known role mappings
        if normalized_lower in self.role_mappings:
            return self.role_mappings[normalized_lower]
        
        # Default to title case
        return self.title_case_string(role)
    
    def resolve_location(self, location_name: str) -> str:
        """
        Resolve location name using normalization mappings.
        
        Args:
            location_name: Location name to resolve
            
        Returns:
            Resolved location name
        """
        if not location_name:
            return ""
        
        normalized = self.normalize_string(location_name)
        
        # Check for known location mappings
        if normalized in self.location_mappings:
            return self.location_mappings[normalized]
        
        # Default to title case
        return self.title_case_string(location_name)
    
    def resolve_organization(self, org_name: str) -> str:
        """
        Resolve organization name using normalization mappings.
        
        Args:
            org_name: Organization name to resolve
            
        Returns:
            Resolved organization name
        """
        if not org_name:
            return ""
        
        normalized = self.normalize_string(org_name)
        
        # Check for known organization mappings
        if normalized in self.organization_mappings:
            return self.organization_mappings[normalized]
        
        # Default to title case
        return self.title_case_string(org_name)
    
    def resolve_entities(self, extraction_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve and deduplicate entities from extraction results.
        
        Args:
            extraction_results: List of extraction results from EntityExtractor
            
        Returns:
            Dictionary with resolved entities and mappings
        """
        print("🔄 Resolving and deduplicating entities...")
        
        # Track entities and their article mentions
        persons_dict = {}  # normalized_name -> {name, roles, articles}
        organizations_dict = {}  # normalized_name -> {name, types, articles}
        locations_dict = {}  # normalized_name -> {name, types, articles}
        events_dict = {}  # normalized_name -> {name, dates, articles}
        topics_set = set()  # All unique topics
        
        # Entity-to-articles mapping
        entity_articles = defaultdict(set)  # entity_id -> set of article_ids
        
        successful_extractions = 0
        
        for result in extraction_results:
            if not result.get('extraction_successful', False):
                continue
            
            successful_extractions += 1
            article_id = result['article_id']
            entities = result['entities']
            
            # Process persons
            for person in entities.get('persons', []):
                name = person.get('name', '').strip()
                role = person.get('role', '').strip()
                
                if not name:
                    continue
                
                normalized_name = self.normalize_person_name(name)
                normalized_key = self.normalize_string(normalized_name)
                
                if normalized_key not in persons_dict:
                    persons_dict[normalized_key] = {
                        'name': normalized_name,
                        'roles': set(),
                        'articles': set()
                    }
                
                if role:
                    normalized_role = self.normalize_role(role)
                    persons_dict[normalized_key]['roles'].add(normalized_role)
                
                persons_dict[normalized_key]['articles'].add(article_id)
                entity_articles[f"person:{normalized_name}"].add(article_id)
            
            # Process organizations
            for org in entities.get('organizations', []):
                name = org.get('name', '').strip()
                org_type = org.get('type', '').strip()
                
                if not name:
                    continue
                
                resolved_name = self.resolve_organization(name)
                normalized_key = self.normalize_string(resolved_name)
                
                if normalized_key not in organizations_dict:
                    organizations_dict[normalized_key] = {
                        'name': resolved_name,
                        'types': set(),
                        'articles': set()
                    }
                
                if org_type:
                    normalized_type = self.title_case_string(org_type)
                    organizations_dict[normalized_key]['types'].add(normalized_type)
                
                organizations_dict[normalized_key]['articles'].add(article_id)
                entity_articles[f"organization:{resolved_name}"].add(article_id)
            
            # Process locations
            for location in entities.get('locations', []):
                name = location.get('name', '').strip()
                loc_type = location.get('type', '').strip()
                
                if not name:
                    continue
                
                resolved_name = self.resolve_location(name)
                normalized_key = self.normalize_string(resolved_name)
                
                if normalized_key not in locations_dict:
                    locations_dict[normalized_key] = {
                        'name': resolved_name,
                        'types': set(),
                        'articles': set()
                    }
                
                if loc_type:
                    normalized_type = self.title_case_string(loc_type)
                    locations_dict[normalized_key]['types'].add(normalized_type)
                
                locations_dict[normalized_key]['articles'].add(article_id)
                entity_articles[f"location:{resolved_name}"].add(article_id)
            
            # Process events
            for event in entities.get('events', []):
                name = event.get('name', '').strip()
                date = event.get('date', '').strip()
                
                if not name:
                    continue
                
                event_name = self.title_case_string(name)
                normalized_key = self.normalize_string(event_name)
                
                if normalized_key not in events_dict:
                    events_dict[normalized_key] = {
                        'name': event_name,
                        'dates': set(),
                        'articles': set()
                    }
                
                if date and date.lower() != 'unknown':
                    events_dict[normalized_key]['dates'].add(date)
                
                events_dict[normalized_key]['articles'].add(article_id)
                entity_articles[f"event:{event_name}"].add(article_id)
            
            # Process topics
            for topic in entities.get('topics', []):
                if topic and topic.strip():
                    normalized_topic = self.title_case_string(topic.strip())
                    topics_set.add(normalized_topic)
                    entity_articles[f"topic:{normalized_topic}"].add(article_id)
        
        # Convert to final format
        unique_persons = [
            {
                'name': data['name'],
                'roles': list(data['roles']),
                'article_count': len(data['articles'])
            }
            for data in persons_dict.values()
        ]
        
        unique_organizations = [
            {
                'name': data['name'],
                'types': list(data['types']),
                'article_count': len(data['articles'])
            }
            for data in organizations_dict.values()
        ]
        
        unique_locations = [
            {
                'name': data['name'],
                'types': list(data['types']),
                'article_count': len(data['articles'])
            }
            for data in locations_dict.values()
        ]
        
        unique_events = [
            {
                'name': data['name'],
                'dates': list(data['dates']),
                'article_count': len(data['articles'])
            }
            for data in events_dict.values()
        ]
        
        unique_topics = list(topics_set)
        
        # Convert entity_articles to serializable format
        entity_articles_dict = {
            entity_id: list(article_ids) 
            for entity_id, article_ids in entity_articles.items()
        }
        
        # Print statistics
        print(f"✅ Entity resolution complete!")
        print(f"   Processed {successful_extractions} successful extractions")
        print(f"   Unique persons: {len(unique_persons)}")
        print(f"   Unique organizations: {len(unique_organizations)}")
        print(f"   Unique locations: {len(unique_locations)}")
        print(f"   Unique events: {len(unique_events)}")
        print(f"   Unique topics: {len(unique_topics)}")
        
        # Show some deduplication examples
        print(f"\n📊 Deduplication Examples:")
        if unique_persons:
            sample_person = unique_persons[0]
            if sample_person['article_count'] > 1:
                print(f"   Person '{sample_person['name']}' appears in {sample_person['article_count']} articles")
        
        if unique_organizations:
            sample_org = unique_organizations[0]
            if sample_org['article_count'] > 1:
                print(f"   Organization '{sample_org['name']}' appears in {sample_org['article_count']} articles")
        
        result = {
            'resolution_metadata': {
                'total_extractions_processed': len(extraction_results),
                'successful_extractions': successful_extractions,
                'unique_entity_counts': {
                    'persons': len(unique_persons),
                    'organizations': len(unique_organizations),
                    'locations': len(unique_locations),
                    'events': len(unique_events),
                    'topics': len(unique_topics)
                }
            },
            'resolved_entities': {
                'persons': unique_persons,
                'organizations': unique_organizations,
                'locations': unique_locations,
                'events': unique_events,
                'topics': unique_topics
            },
            'entity_articles_mapping': entity_articles_dict
        }
        
        return result


def main():
    """Demo function for entity resolution."""
    print("🔄 Entity Resolver Demo")
    print("=" * 40)
    
    # Sample extraction results for testing
    sample_results = [
        {
            'article_id': 'article_001',
            'extraction_successful': True,
            'entities': {
                'persons': [
                    {'name': 'Tim Cook', 'role': 'CEO'},
                    {'name': 'Mr. Tim Cook', 'role': 'chief executive officer'}
                ],
                'organizations': [
                    {'name': 'Apple Inc.', 'type': 'company'},
                    {'name': 'Apple Inc', 'type': 'technology company'}
                ],
                'locations': [
                    {'name': 'US', 'type': 'country'},
                    {'name': 'NYC', 'type': 'city'}
                ],
                'events': [
                    {'name': 'Product Launch', 'date': '2024-01-15'}
                ],
                'topics': ['technology', 'innovation', 'smartphones']
            }
        },
        {
            'article_id': 'article_002',
            'extraction_successful': True,
            'entities': {
                'persons': [
                    {'name': 'Tim Cook', 'role': 'Apple CEO'}
                ],
                'organizations': [
                    {'name': 'Apple', 'type': 'tech company'}
                ],
                'locations': [
                    {'name': 'United States', 'type': 'country'},
                    {'name': 'New York City', 'type': 'city'}
                ],
                'events': [],
                'topics': ['Technology', 'Business']
            }
        }
    ]
    
    resolver = EntityResolver()
    
    print("Sample extraction results:")
    print(f"Articles: {len(sample_results)}")
    
    resolved = resolver.resolve_entities(sample_results)
    
    print(f"\nResolution results:")
    print(f"Unique persons: {len(resolved['resolved_entities']['persons'])}")
    for person in resolved['resolved_entities']['persons']:
        print(f"  - {person['name']}: {person['roles']} (in {person['article_count']} articles)")
    
    print(f"Unique organizations: {len(resolved['resolved_entities']['organizations'])}")
    for org in resolved['resolved_entities']['organizations']:
        print(f"  - {org['name']}: {org['types']} (in {org['article_count']} articles)")
    
    print(f"Unique locations: {len(resolved['resolved_entities']['locations'])}")
    for loc in resolved['resolved_entities']['locations']:
        print(f"  - {loc['name']}: {loc['types']} (in {loc['article_count']} articles)")


if __name__ == "__main__":
    main()