"""
BBC News Dataset Loader

Downloads and processes the BBC News dataset from HuggingFace,
creating a balanced subset for the KG-Enhanced RAG system.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

from ..config import config


class BBCDatasetLoader:
    """
    Handles downloading and processing of the BBC News dataset.
    
    Creates a balanced subset with equal representation from each category.
    """
    
    def __init__(self):
        self.config = config
        self.dataset = None
        self.subset_data = []
    
    def download_dataset(self) -> None:
        """Download the BBC News dataset from HuggingFace."""
        print("📥 Downloading BBC News dataset from HuggingFace...")
        try:
            self.dataset = load_dataset(self.config.BBC_DATASET_NAME)
            print(f"✅ Successfully downloaded dataset with {len(self.dataset['train'])} articles")
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            raise
    
    def create_balanced_subset(self) -> List[Dict]:
        """
        Create a balanced subset with equal articles per category.
        
        Returns:
            List[Dict]: List of articles with metadata
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call download_dataset() first.")
        
        print(f"🔄 Creating balanced subset with {self.config.ARTICLES_PER_CATEGORY} articles per category...")
        
        # Group articles by category
        category_articles = defaultdict(list)
        
        for idx, example in enumerate(self.dataset['train']):
            category = example['label_text'].lower()
            if category in self.config.BBC_CATEGORIES:
                article_data = {
                    'original_index': idx,
                    'category': category,
                    'text': example['text'],
                    'label': example['label'],
                    'label_text': example['label_text']
                }
                category_articles[category].append(article_data)
        
        # Print category distribution in original dataset
        print("\n📊 Original dataset distribution:")
        for category in self.config.BBC_CATEGORIES:
            count = len(category_articles[category])
            print(f"  {category.capitalize()}: {count} articles")
        
        # Sample articles from each category
        random.seed(self.config.RANDOM_SEED)
        subset_articles = []
        
        for category in self.config.BBC_CATEGORIES:
            available_articles = category_articles[category]
            
            if len(available_articles) < self.config.ARTICLES_PER_CATEGORY:
                print(f"⚠️ Warning: Only {len(available_articles)} articles available for {category}")
                selected_articles = available_articles
            else:
                selected_articles = random.sample(available_articles, self.config.ARTICLES_PER_CATEGORY)
            
            # Add unique article IDs
            for idx, article in enumerate(selected_articles):
                article['article_id'] = f"bbc_{category}_{idx:02d}"
                subset_articles.append(article)
        
        self.subset_data = subset_articles
        print(f"✅ Created subset with {len(subset_articles)} articles")
        
        return subset_articles
    
    def save_subset(self, output_path: Path = None) -> Path:
        """
        Save the subset to JSON file.
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path: Path where the file was saved
        """
        if not self.subset_data:
            raise ValueError("No subset data to save. Create subset first.")
        
        if output_path is None:
            output_path = self.config.get_bbc_subset_path()
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving subset to {output_path}...")
        
        # Create metadata
        metadata = {
            'dataset_info': {
                'source': self.config.BBC_DATASET_NAME,
                'total_articles': len(self.subset_data),
                'articles_per_category': self.config.ARTICLES_PER_CATEGORY,
                'categories': self.config.BBC_CATEGORIES,
                'creation_date': pd.Timestamp.now().isoformat()
            },
            'articles': self.subset_data
        }
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Dataset saved successfully to {output_path}")
        return output_path
    
    def display_statistics(self) -> None:
        """Display comprehensive dataset statistics."""
        if not self.subset_data:
            print("❌ No subset data available. Create subset first.")
            return
        
        print("\n📈 Dataset Statistics")
        print("=" * 50)
        
        # Category distribution
        category_counts = defaultdict(int)
        text_lengths = []
        word_counts = []
        
        for article in self.subset_data:
            category_counts[article['category']] += 1
            text_length = len(article['text'])
            word_count = len(article['text'].split())
            text_lengths.append(text_length)
            word_counts.append(word_count)
        
        print(f"Total Articles: {len(self.subset_data)}")
        print(f"Categories: {len(category_counts)}")
        print("\nCategory Distribution:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category.capitalize()}: {count} articles")
        
        # Text statistics
        print(f"\nText Length Statistics (characters):")
        print(f"  Mean: {sum(text_lengths) / len(text_lengths):.0f}")
        print(f"  Min: {min(text_lengths)}")
        print(f"  Max: {max(text_lengths)}")
        
        print(f"\nWord Count Statistics:")
        print(f"  Mean: {sum(word_counts) / len(word_counts):.0f}")
        print(f"  Min: {min(word_counts)}")
        print(f"  Max: {max(word_counts)}")
    
    def display_sample_articles(self, chars_limit: int = 300) -> None:
        """
        Display sample articles from each category.
        
        Args:
            chars_limit: Maximum characters to display per article
        """
        if not self.subset_data:
            print("❌ No subset data available. Create subset first.")
            return
        
        print(f"\n📄 Sample Articles (first {chars_limit} characters)")
        print("=" * 60)
        
        # Group by category for display
        category_articles = defaultdict(list)
        for article in self.subset_data:
            category_articles[article['category']].append(article)
        
        for category in self.config.BBC_CATEGORIES:
            if category in category_articles:
                sample_article = category_articles[category][0]
                print(f"\n🏷️ Category: {category.upper()}")
                print(f"📄 Article ID: {sample_article['article_id']}")
                print(f"📝 Text Preview:")
                preview_text = sample_article['text'][:chars_limit]
                if len(sample_article['text']) > chars_limit:
                    preview_text += "..."
                print(f"   {preview_text}")
                print("-" * 60)


def main():
    """Main function to download and process the BBC News dataset."""
    print("🚀 BBC News Dataset Loader")
    print("=" * 40)
    
    # Initialize loader
    loader = BBCDatasetLoader()
    
    try:
        # Download dataset
        loader.download_dataset()
        
        # Create balanced subset
        loader.create_balanced_subset()
        
        # Save subset
        output_path = loader.save_subset()
        
        # Display statistics
        loader.display_statistics()
        
        # Display sample articles
        loader.display_sample_articles()
        
        print(f"\n🎉 Successfully processed BBC News dataset!")
        print(f"📁 Saved to: {output_path}")
        print(f"📊 Total articles: {len(loader.subset_data)}")
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()