"""
Smart Keyword Extractor - Dependency-free solution
Extracts high-quality keywords from customer reviews using basic Python.
No sentiment analysis - focuses on pure keyword extraction.
"""

import re
import csv
from collections import Counter, defaultdict
import string


class SmartKeywordExtractor:
    def __init__(self):
        # Common English stopwords
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to',
            'was', 'will', 'with', 'we', 'i', 'you', 'they', 'this', 'but', 'or',
            'had', 'have', 'very', 'so', 'can', 'could', 'would', 'should', 'there',
            'their', 'them', 'us', 'our', 'my', 'me', 'his', 'her', 'him', 'were',
            'did', 'do', 'does', 'am', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'than', 'too', 'up', 'down', 'out', 'off', 'over', 'under',
            'again', 'further', 'then', 'once', 'here', 'when', 'where', 'why',
            'how', 'what', 'which', 'who', 'if', 'because', 'until', 'while',
            'about', 'against', 'between', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'now', 'also', 'just', 'get', 'got'
        }

        # Irrelevant/low-value words specific to reviews
        self.irrelevant_words = {
            'day', 'night', 'time', 'times', 'week', 'month', 'year', 'hour',
            'minute', 'today', 'yesterday', 'tomorrow', 'ago', 'later', 'early',
            'late', 'first', 'second', 'third', 'last', 'next', 'previous',
            'every', 'always', 'never', 'sometimes', 'usually', 'often',
            'really', 'quite', 'pretty', 'much', 'many', 'lot', 'lots',
            'bit', 'little', 'big', 'small', 'large', 'huge', 'tiny',
            'place', 'thing', 'things', 'way', 'ways', 'part', 'side',
            'end', 'beginning', 'middle', 'top', 'bottom', 'back', 'front',
            'left', 'right', 'around', 'near', 'far', 'close', 'away',
            'sure', 'yes', 'ok', 'okay', 'well', 'fine', 'alright',
            'maybe', 'perhaps', 'probably', 'definitely', 'certainly'
        }

        # Hotel-specific adjectives for quality descriptions
        self.hotel_adjectives = {
            'excellent', 'great', 'good', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'perfect', 'outstanding', 'brilliant', 'fabulous',
            'superb', 'magnificent', 'beautiful', 'lovely', 'nice', 'pleasant',
            'comfortable', 'clean', 'fresh', 'delicious', 'tasty', 'friendly',
            'helpful', 'professional', 'courteous', 'attentive', 'responsive',
            'quick', 'fast', 'convenient', 'easy', 'smooth', 'efficient',
            'bad', 'poor', 'terrible', 'awful', 'horrible', 'disgusting',
            'dirty', 'filthy', 'noisy', 'loud', 'slow', 'delayed', 'late',
            'rude', 'unhelpful', 'unprofessional', 'disappointing',
            'unsatisfactory', 'unacceptable', 'inconvenient', 'difficult',
            'uncomfortable', 'cramped', 'small', 'tiny', 'old', 'outdated',
            'broken', 'damaged', 'smelly', 'cold', 'hot', 'expensive'
        }

        # Hotel/hospitality specific nouns
        self.hotel_nouns = {
            'hotel', 'room', 'service', 'staff', 'food', 'breakfast', 'dinner',
            'lunch', 'restaurant', 'bar', 'lobby', 'reception', 'check-in',
            'checkout', 'bed', 'bathroom', 'shower', 'wifi', 'internet',
            'parking', 'pool', 'gym', 'spa', 'location', 'view', 'beach',
            'amenities', 'facilities', 'price', 'value', 'money', 'cost',
            'booking', 'reservation', 'stay', 'visit', 'trip', 'vacation',
            'holiday', 'experience', 'management', 'housekeeping', 'maintenance',
            'security', 'concierge', 'porter', 'bellboy', 'valet'
        }

    def clean_text(self, text):
        """Clean and normalize text"""
        if not text or text == '' or str(text).lower() == 'nan':
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Remove special characters but keep spaces and hyphens
        text = re.sub(r'[^\w\s\-]', ' ', text)

        # Replace multiple spaces/hyphens with single space
        text = re.sub(r'[\s\-]+', ' ', text)

        return text.strip()

    def is_valid_word(self, word):
        """Check if word is valid for keyword extraction"""
        if not word or len(word) < 2:
            return False

        # Skip if it's a stopword or irrelevant word
        if word in self.stopwords or word in self.irrelevant_words:
            return False

        # Skip if it's just numbers
        if word.isdigit():
            return False

        # Skip if it contains mostly numbers
        if sum(c.isdigit() for c in word) > len(word) / 2:
            return False

        return True

    def extract_unigrams(self, text):
        """Extract meaningful unigrams (single words)"""
        words = text.split()
        unigrams = []

        for word in words:
            if self.is_valid_word(word):
                # Focus on hotel adjectives and hotel-specific nouns
                if (word in self.hotel_adjectives or
                        word in self.hotel_nouns):
                    unigrams.append(word)

        return unigrams

    def extract_bigrams(self, text):
        """Extract meaningful bigrams"""
        words = text.split()
        bigrams = []

        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]

            if not (self.is_valid_word(word1) and self.is_valid_word(word2)):
                continue

            # Look for adjective + noun patterns
            adjective_noun = (word1 in self.hotel_adjectives and word2 in self.hotel_nouns)

            # Look for noun + noun combinations that make sense
            noun_noun = (word1 in self.hotel_nouns and word2 in self.hotel_nouns)

            if adjective_noun or noun_noun:
                bigram = f"{word1} {word2}"
                bigrams.append(bigram)

        return bigrams

    def extract_trigrams(self, text):
        """Extract meaningful trigrams"""
        words = text.split()
        trigrams = []

        for i in range(len(words) - 2):
            word1, word2, word3 = words[i], words[i + 1], words[i + 2]

            if not (self.is_valid_word(word1) and
                    self.is_valid_word(word2) and
                    self.is_valid_word(word3)):
                continue

            # Look for adjective-adjective-noun or adjective-noun-noun patterns
            adj_adj_noun = (word1 in self.hotel_adjectives and
                            word2 in self.hotel_adjectives and
                            word3 in self.hotel_nouns)

            adj_noun_noun = (word1 in self.hotel_adjectives and
                             word2 in self.hotel_nouns and word3 in self.hotel_nouns)

            if adj_adj_noun or adj_noun_noun:
                trigram = f"{word1} {word2} {word3}"
                trigrams.append(trigram)

        return trigrams

    def extract_keywords_from_reviews(self, csv_file):
        """Extract keywords from CSV file containing reviews"""
        print(f"Loading reviews from {csv_file}...")

        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                reviews = []
                for row in reader:
                    # Try different possible column names
                    review_text = (row.get('HIGHLIGHTED_REVIEW_CONTENT') or
                                   row.get('review') or
                                   row.get('content') or
                                   row.get('text') or '')
                    if review_text.strip():
                        reviews.append(review_text)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return None

        print(f"Loaded {len(reviews)} reviews")

        if not reviews:
            print("No reviews found!")
            return None

        # Collect all keywords
        all_unigrams = []
        all_bigrams = []
        all_trigrams = []

        for review in reviews:
            clean_review = self.clean_text(review)

            all_unigrams.extend(self.extract_unigrams(clean_review))
            all_bigrams.extend(self.extract_bigrams(clean_review))
            all_trigrams.extend(self.extract_trigrams(clean_review))

        # Count frequencies
        unigram_counts = Counter(all_unigrams)
        bigram_counts = Counter(all_bigrams)
        trigram_counts = Counter(all_trigrams)

        # Filter by frequency (at least 2 occurrences)
        filtered_unigrams = {k: v for k, v in unigram_counts.items() if v >= 2}
        filtered_bigrams = {k: v for k, v in bigram_counts.items() if v >= 2}
        filtered_trigrams = {k: v for k, v in trigram_counts.items() if v >= 2}

        print(f"Found keywords:")
        print(f"  - Unigrams: {len(filtered_unigrams)}")
        print(f"  - Bigrams: {len(filtered_bigrams)}")
        print(f"  - Trigrams: {len(filtered_trigrams)}")

        return {
            'unigrams': filtered_unigrams,
            'bigrams': filtered_bigrams,
            'trigrams': filtered_trigrams
        }

    def categorize_keywords(self, keywords):
        """Categorize keywords by business aspects"""
        categories = {
            'Room Quality': [],
            'Service Quality': [],
            'Food & Dining': [],
            'Location & Amenities': [],
            'Value & Pricing': [],
            'Overall Experience': []
        }

        # Room-related keywords
        room_words = ['room', 'bed', 'bathroom', 'shower', 'clean', 'comfortable', 'spacious']

        # Service-related keywords
        service_words = ['service', 'staff', 'reception', 'check', 'helpful', 'friendly', 'professional']

        # Food-related keywords
        food_words = ['food', 'breakfast', 'dinner', 'lunch', 'restaurant', 'delicious', 'tasty']

        # Location & amenities
        location_words = ['location', 'pool', 'gym', 'spa', 'wifi', 'parking', 'beach', 'view']

        # Value-related keywords
        value_words = ['price', 'value', 'money', 'cost', 'expensive', 'cheap', 'affordable']

        # Overall experience
        experience_words = ['stay', 'experience', 'visit', 'trip', 'hotel', 'excellent', 'great', 'poor']

        for keyword_type in ['unigrams', 'bigrams', 'trigrams']:
            for keyword, freq in keywords[keyword_type].items():
                keyword_lower = keyword.lower()

                # Categorize based on word content
                if any(word in keyword_lower for word in room_words):
                    categories['Room Quality'].append((keyword, freq, keyword_type))
                elif any(word in keyword_lower for word in service_words):
                    categories['Service Quality'].append((keyword, freq, keyword_type))
                elif any(word in keyword_lower for word in food_words):
                    categories['Food & Dining'].append((keyword, freq, keyword_type))
                elif any(word in keyword_lower for word in location_words):
                    categories['Location & Amenities'].append((keyword, freq, keyword_type))
                elif any(word in keyword_lower for word in value_words):
                    categories['Value & Pricing'].append((keyword, freq, keyword_type))
                else:
                    categories['Overall Experience'].append((keyword, freq, keyword_type))

        return categories

    def save_to_excel(self, keywords, output_file):
        """Save keywords to Excel-compatible CSV file"""
        print(f"Saving keywords to {output_file}...")

        # Categorize keywords
        categories = self.categorize_keywords(keywords)

        # Prepare data for Excel
        excel_data = []

        # Add header
        excel_data.append(['Category', 'Keyword', 'Type', 'Frequency'])

        for category, keyword_list in categories.items():
            if not keyword_list:
                continue

            # Sort by frequency (highest first)
            keyword_list.sort(key=lambda x: x[1], reverse=True)

            for keyword, freq, kw_type in keyword_list:
                excel_data.append([category, keyword, kw_type, freq])

        # Save to CSV (Excel compatible)
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(excel_data)

            print(f"Successfully saved {len(excel_data) - 1} keywords to {output_file}")

            # Print summary
            print("\nKeyword Summary by Category:")
            for category, keyword_list in categories.items():
                if keyword_list:
                    print(f"  {category}: {len(keyword_list)} keywords")
                    # Show top 3 keywords
                    top_keywords = sorted(keyword_list, key=lambda x: x[1], reverse=True)[:3]
                    for kw, freq, _ in top_keywords:
                        print(f"    - {kw} ({freq}x)")

            return True

        except Exception as e:
            print(f"Error saving file: {e}")
            return False


def main():
    """Main function to run keyword extraction"""
    extractor = SmartKeywordExtractor()

    # Input and output files
    input_file = r"c:\Users\venky\Desktop\full pipeline 2\200 reviews.csv"
    output_file = r"c:\Users\venky\Desktop\full pipeline 2\keywords_domain specfic.csv"

    print("=== Smart Keyword Extractor (No Sentiment) ===")
    print("Extracting high-quality keywords from customer reviews...")

    # Extract keywords
    keywords = extractor.extract_keywords_from_reviews(input_file)

    if keywords is None:
        print("Failed to extract keywords!")
        return

    # Save to Excel
    success = extractor.save_to_excel(keywords, output_file)

    if success:
        print(f"\n✅ Complete! Check the file: {output_file}")
        print("You can open this CSV file in Excel to see all the keywords organized by category.")
    else:
        print("❌ Failed to save keywords!")


if __name__ == "__main__":
    main()
