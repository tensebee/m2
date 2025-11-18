#!/usr/bin/env python3
"""
Dictionary Scraper - 26 Word Cycles
Pulls one random word per letter (A→Z), saves every 26 words.
"""

import json
import requests
import time
import os
import signal
import sys
import random
from typing import Set, Dict, List
from datetime import datetime

class CycleDictionaryScraper:
    def __init__(self, output_file: str = "dictionary_corpus.json"):
        self.output_file = output_file
        self.base_url = "https://api.dictionaryapi.dev/api/v2/entries/en/"
        self.scraped_words: Set[str] = set()
        self.entries: List[Dict] = []
        self.request_delay = 0.8
        self.running = True
        
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.words_by_letter: Dict[str, List[str]] = {}
        
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
    
    def handle_shutdown(self, signum, frame):
        print("\n\n[SHUTDOWN] Saving and exiting...")
        self.running = False
        self.save_corpus()
        print("[EXIT] Done.")
        sys.exit(0)
    
    def load_word_list(self):
        """Load word list and organize by first letter"""
        wordlist_file = "english_words.txt"
        
        if os.path.exists(wordlist_file):
            print(f"[LOAD] Using word list: {wordlist_file}")
            with open(wordlist_file, 'r') as f:
                words = [line.strip().lower() for line in f if line.strip()]
        else:
            print("[DOWNLOAD] Fetching word list...")
            try:
                url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    words = response.text.strip().split('\n')
                    words = [w.strip().lower() for w in words if w.strip()]
                    
                    with open(wordlist_file, 'w') as f:
                        f.write('\n'.join(words))
                    print(f"[OK] Downloaded {len(words)} words")
                else:
                    print("[ERROR] Could not download")
                    sys.exit(1)
            except Exception as e:
                print(f"[ERROR] {e}")
                sys.exit(1)
        
        # Organize by first letter
        for letter in self.alphabet:
            self.words_by_letter[letter] = [w for w in words if w.startswith(letter)]
        
        total = sum(len(words) for words in self.words_by_letter.values())
        print(f"[INFO] Organized {total} words by first letter")
        for letter in self.alphabet:
            print(f"  {letter.upper()}: {len(self.words_by_letter[letter])} words")
    
    def load_existing(self):
        """Load existing corpus"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r') as f:
                    data = json.load(f)
                    if data and len(data) > 0:
                        self.entries = data[0].get('chat_messages', [])
                        for entry in self.entries:
                            text = entry.get('text', '')
                            word = text.split('(')[0].strip().lower()
                            self.scraped_words.add(word)
                
                print(f"\n[LOAD] Existing: {len(self.scraped_words)} words, {len(self.entries)} definitions")
            except Exception as e:
                print(f"[WARN] Could not load: {e}")
    
    def get_random_word_for_letter(self, letter: str) -> str:
        """Get random unscraped word starting with letter"""
        available = [w for w in self.words_by_letter[letter] 
                    if w not in self.scraped_words]
        
        if not available:
            # All words for this letter scraped, pick any
            available = self.words_by_letter[letter]
        
        if not available:
            return None
        
        return random.choice(available)
    
    def fetch_definition(self, word: str) -> bool:
        """Fetch and add definition"""
        try:
            response = requests.get(f"{self.base_url}{word}", timeout=5)
            if response.status_code != 200:
                return False
            
            data = response.json()
            count = self.process_response(word, data)
            
            if count > 0:
                self.scraped_words.add(word)
                return True
            
            return False
            
        except:
            return False
    
    def process_response(self, word: str, data: List[Dict]) -> int:
        """Process API response"""
        count = 0
        
        for entry in data:
            word = entry.get('word', word)
            
            for meaning in entry.get('meanings', []):
                pos = meaning.get('partOfSpeech', '')
                
                for defn in meaning.get('definitions', [])[:2]:  # Max 2 per POS
                    definition = defn.get('definition', '')
                    example = defn.get('example', '')
                    
                    if definition:
                        text = f"{word}"
                        if pos:
                            text += f" ({pos})"
                        text += f": {definition}"
                        
                        if example:
                            text += f" Example: {example}"
                        
                        if not any(e.get('text') == text for e in self.entries):
                            self.entries.append({
                                "uuid": f"def-{word}-{count}",
                                "text": text,
                                "sender": "assistant",
                                "created_at": datetime.now().isoformat(),
                                "updated_at": datetime.now().isoformat()
                            })
                            count += 1
        
        return count
    
    def save_corpus(self):
        """Save corpus"""
        conversation = [{
            "uuid": "dictionary-corpus",
            "name": "Dictionary Definitions Corpus",
            "summary": f"Dictionary with {len(self.scraped_words)} unique words",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": datetime.now().isoformat(),
            "chat_messages": self.entries
        }]
        
        temp_file = self.output_file + ".tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)
        
        os.replace(temp_file, self.output_file)
    
    def run_cycle(self, cycle_num: int) -> int:
        """Run one A-Z cycle, return success count"""
        print(f"\n{'='*70}")
        print(f"CYCLE #{cycle_num}")
        print(f"{'='*70}")
        
        success_count = 0
        
        for letter in self.alphabet:
            if not self.running:
                break
            
            word = self.get_random_word_for_letter(letter)
            if not word:
                print(f"[{letter.upper()}] No words available")
                continue
            
            success = self.fetch_definition(word)
            
            if success:
                success_count += 1
                print(f"[{letter.upper()}] {word:<20} | Defs: {len(self.entries):>5} | Words: {len(self.scraped_words):>5}")
            else:
                print(f"[{letter.upper()}] {word:<20} | [SKIP] not found")
            
            time.sleep(self.request_delay)
        
        return success_count
    
    def run(self):
        """Main loop"""
        print("\n" + "="*70)
        print("CYCLE DICTIONARY SCRAPER")
        print("="*70)
        print("Strategy: One random word per letter (A→Z)")
        print("Saves after each complete 26-word cycle")
        print("="*70)
        
        self.load_word_list()
        self.load_existing()
        
        print(f"\nOutput: {self.output_file}")
        print(f"Already scraped: {len(self.scraped_words)} words")
        print(f"Rate: ~{int(60/self.request_delay)} req/min")
        print("\nPress Ctrl+C to stop")
        
        cycle_num = 1
        start_time = time.time()
        
        while self.running:
            success = self.run_cycle(cycle_num)
            
            if success > 0:
                self.save_corpus()
                elapsed = time.time() - start_time
                print(f"\n[SAVE] Cycle {cycle_num} complete: {success}/26 words added")
                print(f"       Total: {len(self.scraped_words)} words, {len(self.entries)} definitions")
                print(f"       Runtime: {elapsed/60:.1f} min")
            
            cycle_num += 1
        
        # Final summary
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Cycles completed: {cycle_num - 1}")
        print(f"Runtime: {elapsed/3600:.1f} hours")
        print(f"Unique words: {len(self.scraped_words)}")
        print(f"Total definitions: {len(self.entries)}")
        print(f"Saved to: {self.output_file}")
        print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Cycle dictionary scraper')
    parser.add_argument('-o', '--output', default='dictionary_corpus.json')
    parser.add_argument('-d', '--delay', type=float, default=0.8)
    
    args = parser.parse_args()
    
    scraper = CycleDictionaryScraper(output_file=args.output)
    scraper.request_delay = args.delay
    scraper.run()


if __name__ == "__main__":
    main()