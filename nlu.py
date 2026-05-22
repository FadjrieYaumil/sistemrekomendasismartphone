# =========================
# nlu.py - UPDATED NLU WITH ENHANCED SERPAPI
# =========================
import re
import os
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from serpapi_searcher import SerpAPISearcher
from typing import Dict, List, Tuple, Any, Optional

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


# =========================
# DATA LOADER
# =========================
class DataLoader:
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder
        self.smartphones_df = None
        self.typo_dict = {}
        self.slang_dict = {}
        self.stop_words = set()
        self.brands = set()
        self.models = set()
        self.load_all()
    
    def load_all(self):
        print("\n📂 LOADING DATA...")
        
        try:
            path = os.path.join(self.data_folder, "smartphonedatasetfixstock.csv")
            self.smartphones_df = pd.read_csv(path, sep=';')
            self.smartphones_df.columns = self.smartphones_df.columns.str.lower().str.strip()
            
            if 'nama_brand' in self.smartphones_df.columns:
                self.smartphones_df['nama_brand'] = self.smartphones_df['nama_brand'].astype(str).str.lower().str.strip()
                self.brands = set(self.smartphones_df['nama_brand'].dropna().unique())
                print(f"  ✅ Brands in dataset: {sorted(self.brands)}")
            
            if 'model' in self.smartphones_df.columns:
                self.smartphones_df['model'] = self.smartphones_df['model'].astype(str).str.lower().str.strip()
                self.models = set(self.smartphones_df['model'].dropna().unique())
            
            if 'chipset' in self.smartphones_df.columns:
                self.smartphones_df['chipset'] = self.smartphones_df['chipset'].astype(str).str.lower().str.strip()
            
            if 'harga' in self.smartphones_df.columns:
                self.smartphones_df['harga'] = self.smartphones_df['harga'].apply(
                    lambda x: int(str(x).replace('.', '').replace(',', '')) if pd.notna(x) else 0
                )
            
            numeric_cols = ['resolusi_kamera', 'refresh_rate', 'ukuran_layar', 
                           'kapasitas_baterai', 'memori_internal', 'kapasitas_ram']
            for col in numeric_cols:
                if col in self.smartphones_df.columns:
                    self.smartphones_df[col] = pd.to_numeric(
                        self.smartphones_df[col].astype(str).str.replace('.', '').str.replace(',', ''), 
                        errors='coerce'
                    )
            
            if '5g' in self.smartphones_df.columns:
                self.smartphones_df['network_type'] = self.smartphones_df['5g'].apply(
                    lambda x: '5G' if x == 1 else '4G'
                )
            
            print(f"✅ Dataset loaded: {len(self.smartphones_df)} models, {len(self.brands)} brands")
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
            self.smartphones_df = pd.DataFrame()
        
        try:
            path = os.path.join(self.data_folder, "typodataset.csv")
            if os.path.exists(path):
                df_typo = pd.read_csv(path, sep=';', on_bad_lines='skip')
                if 'typo' in df_typo.columns and 'correct' in df_typo.columns:
                    self.typo_dict = dict(zip(df_typo["typo"].str.lower().str.strip(), df_typo["correct"].str.lower().str.strip()))
                    print(f"✅ Typo: {len(self.typo_dict)} entries")
        except: 
            print("ℹ️ Typo dataset not loaded")
        
        try:
            path = os.path.join(self.data_folder, "slangwords.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        self.slang_dict = eval(content)
                        print(f"✅ Slang: {len(self.slang_dict)} entries")
        except: 
            print("ℹ️ Slang words not loaded")
        
        try:
            path = os.path.join(self.data_folder, "stopwords.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    self.stop_words = set(line.strip().lower() for line in f if line.strip())
                    print(f"✅ Stopwords: {len(self.stop_words)} words")
        except: 
            print("ℹ️ Stopwords not loaded")


# =========================
# TEXT PREPROCESSOR
# =========================
class TextPreprocessor:
    def __init__(self, slang_dict, typo_dict, stop_words):
        self.slang_dict = slang_dict
        self.typo_dict = typo_dict
        self.stop_words = stop_words
    
    def preprocess(self, text: str) -> Tuple[str, List[str]]:
        changes = []
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        for i, w in enumerate(words):
            if w in self.slang_dict:
                words[i] = self.slang_dict[w]
                changes.append(f"Slang: '{w}' -> '{words[i]}'")
        text = ' '.join(words)
        
        words = text.split()
        for i, w in enumerate(words):
            if w in self.typo_dict:
                words[i] = self.typo_dict[w]
                changes.append(f"Typo: '{w}' -> '{words[i]}'")
        text = ' '.join(words)
        
        return text, changes


# =========================
# ENTITY EXTRACTOR (FIXED - Better detection)
# =========================
class EntityExtractor:
    def __init__(self, brands, models):
        self.brands = brands
        self.models = models
        self.intent_keywords = {
            'gaming': {'game', 'gaming', 'ngegame', 'ml', 'pubg', 'genshin', 'ff', 'free fire', 
                       'cod', 'valorant', 'aov', 'lol', 'roblox', 'minecraft', 'coc', 'fifa', 
                       'asphalt', 'honkai', 'mobile legend', 'esport', 'mlbb', 'dota', 'wild rift',
                       'codm', 'pubgm', 'fortnite', 'apex', 'among us', 'pokemon', 'gta',
                       'pes', 'efootball', 'farlight', 'undawn', 'tower of fantasy', 'nba',
                       'wuwa', 'wuthering waves', 'zenless', 'zzz', 'snowbreak'},
            'fotografi': {'kamera', 'foto', 'selfie', 'jernih', 'tajam', 'dslr', 'dlsr', 
                          'mirrorless', 'vlog', 'cinema', 'zoom', 'ois', 'stabil', 'potret', 
                          'swafoto', 'fotografi', 'photography', 'digital'},
            'baterai': {'baterai', 'batrai', 'batre', 'tahan lama', 'awet', 'charging', 
                        'fast charging', 'power', 'daya', 'hemat', 'awet'},
            'performa': {'kuat', 'kencang', 'ngebut', 'maksimal', 'lancar', 'cepat', 'performa', 
                         'responsif', 'multitasking', 'halus', 'ngebut'},
            'budget': {'murah', 'dibawah', 'harga', 'budget', 'terjangkau', 'ekonomis', 'miring', 
                       'hemat', 'max', 'maksimal', 'minimal', 'mahal', 'diatas', 'gak mahal',
                       'ga mahal', 'nggak mahal', 'tidak mahal', 'low budget', 'lowbudget'},
            'multimedia': {'layar', 'nonton', 'youtube', 'netflix', 'video', 'streaming', 
                           'bioskop', 'film', 'amoled', 'oled', 'lcd', 'super amoled'},
            'harian': {'standar', 'biasa', 'cukup', 'sehari', 'basic', 'sederhana', 'sosmed', 
                       'wa', 'whatsapp', 'chat', 'telepon', 'browsing', 'tiktok', 'instagram', 
                       'facebook', 'twitter', 'telegram'}
        }
        
        self.static_phrases = {
            'kamera jernih': {'intent': 'fotografi'},
            'kamera tajam': {'intent': 'fotografi'},
            'kamera bagus': {'intent': 'fotografi'},
            'foto jernih': {'intent': 'fotografi'},
            'foto bagus': {'intent': 'fotografi'},
            'kamera ois': {'intent': 'fotografi'},
            'kamera stabil': {'intent': 'fotografi'},
            'video stabil': {'intent': 'fotografi'},
            'kamera zoom': {'intent': 'fotografi'},
            'kamera selfie': {'intent': 'fotografi'},
            'kamera depan': {'intent': 'fotografi'},
            'kamera belakang': {'intent': 'fotografi'},
            'kamera utama': {'intent': 'fotografi'},
            'tidak lag': {'intent': 'performa'},
            'tidak lemot': {'intent': 'performa'},
            'tidak ngelag': {'intent': 'performa'},
            'anti lag': {'intent': 'performa'},
            'anti lemot': {'intent': 'performa'},
            'scroll lancar': {'intent': 'performa'},
            'lancar jaya': {'intent': 'performa'},
            'ngebut': {'intent': 'performa'},
            'kencang': {'intent': 'performa'},
            'responsif': {'intent': 'performa'},
            'halus': {'intent': 'performa'},
            'enteng': {'intent': 'performa'},
            'ringan': {'intent': 'performa'},
            'batre awet': {'intent': 'baterai'},
            'baterai awet': {'intent': 'baterai'},
            'baterai tahan lama': {'intent': 'baterai'},
            'tahan lama': {'intent': 'baterai'},
            'batre besar': {'intent': 'baterai'},
            'baterai besar': {'intent': 'baterai'},
            'fast charging': {'intent': 'baterai'},
            'ngecas cepat': {'intent': 'baterai'},
            'charging cepat': {'intent': 'baterai'},
            'hemat baterai': {'intent': 'baterai'},
            'hemat batre': {'intent': 'baterai'},
            'irit baterai': {'intent': 'baterai'},
            'daya tahan': {'intent': 'baterai'},
            'layar jernih': {'intent': 'multimedia'},
            'layar bagus': {'intent': 'multimedia'},
            'layar besar': {'intent': 'multimedia'},
            'layar lebar': {'intent': 'multimedia'},
            'layar amoled': {'intent': 'multimedia'},
            'nonton youtube': {'intent': 'multimedia'},
            'nonton film': {'intent': 'multimedia'},
            'nonton video': {'intent': 'multimedia'},
            'streaming': {'intent': 'multimedia'},
            'nonton netflix': {'intent': 'multimedia'},
            'sosmed': {'intent': 'harian'},
            'sosial media': {'intent': 'harian'},
            'medsos': {'intent': 'harian'},
            'chat wa': {'intent': 'harian'},
            'whatsapp': {'intent': 'harian'},
            'telepon': {'intent': 'harian'},
            'telpon': {'intent': 'harian'},
            'tiktok': {'intent': 'harian'},
            'instagram': {'intent': 'harian'},
            'facebook': {'intent': 'harian'},
            'browsing': {'intent': 'harian'},
            'sehari hari': {'intent': 'harian'},
            'kebutuhan sehari': {'intent': 'harian'},
            'paket lengkap': {'intent': 'performa'},
            'all rounder': {'intent': 'performa'},
        }
        
        self.ignore_terms = {
            'hp', 'handphone', 'smartphone', 'android', 'ios', 'iphone',
            'seluler', 'ponsel', 'telepon', 'genggam', 'device', 'perangkat',
            'juta', 'jt', 'ribu', 'rb', 'rupiah', 'rp',
            'harga', 'budget', 'murah', 'mahal',
            'maksimal', 'minimal', 'max', 'min', 'dibawah', 'diatas',
            'rekomendasi', 'saran', 'pilihan',
            'beli', 'membeli', 'cari', 'mencari',
            'dan', 'bisa', 'juga', 'sama', 'kalo', 'kalau',
            'banget', 'sih', 'nih', 'tuh', 'dong', 'deh', 'kok',
            'ini', 'itu', 'untuk', 'dengan', 'ada', 'buat',
            'yang', 'di', 'ke', 'dari', 'atau', 'tapi',
            'bagus', 'oke', 'sip', 'mantap', 'jos', 'gokil',
            'aja', 'dong', 'kan', 'ya', 'yah', 'ga', 'gak', 'nggak',
            '4g', '5g', 'lte', 'inci', 'inch', 'cm', 'mm',
            'mah', 'mp', 'gb', 'tb',
            'setara', 'kayak', 'seperti', 'mirip',
            'spesifikasi', 'spek', 'tipe', 'jenis', 'macam',
            'nonton', 'beli', 'cari', 'pakai', 'pake',
            'khusus', 'spesial', 'tertentu',
            'dapat', 'dapet', 'mendapatkan',
            'mendukung', 'support', 'supporting',
            'berat', 'ringan', 'sedang',
        }
    
    def extract(self, text: str) -> Dict:
        text_lower = text.lower()
        entities = {
            'ram': None, 'rom': None, 'battery': None, 'camera': None,
            'screen_size': None, 'refresh_rate': None, 'network': None,
            'price': None, 'price_operator': None, 'brand': None,
            'model': None, 'unknown_terms': [],
            'static_matches': [],
            'has_budget_mention': False,
            'has_explicit_price': False
        }
        
        # Extract RAM
        m = re.search(r'(\d+)\s*gb\s*ram|ram\s*(\d+)', text_lower)
        if m:
            entities['ram'] = int(m.group(1) or m.group(2))
        
        # Extract ROM/Storage
        m = re.search(r'(\d+)\s*gb\s*(?:rom|storage|internal)|(?:rom|storage)\s*(\d+)', text_lower)
        if m:
            entities['rom'] = int(m.group(1) or m.group(2))
        
        # Extract Battery
        m = re.search(r'(\d+)\s*mah|baterai\s*(\d+)', text_lower)
        if m:
            entities['battery'] = int(m.group(1) or m.group(2))
        
        # Extract Camera
        m = re.search(r'(\d+)\s*mp|kamera\s*(\d+)', text_lower)
        if m:
            entities['camera'] = int(m.group(1) or m.group(2))
        
        # Extract Screen Size
        m = re.search(r'(\d+[.,]?\d*)\s*(?:inci|inch|")|layar\s*(\d+[.,]?\d*)', text_lower)
        if m:
            try:
                val = float((m.group(1) or m.group(2)).replace(',', '.'))
                entities['screen_size'] = val
            except: pass
        
        # Extract Refresh Rate
        m = re.search(r'(\d+)\s*hz|refresh\s*rate\s*(\d+)', text_lower)
        if m:
            entities['refresh_rate'] = int(m.group(1) or m.group(2))
        
        # Budget detection
        budget_keywords = ['murah', 'terjangkau', 'ekonomis', 'hemat', 'budget', 'miring',
                          'gak mahal', 'ga mahal', 'nggak mahal', 'tidak mahal',
                          'low budget', 'lowbudget', 'dibawah', 'max', 'maksimal', 
                          'maks', 'maximal', 'minimal', 'min', 'diatas',
                          'harga', 'juta', 'jt', 'ribu', 'rb']
        
        if any(w in text_lower for w in budget_keywords):
            entities['has_budget_mention'] = True
            
            if any(w in text_lower for w in ['dibawah', 'max', 'maksimal', 'maks', 'maximal']):
                entities['price_operator'] = 'below'
            elif any(w in text_lower for w in ['diatas', 'min', 'minimal']):
                entities['price_operator'] = 'above'
            else:
                entities['price_operator'] = 'below'
        
        m = re.search(r'(\d+[.,]?\d*)\s*(?:juta|jt)', text_lower)
        if m:
            try:
                val = float(m.group(1).replace(',', '.'))
                entities['price'] = val * 1000000
                entities['has_explicit_price'] = True
                entities['has_budget_mention'] = True
                if not entities.get('price_operator'):
                    entities['price_operator'] = 'below'
            except: pass
        
        # Network
        if '5g' in text_lower: entities['network'] = '5g'
        elif '4g' in text_lower: entities['network'] = '4g'
        
        # Brand detection
        print(f"\n   Detecting brand from: '{text_lower}'")
        print(f"  Available brands: {sorted(self.brands)}")
        
        for brand in sorted(self.brands, key=len, reverse=True):
            pattern = r'\b' + re.escape(brand) + r'\b'
            if re.search(pattern, text_lower):
                entities['brand'] = brand
                print(f"  ✅ Brand detected (exact): '{brand}'")
                break
        
        if not entities['brand']:
            for brand in sorted(self.brands, key=len, reverse=True):
                if brand in text_lower:
                    entities['brand'] = brand
                    print(f"  ✅ Brand detected (contains): '{brand}'")
                    break
        
        print(f"   Final brand: {entities['brand']}")
        
        # Static phrase matching
        for phrase, info in self.static_phrases.items():
            if phrase in text_lower:
                entities['static_matches'].append({
                    'phrase': phrase,
                    'intent': info['intent']
                })
        
        # ENHANCED: Extract context terms
        entities['unknown_terms'] = self._extract_context_terms(text_lower)
        
        return entities
    
    def _extract_context_terms(self, text: str) -> List[str]:
        """Enhanced context term extraction - FIXED VERSION"""
        found_terms = []
        text_lower = text.lower()
        
        # Known games database (expanded)
        known_games = {
            'snowbreak': 'Snowbreak: Containment Zone',
            'snowbreak containment zone': 'Snowbreak: Containment Zone',
            'wuwa': 'Wuthering Waves',
            'wuthering waves': 'Wuthering Waves',
            'genshin': 'Genshin Impact',
            'genshin impact': 'Genshin Impact',
            'honkai': 'Honkai Star Rail',
            'honkai star rail': 'Honkai Star Rail',
            'ml': 'Mobile Legends',
            'mlbb': 'Mobile Legends',
            'mobile legend': 'Mobile Legends',
            'mobile legends': 'Mobile Legends',
            'aov': 'Arena of Valor',
            'pubg': 'PUBG Mobile',
            'pubg mobile': 'PUBG Mobile',
            'ff': 'Free Fire',
            'free fire': 'Free Fire',
            'cod': 'Call of Duty Mobile',
            'codm': 'Call of Duty Mobile',
            'valorant': 'Valorant Mobile',
            'lol': 'League of Legends Wild Rift',
            'wild rift': 'League of Legends Wild Rift',
            'fortnite': 'Fortnite Mobile',
            'roblox': 'Roblox',
            'minecraft': 'Minecraft',
            'among us': 'Among Us',
            'fifa': 'FIFA Mobile',
            'efootball': 'eFootball',
            'asphalt': 'Asphalt Legends',
            'coc': 'Clash of Clans',
            'tower of fantasy': 'Tower of Fantasy',
            'zenless': 'Zenless Zone Zero',
            'zzz': 'Zenless Zone Zero',
            'apex': 'Apex Legends Mobile',
            'pokemon': 'Pokemon Unite',
            'gta': 'GTA San Andreas Mobile',
            'nba': 'NBA 2K Mobile'
        }
        
        # FIRST: Direct game detection
        for game_key, game_full in known_games.items():
            if game_key in text_lower and game_full.lower() not in [t.lower() for t in found_terms]:
                found_terms.append(game_full)
                print(f"     🎮 Direct game detected: {game_full}")
        
        # SECOND: Camera/Photography detection
        camera_patterns = [
            r'kamera\s+(?:setara|seperti|mirip|kayak)\s+([a-zA-Z\s]+)',
            r'(?:foto|photography)\s+(?:setara|seperti)\s+([a-zA-Z\s]+)',
            r'setara\s+kamera\s+([a-zA-Z\s]+)',
            r'kamera\s+digital',
            r'dslr',
            r'mirrorless',
        ]
        
        for pattern in camera_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                term = match.strip()
                if term and len(term) > 2:
                    if 'kamera' in term:
                        term = term.replace('kamera', '').strip()
                    if term and term not in [t.lower() for t in found_terms]:
                        found_terms.append(term)
                        print(f"      Camera term detected: {term}")
        
        # THIRD: Extract after trigger words
        context_triggers = [
            'main', 'game', 'gaming', 'ngegame', 'spek', 'setara', 'seperti',
            'persis', 'mirip', 'kayak', 'macam', 'khusus', 'buat', 'untuk',
            'spesifikasi', 'tipe', 'jenis', 'seri', 'kelas', 'selevel',
            'kuat', 'kencang', 'ngebut', 'lancar', 'maksimal', 'banding'
        ]
        
        for trigger in context_triggers:
            pattern = r'(?:^|\s)' + re.escape(trigger) + r'\s+([a-zA-Z0-9\s]+?)(?:\s+(?:dan|atau|tapi|yang|di|ke|dari|untuk|dengan|harga|$)|\s*$)'
            matches = re.findall(pattern, text_lower)
            
            for match in matches:
                words = match.strip().split()
                for i in range(1, min(4, len(words) + 1)):
                    term = ' '.join(words[:i])
                    if len(term) > 2 and term not in self.ignore_terms:
                        if term not in [t.lower() for t in found_terms]:
                            if term in known_games:
                                found_terms.append(known_games[term])
                            else:
                                found_terms.append(term)
                            print(f"      Context term detected: {term}")
        
        # FOURTH: Special case - "kamera digital" direct
        if 'kamera digital' in text_lower and 'kamera digital' not in [t.lower() for t in found_terms]:
            found_terms.append('kamera digital')
            print(f"      Direct camera digital detected")
        
        # Clean and deduplicate
        seen = set()
        unique_terms = []
        for term in found_terms:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                unique_terms.append(term)
        
        print(f"      Final unknown terms: {unique_terms}")
        return unique_terms[:3]
    
    def detect_intents(self, text: str) -> Dict:
        text_lower = text.lower()
        scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(0.2 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[intent] = min(score, 1.0)
        
        if scores:
            total = sum(scores.values())
            scores = {k: v/total for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)}
        
        return scores
    
    def get_static_intents(self, static_matches):
        extra_intents = {}
        for match in static_matches:
            intent = match.get('intent', '')
            if intent:
                if intent not in extra_intents:
                    extra_intents[intent] = 0
                extra_intents[intent] += 0.3
        
        return extra_intents


# =========================
# RELATION MATRIX
# =========================
class RelationMatrix:
    MATRIX = {
        'gaming': {
            'RAM': 1.0, 'ROM': 0.5, 'Battery': 0.7, 'Camera': 0.1,
            'Chipset_Score': 1.0, 'Screen_Size': 0.5, 'Refresh_Rate': 0.7,
            'Network': 0.5, 'Price': 0.5
        },
        'fotografi': {
            'RAM': 0.3, 'ROM': 0.5, 'Battery': 0.5, 'Camera': 1.0,
            'Chipset_Score': 0.5, 'Screen_Size': 0.3, 'Refresh_Rate': 0.1,
            'Network': 0.1, 'Price': 0.5
        },
        'baterai': {
            'RAM': 0.3, 'ROM': 0.3, 'Battery': 1.0, 'Camera': 0.1,
            'Chipset_Score': 0.5, 'Screen_Size': 0.3, 'Refresh_Rate': 0.1,
            'Network': 0.3, 'Price': 0.5
        },
        'performa': {
            'RAM': 1.0, 'ROM': 0.5, 'Battery': 0.5, 'Camera': 0.3,
            'Chipset_Score': 1.0, 'Screen_Size': 0.3, 'Refresh_Rate': 0.5,
            'Network': 0.3, 'Price': 0.5
        },
        'multimedia': {
            'RAM': 0.5, 'ROM': 0.7, 'Battery': 0.5, 'Camera': 0.5,
            'Chipset_Score': 0.5, 'Screen_Size': 1.0, 'Refresh_Rate': 0.7,
            'Network': 0.5, 'Price': 0.5
        },
        'harian': {
            'RAM': 0.5, 'ROM': 0.5, 'Battery': 0.5, 'Camera': 0.5,
            'Chipset_Score': 0.5, 'Screen_Size': 0.5, 'Refresh_Rate': 0.3,
            'Network': 0.5, 'Price': 0.5
        },
        'budget': {
            'RAM': 0.5, 'ROM': 0.5, 'Battery': 0.5, 'Camera': 0.5,
            'Chipset_Score': 0.5, 'Screen_Size': 0.5, 'Refresh_Rate': 0.3,
            'Network': 0.5, 'Price': 1.0
        }
    }
    
    @classmethod
    def get_relation(cls, intent: str, criterion: str) -> float:
        if intent in cls.MATRIX and criterion in cls.MATRIX[intent]:
            return cls.MATRIX[intent][criterion]
        return 0.5
    
    @classmethod
    def get_combined_relations(cls, intents: Dict[str, float]) -> Dict[str, float]:
        if not intents:
            return {c: 0.5 for c in cls.MATRIX['harian'].keys()}
        
        criteria = list(cls.MATRIX['harian'].keys())
        combined = {}
        
        for criterion in criteria:
            total_relation = 0.0
            for intent, weight in intents.items():
                relation = cls.get_relation(intent, criterion)
                total_relation += relation * weight
            combined[criterion] = total_relation
        
        return combined
    
    @classmethod
    def display_matrix(cls, intents: Dict[str, float], combined: Dict[str, float]):
        if tabulate:
            print(f"\ MATRIKS RELASI:")
            criteria = list(cls.MATRIX['harian'].keys())
            headers = ['Kriteria'] + [f"{i} ({w:.0%})" for i, w in intents.items()] + ['Gabungan']
            data = []
            for c in criteria:
                row = [c]
                for intent in intents.keys():
                    row.append(f"{cls.get_relation(intent, c):.1f}")
                row.append(f"{combined[c]:.2f}")
                data.append(row)
            print(tabulate(data, headers=headers, tablefmt='grid'))
        print()


# =========================
# SLOT FILLER
# =========================
class SlotFiller:
    def __init__(self):
        self.base_weights = {
            'RAM': 10, 'ROM': 10, 'Battery': 10, 'Camera': 10,
            'Chipset_Score': 10, 'Screen_Size': 10, 'Refresh_Rate': 10,
            'Network': 10, 'Price': 20
        }
        
        self.default_slots = {
            'RAM': {'target': 6, 'weight': 0.10, 'type': 'benefit', 'unit': 'GB'},
            'ROM': {'target': 128, 'weight': 0.10, 'type': 'benefit', 'unit': 'GB'},
            'Battery': {'target': 5000, 'weight': 0.10, 'type': 'benefit', 'unit': 'mAh'},
            'Camera': {'target': 48, 'weight': 0.10, 'type': 'benefit', 'unit': 'MP'},
            'Chipset_Score': {'target': 55, 'weight': 0.10, 'type': 'benefit', 'unit': 'score'},
            'Screen_Size': {'target': 6.5, 'weight': 0.10, 'type': 'benefit', 'unit': 'inch'},
            'Refresh_Rate': {'target': 90, 'weight': 0.10, 'type': 'benefit', 'unit': 'Hz'},
            'Network': {'target': 4, 'weight': 0.10, 'type': 'benefit', 'unit': 'G'},
            'Price': {'target': 3500000, 'weight': 0.20, 'type': 'cost', 'unit': 'Rp'}
        }
    
    def get_relations(self, intents):
        return RelationMatrix.get_combined_relations(intents)
    
    def fill(self, intents, entities, serpapi_results=None):
        slots = {}
        
        relations = RelationMatrix.get_combined_relations(intents)
        RelationMatrix.display_matrix(intents, relations)
        
        final_weights = {}
        for criterion, base_w in self.base_weights.items():
            relation = relations.get(criterion, 0.5)
            final_weights[criterion] = base_w * relation
        
        total = sum(final_weights.values())
        normalized = {k: v/total for k, v in final_weights.items()} if total > 0 else {k: 1.0/len(final_weights) for k in final_weights}
        
        print(f"\ PERHITUNGAN BOBOT (Relationship-Based):")
        if tabulate:
            data = []
            for c in self.base_weights.keys():
                data.append([
                    c, 
                    f"{self.base_weights[c]}%",
                    f"{relations.get(c, 0.5):.2f}",
                    f"{final_weights[c]:.2f}",
                    f"{normalized[c]*100:.1f}%"
                ])
            print(tabulate(data, headers=['Kriteria', 'Base', 'Relasi', 'Bobot×Relasi', 'Normalisasi'], tablefmt='grid'))
        print()
        
        for criterion in self.base_weights.keys():
            target = self.default_slots[criterion]['target']
            unit = self.default_slots[criterion]['unit']
            slot_type = self.default_slots[criterion]['type']
            
            slots[criterion] = {
                'target': target,
                'weight': normalized[criterion],
                'type': slot_type,
                'unit': unit
            }
        
        has_budget = entities.get('has_budget_mention', False)
        
        if 'gaming' in intents:
            slots['RAM']['target'] = max(8, slots['RAM']['target'])
            slots['Chipset_Score']['target'] = max(80, slots['Chipset_Score']['target'])
            slots['Refresh_Rate']['target'] = max(120, slots['Refresh_Rate']['target'])
        
        if 'fotografi' in intents:
            slots['Camera']['target'] = max(64, slots['Camera']['target'])
        
        if 'baterai' in intents:
            slots['Battery']['target'] = max(6000, slots['Battery']['target'])
        
        if 'performa' in intents:
            slots['Chipset_Score']['target'] = max(85, slots['Chipset_Score']['target'])
            slots['RAM']['target'] = max(12, slots['RAM']['target'])
        
        if 'multimedia' in intents:
            slots['Screen_Size']['target'] = max(6.5, slots['Screen_Size']['target'])
            slots['Refresh_Rate']['target'] = max(90, slots['Refresh_Rate']['target'])
        
        if entities.get('has_explicit_price') and entities.get('price'):
            slots['Price']['target'] = entities['price']
        elif has_budget and not entities.get('has_explicit_price'):
            slots['Price']['target'] = 2500000
        
        if entities.get('ram'): slots['RAM']['target'] = entities['ram']
        if entities.get('rom'): slots['ROM']['target'] = entities['rom']
        if entities.get('battery'): slots['Battery']['target'] = entities['battery']
        if entities.get('camera'): slots['Camera']['target'] = entities['camera']
        if entities.get('screen_size'): slots['Screen_Size']['target'] = entities['screen_size']
        if entities.get('refresh_rate'): slots['Refresh_Rate']['target'] = entities['refresh_rate']
        if entities.get('network'): slots['Network']['target'] = 5 if entities['network'] == '5g' else 4
        if entities.get('price'): slots['Price']['target'] = entities['price']
        
        if serpapi_results:
            specs = serpapi_results.get('specs', {})
            if specs.get('rec_ram'): 
                slots['RAM']['target'] = specs['rec_ram']
                print(f"   SerpAPI REC RAM: {specs['rec_ram']} GB")
            elif specs.get('min_ram'): 
                slots['RAM']['target'] = specs['min_ram']
                print(f"   SerpAPI MIN RAM: {specs['min_ram']} GB")
            
            if specs.get('min_storage'): 
                slots['ROM']['target'] = specs['min_storage']
                print(f"   SerpAPI Storage: {specs['min_storage']} GB")
            
            if specs.get('min_battery'): 
                slots['Battery']['target'] = specs['min_battery']
                print(f"   SerpAPI Battery: {specs['min_battery']} mAh")
            
            if specs.get('min_camera'): 
                slots['Camera']['target'] = specs['min_camera']
                print(f"   SerpAPI Camera: {specs['min_camera']} MP")
        
        return slots
    
    def display_slots(self, slots):
        print(f"\n BOBOT KRITERIA (Final):")
        if tabulate:
            data = [[k, v['target'], v['unit'], f"{v['weight']*100:.1f}%", v['type'].upper()] 
                    for k, v in slots.items()]
            print(tabulate(data, headers=['Kriteria', 'Target', 'Unit', 'Bobot', 'Tipe'], tablefmt='grid'))
        else:
            for k, v in slots.items():
                print(f"   {k}: target={v['target']}, bobot={v['weight']*100:.1f}%, tipe={v['type']}")
        print()


# =========================
# DATASET FILTER
# =========================
class DatasetFilter:
    def __init__(self, df): 
        self.df = df
    
    def filter(self, entities):
        df = self.df.copy()
        brand = entities.get('brand')
        model = entities.get('model')
        
        if brand:
            brand_col = 'nama_brand' if 'nama_brand' in df.columns else 'brand_name'
            before = len(df)
            df = df[df[brand_col].str.lower() == brand.lower()]
            print(f"   Filter brand '{brand}': {before} → {len(df)} models")
        
        if model and brand and len(df) > 0:
            if 'model' in df.columns:
                model_clean = model.lower().replace(brand.lower(), '').strip()
                if model_clean:
                    before = len(df)
                    df = df[df['model'].str.lower().str.contains(model_clean, na=False)]
                    print(f"  📱 Filter model '{model_clean}': {before} → {len(df)} models")
        
        return df


# =========================
# CHIPSET SCORER
# =========================
class ChipsetScorer:
    def __init__(self):
        self.scores = {
            'snapdragon 8 gen 4': 95, 'snapdragon 8 gen 3': 90, 'snapdragon 8s gen 3': 88,
            'snapdragon 7s gen 3': 75, 'snapdragon 7 gen 3': 78, 'snapdragon 7 gen 2': 75,
            'snapdragon 6 gen 4': 70, 'snapdragon 6 gen 1': 65, 'snapdragon 685': 50,
            'dimensity 9400e': 92, 'dimensity 9300+': 90, 'dimensity 9300': 88,
            'dimensity 8400 max': 85, 'dimensity 8300': 82, 'dimensity 8200': 80,
            'dimensity 8050': 78, 'dimensity 7300': 72, 'dimensity 7050': 70,
            'dimensity 7025': 68, 'dimensity 7020': 66, 'dimensity 6300': 60,
            'dimensity 6100+': 58, 'dimensity 6080': 55,
            'kirin 9100': 88, 'kirin 9010': 85, 'kirin 9000s': 82,
            'helio g100': 55, 'helio g99': 52, 'helio g88': 48, 'helio g85': 45,
            'unisoc t616': 42, 'unisoc t606': 38, 'unisoc t612': 35,
            'unisoc t603': 30, 'unisoc sc9863a': 25, 'unisoc sc9863a1': 25, 't7300': 60
        }
    
    def score_dataframe(self, df):
        if df.empty:
            df['chipset_score'] = 30
            return df
        
        chipset_col = 'chipset' if 'chipset' in df.columns else 'processor_brand'
        
        def calc(chipset_name):
            if pd.isna(chipset_name) or not chipset_name:
                return 30
            chipset_name = str(chipset_name).lower().strip()
            if chipset_name in self.scores:
                return self.scores[chipset_name]
            for key, value in self.scores.items():
                if key in chipset_name:
                    return value
            if 'snapdragon' in chipset_name: return 70
            elif 'dimensity' in chipset_name: return 65
            elif 'kirin' in chipset_name: return 72
            elif 'helio' in chipset_name: return 50
            elif 'unisoc' in chipset_name: return 35
            else: return 40
        
        df['chipset_score'] = df[chipset_col].apply(calc)
        return df


# =========================
# NLU ENGINE
# =========================
class NLUEngine:
    def __init__(self, serpapi_key: str = ""):
        print("\n INITIALIZING NLU ENGINE WITH ENHANCED SERPAPI...")
        self.data_loader = DataLoader()
        self.preprocessor = TextPreprocessor(self.data_loader.slang_dict, self.data_loader.typo_dict, self.data_loader.stop_words)
        self.entity_extractor = EntityExtractor(self.data_loader.brands, self.data_loader.models)
        self.serpapi = SerpAPISearcher(serpapi_key) if serpapi_key else None
        self.slot_filler = SlotFiller()
        self.dataset_filter = DatasetFilter(self.data_loader.smartphones_df)
        self.chipset_scorer = ChipsetScorer()
        print("✅ Ready with AI Overview support!\n")
    
    def process(self, query: str) -> Dict:
        print(f"\n{'#'*50}")
        print(f"# PROCESSING: \"{query}\"")
        print(f"{'#'*50}\n")
        
        print("STEP 1: PREPROCESSING")
        clean_text, changes = self.preprocessor.preprocess(query)
        if changes:
            for c in changes: print(f"  • {c}")
        print(f"  Hasil: \"{clean_text}\"\n")
        
        print("STEP 2: ENTITY & INTENT")
        entities = self.entity_extractor.extract(clean_text)
        intents = self.entity_extractor.detect_intents(clean_text)
        
        static_intents = self.entity_extractor.get_static_intents(entities.get('static_matches', []))
        for intent, score in static_intents.items():
            if intent in intents:
                intents[intent] += score
            else:
                intents[intent] = score
        
        if intents:
            total = sum(intents.values())
            intents = {k: min(v/total, 1.0) for k, v in sorted(intents.items(), key=lambda x: x[1], reverse=True)}
        
        if intents:
            for intent, score in intents.items():
                print(f"  • {intent}: {score*100:.0f}%")
        if entities.get('brand'): 
            print(f"   Brand: {entities['brand']}")
        if entities.get('has_budget_mention'): 
            print(f"   Budget Mention: YES")
        print()
        
        print("STEP 3: ENHANCED SERPAPI SEARCH (with AI Overview)")
        serpapi_results = {}
        if entities['unknown_terms'] and self.serpapi:
            print(f"   Unknown terms detected: {entities['unknown_terms']}")
            
            for term in entities['unknown_terms'][:2]:
                print(f"\n  {'='*40}")
                print(f"  Searching for: '{term}'")
                r = self.serpapi.search_term(term, clean_text)
                if r:
                    serpapi_results[term] = r
                    self.serpapi.display_results(r)
                else:
                    print(f"  ⚠️ No results for '{term}'")
        else:
            if not entities['unknown_terms']:
                print("  ✅ No unknown terms to search")
            elif not self.serpapi:
                print("  ⚠️ SerpAPI not available (no API key)")
        print()
        
        print("STEP 4: SLOTS (Relationship-Based Weighting)")
        merged = list(serpapi_results.values())[0] if serpapi_results else None
        slots = self.slot_filler.fill(intents, entities, merged)
        self.slot_filler.display_slots(slots)
        
        print("STEP 5: FILTER DATASET")
        filtered_df = self.dataset_filter.filter(entities)
        print(f"  Filtered: {len(filtered_df)} models\n")
        
        print("STEP 6: CHIPSET SCORING")
        scored_df = self.chipset_scorer.score_dataframe(filtered_df)
        print(f"  Scored: {len(scored_df)} models\n")
        
        return {
            'clean_text': clean_text,
            'entities': entities,
            'intents': intents,
            'serpapi_results': serpapi_results,
            'slots': slots,
            'relations': self.slot_filler.get_relations(intents),
            'scored_df': scored_df
        }