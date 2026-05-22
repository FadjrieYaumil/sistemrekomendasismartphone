# =========================
# serpapi_searcher.py - ENHANCED SERPAPI WITH AI OVERVIEW
# =========================
import re
import json
from serpapi import GoogleSearch
from typing import Dict, Optional, List, Tuple


class SerpAPISearcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.available = bool(api_key)
        
        self.context_triggers = [
            'main', 'game', 'gaming', 'ngegame', 'spek', 'setara', 'seperti',
            'persis', 'mirip', 'kayak', 'macam', 'khusus', 'buat', 'untuk',
            'spesifikasi', 'tipe', 'jenis', 'seri', 'kelas', 'selevel',
            'sebanding', 'saingan', 'kompetitor', 'rival', 'kuat', 'kencang',
            'ngebut', 'lancar', 'maksimal', 'banding', 'tanding', 'lawan'
        ]
        
        self.known_games = {
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
            'arena of valor': 'Arena of Valor',
            'pubg': 'PUBG Mobile',
            'pubg mobile': 'PUBG Mobile',
            'pubgm': 'PUBG Mobile',
            'ff': 'Free Fire',
            'free fire': 'Free Fire',
            'cod': 'Call of Duty Mobile',
            'codm': 'Call of Duty Mobile',
            'call of duty': 'Call of Duty Mobile',
            'valorant': 'Valorant Mobile',
            'lol': 'League of Legends Wild Rift',
            'wild rift': 'League of Legends Wild Rift',
            'fortnite': 'Fortnite Mobile',
            'roblox': 'Roblox',
            'minecraft': 'Minecraft',
            'among us': 'Among Us',
            'pokemon': 'Pokemon Unite',
            'fifa': 'FIFA Mobile',
            'efootball': 'eFootball',
            'pes': 'eFootball',
            'asphalt': 'Asphalt Legends',
            'coc': 'Clash of Clans',
            'clash of clans': 'Clash of Clans',
            'clash royale': 'Clash Royale',
            'brawl stars': 'Brawl Stars',
            'nikke': 'Goddess of Victory Nikke',
            'tower of fantasy': 'Tower of Fantasy',
            'zenless': 'Zenless Zone Zero',
            'zzz': 'Zenless Zone Zero',
            'apex': 'Apex Legends Mobile',
            'dota': 'Dota Underlords',
            'undawn': 'Undawn',
            'farlight': 'Farlight 84',
            'gta': 'GTA San Andreas Mobile',
            'nba': 'NBA 2K Mobile'
        }
    
    def search_term(self, term: str, context: str = "") -> Optional[Dict]:
        if not self.available:
            return None
        
        if any(game_word in term.lower() for game_word in ['game', 'gaming', 'wuwa', 'genshin', 'pubg', 'ml', 'ff', 'snowbreak']):
            query = f"{term} spesifikasi minimum RAM rekomendasi HP Android system requirements"
        else:
            query = f"{term} adalah pengertian spesifikasi minimum RAM rekomendasi"
        
        print(f"\n🔍 SerpAPI: \"{query}\"")
        
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.api_key,
                "num": 10,
                "gl": "id",
                "hl": "id",
                "location": "Indonesia",
                "google_domain": "google.co.id",
                "safe": "off"
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            print(f"  📋 Available keys: {list(results.keys())}")
            
            ai_overview = self._extract_ai_overview(results)
            organic_results = results.get("organic_results", [])
            specs = self._extract_specs_from_results(organic_results, term)
            knowledge_graph = results.get("knowledge_graph", {})
            if knowledge_graph:
                print(f"  📚 Knowledge Graph found: {knowledge_graph.get('title', '')}")
            
            related_questions = results.get("related_questions", [])
            definition = ai_overview if ai_overview else self._build_definition_from_results(organic_results, term)
            
            return {
                'term': term,
                'query': query,
                'specs': specs,
                'definition': definition,
                'ai_overview': ai_overview,
                'has_specs': any([specs.get('min_ram'), specs.get('rec_ram'),
                                  specs.get('min_storage'), specs.get('min_battery'),
                                  specs.get('min_camera'), specs.get('recommended_chipset')]),
                'knowledge_graph': knowledge_graph,
                'related_questions': related_questions[:3] if related_questions else [],
                'organic_results_count': len(organic_results)
            }
            
        except Exception as e:
            print(f"  ⚠️ SerpAPI Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_ai_overview(self, results: Dict) -> Optional[str]:
        """Extract AI Overview - ENHANCED VERSION"""
        
        if "ai_overview" in results:
            overview = results["ai_overview"]
            if isinstance(overview, dict):
                for key in ['text', 'snippet', 'content', 'description']:
                    if key in overview:
                        print(f"  ✅ AI Overview found (direct): {len(str(overview[key]))} chars")
                        return overview[key]
            elif isinstance(overview, str):
                print(f"  ✅ AI Overview found (string): {len(overview)} chars")
                return overview
        
        if "answer_box" in results:
            answer = results["answer_box"]
            if isinstance(answer, dict):
                if answer.get("source") == "AI" or answer.get("type") == "ai_answer":
                    snippet = answer.get("snippet") or answer.get("answer") or answer.get("text")
                    if snippet:
                        print(f"  ✅ AI answer box found")
                        return snippet
                snippet = answer.get("snippet") or answer.get("answer")
                if snippet:
                    print(f"  ✅ Answer box found")
                    return snippet
        
        if "featured_snippet" in results:
            snippet = results["featured_snippet"]
            if isinstance(snippet, dict) and "description" in snippet:
                print(f"  ✅ Featured snippet found")
                return snippet["description"]
        
        if "knowledge_panel" in results:
            panel = results["knowledge_panel"]
            if isinstance(panel, dict):
                desc = panel.get("description") or panel.get("snippet")
                if desc:
                    print(f"  ✅ Knowledge panel found")
                    return desc
        
        organic = results.get("organic_results", [])
        for result in organic[:3]:
            snippet = result.get("snippet", "")
            if len(snippet) > 150 and any(keyword in snippet.lower() for keyword in 
                                          ['spesifikasi', 'minimum', 'rekomendasi', 'ram', 
                                           'storage', 'chipset', 'prosesor', 'persyaratan']):
                print(f"  ✅ Spec overview found in organic results")
                return snippet
        
        print(f"  ⚠️ No AI Overview found")
        return None
    
    def _extract_specs_from_results(self, organic_results: List[Dict], term: str) -> Dict:
        specs = {}
        
        if not organic_results:
            return specs
        
        combined_text = " ".join([
            r.get("snippet", "") + " " + r.get("title", "")
            for r in organic_results[:10]
        ])
        
        for result in organic_results:
            if "rich_snippet" in result:
                rich = result["rich_snippet"]
                if "top" in rich:
                    for item in rich["top"].get("detected_extensions", []):
                        combined_text += " " + str(item)
        
        print(f"  📝 Combined text length: {len(combined_text)} chars")
        
        ram_patterns = [
            r'(\d+)\s*GB\s*RAM',
            r'RAM\s*(\d+)\s*GB',
            r'ram\s*(\d+)\s*gb',
            r'(\d+)\s*gb\s*ram',
        ]
        for pattern in ram_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            if matches:
                rams = [int(m) for m in matches if 1 <= int(m) <= 32]
                if rams:
                    specs['min_ram'] = min(rams)
                    specs['rec_ram'] = max(rams)
                    print(f"  📊 RAM found: min={specs['min_ram']}, rec={specs['rec_ram']}")
                break
        
        rec_ram_patterns = [
            r'(?:recommended|rekomendasi|disarankan|minimal)\s*(\d+)\s*GB\s*RAM',
            r'RAM\s*(?:recommended|rekomendasi|disarankan|minimal)\s*(\d+)\s*GB',
            r'(?:butuh|perlu|memerlukan|minimal)\s*RAM\s*(\d+)\s*GB',
        ]
        for pattern in rec_ram_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                val = int(match.group(1))
                if 1 <= val <= 32:
                    specs['rec_ram'] = val
                    print(f"  📊 Rec RAM found: {specs['rec_ram']}")
                break
        
        storage_patterns = [
            r'(\d+)\s*GB\s*(?:storage|penyimpanan|internal|ROM|ruang)',
            r'(?:storage|penyimpanan|internal|ROM)\s*(\d+)\s*GB',
            r'(?:butuh|perlu)\s*(?:storage|penyimpanan)\s*(\d+)\s*GB',
        ]
        for pattern in storage_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            if matches:
                storages = [int(m) for m in matches if 16 <= int(m) <= 1024]
                if storages:
                    specs['min_storage'] = min(storages)
                    print(f"  📊 Storage found: {specs['min_storage']}")
                break
        
        battery_patterns = [
            r'(\d+)\s*mAh',
            r'(\d+)\s*mah',
            r'baterai\s*(\d+)\s*mAh',
        ]
        for pattern in battery_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            if matches:
                batteries = [int(m) for m in matches if 1000 <= int(m) <= 10000]
                if batteries:
                    specs['min_battery'] = min(batteries)
                    print(f"  📊 Battery found: {specs['min_battery']}")
                break
        
        camera_patterns = [
            r'(\d+)\s*MP',
            r'(\d+)\s*mp',
            r'kamera\s*(\d+)\s*MP',
        ]
        for pattern in camera_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            if matches:
                cameras = [int(m) for m in matches if 8 <= int(m) <= 200]
                if cameras:
                    specs['min_camera'] = min(cameras)
                    print(f"  📊 Camera found: {specs['min_camera']}")
                break
        
        chipset_patterns = [
            r'(Snapdragon\s*[\w\s\d\+]*?)(?:\s|,|\.|$)',
            r'(Dimensity\s*[\w\s\d\+]*?)(?:\s|,|\.|$)',
            r'(Kirin\s*[\w\s\d\+]*?)(?:\s|,|\.|$)',
            r'(Helio\s*[\w\s\d\+]*?)(?:\s|,|\.|$)',
            r'(Exynos\s*[\w\s\d\+]*?)(?:\s|,|\.|$)',
            r'(Unisoc\s*[\w\s\d\+]*?)(?:\s|,|\.|$)',
        ]
        for pattern in chipset_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                chipset = match.group(0).strip()
                if len(chipset) > 3:
                    specs['recommended_chipset'] = chipset
                    print(f"  📊 Chipset found: {chipset}")
                break
        
        if 'android' in combined_text.lower():
            os_match = re.search(r'Android\s*(\d+\.?\d*)', combined_text, re.IGNORECASE)
            if os_match:
                specs['min_os'] = f"Android {os_match.group(1)}"
        
        if 'ios' in combined_text.lower():
            ios_match = re.search(r'iOS\s*(\d+\.?\d*)', combined_text, re.IGNORECASE)
            if ios_match:
                specs['min_os'] = f"iOS {ios_match.group(1)}"
        
        return specs
    
    def _build_definition_from_results(self, organic_results: List[Dict], term: str) -> str:
        if not organic_results:
            return f"Tidak ada informasi yang ditemukan untuk '{term}'"
        
        relevant_snippets = []
        for result in organic_results[:5]:
            snippet = result.get("snippet", "")
            title = result.get("title", "")
            
            if any(keyword in (snippet + title).lower() for keyword in 
                  ['spesifikasi', 'minimum', 'rekomendasi', 'requirement', 'system']):
                relevant_snippets.append(snippet)
        
        if relevant_snippets:
            return " ".join(relevant_snippets[:3])
        
        return organic_results[0].get("snippet", f"Informasi tentang {term}")
    
    def display_results(self, result: Dict):
        if not result:
            return
        
        print(f"\n📊 HASIL SERPAPI: {result['term'].upper()}")
        print(f"  Query: {result.get('query', '')}")
        
        if result.get('ai_overview'):
            print(f"\n  🤖 AI OVERVIEW:")
            overview = result['ai_overview']
            if len(overview) > 300:
                print(f"  {overview[:300]}...")
            else:
                print(f"  {overview}")
        
        if result.get('definition') and result['definition'] != result.get('ai_overview'):
            print(f"\n  📖 DEFINISI:")
            definition = result['definition']
            if len(definition) > 200:
                print(f"  {definition[:200]}...")
            else:
                print(f"  {definition}")
        
        specs = result.get('specs', {})
        if specs:
            print(f"\n  ⚙️ SPESIFIKASI:")
            if specs.get('min_ram'):
                print(f"    RAM Minimal: {specs['min_ram']} GB")
            if specs.get('rec_ram'):
                print(f"    RAM Rekomendasi: {specs['rec_ram']} GB")
            if specs.get('min_storage'):
                print(f"    Storage Minimal: {specs['min_storage']} GB")
            if specs.get('min_battery'):
                print(f"    Baterai Minimal: {specs['min_battery']} mAh")
            if specs.get('min_camera'):
                print(f"    Kamera Minimal: {specs['min_camera']} MP")
            if specs.get('recommended_chipset'):
                print(f"    Chipset Rekomendasi: {specs['recommended_chipset']}")
            if specs.get('min_os'):
                print(f"    OS Minimal: {specs['min_os']}")
        
        print()