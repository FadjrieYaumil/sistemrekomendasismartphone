# =========================
# main.py - MAIN APPLICATION
# =========================
import os
import sys
import pandas as pd
from nlu import NLUEngine
from copras import COPRAS


class SmartphoneRecommendationSystem:
    def __init__(self, serpapi_key=""):
        print("\n" + "="*50)
        print(" SMARTPHONE RECOMMENDATION SYSTEM")
        print("   NLU + COPRAS Method")
        print("="*50)
        self.nlu = NLUEngine(serpapi_key)
        self.top_recommendations = None
    
    def process_query(self, query: str):
        if query.lower() in ['exit', 'quit', 'keluar']:
            return None
        
        # Phase 1: NLU
        print("PHASE 1: NLU Processing")
        print("="*50)
        nlu_result = self.nlu.process(query)
        
        if nlu_result['scored_df'].empty:
            print("❌ Tidak ada data yang sesuai!")
            return None
        
        # Phase 2: COPRAS
        print("\nPHASE 2: COPRAS Calculation")
        print("="*50)
        copras = COPRAS(nlu_result['scored_df'], nlu_result['slots'])
        copras.calculate()
        self.top_recommendations = copras.get_top_n(5)
        copras.display_results(5)
        
        return {'top_5': self.top_recommendations}
    
    def run_interactive(self):
        print("\n Masukkan query (contoh: 'hp gaming murah 3 juta')")
        print("   Ketik 'exit' untuk keluar\n")
        
        while True:
            query = input(" Query: ").strip()
            if query.lower() in ['exit', 'quit', 'keluar']:
                print("\n Terima kasih!")
                break
            if not query:
                continue
            
            self.process_query(query)
            input("\n⏭ Enter untuk lanjut...")


def main():
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
    
    if not SERPAPI_KEY:
        print(" SERPAPI_KEY tidak diset")
        print("   Set: $env:SERPAPI_KEY='your_key'\n")
    
    system = SmartphoneRecommendationSystem(SERPAPI_KEY)
    
    if len(sys.argv) > 1:
        system.process_query(' '.join(sys.argv[1:]))
    else:
        try:
            system.run_interactive()
        except KeyboardInterrupt:
            print("\n Selesai.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()