# =========================
# app.py - FLASK BACKEND (FULLY FIXED WITH BRAND FILTER)
# =========================
import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify
from nlu import NLUEngine
from copras import COPRAS

app = Flask(__name__)

nlu_engine = None

def init_engine():
    global nlu_engine
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "c78fcf95c8fdd41787fa0a368e57e65895e03c418318ae4f8260054fcf43970d")
    print("\n" + "="*50)
    print(" SMARTPHONE RECOMMENDATION SYSTEM")
    print("   NLU + COPRAS Method - Web Version")
    print("="*50)
    
    if not SERPAPI_KEY:
        print("⚠️  SERPAPI_KEY tidak diset (fitur pencarian game terbatas)")
    else:
        print(f"✅ SerpAPI Key: {SERPAPI_KEY[:10]}...")
    
    nlu_engine = NLUEngine(SERPAPI_KEY)
    print("✅ Engine siap!\n")


def format_serpapi_results(serpapi_results):
    if not serpapi_results:
        return []
    
    formatted = []
    for term, result in serpapi_results.items():
        specs = result.get('specs', {})
        
        item = {
            'term': term,
            'query': result.get('query', ''),
            'definition': result.get('definition', None),
            'min_ram': specs.get('min_ram'),
            'rec_ram': specs.get('rec_ram'),
            'min_storage': specs.get('min_storage'),
            'min_battery': specs.get('min_battery'),
            'min_camera': specs.get('min_camera'),
            'recommended_chipset': specs.get('recommended_chipset'),
            'has_specs': result.get('has_specs', False)
        }
        formatted.append(item)
    
    return formatted


def extract_links(row):
    links = {'website': '', 'shopee': '', 'tokopedia': ''}
    link_mappings = {
        'website': ['link_website', 'website', 'official_link'],
        'shopee': ['link_shopee', 'shopee', 'shopee_link'],
        'tokopedia': ['link_tokopedia', 'tokopedia', 'tokopedia_link']
    }
    for key, possible_cols in link_mappings.items():
        for col in possible_cols:
            if col in row.index:
                val = row[col]
                if pd.notna(val) and str(val).strip() and str(val).strip() != '' and str(val).strip() != 'nan':
                    links[key] = str(val).strip()
                break
    return links


def extract_specs(row, slots):
    specs = {}
    spec_mappings = {
        'RAM': ['kapasitas_ram', 'ram'],
        'ROM': ['memori_internal', 'rom', 'storage'],
        'Battery': ['kapasitas_baterai', 'battery'],
        'Camera': ['resolusi_kamera', 'camera'],
        'Chipset_Score': ['chipset_score'],
        'Screen_Size': ['ukuran_layar', 'screen_size'],
        'Refresh_Rate': ['refresh_rate'],
        'Network': ['network_type', '5g', 'network'],
        'Price': ['harga', 'price']
    }
    for spec_name, possible_cols in spec_mappings.items():
        value = None
        for col in possible_cols:
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    try:
                        if spec_name == 'Network':
                            if col == '5g':
                                value = 5 if int(float(val)) == 1 else 4
                            else:
                                v = str(val).lower()
                                value = 5 if '5g' in v else 4
                        elif spec_name == 'Price':
                            value = int(float(str(val).replace('.', '').replace(',', '')))
                        elif spec_name == 'Chipset_Score':
                            value = int(float(val))
                        else:
                            value = int(float(str(val).replace('.', '').replace(',', '')))
                    except:
                        value = None
                break
        if value is not None:
            specs[spec_name] = value
        elif spec_name in slots:
            specs[spec_name] = slots[spec_name]['target']
    return specs


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Maaf, silakan input terlebih dahulu'}), 400
        
        print(f"\n Processing query: \"{query}\"")
        
        nlu_result = nlu_engine.process(query)
        serpapi_formatted = format_serpapi_results(nlu_result.get('serpapi_results', {}))
        
        # ============ KRITIKAL: Hard Filter Brand ============
        df_filtered = nlu_result['scored_df'].copy()
        brand_mentioned = nlu_result['entities'].get('brand')
        
        print(f"\n{'='*50}")
        print(f" DEBUG INFO:")
        print(f"  Query: {query}")
        print(f"  Brand detected: {brand_mentioned}")
        print(f"  Total HP before brand filter: {len(df_filtered)}")
        print(f"{'='*50}\n")
        
        # HARD FILTER berdasarkan brand
        if brand_mentioned:
            brand_col = 'nama_brand' if 'nama_brand' in df_filtered.columns else 'brand_name'
            before_filter = len(df_filtered)
            
            # Filter ketat (case-insensitive)
            df_filtered = df_filtered[df_filtered[brand_col].str.lower() == brand_mentioned.lower()]
            
            print(f"   HARDFILTER brand '{brand_mentioned}': {before_filter} → {len(df_filtered)} HP")
            
            # Jika tidak ada HP dari brand tersebut
            if len(df_filtered) == 0:
                available_brands = sorted(list(nlu_engine.data_loader.brands))
                return jsonify({
                    'error': f'Maaf, tidak ditemukan smartphone dari brand "{brand_mentioned}" di database kami.\n\nBrand yang tersedia: {", ".join(available_brands[:15])}...',
                    'success': False
                }), 404
        
        # Filter budget jika ada
        if nlu_result['entities'].get('has_budget_mention') and len(df_filtered) > 0:
            price_target = nlu_result['slots']['Price']['target']
            price_operator = nlu_result['entities'].get('price_operator', 'below')
            
            if 'harga' in df_filtered.columns:
                if price_operator == 'below':
                    max_price = price_target * 1.3
                    before = len(df_filtered)
                    df_filtered = df_filtered[df_filtered['harga'] <= max_price]
                    print(f"   Budget filter (<={max_price:,.0f}): {before} → {len(df_filtered)} HP")
                elif price_operator == 'above':
                    min_price = price_target * 0.7
                    before = len(df_filtered)
                    df_filtered = df_filtered[df_filtered['harga'] >= min_price]
                    print(f"   Budget filter (>={min_price:,.0f}): {before} → {len(df_filtered)} HP")
        
        # Hapus HP dengan harga 0
        if 'harga' in df_filtered.columns and len(df_filtered) > 0:
            before = len(df_filtered)
            df_filtered = df_filtered[df_filtered['harga'] > 0]
            if before - len(df_filtered) > 0:
                print(f"   Hapus {before - len(df_filtered)} HP dengan harga 0")
        
        # Jika setelah filter kosong
        if len(df_filtered) == 0:
            return jsonify({
                'error': f'Tidak ada smartphone yang sesuai dengan kriteria Anda.\nBrand: {brand_mentioned if brand_mentioned else "semua"}\nBudget: {nlu_result["slots"]["Price"]["target"]:,.0f}',
                'success': False
            }), 404
        
        # Update untuk COPRAS
        nlu_result['scored_df'] = df_filtered
        
        # ============ COPRAS Calculation ============
        print(f"\n Running COPRAS on {len(df_filtered)} HP...")
        copras = COPRAS(nlu_result['scored_df'], nlu_result['slots'])
        copras.calculate()
        top_recommendations = copras.get_top_n(10)
        
        results = []
        for idx, row in top_recommendations.iterrows():
            brand_col = 'nama_brand' if 'nama_brand' in row.index else 'brand_name'
            brand = str(row.get(brand_col, 'Unknown'))
            
            ram_val = row.get('kapasitas_ram', row.get('ram', None))
            rom_val = row.get('memori_internal', row.get('rom', row.get('storage', None)))
            battery_val = row.get('kapasitas_baterai', row.get('battery', None))
            camera_val = row.get('resolusi_kamera', row.get('camera', None))
            price_val = row.get('harga', row.get('price', 0))
            
            try: ram_val = int(float(str(ram_val).replace('.', ''))) if pd.notna(ram_val) else None
            except: ram_val = None
            try: rom_val = int(float(str(rom_val).replace('.', ''))) if pd.notna(rom_val) else None
            except: rom_val = None
            try: battery_val = int(float(str(battery_val).replace('.', ''))) if pd.notna(battery_val) else None
            except: battery_val = None
            try: camera_val = int(float(str(camera_val).replace('.', ''))) if pd.notna(camera_val) else None
            except: camera_val = None
            try: price_val = int(float(str(price_val).replace('.', '').replace(',', ''))) if pd.notna(price_val) else 0
            except: price_val = 0
            
            links = extract_links(row)
            specs = extract_specs(row, nlu_result['slots'])
            
            result = {
                'rank': int(row.get('Rank', 0)),
                'brand': brand.capitalize(),
                'nama_brand': brand,
                'model': str(row.get('model', 'Unknown')),
                'full_name': f"{brand.capitalize()} {str(row.get('model', 'Unknown'))}",
                'ram': ram_val, 'rom': rom_val, 'battery': battery_val,
                'camera': camera_val, 'processor': str(row.get('chipset', 'N/A')),
                'price': price_val,
                'chipset_score': int(float(row.get('chipset_score', 0))) if pd.notna(row.get('chipset_score')) else 0,
                'qi_score': float(row.get('Qi_Score', 0)),
                'utility': float(row.get('Utility_%', 0)),
                'screen_size': float(row.get('ukuran_layar', 0)) if pd.notna(row.get('ukuran_layar')) else None,
                'refresh_rate': int(float(row.get('refresh_rate', 0))) if pd.notna(row.get('refresh_rate')) else None,
                'network': str(row.get('network_type', '4G')),
                'links': links, 'specs': specs
            }
            results.append(result)
        
        slots_info = {}
        for k, v in nlu_result['slots'].items():
            slots_info[k] = {
                'target': v['target'], 'weight': round(v['weight'], 4),
                'type': v['type'], 'unit': v['unit']
            }
        
        base_weights = {
            'RAM': 10, 'ROM': 10, 'Battery': 10, 'Camera': 10,
            'Chipset_Score': 10, 'Screen_Size': 10, 'Refresh_Rate': 10,
            'Network': 10, 'Price': 20
        }
        
        response = {
            'success': True, 'query': query,
            'clean_text': nlu_result.get('clean_text', query),
            'intents': nlu_result.get('intents', {}),
            'entities': {k: v for k, v in nlu_result.get('entities', {}).items() 
                        if v is not None and k not in ['unknown_terms', 'static_matches']},
            'unknown_terms': nlu_result.get('entities', {}).get('unknown_terms', []),
            'static_matches': nlu_result.get('entities', {}).get('static_matches', []),
            'serpapi_results': serpapi_formatted,
            'slots': slots_info,
            'relations': nlu_result.get('relations', {}),
            'base_weights': base_weights,
            'recommendations': results,
            'total_alternatives': len(nlu_result['scored_df'])
        }
        
        print(f"\n✅ FINAL RESULT: {len(results)} rekomendasi dari brand {brand_mentioned if brand_mentioned else 'semua'}")
        print(f"   Top 1: {results[0]['full_name']} - {results[0]['utility']:.1f}%")
        print(f"{'='*50}\n")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Sistem Rekomendasi Smartphone siap'})


if __name__ == '__main__':
    init_engine()
    print("\n" + "="*50)
    print("🌐 Web server berjalan di: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)