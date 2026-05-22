# =========================
# copras.py - COPRAS METHOD (DETAILED TERMINAL OUTPUT + DUAL TABLE)
# =========================
import pandas as pd
import numpy as np
from typing import Dict, Any

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


class COPRAS:
    def __init__(self, dataframe, slots):
        self.df = dataframe.copy()
        self.slots = slots
        self.criteria = list(slots.keys())
        self.benefit_criteria = [c for c in self.criteria if slots[c]['type'] == 'benefit']
        self.cost_criteria = [c for c in self.criteria if slots[c]['type'] == 'cost']
        self.final_ranking = None
    
    def _convert_network(self, value):
        if pd.isna(value):
            return 4
        v = str(value).lower()
        if '5g' in v:
            return 5
        elif '4g' in v:
            return 4
        return 4
    
    def _prepare_matrix(self):
        mapping = {
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
        
        data = {}
        missing_penalty = {}
        
        for criterion in self.criteria:
            found = False
            for col in mapping.get(criterion, [criterion.lower()]):
                if col in self.df.columns:
                    raw_values = pd.to_numeric(self.df[col], errors='coerce')
                    missing_penalty[criterion] = raw_values.isna()
                    data[criterion] = raw_values.fillna(0)
                    found = True
                    break
            if not found:
                data[criterion] = np.zeros(len(self.df))
                missing_penalty[criterion] = np.ones(len(self.df), dtype=bool)
        
        dm = pd.DataFrame(data)
        
        for criterion in self.benefit_criteria:
            if criterion in missing_penalty:
                penalty_mask = missing_penalty[criterion]
                if penalty_mask.any():
                    target = self.slots[criterion]['target']
                    dm.loc[penalty_mask, criterion] = target * 0.3
                    print(f"⚠️  {criterion}: {penalty_mask.sum()} data hilang → diisi 30% target ({target * 0.3})")
        
        if 'Price' in missing_penalty:
            penalty_mask = missing_penalty['Price']
            if penalty_mask.any():
                price_target = self.slots['Price']['target']
                dm.loc[penalty_mask, 'Price'] = price_target * 2.0
                print(f"⚠️  Price: {penalty_mask.sum()} data hilang → penalti 200% target ({price_target * 2.0})")
        
        if 'Network' in dm.columns:
            if '5g' in self.df.columns and '5g' in mapping.get('Network', []):
                dm['Network'] = dm['Network'].apply(lambda x: 5 if x == 1 else 4)
            else:
                dm['Network'] = dm['Network'].apply(self._convert_network)
        
        return dm
    
    def calculate(self):
        print(f"\n{'='*70}")
        print(f"🧮 COPRAS CALCULATION - {len(self.df)} ALTERNATIVES")
        print(f"{'='*70}")
        
        # Get brand and model columns
        brand_col = 'nama_brand' if 'nama_brand' in self.df.columns else 'brand_name'
        model_col = 'model' if 'model' in self.df.columns else None
        
        # ================================================================
        # STEP 1: DECISION MATRIX
        # ================================================================
        dm = self._prepare_matrix()
        
        print(f"\n{'─'*70}")
        print(f"📊 STEP 1: DECISION MATRIX (Data Awal)")
        print(f"{'─'*70}")
        print(f"   Formula: X = nilai asli dari dataset")
        print(f"   Benefit Criteria (max better): {', '.join(self.benefit_criteria)}")
        print(f"   Cost Criteria (min better): {', '.join(self.cost_criteria)}")
        print()
        
        # Tampilkan decision matrix (semua alternatif)
        dm_display = dm.copy()
        dm_display.insert(0, 'No', range(1, len(dm_display) + 1))
        if brand_col in self.df.columns:
            dm_display.insert(1, 'Brand', self.df[brand_col].values)
        if model_col and model_col in self.df.columns:
            dm_display.insert(2, 'Model', self.df[model_col].values)
        
        # Format angka
        for col in dm_display.columns:
            if col not in ['No', 'Brand', 'Model']:
                dm_display[col] = dm_display[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        
        if tabulate:
            print(tabulate(dm_display, headers='keys', tablefmt='grid', showindex=False))
        else:
            print(dm_display.to_string(index=False))
        
        # ================================================================
        # STEP 2: PENALTY MATRIX
        # ================================================================
        penalty_matrix = dm.copy()
        
        print(f"\n{'─'*70}")
        print(f"⚡ STEP 2: PENALTY MATRIX (Sensitivitas)")
        print(f"{'─'*70}")
        print(f"   Formula Price: X' = X × (1 + max(0, (X - target)/target × 2))")
        print(f"   Formula Benefit: X' = X × (1 - max(0, min(0.8, (target - X)/target × 0.5)))")
        print(f"   Target Price: {self.slots['Price']['target']:,.0f}")
        print()
        
        if 'Price' in penalty_matrix.columns:
            price_target = self.slots['Price']['target']
            before_price = penalty_matrix['Price'].copy()
            penalty_matrix['Price'] = penalty_matrix['Price'].apply(
                lambda x: x * (1 + max(0, (x - price_target) / price_target * 2))
            )
        
        for c in self.benefit_criteria:
            if c in penalty_matrix.columns:
                target = self.slots[c]['target']
                if target > 0:
                    before = penalty_matrix[c].copy()
                    penalty_matrix[c] = penalty_matrix[c].apply(
                        lambda x: x * (1 - max(0, min(0.8, (target - x) / target * 0.5)))
                    )
                    # Tampilkan perubahan
                    changed = (before != penalty_matrix[c]).sum()
                    if changed > 0:
                        print(f"   {c}: {changed} nilai berubah (target={target})")
        
        if 'Price' in penalty_matrix.columns:
            changed = (before_price != penalty_matrix['Price']).sum()
            if changed > 0:
                print(f"   Price: {changed} nilai berubah (target={price_target:,.0f})")
        
        # Tampilkan penalty matrix
        penalty_display = penalty_matrix.copy()
        penalty_display.insert(0, 'No', range(1, len(penalty_display) + 1))
        if brand_col in self.df.columns:
            penalty_display.insert(1, 'Brand', self.df[brand_col].values)
        if model_col and model_col in self.df.columns:
            penalty_display.insert(2, 'Model', self.df[model_col].values)
        
        for col in penalty_display.columns:
            if col not in ['No', 'Brand', 'Model']:
                penalty_display[col] = penalty_display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        
        if tabulate:
            print(f"\n{tabulate(penalty_display, headers='keys', tablefmt='grid', showindex=False)}")
        else:
            print(f"\n{penalty_display.to_string(index=False)}")
        
        # ================================================================
        # STEP 3: NORMALIZATION
        # ================================================================
        norm = penalty_matrix.copy()
        
        print(f"\n{'─'*70}")
        print(f"📐 STEP 3: NORMALIZATION MATRIX")
        print(f"{'─'*70}")
        print(f"   Formula: n_ij = X_ij / Σ X_j")
        print()
        
        norm_sums = {}
        for c in self.criteria:
            s = norm[c].sum()
            norm_sums[c] = s
            if s > 0:
                norm[c] = norm[c] / s
            print(f"   Σ {c}: {s:.4f}")
        
        # Tampilkan normalized matrix
        norm_display = norm.copy()
        norm_display.insert(0, 'No', range(1, len(norm_display) + 1))
        if brand_col in self.df.columns:
            norm_display.insert(1, 'Brand', self.df[brand_col].values)
        if model_col and model_col in self.df.columns:
            norm_display.insert(2, 'Model', self.df[model_col].values)
        
        for col in norm_display.columns:
            if col not in ['No', 'Brand', 'Model']:
                norm_display[col] = norm_display[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
        
        if tabulate:
            print(f"\n{tabulate(norm_display, headers='keys', tablefmt='grid', showindex=False)}")
        else:
            print(f"\n{norm_display.to_string(index=False)}")
        
        # ================================================================
        # STEP 4: WEIGHTED MATRIX
        # ================================================================
        print(f"\n{'─'*70}")
        print(f"⚖️  STEP 4: WEIGHTED MATRIX")
        print(f"{'─'*70}")
        print(f"   Formula: w_ij = n_ij × weight_j")
        print()
        
        for c in self.criteria:
            weight = self.slots[c]['weight']
            print(f"   Weight {c}: {weight:.4f}")
            norm[c] = norm[c] * weight
        
        # Tampilkan weighted matrix
        weighted_display = norm.copy()
        weighted_display.insert(0, 'No', range(1, len(weighted_display) + 1))
        if brand_col in self.df.columns:
            weighted_display.insert(1, 'Brand', self.df[brand_col].values)
        if model_col and model_col in self.df.columns:
            weighted_display.insert(2, 'Model', self.df[model_col].values)
        
        for col in weighted_display.columns:
            if col not in ['No', 'Brand', 'Model']:
                weighted_display[col] = weighted_display[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
        
        if tabulate:
            print(f"\n{tabulate(weighted_display, headers='keys', tablefmt='grid', showindex=False)}")
        else:
            print(f"\n{weighted_display.to_string(index=False)}")
        
        # ================================================================
        # STEP 5: S+ and S- CALCULATION
        # ================================================================
        scores = pd.DataFrame(index=dm.index)
        
        print(f"\n{'─'*70}")
        print(f"📈 STEP 5: S+ (BENEFIT) & S- (COST) CALCULATION")
        print(f"{'─'*70}")
        print(f"   Formula S+: Σ weighted_benefit_criteria")
        print(f"   Formula S-: Σ weighted_cost_criteria")
        print(f"   Benefit: {', '.join(self.benefit_criteria)}")
        print(f"   Cost: {', '.join(self.cost_criteria)}")
        print()
        
        if self.benefit_criteria:
            scores['S_plus'] = norm[self.benefit_criteria].sum(axis=1)
        else:
            scores['S_plus'] = 0
        
        if self.cost_criteria:
            scores['S_minus'] = norm[self.cost_criteria].sum(axis=1)
        else:
            scores['S_minus'] = 0
        
        # Tampilkan S+ dan S-
        s_display = scores[['S_plus', 'S_minus']].copy()
        s_display.insert(0, 'No', range(1, len(s_display) + 1))
        if brand_col in self.df.columns:
            s_display.insert(1, 'Brand', self.df[brand_col].values)
        if model_col and model_col in self.df.columns:
            s_display.insert(2, 'Model', self.df[model_col].values)
        
        for col in ['S_plus', 'S_minus']:
            s_display[col] = s_display[col].apply(lambda x: f"{x:.6f}")
        
        if tabulate:
            print(tabulate(s_display, headers='keys', tablefmt='grid', showindex=False))
        else:
            print(s_display.to_string(index=False))
        
        # ================================================================
        # STEP 6: Qi CALCULATION
        # ================================================================
        S_minus_sum = scores['S_minus'].sum()
        S_minus_min = scores['S_minus'].min()
        
        print(f"\n{'─'*70}")
        print(f"🔢 STEP 6: Qi CALCULATION")
        print(f"{'─'*70}")
        print(f"   Formula: Qi = S+_i + (S-_min × Σ S-) / (S-_i × Σ(S-_min/S-_i))")
        print(f"   Σ S- (total S_minus): {S_minus_sum:.6f}")
        print(f"   S-_min (nilai S_minus terkecil): {S_minus_min:.6f}")
        
        if S_minus_sum > 0 and S_minus_min > 0:
            sum_div = np.sum(S_minus_min / scores['S_minus'].replace(0, np.nan))
            print(f"   Σ(S-_min/S-_i): {sum_div:.6f}")
            
            if sum_div > 0:
                scores['Qi'] = scores['S_plus'] + (S_minus_min * S_minus_sum) / (scores['S_minus'] * sum_div)
                
                # Tampilkan detail Qi untuk setiap alternatif
                print(f"\n   Detail Perhitungan Qi:")
                for i in scores.index:
                    s_plus = scores.loc[i, 'S_plus']
                    s_minus = scores.loc[i, 'S_minus']
                    term1 = s_plus
                    term2 = (S_minus_min * S_minus_sum) / (s_minus * sum_div) if s_minus > 0 else 0
                    qi = scores.loc[i, 'Qi']
                    
                    brand = self.df.loc[i, brand_col] if brand_col in self.df.columns else '?'
                    model = self.df.loc[i, model_col] if model_col and model_col in self.df.columns else '?'
                    
                    print(f"   {i+1}. {brand} {model}")
                    print(f"      Qi = {term1:.6f} + ({S_minus_min:.6f} × {S_minus_sum:.6f}) / ({s_minus:.6f} × {sum_div:.6f})")
                    print(f"      Qi = {term1:.6f} + {term2:.6f} = {qi:.6f}")
            else:
                scores['Qi'] = scores['S_plus']
                print(f"   Σ(S-_min/S-_i) = 0, Qi = S_plus")
        else:
            scores['Qi'] = scores['S_plus']
            print(f"   S_minus_sum atau S_minus_min = 0, Qi = S_plus")
        
        scores['Qi'] = scores['Qi'].fillna(0)
        
        # ================================================================
        # STEP 7: UTILITY CALCULATION
        # ================================================================
        Q_max = scores['Qi'].max()
        
        print(f"\n{'─'*70}")
        print(f"📊 STEP 7: UTILITY CALCULATION")
        print(f"{'─'*70}")
        print(f"   Formula: U_i = (Qi / Q_max) × 100%")
        print(f"   Q_max (nilai Qi tertinggi): {Q_max:.6f}")
        print()
        
        scores['Utility'] = (scores['Qi'] / Q_max * 100) if Q_max > 0 else 0
        
        # Tampilkan Qi dan Utility
        qu_display = scores[['Qi', 'Utility']].copy()
        qu_display.insert(0, 'No', range(1, len(qu_display) + 1))
        if brand_col in self.df.columns:
            qu_display.insert(1, 'Brand', self.df[brand_col].values)
        if model_col and model_col in self.df.columns:
            qu_display.insert(2, 'Model', self.df[model_col].values)
        
        qu_display['Qi'] = qu_display['Qi'].apply(lambda x: f"{x:.6f}")
        qu_display['Utility'] = qu_display['Utility'].apply(lambda x: f"{x:.2f}%")
        
        if tabulate:
            print(tabulate(qu_display, headers='keys', tablefmt='grid', showindex=False))
        else:
            print(qu_display.to_string(index=False))
        
        # ================================================================
        # BUILD FINAL RANKING
        # ================================================================
        self.final_ranking = self.df.copy()
        self.final_ranking['Qi_Score'] = scores['Qi']
        self.final_ranking['Utility_%'] = scores['Utility']
        self.final_ranking['S_plus'] = scores['S_plus']
        self.final_ranking['S_minus'] = scores['S_minus']
        self.final_ranking = self.final_ranking.sort_values('Qi_Score', ascending=False)
        self.final_ranking['Rank'] = range(1, len(self.final_ranking) + 1)
        
        # ================================================================
        # FINAL OUTPUT: 2 TABLES
        # ================================================================
        print(f"\n{'='*70}")
        print(f"✅ COPRAS COMPLETE!")
        print(f"{'='*70}")
        
        # TABLE 1: Top 10 WITH Qi and Utility
        print(f"\n{'─'*70}")
        print(f"🏆 TOP 10 REKOMENDASI (DENGAN Qi SCORE & UTILITY)")
        print(f"{'─'*70}")
        
        top10 = self.final_ranking.head(10)
        table1_cols = ['Rank']
        if brand_col in top10.columns:
            table1_cols.append(brand_col)
        if 'model' in top10.columns:
            table1_cols.append('model')
        for c in ['kapasitas_ram', 'memori_internal', 'kapasitas_baterai', 'resolusi_kamera', 'harga', 'chipset_score', 'Qi_Score', 'Utility_%']:
            if c in top10.columns:
                table1_cols.append(c)
        
        table1 = top10[table1_cols].copy()
        
        # Format table1
        for c in table1.columns:
            if c == 'Qi_Score':
                table1[c] = table1[c].apply(lambda x: f"{x:.4f}")
            elif c == 'Utility_%':
                table1[c] = table1[c].apply(lambda x: f"{x:.1f}%")
            elif c == 'harga':
                table1[c] = table1[c].apply(lambda x: f"Rp{x:,.0f}" if pd.notna(x) else "N/A")
            elif c == 'kapasitas_ram':
                table1[c] = table1[c].apply(lambda x: f"{x} GB" if pd.notna(x) else "N/A")
            elif c == 'memori_internal':
                table1[c] = table1[c].apply(lambda x: f"{x} GB" if pd.notna(x) else "N/A")
            elif c == 'kapasitas_baterai':
                table1[c] = table1[c].apply(lambda x: f"{x} mAh" if pd.notna(x) else "N/A")
            elif c == 'resolusi_kamera':
                table1[c] = table1[c].apply(lambda x: f"{x} MP" if pd.notna(x) else "N/A")
            elif c == 'chipset_score':
                table1[c] = table1[c].apply(lambda x: f"{x}/100" if pd.notna(x) else "N/A")
            elif c == brand_col:
                table1[c] = table1[c].apply(lambda x: str(x).capitalize())
            elif c == 'model':
                table1[c] = table1[c].apply(lambda x: str(x))
        
        renames1 = {
            brand_col: 'Brand', 'model': 'Model',
            'kapasitas_ram': 'RAM', 'memori_internal': 'Storage',
            'kapasitas_baterai': 'Battery', 'resolusi_kamera': 'Camera',
            'harga': 'Price', 'chipset_score': 'Chipset',
            'Qi_Score': 'Qi_Score', 'Utility_%': 'Utility'
        }
        table1 = table1.rename(columns={k: v for k, v in renames1.items() if k in table1.columns})
        
        if tabulate:
            print(tabulate(table1, headers='keys', tablefmt='grid', showindex=False))
        else:
            print(table1.to_string(index=False))
        
        # TABLE 2: Top 10 WITHOUT Qi and Utility
        print(f"\n{'─'*70}")
        print(f"📱 TOP 10 REKOMENDASI (TANPA Qi SCORE & UTILITY)")
        print(f"{'─'*70}")
        
        table2_cols = ['Rank']
        if brand_col in top10.columns:
            table2_cols.append(brand_col)
        if 'model' in top10.columns:
            table2_cols.append('model')
        for c in ['kapasitas_ram', 'memori_internal', 'kapasitas_baterai', 'resolusi_kamera', 'harga', 'chipset_score']:
            if c in top10.columns:
                table2_cols.append(c)
        
        table2 = top10[table2_cols].copy()
        
        # Format table2
        for c in table2.columns:
            if c == 'harga':
                table2[c] = table2[c].apply(lambda x: f"Rp{x:,.0f}" if pd.notna(x) else "N/A")
            elif c == 'kapasitas_ram':
                table2[c] = table2[c].apply(lambda x: f"{x} GB" if pd.notna(x) else "N/A")
            elif c == 'memori_internal':
                table2[c] = table2[c].apply(lambda x: f"{x} GB" if pd.notna(x) else "N/A")
            elif c == 'kapasitas_baterai':
                table2[c] = table2[c].apply(lambda x: f"{x} mAh" if pd.notna(x) else "N/A")
            elif c == 'resolusi_kamera':
                table2[c] = table2[c].apply(lambda x: f"{x} MP" if pd.notna(x) else "N/A")
            elif c == 'chipset_score':
                table2[c] = table2[c].apply(lambda x: f"{x}/100" if pd.notna(x) else "N/A")
            elif c == brand_col:
                table2[c] = table2[c].apply(lambda x: str(x).capitalize())
            elif c == 'model':
                table2[c] = table2[c].apply(lambda x: str(x))
        
        renames2 = {
            brand_col: 'Brand', 'model': 'Model',
            'kapasitas_ram': 'RAM', 'memori_internal': 'Storage',
            'kapasitas_baterai': 'Battery', 'resolusi_kamera': 'Camera',
            'harga': 'Price', 'chipset_score': 'Chipset'
        }
        table2 = table2.rename(columns={k: v for k, v in renames2.items() if k in table2.columns})
        
        if tabulate:
            print(tabulate(table2, headers='keys', tablefmt='grid', showindex=False))
        else:
            print(table2.to_string(index=False))
        
        print(f"\n{'='*70}")
        print(f"📊 RANKING COMPLETE - {len(self.final_ranking)} smartphones ranked")
        print(f"{'='*70}\n")
        
        return self.final_ranking
    
    def get_top_n(self, n=5):
        return self.final_ranking.head(n)
    
    def get_formatted_results(self, n=5):
        top = self.get_top_n(n)
        results = []
        
        brand_col = 'nama_brand' if 'nama_brand' in top.columns else 'brand_name'
        
        for idx, row in top.iterrows():
            phone_data = {
                'brand': str(row.get(brand_col, 'Unknown')),
                'nama_brand': str(row.get(brand_col, 'Unknown')),
                'model': str(row.get('model', 'Unknown')),
                'price': int(row.get('harga', row.get('price', 0))) if pd.notna(row.get('harga', row.get('price', 0))) else 0,
                'utility': round(float(row.get('Utility_%', 0)), 1),
                'qi_score': round(float(row.get('Qi_Score', 0)), 4),
                's_plus': round(float(row.get('S_plus', 0)), 6),
                's_minus': round(float(row.get('S_minus', 0)), 6),
                'rank': int(row.get('Rank', 0)),
                'links': self._extract_links(row),
                'specs': self._extract_specs(row)
            }
            results.append(phone_data)
        
        return results
    
    def _extract_links(self, row):
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
    
    def _extract_specs(self, row):
        specs = {}
        
        spec_mappings = {
            'RAM': ['kapasitas_ram', 'ram'],
            'ROM': ['memori_internal', 'rom', 'storage'],
            'Battery': ['kapasitas_baterai', 'battery'],
            'Camera': ['resolusi_kamera', 'camera'],
            'Chipset_Score': ['chipset_score'],
            'Screen_Size': ['ukuran_layar', 'screen_size'],
            'Refresh_Rate': ['refresh_rate'],
            'Network': ['network_type', '5g'],
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
        
        return specs


def run_copras(df, slots, n=5):
    c = COPRAS(df, slots)
    c.calculate()
    return c.get_top_n(n)