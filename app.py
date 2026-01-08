import requests
import pandas as pd
import toml
import json

class DhanOptionAnalyzer:
    def __init__(self, secrets_file="secrets.toml"):
        # 1. Load Credentials
        try:
            secrets = toml.load(secrets_file)
            self.client_id = secrets["dhan"]["client_id"]
            self.access_token = secrets["dhan"]["access_token"]
        except Exception as e:
            raise Exception(f"Error loading secrets.toml: {e}")

        self.base_url = "https://api.dhan.co/v2/optionchain"
        
    def fetch_option_chain(self, underlying_scrip=13, underlying_seg="IDX_I", expiry="2024-10-31"):
        """
        Fetches option chain data using the exact raw API request structure provided.
        """
        headers = {
            'access-token': self.access_token,
            'client-id': self.client_id,
            'Content-Type': 'application/json'
        }
        
        payload = {
            "UnderlyingScrip": underlying_scrip,
            "UnderlyingSeg": underlying_seg,
            "Expiry": expiry
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Request Failed: {e}")
            return None

    def process_data(self, raw_data):
        """
        Converts the nested JSON response into a specialized DataFrame for analysis.
        """
        if not raw_data or 'data' not in raw_data or 'oc' not in raw_data['data']:
            print("Invalid Data Format Received")
            return None, None

        spot_price = raw_data['data'].get('last_price', 0)
        oc_data = raw_data['data']['oc']
        
        processed_rows = []

        for strike_str, details in oc_data.items():
            strike_price = float(strike_str)
            
            # Extract Call (CE) Data
            ce = details.get('ce', {})
            ce_oi = ce.get('oi', 0)
            ce_ltp = ce.get('last_price', 0)
            ce_vol = ce.get('volume', 0)
            ce_iv = ce.get('implied_volatility', 0)
            ce_delta = ce.get('greeks', {}).get('delta', 0)
            # Calculate Change in OI (Current - Previous)
            ce_oi_change = ce_oi - ce.get('previous_oi', 0)

            # Extract Put (PE) Data
            pe = details.get('pe', {})
            pe_oi = pe.get('oi', 0)
            pe_ltp = pe.get('last_price', 0)
            pe_vol = pe.get('volume', 0)
            pe_iv = pe.get('implied_volatility', 0)
            pe_delta = pe.get('greeks', {}).get('delta', 0)
            pe_oi_change = pe_oi - pe.get('previous_oi', 0)

            processed_rows.append({
                'strike_price': strike_price,
                'ce_oi': ce_oi, 'ce_oi_change': ce_oi_change, 'ce_ltp': ce_ltp, 'ce_iv': ce_iv, 'ce_delta': ce_delta,
                'pe_oi': pe_oi, 'pe_oi_change': pe_oi_change, 'pe_ltp': pe_ltp, 'pe_iv': pe_iv, 'pe_delta': pe_delta
            })

        df = pd.DataFrame(processed_rows)
        return df.sort_values('strike_price'), spot_price

    def generate_signal(self, df, spot_price):
        """
        Analyzes Greeks, IV, OI, and Price to determine direction, support, and resistance.
        """
        if df is None or df.empty:
            return

        # 1. Macro Analysis (PCR)
        total_ce_oi = df['ce_oi'].sum()
        total_pe_oi = df['pe_oi'].sum()
        pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0

        # 2. Key Levels (Support & Resistance)
        # Resistance = Strike with Max Call OI
        res_row = df.loc[df['ce_oi'].idxmax()]
        resistance = res_row['strike_price']
        
        # Support = Strike with Max Put OI
        sup_row = df.loc[df['pe_oi'].idxmax()]
        support = res_row['strike_price']

        # 3. Trend Decider (ATM Analysis)
        # Find the ATM strike (closest to spot)
        atm_row = df.iloc[(df['strike_price'] - spot_price).abs().argsort()[:1]].iloc[0]
        
        bias = "NEUTRAL"
        reason = []

        # PCR Logic
        if pcr > 1.2:
            reason.append("High PCR (Bullish Sentiment)")
        elif pcr < 0.7:
            reason.append("Low PCR (Bearish Sentiment)")

        # ATM Buildup Logic
        if atm_row['ce_oi_change'] > 0 and atm_row['pe_oi_change'] < 0:
            bias = "BEARISH"
            reason.append("Call Writing at ATM (Resistance Building)")
        elif atm_row['pe_oi_change'] > 0 and atm_row['ce_oi_change'] < 0:
            bias = "BULLISH"
            reason.append("Put Writing at ATM (Support Building)")
        elif atm_row['ce_oi_change'] > 0 and atm_row['pe_oi_change'] > 0:
            bias = "SIDEWAYS / VOLATILE"
            reason.append("Both Sides Adding Positions (Straddle/Strangle Buildup)")

        # 4. IV Check
        avg_iv = (atm_row['ce_iv'] + atm_row['pe_iv']) / 2
        iv_status = "High" if avg_iv > 20 else "Normal"

        # --- OUTPUT REPORT ---
        print("\n" + "="*50)
        print(f" MARKET SNAPSHOT (Spot: {spot_price})")
        print("="*50)
        print(f"Detected Trend:   {bias}")
        print(f"Reasoning:        {', '.join(reason)}")
        print(f"PCR:              {pcr}")
        print(f"Avg ATM IV:       {avg_iv:.2f}% ({iv_status})")
        print("-" * 50)
        print(f"Immediate Support:    {support} (Max Put OI: {int(sup_row['pe_oi'])})")
        print(f"Immediate Resistance: {resistance} (Max Call OI: {int(res_row['ce_oi'])})")
        print("-" * 50)
        print(f"ATM Strike ({atm_row['strike_price']}) Activity:")
        print(f"  Call OI Chg: {int(atm_row['ce_oi_change'])}")
        print(f"  Put OI Chg:  {int(atm_row['pe_oi_change'])}")
        print("="*50)

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    tool = DhanOptionAnalyzer()
    
    # Configuration
    # Note: Update expiry to a valid future date if fetching live
    EXPIRY_DATE = "2026-01-13" 
    
    print(f"Fetching NIFTY Option Chain for Expiry: {EXPIRY_DATE}...")
    raw_json = tool.fetch_option_chain(underlying_scrip=13, expiry=EXPIRY_DATE)
    
    if raw_json:
        df, spot = tool.process_data(raw_json)
        tool.generate_signal(df, spot)
