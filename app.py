import streamlit as st
import pandas as pd
from dhanhq import dhanhq
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Dhan Option Chain Analyzer",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card { background-color: #0e1117; border: 1px solid #262730; border-radius: 5px; padding: 15px; text-align: center; }
    .bullish { color: #00FF00; font-weight: bold; }
    .bearish { color: #FF0000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONFIGURATION & MAPS
# ==========================================

# Verified IDs for Dhan
INDEX_MAP = {
    'NIFTY':     {'id': '13', 'name': 'NIFTY 50', 'default_spot': 26000}, 
    'BANKNIFTY': {'id': '25', 'name': 'BANK NIFTY', 'default_spot': 54000}, 
    'SENSEX':    {'id': '51', 'name': 'SENSEX',     'default_spot': 84000}
}

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

@st.cache_data(ttl=300) # Cache expiry list for 5 mins
def fetch_expiry_dates(client_id, access_token, security_id):
    """
    Fetches the list of ACTIVE expiry dates from Dhan.
    This prevents 'Invalid Request' errors by ensuring we only ask for valid dates.
    """
    try:
        dhan = dhanhq(client_id, access_token)
        # Use IDX_I for Index Underlying
        response = dhan.expiry_list(
            under_security_id=security_id,
            under_exchange_segment="IDX_I" 
        )
        if response['status'] == 'success':
            return response['data'] # Returns list of date strings ['2026-01-13', ...]
        else:
            st.error(f"Expiry Fetch Error: {response}")
            return []
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return []

@st.cache_data(ttl=15)
def fetch_option_chain(client_id, access_token, security_id, expiry_str):
    """Fetches Option Chain for a SPECIFIC verified date."""
    try:
        dhan = dhanhq(client_id, access_token)
        response = dhan.option_chain(
            under_exchange_segment="IDX_I", 
            under_security_id=security_id,            
            expiry=expiry_str
        )
        return response
    except Exception as e:
        return {"status": "failure", "remarks": {"error_message": str(e)}}

def process_data(df, spot):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty: return None
    df.columns = [c.lower() for c in df.columns]
    
    required = ['option_type', 'strike_price', 'oi', 'last_price']
    if not all(col in df.columns for col in required): return None

    ce = df[df['option_type'] == 'CALL'].copy()
    pe = df[df['option_type'] == 'PUT'].copy()
    
    chain = pd.merge(ce, pe, on='strike_price', suffixes=('_ce', '_pe'))
    chain = chain.sort_values('strike_price')
    
    # Filter Strikes around ATM
    atm_idx = (chain['strike_price'] - spot).abs().idxmin()
    start = max(0, atm_idx - 8)
    end = min(len(chain), atm_idx + 9)
    
    return chain.iloc[start:end], chain

# ==========================================
# 4. SIDEBAR & LOGIC
# ==========================================

try:
    CLIENT_ID = st.secrets["dhan"]["client_id"]
    ACCESS_TOKEN = st.secrets["dhan"]["access_token"]
except:
    st.error("❌ Secrets not found. Please check `.streamlit/secrets.toml`")
    st.stop()

st.sidebar.header("⚙️ Settings")
selected_index = st.sidebar.radio("Select Index", list(INDEX_MAP.keys()))

idx_config = INDEX_MAP[selected_index]
security_id = idx_config['id']

# --- STEP 1: FETCH AVAILABLE EXPIRIES ---
# This eliminates the guessing game.
available_expiries = fetch_expiry_dates(CLIENT_ID, ACCESS_TOKEN, security_id)

if available_expiries:
    # Default to the nearest date (first in the list)
    selected_expiry = st.sidebar.selectbox("Select Expiry Date", available_expiries)
else:
    st.sidebar.warning("Could not fetch expiry list. Check connectivity.")
    selected_expiry = None

# Spot Price Placeholder
if 'last_spot' not in st.session_state:
    st.session_state['last_spot'] = idx_config['default_spot']

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 5. DASHBOARD
# ==========================================
st.title(f"📊 {idx_config['name']} Analysis")

if selected_expiry:
    # --- STEP 2: FETCH OPTION CHAIN ---
    resp = fetch_option_chain(CLIENT_ID, ACCESS_TOKEN, security_id, selected_expiry)

    if isinstance(resp, dict) and resp.get('status') == 'success':
        data = resp.get('data', {})
        
        # 1. Extract Spot Price (Auto-Update)
        api_spot = data.get('last_price')
        if api_spot:
            st.session_state['last_spot'] = api_spot
            st.sidebar.success(f"🟢 Live Spot: {api_spot}")
        
        # 2. Convert to DataFrame
        # Handle nested 'data' structure if present (common in v2)
        chain_data = data.get('oc', data) 
        if isinstance(chain_data, dict) and 'data' in chain_data:
            chain_data = chain_data['data']
            
        full_df = pd.DataFrame(chain_data) if isinstance(chain_data, list) else pd.DataFrame(chain_data)

        # 3. Process
        view_df, processed_chain = process_data(full_df, st.session_state['last_spot'])
        
        if view_df is not None:
            # KPI Calculations
            ce_oi = processed_chain['oi_ce'].sum()
            pe_oi = processed_chain['oi_pe'].sum()
            pcr = round(pe_oi / ce_oi, 2) if ce_oi > 0 else 0
            
            sup = processed_chain.loc[processed_chain['oi_pe'].idxmax(), 'strike_price']
            res = processed_chain.loc[processed_chain['oi_ce'].idxmax(), 'strike_price']
            
            sent = "BULLISH" if pcr > 1.2 else ("BEARISH" if pcr < 0.6 else "NEUTRAL")
            col = "bullish" if pcr > 1.2 else ("bearish" if pcr < 0.6 else "neutral")

            # Metrics Row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PCR", pcr)
            c2.metric("Support", sup)
            c3.metric("Resistance", res)
            c4.markdown(f"**Sentiment**<br><span class='{col}'>{sent}</span>", unsafe_allow_html=True)

            st.divider()

            # Chart Row
            fig = go.Figure()
            fig.add_trace(go.Bar(x=view_df['strike_price'], y=view_df['oi_ce'], name='Call OI', marker_color='#ff4b4b'))
            fig.add_trace(go.Bar(x=view_df['strike_price'], y=view_df['oi_pe'], name='Put OI', marker_color='#00cc96'))
            fig.add_vline(x=st.session_state['last_spot'], line_dash="dash", line_color="yellow", annotation_text="Spot")
            fig.update_layout(xaxis_title="Strike", yaxis_title="Open Interest", barmode='group', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Table Row
            cols = ['oi_ce', 'last_price_ce', 'strike_price', 'last_price_pe', 'oi_pe']
            safe_cols = [c for c in cols if c in view_df.columns]
            st.dataframe(view_df[safe_cols], hide_index=True, use_container_width=True)
        else:
            st.error("Data received but columns missing. API response structure might have changed.")

    elif isinstance(resp, dict):
        st.error(f"⚠️ API Error: {resp.get('remarks', resp)}")
        st.info(f"Requested: {selected_index} ({security_id}) for {selected_expiry}")

else:
    st.info("👈 Waiting for Expiry Date selection...")
