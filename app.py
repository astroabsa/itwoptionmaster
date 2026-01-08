import streamlit as st
import pandas as pd
from dhanhq import dhanhq
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. SETUP
# ==========================================
st.set_page_config(page_title="Dhan Option Chain", page_icon="🚀", layout="wide")

st.markdown("""
<style>
    .metric-card { background-color: #0e1117; border: 1px solid #262730; border-radius: 5px; padding: 15px; text-align: center; }
    .bullish { color: #00FF00; font-weight: bold; }
    .bearish { color: #FF0000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. VERIFIED CONFIGURATION (FROM YOUR CSV)
# ==========================================
# These IDs come directly from your 'api-scrip-master (1).csv' file
INDEX_MAP = {
    'NIFTY': {
        'id': '13',           # Verified: CSV Row for 'NIFTY'
        'segment': 'IDX_I',   # Standard Index Segment
        'name': 'NIFTY 50',
        'default_spot': 26000
    },
    'SENSEX': {
        'id': '51',           # Verified: CSV Row for 'SENSEX' (NOT 51005)
        'segment': 'IDX_I',
        'name': 'SENSEX',
        'default_spot': 84000
    },
    'BANKNIFTY': {
        'id': '25',           # Verified: CSV Row for 'BANKNIFTY'
        'segment': 'IDX_I',
        'name': 'BANK NIFTY',
        'default_spot': 54000
    },
    'FINNIFTY': {
        'id': '27',           # Verified: CSV Row for 'FINNIFTY'
        'segment': 'IDX_I',
        'name': 'FINNIFTY',
        'default_spot': 24000
    }
}

# ==========================================
# 3. CORE FUNCTIONS
# ==========================================

@st.cache_data(ttl=60)
def fetch_option_chain(client_id, access_token, security_id, expiry_str):
    try:
        dhan = dhanhq(client_id, access_token)
        # Using the verified ID and Segment
        response = dhan.option_chain(
            under_exchange_segment="IDX_I",
            under_security_id=security_id,
            expiry=expiry_str
        )
        if response['status'] == 'success':
            return pd.DataFrame(response['data'])
        else:
            return response # Return error dict
    except Exception as e:
        return {"error": str(e)}

def process_data(df, spot):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty: return None
    df.columns = [c.lower() for c in df.columns]
    
    # Nested data handling (v2 API structure)
    if 'oc' in df.columns: 
        pass 

    if 'option_type' not in df.columns: return None

    ce = df[df['option_type'] == 'CALL'].copy()
    pe = df[df['option_type'] == 'PUT'].copy()
    
    chain = pd.merge(ce, pe, on='strike_price', suffixes=('_ce', '_pe'))
    chain = chain.sort_values('strike_price')
    
    # Smart Filter (ATM +/- 8)
    atm_idx = (chain['strike_price'] - spot).abs().idxmin()
    start = max(0, atm_idx - 8)
    end = min(len(chain), atm_idx + 9)
    
    return chain.iloc[start:end]

# ==========================================
# 4. APP LOGIC
# ==========================================

try:
    CLIENT_ID = st.secrets["dhan"]["client_id"]
    ACCESS_TOKEN = st.secrets["dhan"]["access_token"]
except:
    st.error("❌ Secrets not found. Create `.streamlit/secrets.toml` with [dhan] section.")
    st.stop()

st.sidebar.header("⚙️ Configuration")
idx_name = st.sidebar.radio("Select Index", list(INDEX_MAP.keys()))
idx_config = INDEX_MAP[idx_name]

# Date Picker 
today = datetime.now().date()
expiry = st.sidebar.date_input("Expiry Date", today)

if st.sidebar.button("Refresh"):
    st.cache_data.clear()
    st.rerun()

st.title(f"📊 {idx_config['name']} Analyzer")

# 1. Fetch
data_resp = fetch_option_chain(CLIENT_ID, ACCESS_TOKEN, idx_config['id'], str(expiry))

# 2. Process & Display
if isinstance(data_resp, pd.DataFrame):
    # Extract Spot Price from the API response itself
    if 'last_price' in data_resp.columns:
        api_spot = data_resp['last_price'].iloc[0] if not data_resp.empty else idx_config['default_spot']
    else:
        api_spot = idx_config['default_spot']

    view_df = process_data(data_resp, api_spot)
    
    if view_df is not None:
        st.success(f"🟢 Connection Successful! (ID: {idx_config['id']})")
        st.metric("Underlying Spot Price", api_spot)

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(x=view_df['strike_price'], y=view_df['oi_ce'], name='Call OI', marker_color='#ff4b4b'))
        fig.add_trace(go.Bar(x=view_df['strike_price'], y=view_df['oi_pe'], name='Put OI', marker_color='#00cc96'))
        fig.add_vline(x=api_spot, line_dash="dash", line_color="yellow", annotation_text="Spot")
        fig.update_layout(height=400, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Table
        cols = ['oi_ce', 'last_price_ce', 'strike_price', 'last_price_pe', 'oi_pe']
        st.dataframe(view_df[[c for c in cols if c in view_df.columns]], hide_index=True, use_container_width=True)
    else:
        st.warning("Data fetched but processing failed (Check API response structure).")

elif isinstance(data_resp, dict):
    st.error(f"⚠️ API Error: {data_resp.get('remarks', data_resp)}")
    st.info(f"Verified Config Used: ID={idx_config['id']} | Segment={idx_config['segment']}")
