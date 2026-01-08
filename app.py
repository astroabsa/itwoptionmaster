import streamlit as st
import pandas as pd
from dhanhq import dhanhq
import plotly.graph_objects as go
from datetime import datetime, timedelta

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

# MAPPING: Now includes 'quote_segment' for fetching Spot Price
INDEX_MAP = {
    'NIFTY': {
        'id': '13', 
        'name': 'NIFTY 50', 
        'quote_segment': 'NSE' # Segment for Spot Price
    }, 
    'BANKNIFTY': {
        'id': '25', 
        'name': 'BANK NIFTY', 
        'quote_segment': 'NSE'
    }, 
    'SENSEX': {
        'id': '51', 
        'name': 'SENSEX',     
        'quote_segment': 'BSE'
    }
}

EXPIRY_DAYS = {
    'NIFTY': 1,      # Tuesday
    'BANKNIFTY': 2,  # Wednesday
    'SENSEX': 3      # Thursday
}

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def get_next_expiry(index_name):
    """Auto-calculates the correct expiry date."""
    target_weekday = EXPIRY_DAYS.get(index_name, 3) 
    today = datetime.now().date()
    days_ahead = target_weekday - today.weekday()
    if days_ahead <= 0: days_ahead += 7
    return today + timedelta(days=days_ahead)

@st.cache_data(ttl=5) # Cache spot price for 5 seconds only
def fetch_live_spot(client_id, access_token, segment, security_id):
    """Fetches the LIVE spot price of the index."""
    try:
        dhan = dhanhq(client_id, access_token)
        # Note: We use the QUOTE segment (NSE/BSE), not IDX_I
        response = dhan.quote(
            exchange_segment=segment, 
            security_id=security_id
        )
        if response['status'] == 'success':
            return float(response['data']['last_price'])
        else:
            return None
    except Exception:
        return None

@st.cache_data(ttl=60)
def fetch_option_chain(client_id, access_token, security_id, expiry_str):
    """Fetches Option Chain."""
    try:
        dhan = dhanhq(client_id, access_token)
        # Note: We use IDX_I for the Option Chain underlying
        response = dhan.option_chain(
            under_exchange_segment="IDX_I", 
            under_security_id=security_id,            
            expiry=expiry_str
        )
        if response['status'] == 'success':
            return pd.DataFrame(response['data'])
        else:
            return response
    except Exception as e:
        return {"error": str(e)}

def process_data(df, spot):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty: return None
    df.columns = [c.lower() for c in df.columns]
    if 'option_type' not in df.columns: return None

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
    st.error("❌ Secrets not found.")
    st.stop()

st.sidebar.header("⚙️ Settings")
selected_index = st.sidebar.radio("Select Index", list(INDEX_MAP.keys()))

# Config
idx_config = INDEX_MAP[selected_index]
security_id = idx_config['id']
quote_segment = idx_config['quote_segment']

# 1. Fetch Live Spot Price (Dynamic)
live_price = fetch_live_spot(CLIENT_ID, ACCESS_TOKEN, quote_segment, security_id)

if live_price:
    st.sidebar.success(f"🟢 Live Spot: {live_price}")
    # Allow override but default to live price
    spot = st.sidebar.number_input("Spot Price", value=live_price)
else:
    st.sidebar.warning("⚠️ Could not fetch live price")
    spot = st.sidebar.number_input("Spot Price (Manual)", value=25000.0)

# 2. Set Date
auto_date = get_next_expiry(selected_index)
expiry = st.sidebar.date_input(f"Expiry", auto_date)

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 5. DASHBOARD
# ==========================================
st.title(f"📊 {idx_config['name']} Analysis")
st.caption(f"Spot Price: {spot} | Expiry: {expiry}")

# Fetch Option Chain
data_resp = fetch_option_chain(CLIENT_ID, ACCESS_TOKEN, security_id, str(expiry))

if isinstance(data_resp, pd.DataFrame):
    view_df, full_df = process_data(data_resp, spot)
    
    if view_df is not None:
        # Stats
        ce_oi = full_df['oi_ce'].sum()
        pe_oi = full_df['oi_pe'].sum()
        pcr = round(pe_oi / ce_oi, 2) if ce_oi > 0 else 0
        
        sup = full_df.loc[full_df['oi_pe'].idxmax(), 'strike_price']
        res = full_df.loc[full_df['oi_ce'].idxmax(), 'strike_price']
        
        sent = "BULLISH" if pcr > 1.2 else ("BEARISH" if pcr < 0.6 else "NEUTRAL")
        col = "bullish" if pcr > 1.2 else ("bearish" if pcr < 0.6 else "neutral")

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PCR", pcr)
        c2.metric("Support", sup)
        c3.metric("Resistance", res)
        c4.markdown(f"**Sentiment**<br><span class='{col}'>{sent}</span>", unsafe_allow_html=True)

        st.divider()

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(x=view_df['strike_price'], y=view_df['oi_ce'], name='Call OI', marker_color='#ff4b4b'))
        fig.add_trace(go.Bar(x=view_df['strike_price'], y=view_df['oi_pe'], name='Put OI', marker_color='#00cc96'))
        fig.add_vline(x=spot, line_dash="dash", line_color="yellow", annotation_text="Spot")
        fig.update_layout(xaxis_title="Strike", yaxis_title="OI", barmode='group', height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        cols = ['oi_ce', 'last_price_ce', 'strike_price', 'last_price_pe', 'oi_pe']
        safe_cols = [c for c in cols if c in view_df.columns]
        st.dataframe(view_df[safe_cols], hide_index=True, use_container_width=True)

    else:
        st.error("Data fetched but processing failed. (Check market hours)")

elif isinstance(data_resp, dict):
    st.error(f"API Error: {data_resp.get('error')}")
