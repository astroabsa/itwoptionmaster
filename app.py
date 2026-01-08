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
    .neutral { color: #FFFF00; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONFIGURATION & MAPS
# ==========================================

INDEX_MAP = {
    'NIFTY':     {'id': '13', 'name': 'NIFTY 50', 'default_spot': 26000}, 
    'BANKNIFTY': {'id': '25', 'name': 'BANK NIFTY', 'default_spot': 54000}, 
    'SENSEX':    {'id': '51', 'name': 'SENSEX',     'default_spot': 84000}
}

# EXPIRY LOGIC (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri)
# Adjusted based on your platform screenshots (Nifty=Thu)
EXPIRY_DAYS = {
    'NIFTY': 3,      # Thursday
    'BANKNIFTY': 2,  # Wednesday
    'SENSEX': 3      # Thursday (BSE changed to Thu recently)
}

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def get_next_expiry(index_name):
    """Auto-calculates the next valid expiry day."""
    target_weekday = EXPIRY_DAYS.get(index_name, 3) 
    today = datetime.now().date()
    days_ahead = target_weekday - today.weekday()
    
    # If today is the expiry day, we might want today or next week depending on time
    # For simplicity, if today is target, keep today. If passed, add 7.
    if days_ahead < 0: 
        days_ahead += 7
    return today + timedelta(days=days_ahead)

@st.cache_data(ttl=15)
def fetch_option_chain(client_id, access_token, security_id, expiry_str):
    """Fetches Option Chain and extracts Spot Price."""
    try:
        dhan = dhanhq(client_id, access_token)
        response = dhan.option_chain(
            under_exchange_segment="IDX_I", 
            under_security_id=security_id,            
            expiry=expiry_str
        )
        # Return the whole response to handle status checks in main
        return response
    except Exception as e:
        return {"status": "failure", "remarks": {"error_message": str(e)}}

def process_data(df, spot):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty: return None
    df.columns = [c.lower() for c in df.columns]
    
    # Safety check for required columns
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
# 4. SIDEBAR & SETUP
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

# Auto-set Date
auto_date = get_next_expiry(selected_index)
expiry = st.sidebar.date_input(f"Expiry Date", auto_date)

# Spot Price Placeholder (Will update if API succeeds)
if 'last_spot' not in st.session_state:
    st.session_state['last_spot'] = idx_config['default_spot']

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 5. DASHBOARD & LOGIC
# ==========================================
st.title(f"📊 {idx_config['name']} Analysis")

# 1. Fetch Data
resp = fetch_option_chain(CLIENT_ID, ACCESS_TOKEN, security_id, str(expiry))

# 2. Check Success
if isinstance(resp, dict) and resp.get('status') == 'success':
    data = resp.get('data', {})
    
    # EXTRACT SPOT PRICE FROM CHAIN RESPONSE
    # Dhan API usually returns 'last_price' in the root of data object for the underlying
    api_spot = data.get('last_price')
    
    if api_spot:
        st.session_state['last_spot'] = api_spot
        st.success(f"🟢 Live Spot Price: {api_spot}")
    else:
        st.warning("⚠️ Spot price not found in response, using manual/default.")

    # Convert chain data to DataFrame
    # Note: 'oc' key usually holds the list of options in v2
    chain_list = data.get('oc', data) # Fallback if structure varies
    if isinstance(chain_list, dict): # Handle nested 'data' case if exists
        chain_list = chain_list.get('data', [])
        
    full_df = pd.DataFrame(chain_list) if isinstance(chain_list, list) else pd.DataFrame(data)

    # 3. Process & Visualize
    view_df, processed_chain = process_data(full_df, st.session_state['last_spot'])
    
    if view_df is not None:
        # Stats
        ce_oi = processed_chain['oi_ce'].sum()
        pe_oi = processed_chain['oi_pe'].sum()
        pcr = round(pe_oi / ce_oi, 2) if ce_oi > 0 else 0
        
        sup = processed_chain.loc[processed_chain['oi_pe'].idxmax(), 'strike_price']
        res = processed_chain.loc[processed_chain['oi_ce'].idxmax(), 'strike_price']
        
        sent = "BULLISH" if pcr > 1.2 else ("BEARISH" if pcr < 0.6 else "NEUTRAL")
        col = "bullish" if pcr > 1.2 else ("bearish" if pcr < 0.6 else "neutral")

        # Metrics
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
        fig.add_vline(x=st.session_state['last_spot'], line_dash="dash", line_color="yellow", annotation_text="Spot")
        fig.update_layout(xaxis_title="Strike", yaxis_title="Open Interest", barmode='group', height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        cols = ['oi_ce', 'last_price_ce', 'strike_price', 'last_price_pe', 'oi_pe']
        safe_cols = [c for c in cols if c in view_df.columns]
        st.dataframe(view_df[safe_cols], hide_index=True, use_container_width=True)
    else:
        st.error("Data fetched but format was unexpected. (Columns missing)")

elif isinstance(resp, dict):
    # FAILURE HANDLING
    st.error("⚠️ API Request Failed")
    # Print the ACTUAL error message from Dhan
    st.json(resp) 
    st.info(f"Debug: Requested {selected_index} (ID: {security_id}) for Expiry {expiry}")

else:
    st.error("Unknown API Error")
