import streamlit as st
import pandas as pd
from dhanhq import dhanhq
import plotly.graph_objects as go

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Dhan Option Chain (Docs Compliant)", page_icon="📘", layout="wide")

# VERIFIED CONFIG (From your uploaded CSV)
# Nifty ID is 13, Sensex is 51. Segment is IDX_I (Index).
INDEX_MAP = {
    'NIFTY':     {'id': '13', 'segment': 'IDX_I', 'name': 'NIFTY 50'},
    'SENSEX':    {'id': '51', 'segment': 'IDX_I', 'name': 'SENSEX'},
    'BANKNIFTY': {'id': '25', 'segment': 'IDX_I', 'name': 'BANK NIFTY'},
    'FINNIFTY':  {'id': '27', 'segment': 'IDX_I', 'name': 'FINNIFTY'}
}

# ==========================================
# 2. API FUNCTIONS (Following Docs)
# ==========================================

@st.cache_data(ttl=300)
def fetch_expiry_list(client_id, access_token, security_id, segment):
    """
    Implements: https://dhanhq.co/docs/v2/option-chain/#expiry-list
    Fetches valid dates so we don't send invalid requests.
    """
    try:
        dhan = dhanhq(client_id, access_token)
        response = dhan.expiry_list(
            under_security_id=security_id,
            under_exchange_segment=segment
        )
        if response['status'] == 'success':
            return response['data'] # Returns list of dates
        else:
            st.error(f"Expiry Fetch Failed: {response}")
            return []
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return []

@st.cache_data(ttl=15)
def fetch_option_chain(client_id, access_token, security_id, segment, expiry):
    """
    Implements: https://dhanhq.co/docs/v2/option-chain/
    """
    try:
        dhan = dhanhq(client_id, access_token)
        response = dhan.option_chain(
            under_security_id=security_id,
            under_exchange_segment=segment,
            expiry=expiry
        )
        if response['status'] == 'success':
            return response['data']
        else:
            return response # Return error dict
    except Exception as e:
        return {"status": "failure", "remarks": {"error_message": str(e)}}

def process_data(data, spot_price=None):
    # Handle v2 response structure (sometimes data is nested)
    if isinstance(data, dict) and 'oc' in data:
        # If API returns {last_price: x, oc: {...}}
        spot_price = data.get('last_price', spot_price)
        chain_data = data.get('oc', {})
        if isinstance(chain_data, dict):
            chain_data = chain_data.get('data', [])
        df = pd.DataFrame(chain_data)
    else:
        # Flat structure
        df = pd.DataFrame(data)
    
    if df.empty: return None, spot_price

    df.columns = [c.lower() for c in df.columns]
    if 'option_type' not in df.columns: return None, spot_price

    ce = df[df['option_type'] == 'CALL'].copy()
    pe = df[df['option_type'] == 'PUT'].copy()
    
    chain = pd.merge(ce, pe, on='strike_price', suffixes=('_ce', '_pe'))
    chain = chain.sort_values('strike_price')
    
    # Filter Strikes around ATM
    if spot_price:
        atm_idx = (chain['strike_price'] - spot_price).abs().idxmin()
        start = max(0, atm_idx - 8)
        end = min(len(chain), atm_idx + 9)
        chain = chain.iloc[start:end]
    
    return chain, spot_price

# ==========================================
# 3. APP INTERFACE
# ==========================================

# Login
try:
    CLIENT_ID = st.secrets["dhan"]["client_id"]
    ACCESS_TOKEN = st.secrets["dhan"]["access_token"]
except:
    st.error("❌ Secrets not found. Check `.streamlit/secrets.toml`")
    st.stop()

st.sidebar.header("⚙️ Settings")
idx_key = st.sidebar.radio("Select Index", list(INDEX_MAP.keys()))
config = INDEX_MAP[idx_key]

# --- STEP 1: Get Valid Expiry Dates (The "Fix") ---
# This prevents Error 814 by asking the API "What dates do you have?" first.
valid_dates = fetch_expiry_list(CLIENT_ID, ACCESS_TOKEN, config['id'], config['segment'])

if valid_dates:
    expiry_date = st.sidebar.selectbox("Select Expiry", valid_dates)
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.title(f"📊 {config['name']} Analysis")

    # --- STEP 2: Get Option Chain ---
    resp_data = fetch_option_chain(CLIENT_ID, ACCESS_TOKEN, config['id'], config['segment'], expiry_date)

    # Check for errors
    if isinstance(resp_data, dict) and resp_data.get('status') == 'failure':
        st.error(f"API Error: {resp_data.get('remarks', resp_data)}")
        st.write("Debug Info:", config, expiry_date)
    
    elif resp_data:
        # Processing
        view_df, spot = process_data(resp_data)
        
        if view_df is not None:
            # Display Spot if found
            if spot: st.metric("Underlying Spot Price", spot)

            # Chart
            fig = go.Figure()
            fig.add_trace(go.Bar(x=view_df['strike_price'], y=view_df['oi_ce'], name='Call OI', marker_color='#ff4b4b'))
            fig.add_trace(go.Bar(x=view_df['strike_price'], y=view_df['oi_pe'], name='Put OI', marker_color='#00cc96'))
            if spot:
                fig.add_vline(x=spot, line_dash="dash", line_color="yellow", annotation_text="Spot")
            st.plotly_chart(fig, use_container_width=True)

            # Table
            cols = ['oi_ce', 'last_price_ce', 'strike_price', 'last_price_pe', 'oi_pe']
            st.dataframe(view_df[[c for c in cols if c in view_df.columns]], hide_index=True, use_container_width=True)
        else:
            st.warning("Data fetched but could not be processed. (Empty chain?)")
else:
    st.warning("Could not fetch expiry dates. Check if Market is Open or API Key is valid.")
