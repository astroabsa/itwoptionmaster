import streamlit as st
import pandas as pd
from dhanhq import dhanhq
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. SETUP
# ==========================================
st.set_page_config(page_title="Dhan Diagnostic Tool", page_icon="🛠️", layout="wide")

st.markdown("""
<style>
    .success-box { padding: 10px; background-color: #d4edda; color: #155724; border-radius: 5px; border: 1px solid #c3e6cb; }
    .error-box { padding: 10px; background-color: #f8d7da; color: #721c24; border-radius: 5px; border: 1px solid #f5c6cb; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DIAGNOSTIC FUNCTIONS
# ==========================================

def test_connection(client_id, access_token, segment, security_id, expiry_date):
    """
    Tries to fetch data with a specific combination of parameters.
    Returns (Success_Bool, Message/Data)
    """
    try:
        dhan = dhanhq(client_id, access_token)
        # Test 1: Option Chain Fetch
        response = dhan.option_chain(
            under_exchange_segment=segment,
            under_security_id=security_id,
            expiry=str(expiry_date)
        )
        if response['status'] == 'success':
            return True, response['data']
        else:
            return False, response # Return error dict
    except Exception as e:
        return False, str(e)

def process_data(data, spot_price):
    """Simple processor for the data."""
    if not data: return None
    # Handle v2 nested structure
    chain_list = data.get('oc', data)
    if isinstance(chain_list, dict) and 'data' in chain_list:
        chain_list = chain_list['data']
    
    df = pd.DataFrame(chain_list)
    df.columns = [c.lower() for c in df.columns]
    
    if 'option_type' not in df.columns: return None
    
    ce = df[df['option_type'] == 'CALL'].copy()
    pe = df[df['option_type'] == 'PUT'].copy()
    chain = pd.merge(ce, pe, on='strike_price', suffixes=('_ce', '_pe'))
    chain = chain.sort_values('strike_price')
    
    # Filter ATM
    atm_idx = (chain['strike_price'] - spot_price).abs().idxmin()
    return chain.iloc[max(0, atm_idx-8):min(len(chain), atm_idx+9)]

# ==========================================
# 3. MAIN APP
# ==========================================

# Secrets Load
try:
    CLIENT_ID = st.secrets["dhan"]["client_id"]
    ACCESS_TOKEN = st.secrets["dhan"]["access_token"]
except:
    st.error("❌ Secrets not found.")
    st.stop()

st.title("🛠️ Dhan API Auto-Diagnostics")
st.write("This tool will test multiple connection methods to find the one that works for your account.")

c1, c2, c3 = st.columns(3)
with c1:
    idx_name = st.selectbox("Index", ["NIFTY", "SENSEX", "BANKNIFTY"])
with c2:
    # Manual date picker to ensure we match the screenshot exactly
    # Defaulting to the date seen in your screenshot: Jan 13, 2026
    default_date = datetime(2026, 1, 13)
    expiry_input = st.date_input("Expiry Date (Match your Dhan App)", default_date)
with c3:
    run_test = st.button("🚀 Run Connection Test", type="primary")

# Configuration Map
config = {
    "NIFTY": {"id": "13", "spot": 26000},
    "SENSEX": {"id": "51", "spot": 84000},
    "BANKNIFTY": {"id": "25", "spot": 54000}
}
current_config = config[idx_name]

if run_test:
    st.divider()
    st.subheader(f"Testing Connectivity for {idx_name} on {expiry_input}")
    
    # SEGMENTS TO TEST
    # We suspect IDX_I is standard, but NSE/NSE_FNO might be required for your specific setup
    segments_to_test = ["IDX_I", "NSE", "NSE_FNO", "BSE"]
    
    success_found = False
    valid_data = None
    working_segment = None

    # --- THE LOOP ---
    for seg in segments_to_test:
        with st.status(f"Testing Segment: **{seg}**...", expanded=False) as status:
            is_ok, result = test_connection(CLIENT_ID, ACCESS_TOKEN, seg, current_config['id'], expiry_input)
            
            if is_ok:
                status.update(label=f"✅ Segment **{seg}** : SUCCESS!", state="complete")
                st.markdown(f"<div class='success-box'><b>Success!</b> The API accepted segment <code>{seg}</code>.</div>", unsafe_allow_html=True)
                success_found = True
                working_segment = seg
                valid_data = result
                break # Stop testing if one works
            else:
                status.update(label=f"❌ Segment **{seg}** : FAILED ({result.get('data', {}).get('data', 'Unknown Error')})", state="error")
    
    # --- RESULT DISPLAY ---
    st.divider()
    if success_found and valid_data:
        st.success(f"🎉 Connection Established using Segment: {working_segment}")
        
        # Extract Spot
        api_spot = valid_data.get('last_price', current_config['spot'])
        st.metric("Live Spot Price", api_spot)
        
        # Show Data
        df_view = process_data(valid_data, api_spot)
        if df_view is not None:
            st.write("### Option Chain Data Preview")
            st.dataframe(df_view[['oi_ce', 'last_price_ce', 'strike_price', 'last_price_pe', 'oi_pe']], hide_index=True)
            
            # Simple Chart
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_view['strike_price'], y=df_view['oi_ce'], name='Call OI', marker_color='red'))
            fig.add_trace(go.Bar(x=df_view['strike_price'], y=df_view['oi_pe'], name='Put OI', marker_color='green'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Connection succeeded, but data processing failed. (Check dataframe columns)")
            
    else:
        st.error("❌ All connection attempts failed.")
        st.write("### Troubleshooting Steps:")
        st.markdown(f"""
        1. **Check Expiry Date:** You selected **{expiry_input}**.
           - Open your Dhan App. Does NIFTY expire on this *exact* date?
           - If it is a holiday, the date might be Jan 12 or 14.
        2. **Check ID:** We used ID `{current_config['id']}`.
           - This is the standard ID. If Dhan changed IDs for 2026, you may need to check their Scrip Master CSV.
        """)
