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

# Custom CSS
st.markdown("""
<style>
    .metric-card { background-color: #0e1117; border: 1px solid #262730; border-radius: 5px; padding: 15px; text-align: center; }
    .bullish { color: #00FF00; font-weight: bold; }
    .bearish { color: #FF0000; font-weight: bold; }
    .neutral { color: #FFFF00; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

@st.cache_data(ttl=60)
def fetch_dhan_data(client_id, access_token, under_segment, under_id, expiry):
    """
    Fetches Option Chain data.
    """
    try:
        dhan = dhanhq(client_id, access_token)
        
        # Calling API with corrected parameters
        response = dhan.option_chain(
            under_exchange_segment=under_segment, 
            under_security_id=under_id,            
            expiry=expiry                          
        )
        
        if response['status'] == 'success':
            return pd.DataFrame(response['data'])
        else:
            return response # Return the error dict to display it
            
    except Exception as e:
        return {"error": str(e)}

def process_chain_data(df, spot_price):
    """Processes the raw dataframe."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty: return None

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    if 'option_type' not in df.columns:
        return None

    ce_df = df[df['option_type'] == 'CALL'].copy()
    pe_df = df[df['option_type'] == 'PUT'].copy()
    
    chain = pd.merge(ce_df, pe_df, on='strike_price', suffixes=('_CE', '_PE'))
    chain = chain.sort_values('strike_price')
    
    # Filter 10 strikes around ATM
    atm_idx = (chain['strike_price'] - spot_price).abs().idxmin()
    start_idx = max(0, atm_idx - 10)
    end_idx = min(len(chain), atm_idx + 10)
    
    return chain.iloc[start_idx:end_idx], chain

def calculate_stats(full_chain_df):
    """Calculates PCR and Max OI."""
    total_ce_oi = full_chain_df['oi_ce'].sum()
    total_pe_oi = full_chain_df['oi_pe'].sum()
    
    pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0
    
    res_strike = full_chain_df.loc[full_chain_df['oi_ce'].idxmax(), 'strike_price']
    sup_strike = full_chain_df.loc[full_chain_df['oi_pe'].idxmax(), 'strike_price']
    
    sentiment = "NEUTRAL"
    color = "neutral"
    
    if pcr >= 1.2:
        sentiment = "BULLISH"
        color = "bullish"
    elif pcr <= 0.6:
        sentiment = "BEARISH"
        color = "bearish"
        
    return pcr, sup, res, sentiment, color

# ==========================================
# 3. SECRETS & SETUP
# ==========================================

try:
    CLIENT_ID = st.secrets["dhan"]["client_id"]
    ACCESS_TOKEN = st.secrets["dhan"]["access_token"]
except Exception:
    st.error("❌ Secrets not found in `.streamlit/secrets.toml`")
    st.stop()

st.sidebar.header("⚙️ Settings")
market = st.sidebar.radio("Index", ["NIFTY", "SENSEX"])

# --- FIX: CORRECTED SEGMENTS (NSE/BSE) ---
sec_map = {
    "NIFTY":  {"id": "13",    "segment": "NSE", "spot": 23500},  # Segment is NSE for Nifty Underlying
    "SENSEX": {"id": "51005", "segment": "BSE", "spot": 77000}   # Segment is BSE for Sensex Underlying
}

sel_sec = sec_map[market]

# Date Picker
# Note: Users MUST pick the valid expiry date (e.g., Thursday for Nifty)
expiry = st.sidebar.date_input("Expiry Date (Must be exact)", datetime.now())

if st.sidebar.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 4. DASHBOARD
# ==========================================
st.title(f"📊 {market} Option Chain")

spot = st.sidebar.number_input("Spot Price", value=sel_sec["spot"])

data_response = fetch_dhan_data(CLIENT_ID, ACCESS_TOKEN, sel_sec['segment'], sel_sec['id'], str(expiry))

# Check if response is a DataFrame (Success) or Dict (Error)
if isinstance(data_response, pd.DataFrame):
    view_df, full_df = process_chain_data(data_response, spot)
    
    if view_df is not None:
        pcr, sup, res, sent, col = calculate_stats(full_df)

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
        st.write("### Data Table")
        st.dataframe(view_df[['oi_ce', 'last_price_ce', 'strike_price', 'last_price_pe', 'oi_pe']], hide_index=True)
    else:
        st.error("Data processing failed. The API response format might be unexpected.")

elif isinstance(data_response, dict):
    # Display friendly error
    st.error("⚠️ **API Request Failed**")
    st.write(f"**Error Details:** {data_response}")
    st.info(f"💡 **Tip:** Ensure '{expiry}' is a valid expiry date for {market}. If it is a holiday or non-expiry day, the API returns Error 814.")
else:
    st.warning("No data returned.")
