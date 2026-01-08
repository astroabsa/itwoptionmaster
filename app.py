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

# Custom CSS for visual tags
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
    Fetches Option Chain data using NEW DhanHQ v2.0 parameters.
    """
    try:
        dhan = dhanhq(client_id, access_token)
        
        # --- FIX 1: Updated Parameter Names for v2.0+ ---
        response = dhan.option_chain(
            under_exchange_segment=under_segment,  # WAS: exchange_segment
            under_security_id=under_id,            # WAS: security_id
            expiry=expiry                          # WAS: expiry_date
        )
        
        if response['status'] == 'success':
            return pd.DataFrame(response['data'])
        else:
            st.error(f"API Error: {response}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

def process_chain_data(df, spot_price):
    """Processes the raw dataframe into analysis-ready format."""
    if df is None or df.empty: return None

    # Normalize column names to lowercase to handle API variations
    df.columns = [c.lower() for c in df.columns]

    # Verify expected columns exist
    if 'option_type' not in df.columns:
        st.error("Data received but 'option_type' column is missing. API structure may have changed.")
        return None

    ce_df = df[df['option_type'] == 'CALL'].copy()
    pe_df = df[df['option_type'] == 'PUT'].copy()
    
    # Merge Call and Put data on Strike Price
    chain = pd.merge(ce_df, pe_df, on='strike_price', suffixes=('_CE', '_PE'))
    chain = chain.sort_values('strike_price')
    
    # Filter Strikes: Keep only 10 strikes above/below ATM
    atm_idx = (chain['strike_price'] - spot_price).abs().idxmin()
    start_idx = max(0, atm_idx - 10)
    end_idx = min(len(chain), atm_idx + 10)
    
    return chain.iloc[start_idx:end_idx], chain

def calculate_stats(full_chain_df):
    """Calculates PCR and identifies Max OI Support/Resistance."""
    total_ce_oi = full_chain_df['oi_ce'].sum()
    total_pe_oi = full_chain_df['oi_pe'].sum()
    
    pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0
    
    # Find Strike with Max Call OI (Resistance) and Max Put OI (Support)
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
# 3. SECRETS & CONFIGURATION
# ==========================================

# Load Secrets
try:
    CLIENT_ID = st.secrets["dhan"]["client_id"]
    ACCESS_TOKEN = st.secrets["dhan"]["access_token"]
except Exception:
    st.error("❌ Secrets not found. Please create `.streamlit/secrets.toml` with a [dhan] section.")
    st.stop()

# Sidebar Settings
st.sidebar.header("⚙️ Settings")
market = st.sidebar.radio("Select Index", ["NIFTY", "SENSEX"])

# --- FIX 2: Correct Segment is IDX_I for Index Options ---
sec_map = {
    "NIFTY":  {"id": "13",    "segment": "IDX_I", "spot": 23500}, 
    "SENSEX": {"id": "51005", "segment": "IDX_I", "spot": 77000}
}
# Note: 'IDX_I' tells Dhan to fetch options for the UNDERLYING Index.

sel_sec = sec_map[market]
expiry = st.sidebar.date_input("Expiry Date", datetime.now())

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 4. DASHBOARD RENDER
# ==========================================
st.title(f"📊 {market} Option Chain")

# Simulated Spot Price Input (In production, fetch this via API)
spot = st.sidebar.number_input("Current Spot Price", value=sel_sec["spot"])

# Fetch Data
data = fetch_dhan_data(CLIENT_ID, ACCESS_TOKEN, sel_sec['segment'], sel_sec['id'], str(expiry))

if data is not None:
    view_df, full_df = process_chain_data(data, spot)
    
    if view_df is not None:
        pcr, sup, res, sent, col = calculate_stats(full_df)

        # 1. KPI Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PCR", pcr)
        c2.metric("Support (Max PE OI)", sup)
        c3.metric("Resistance (Max CE OI)", res)
        c4.markdown(f"**Sentiment**<br><span class='{col}'>{sent}</span>", unsafe_allow_html=True)

        st.divider()

        # 2. Open Interest Chart
        st.subheader("Open Interest Structure")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=view_df['strike_price'], y=view_df['oi_ce'], name='Call OI (Res)', marker_color='#FF4B4B'))
        fig.add_trace(go.Bar(x=view_df['strike_price'], y=view_df['oi_pe'], name='Put OI (Sup)', marker_color='#00CC96'))
        fig.add_vline(x=spot, line_dash="dash", line_color="yellow", annotation_text="Spot")
        
        fig.update_layout(xaxis_title="Strike Price", yaxis_title="Open Interest", barmode='group', height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. Data Table
        st.subheader("Deep Dive Data")
        # Display simplified table
        table_cols = ['oi_ce', 'last_price_ce', 'strike_price', 'last_price_pe', 'oi_pe']
        st.dataframe(view_df[table_cols], hide_index=True, use_container_width=True)

    else:
        st.warning("Data fetched but empty. Check if Expiry Date matches active contracts.")
