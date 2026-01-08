import streamlit as st
import pandas as pd
import numpy as np
from dhanhq import dhanhq
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Dhan Option Chain Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
    }
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
    """Fetches and cleans Option Chain data."""
    try:
        dhan = dhanhq(client_id, access_token)
        
        # CORRECTED API CALL --------------------------------------
        response = dhan.option_chain(
            under_exchange_segment=under_segment,  # Changed from exchange_segment
            under_security_id=under_id,            # Changed from security_id
            expiry=expiry                          # Changed from expiry_date
        )
        # ---------------------------------------------------------
        
        if response['status'] == 'success':
            return pd.DataFrame(response['data'])
        else:
            st.error(f"API Error: {response}")
            return None
    except Exception as e:
        st.error(f"Connection Failed: {e}")
        return None

def process_chain_data(df, spot_price):
    """Processes the raw dataframe into analysis ready format."""
    if df is None or df.empty: return None

    # Using 'oc' key if data comes in nested format, otherwise direct frame
    # Dhan V2 often returns a nested structure in 'data', we might need to normalize it
    # But often the dataframe conversion handles flat lists. 
    # If the dataframe is empty, we return None.
    
    # Filter for Calls and Puts based on column names
    # Note: Ensure your API response actually has these columns. 
    # Sometimes keys are lowercase 'call' / 'put' or 'CE' / 'PE'
    
    # Standardize column names if needed (Optional, depends on exact API response structure)
    # df.columns = [c.lower() for c in df.columns]

    ce_df = df[df['option_type'] == 'CALL'].copy()
    pe_df = df[df['option_type'] == 'PUT'].copy()
    
    chain = pd.merge(ce_df, pe_df, on='strike_price', suffixes=('_CE', '_PE'))
    chain = chain.sort_values('strike_price')
    
    # Filter Strikes: Keep only 10 strikes above and below ATM
    atm_idx = (chain['strike_price'] - spot_price).abs().idxmin()
    start_idx = max(0, atm_idx - 10)
    end_idx = min(len(chain), atm_idx + 10)
    filtered_chain = chain.iloc[start_idx:end_idx]
    
    return filtered_chain, chain

def calculate_market_stats(full_chain_df, spot_price):
    """Calculates PCR, Support, Resistance, and Sentiment."""
    total_ce_oi = full_chain_df['oi_CE'].sum()
    total_pe_oi = full_chain_df['oi_PE'].sum()
    
    pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0
    
    resistance_strike = full_chain_df.loc[full_chain_df['oi_CE'].idxmax(), 'strike_price']
    support_strike = full_chain_df.loc[full_chain_df['oi_PE'].idxmax(), 'strike_price']
    
    sentiment = "Neutral"
    color = "neutral"
    
    if pcr >= 1.2:
        sentiment = "BULLISH (Strong Support)"
        color = "bullish"
    elif pcr <= 0.6:
        sentiment = "BEARISH (Strong Resistance)"
        color = "bearish"
    elif 0.8 <= pcr <= 1.1:
        sentiment = "SIDEWAYS / CONSOLIDATION"
    else:
        sentiment = "MILDLY BULLISH/BEARISH"
        
    return {
        "pcr": pcr,
        "support": support_strike,
        "resistance": resistance_strike,
        "sentiment": sentiment,
        "color": color,
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi
    }

# ==========================================
# 3. CONFIGURATION & SECRETS LOADING
# ==========================================

try:
    CLIENT_ID = st.secrets["dhan"]["client_id"]
    ACCESS_TOKEN = st.secrets["dhan"]["access_token"]
except FileNotFoundError:
    st.error("❌ `secrets.toml` not found. Please create it in the `.streamlit` folder.")
    st.stop()
except KeyError:
    st.error("❌ Missing keys in `secrets.toml`.")
    st.stop()

# Sidebar Config
st.sidebar.header("⚙️ Market Settings")
market_type = st.sidebar.radio("Select Index", ["SENSEX", "NIFTY"])

# CORRECTED SECURITY MAPPING ----------------------------------
# Note: For Option Chain, we usually pass the UNDERLYING segment.
# For Indices (Nifty/Sensex), the segment is usually 'IDX_I'.
security_map = {
    "SENSEX": {"id": "51005", "segment": "IDX_I", "spot": 84331}, 
    "NIFTY": {"id": "13", "segment": "IDX_I", "spot": 21500}      
}
# -------------------------------------------------------------

expiry_date = st.sidebar.date_input("Expiry Date", datetime.now())
selected_security = security_map[market_type]

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
st.title(f"📊 {market_type} Option Chain Analyzer")

spot_price = st.sidebar.number_input("Spot Price (Live/Simulated)", value=selected_security["spot"])

raw_df = fetch_dhan_data(
    CLIENT_ID, 
    ACCESS_TOKEN, 
    selected_security['segment'], # Now passes 'IDX_I'
    selected_security['id'], 
    str(expiry_date)
)

if raw_df is not None:
    view_df, full_df = process_chain_data(raw_df, spot_price)
    
    if view_df is not None:
        stats = calculate_market_stats(full_df, spot_price)

        # 2. KPI METRICS
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Put Call Ratio (PCR)", f"{stats['pcr']}")
        with c2: st.metric("Support (Max Put OI)", f"{stats['support']}")
        with c3: st.metric("Resistance (Max Call OI)", f"{stats['resistance']}")
        with c4: 
            st.markdown("**Market Sentiment**")
            st.markdown(f"<h4 class='{stats['color']}'>{stats['sentiment']}</h4>", unsafe_allow_html=True)

        st.divider()

        # 3. INTERACTIVE OI CHART
        st.subheader("Open Interest Structure")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=view_df['strike_price'], y=view_df['oi_CE'], name='Call OI (Res)', marker_color='#FF4B4B'))
        fig.add_trace(go.Bar(x=view_df['strike_price'], y=view_df['oi_PE'], name='Put OI (Sup)', marker_color='#00CC96'))
        fig.add_vline(x=spot_price, line_dash="dash", line_color="yellow", annotation_text="Spot")

        fig.update_layout(
            barmode='group', 
            xaxis_title='Strike Price', 
            yaxis_title='Open Interest',
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # 4. HEATMAP TABLE
        st.subheader("Option Chain Data")
        table_view = view_df[['oi_CE', 'last_price_CE', 'strike_price', 'last_price_PE', 'oi_PE']].copy()
        table_view.columns = ['Call OI', 'Call LTP', 'STRIKE', 'Put LTP', 'Put OI']
        
        st.dataframe(
            table_view.style.background_gradient(subset=['Call OI'], cmap='Reds')
                             .background_gradient(subset=['Put OI'], cmap='Greens')
                             .format("{:.2f}", subset=['Call LTP', 'Put LTP'])
                             .format("{:.0f}", subset=['Call OI', 'Put OI']),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.error("Data processed but result is empty. Check Expiry Date match.")
else:
    st.info("No data available. Check if the market is open or Expiry Date is correct.")
