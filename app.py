import streamlit as st
import pandas as pd
from dhanhq import dhanhq
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. SETUP
# ==========================================
st.set_page_config(page_title="Dhan Option Chain (2026 Rules)", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .metric-card { background-color: #0e1117; border: 1px solid #262730; border-radius: 5px; padding: 15px; text-align: center; }
    .bullish { color: #00FF00; font-weight: bold; }
    .bearish { color: #FF0000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INTELLIGENT HELPERS
# ==========================================

def get_next_expiry(target_weekday):
    """
    Finds the next date for a specific weekday.
    0=Mon, 1=Tue (NIFTY), 2=Wed, 3=Thu (SENSEX), 4=Fri, 5=Sat, 6=Sun
    """
    today = datetime.now().date()
    days_ahead = target_weekday - today.weekday()
    if days_ahead <= 0: 
        days_ahead += 7
    return today + timedelta(days=days_ahead)

@st.cache_data(ttl=60)
def fetch_dhan_data(client_id, access_token, under_segment, under_id, expiry_str):
    try:
        dhan = dhanhq(client_id, access_token)
        # CRITICAL FIX: Using 'IDX_I' for Indices + New Parameter Names
        response = dhan.option_chain(
            under_exchange_segment=under_segment, 
            under_security_id=under_id,            
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
    
    # Normalize columns (handle case sensitivity)
    df.columns = [c.lower() for c in df.columns]
    
    # Ensure critical columns exist
    if 'option_type' not in df.columns: return None

    ce = df[df['option_type'] == 'CALL'].copy()
    pe = df[df['option_type'] == 'PUT'].copy()
    
    # Merge on Strike
    chain = pd.merge(ce, pe, on='strike_price', suffixes=('_ce', '_pe'))
    chain = chain.sort_values('strike_price')
    
    # Filter 8 Strikes around ATM
    atm_idx = (chain['strike_price'] - spot).abs().idxmin()
    start = max(0, atm_idx - 8)
    end = min(len(chain), atm_idx + 9)
    
    return chain.iloc[start:end], chain

# ==========================================
# 3. SIDEBAR & CONFIG
# ==========================================

try:
    CLIENT_ID = st.secrets["dhan"]["client_id"]
    ACCESS_TOKEN = st.secrets["dhan"]["access_token"]
except:
    st.error("❌ Secrets not found. Check `.streamlit/secrets.toml`")
    st.stop()

st.sidebar.header("⚙️ Configuration")
index_choice = st.sidebar.radio("Select Index", ["NIFTY 50", "SENSEX"])

# --- 2026 REGULATORY CONFIGURATION ---
# Effective Sep 2025: Nifty = TUESDAY, Sensex = THURSDAY
if index_choice == "NIFTY 50":
    sec_id = "13"
    segment = "IDX_I"  # Indices are ALWAYS 'IDX_I'
    spot_default = 25900 # Based on your screenshot
    target_day = 1 # Tuesday
elif index_choice == "SENSEX":
    sec_id = "51005"
    segment = "IDX_I" # Indices are ALWAYS 'IDX_I'
    spot_default = 84000
    target_day = 3 # Thursday

# Auto-calculate the correct date
suggested_date = get_next_expiry(target_day)
expiry = st.sidebar.date_input(f"Expiry ({suggested_date.strftime('%A')})", suggested_date)

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 4. DASHBOARD
# ==========================================
st.title(f"📊 {index_choice} Analyzer")

spot = st.sidebar.number_input("Spot Price", value=spot_default)

data_resp = fetch_dhan_data(CLIENT_ID, ACCESS_TOKEN, segment, sec_id, str(expiry))

if isinstance(data_resp, pd.DataFrame):
    view_df, full_df = process_data(data_resp, spot)
    
    if view_df is not None:
        # Stats
        ce_oi = full_df['oi_ce'].sum()
        pe_oi = full_df['oi_pe'].sum()
        pcr = round(pe_oi / ce_oi, 2) if ce_oi > 0 else 0
        
        sup = full_df.loc[full_df['oi_pe'].idxmax(), 'strike_price']
        res = full_df.loc[full_df['oi_ce'].idxmax(), 'strike_price']
        
        # Simple Sentiment Logic
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
        st.subheader("OI Structure")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=view_df['strike_price'], y=view_df['oi_ce'], name='Call OI', marker_color='#ff4b4b'))
        fig.add_trace(go.Bar(x=view_df['strike_price'], y=view_df['oi_pe'], name='Put OI', marker_color='#00cc96'))
        fig.add_vline(x=spot, line_dash="dash", line_color="yellow", annotation_text="Spot")
        fig.update_layout(xaxis_title="Strike", yaxis_title="OI", barmode='group', height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.write("### Data Table")
        # Ensure columns exist before display
        cols = ['oi_ce', 'last_price_ce', 'strike_price', 'last_price_pe', 'oi_pe']
        safe_cols = [c for c in cols if c in view_df.columns]
        st.dataframe(view_df[safe_cols], hide_index=True)

    else:
        st.error("Data fetched but processing failed. Structure might be unexpected.")

elif isinstance(data_resp, dict):
    st.error(f"⚠️ **API Error:** {data_resp.get('error', 'Unknown')}")
    st.info(f"""
    **Why am I seeing Error 814?**
    1. **Date:** For Nifty, ensure the date is a **Tuesday** (e.g., Jan 13, 2026).
    2. **Segment:** The code now correctly uses `IDX_I`. If it still fails, the date is the likely culprit.
    """)
