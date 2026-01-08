import streamlit as st
from dhanhq import dhanhq
import pandas as pd
import pandas_ta as ta
import pytz
from datetime import datetime, timedelta
import time
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="iTW's Live F&O Screener Pro", layout="wide")
IST = pytz.timezone('Asia/Kolkata') # Force IST Timezone

# --- 2. AUTHENTICATION ---
AUTH_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSEan21a9IVnkdmTFP2Q9O_ILI3waF52lFWQ5RTDtXDZ5MI4_yTQgFYcCXN5HxgkCxuESi5Dwe9iROB/pub?gid=0&single=true&output=csv"

def authenticate_user(user_in, pw_in):
    try:
        df = pd.read_csv(AUTH_CSV_URL)
        df['username'] = df['username'].astype(str).str.strip().str.lower()
        df['password'] = df['password'].astype(str).str.strip()
        match = df[(df['username'] == str(user_in).strip().lower()) & (df['password'] == str(pw_in).strip())]
        return not match.empty
    except: return False

if "authenticated" not in st.session_state: st.session_state["authenticated"] = False
if not st.session_state["authenticated"]:
    st.title("🔐 iTW's F&O Pro Login")
    with st.form("login_form"):
        u = st.text_input("Username"); p = st.text_input("Password", type="password")
        if st.form_submit_button("Log In"):
            if authenticate_user(u, p): st.session_state["authenticated"] = True; st.rerun()
            else: st.error("Invalid Credentials")
    st.stop()

# --- 3. MAIN UI ---
st.title("🚀 iTW's Live F&O Screener Pro")
if st.sidebar.button("Log out"): st.session_state["authenticated"] = False; st.rerun()

# --- 4. API CONNECTION ---
dhan = None
try:
    client_id = st.secrets["DHAN_CLIENT_ID"]
    access_token = st.secrets["DHAN_ACCESS_TOKEN"]
    dhan = dhanhq(client_id, access_token)
except Exception as e: st.error(f"API Error: {e}"); st.stop()

# --- 5. INDEX MAP (Standardized) ---
INDEX_MAP = {
    'NIFTY': {'id': '13', 'name': 'NIFTY 50'}, 
    'BANKNIFTY': {'id': '25', 'name': 'BANK NIFTY'}, 
    'SENSEX': {'id': '51', 'name': 'SENSEX'}
}

# --- 6. MASTER LIST LOADER ---
@st.cache_data(ttl=3600*4)
def get_fno_stock_map():
    fno_map = {}
    if not os.path.exists("dhan_master.csv"):
        st.error("❌ 'dhan_master.csv' NOT FOUND."); return fno_map 

    try:
        df = pd.read_csv("dhan_master.csv", on_bad_lines='skip', low_memory=False)
        df.columns = df.columns.str.strip() 
        
        col_exch = 'SEM_EXM_EXCH_ID'
        col_id = 'SEM_SMST_SECURITY_ID'
        col_name = 'SEM_TRADING_SYMBOL'
        col_inst = 'SEM_INSTRUMENT_NAME'
        col_expiry = 'SEM_EXPIRY_DATE'
        
        if col_name in df.columns: df[col_name] = df[col_name].astype(str).str.upper().str.strip()
        if col_exch in df.columns: df[col_exch] = df[col_exch].astype(str).str.strip()
        if col_inst in df.columns: df[col_inst] = df[col_inst].astype(str).str.strip()
        
        if col_exch in df.columns and col_inst in df.columns:
            stk_df = df[(df[col_exch] == 'NSE') & (df[col_inst] == 'FUTSTK')].copy()
            
            if col_expiry in stk_df.columns:
                stk_df[col_expiry] = stk_df[col_expiry].astype(str)
                stk_df['dt_parsed'] = pd.to_datetime(stk_df[col_expiry], dayfirst=True, errors='coerce')
                
                today = pd.Timestamp.now().normalize()
                valid_futures = stk_df[stk_df['dt_parsed'] >= today]
                valid_futures = valid_futures.sort_values(by=[col_name, 'dt_parsed'])
                curr_stk = valid_futures.drop_duplicates(subset=[col_name], keep='first')
                
                for _, row in curr_stk.iterrows():
                    base_sym = row[col_name].split('-')[0]
                    disp_name = row.get('SEM_CUSTOM_SYMBOL', row[col_name])
                    fno_map[base_sym] = {'id': str(row[col_id]), 'name': disp_name}
    except Exception as e: st.error(f"Error reading CSV: {e}")
    return fno_map

with st.spinner("Loading Stock List..."):
    FNO_MAP = get_fno_stock_map()

# --- 7. HELPER: GET YESTERDAY'S CLOSE (The "Simple" Way) ---
def get_prev_close(security_id):
    try:
        # Fetch last 10 days of DAILY candles. The API handles the "Daily" logic.
        to_d = datetime.now(IST).strftime('%Y-%m-%d')
        from_d = (datetime.now(IST) - timedelta(days=10)).strftime('%Y-%m-%d')
        
        # IDX_I works for all indices (NSE & BSE)
        res = dhan.historical_daily_data(str(security_id), "IDX_I", "INDEX", from_d, to_d)
        
        if res['status'] == 'success' and 'data' in res:
            df = pd.DataFrame(res['data'])
            if df.empty: return 0.0
            
            # Helper to get date string
            time_col = 'start_Time' if 'start_Time' in df.columns else 'timestamp'
            df['date_str'] = df[time_col].astype(str).str[:10]
            
            # Filter out "Today" to find the last completed day
            today_str = datetime.now(IST).strftime('%Y-%m-%d')
            past_df = df[df['date_str'] != today_str]
            
            if not past_df.empty:
                return float(past_df.iloc[-1]['close'])
                
    except: pass
    return 0.0

# --- 8. HELPER: GET LIVE PRICE (The "Simple" Way) ---
def get_live_price(security_id):
    try:
        # Intraday chart for just today/yesterday gives the absolute latest price
        to_d = datetime.now(IST).strftime('%Y-%m-%d')
        from_d = (datetime.now(IST) - timedelta(days=3)).strftime('%Y-%m-%d')
        
        res = dhan.intraday_minute_data(str(security_id), "IDX_I", "INDEX", from_d, to_d, 1)
        
        if res['status'] == 'success' and 'data' in res:
            closes = res['data']['close']
            if len(closes) > 0:
                return float(closes[-1])
    except: pass
    return 0.0

# --- 9. DASHBOARD ---
@st.fragment(run_every=5)
def refreshable_dashboard():
    data = {}
    
    for key, info in INDEX_MAP.items():
        sid = info['id']
        
        # 1. Direct API call for Yesterday
        prev = get_prev_close(sid)
        
        # 2. Direct API call for Today
        ltp = get_live_price(sid)
        
        # 3. Simple Math
        if ltp == 0: ltp = prev # Fallback if live fetch fails
        if prev == 0: prev = ltp # Fallback if history fetch fails
        
        chg = 0.0
        pct = 0.0
        
        if prev > 0:
            chg = ltp - prev
            pct = (chg / prev) * 100
            
        data[info['name']] = {"ltp": ltp, "chg": chg, "pct": pct}

    c1, c2, c3, c4 = st.columns([1,1,1,1.2])
    with c1: d=data.get("NIFTY 50"); st.metric("NIFTY 50", f"{d['ltp']:,.2f}", f"{d['chg']:.2f} ({d['pct']:.2f}%)")
    with c2: d=data.get("BANK NIFTY"); st.metric("BANK NIFTY", f"{d['ltp']:,.2f}", f"{d['chg']:.2f} ({d['pct']:.2f}%)")
    with c3: d=data.get("SENSEX"); st.metric("SENSEX", f"{d['ltp']:,.2f}", f"{d['chg']:.2f} ({d['pct']:.2f}%)")
    with c4:
        bias, color = ("SIDEWAYS ↔️", "gray")
        nifty_pct = data.get("NIFTY 50", {}).get('pct', 0)
        if nifty_pct > 0.25: bias, color = ("BULLISH 🚀", "green")
        elif nifty_pct < -0.25: bias, color = ("BEARISH 📉", "red")
        st.markdown(f"<div style='text-align:center; padding:10px; border:1px solid {color}; border-radius:10px; color:{color}'><h3>Bias: {bias}</h3></div>", unsafe_allow_html=True)
if dhan: refreshable_dashboard()
