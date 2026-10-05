import os
import time
import threading
import io
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SERVIDOR WEB HEALTH CHECK (RENDER FREE)
# ==========================================
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot Profesional con Paper Trading Operativo")

    def log_message(self, format, *args):
        return

def run_health_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    server.serve_forever()

# ==========================================
# 2. CONFIGURACIÓN Y TELEGRAM
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
CHECK_INTERVAL = 900  # 15 minutos

ACCOUNT_CAPITAL_USD = 1000.0
MAX_RISK_PER_TRADE_PCT = 0.015 
STATE_FILE = "paper_trading_state.json"

HTTP_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Error Telegram: {e}")

def send_telegram_photo(image_bytes, caption=""):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {'photo': ('pro_chart.png', image_bytes, 'image/png')}
    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, data=data, files=files, timeout=15)
    except Exception as e:
        print(f"Error Enviando Gráfico: {e}")

# ==========================================
# 3. GESTOR DE ESTADO DE SIMULACIÓN (PAPER TRADING)
# ==========================================
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "active_trade": None,  # {type, entry, sl, tp1, tp2, size_btc, tp1_hit}
        "history": [],        # list of closed trades
        "balance": ACCOUNT_CAPITAL_USD
    }

def save_state(state):
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Error guardando estado: {e}")

# ==========================================
# 4. CONEXIÓN API Y DATOS
# ==========================================
def fetch_cryptocompare_candles(symbol="BTC", convert="USD", limit=150, aggregate=1):
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={symbol}&tsym={convert}&limit={limit}&aggregate={aggregate}"
        res = requests.get(url, headers=HTTP_HEADERS, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data.get("Response") == "Success":
                candles = data.get("Data", {}).get("Data", [])
                if candles:
                    df = pd.DataFrame(candles)
                    df = df[['open', 'high', 'low', 'close', 'volumefrom']].rename(columns={'volumefrom': 'volume'})
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    return df
    except Exception as e:
        print(f"[FETCH ERROR] CryptoCompare: {e}")
    return None

def fetch_kraken_candles(pair="XBTUSD", interval=60, limit=150):
    try:
        url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
        res = requests.get(url, headers=HTTP_HEADERS, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if not data.get("error"):
                result = data.get("result", {})
                key = [k for k in result.keys() if k != 'last'][0]
                raw = result[key]
                df = pd.DataFrame(raw, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                return df[['open', 'high', 'low', 'close', 'volume']].tail(limit).reset_index(drop=True)
    except Exception as e:
        print(f"[FETCH ERROR] Kraken: {e}")
    return None

def fetch_candles_with_fallback(timeframe="1H"):
    if timeframe == "1H":
        df = fetch_cryptocompare_candles(aggregate=1)
        if df is not None and not df.empty: return df
        df = fetch_kraken_candles(interval=60)
        if df is not None and not df.empty: return df
    elif timeframe == "4H":
        df = fetch_cryptocompare_candles(aggregate=4)
        if df is not None and not df.empty: return df
        df = fetch_kraken_candles(interval=240)
        if df is not None and not df.empty: return df
    return None

def get_multiframe_data():
    df_1h = fetch_candles_with_fallback("1H")
    df_4h = fetch_candles_with_fallback("4H")
    if df_1h is None or df_4h is None or df_1h.empty or df_4h.empty:
        raise ValueError("No se pudieron obtener datos institucionales.")
    return df_1h, df_4h

# ==========================================
# 5. ESTRUCTURA Y VOLUMEN
# ==========================================
def calculate_poc(df, bins=20):
    price_min = df['low'].min()
    price_max = df['high'].max()
    counts, bin_edges = np.histogram(df['close'], bins=bins, weights=df['volume'], range=(price_min, price_max))
    max_idx = np.argmax(counts)
    return (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2.0

def detect_structure_and_bos(df, window=4):
    pivots = []
    for i in range(window, len(df) - window):
        high_r = df['high'].iloc[i-window:i+window+1]
        low_r = df['low'].iloc[i-window:i+window+1]
        
        is_high = df['high'].iloc[i] == high_r.max()
        is_low = df['low'].iloc[i] == low_r.min()
        
        if is_high and not is_low: pivots.append(('HIGH', i, df['high'].iloc[i]))
        elif is_low and not is_high: pivots.append(('LOW', i, df['low'].iloc[i]))
            
    recent_highs = [p[2] for p in pivots if p[0] == 'HIGH']
    recent_lows = [p[2] for p in pivots if p[0] == 'LOW']
    
    last_close = df['close'].iloc[-1]
    bos_status = "RANGO / SIN RUPTURA"
    
    if recent_highs and last_close > recent_highs[-1]:
        bos_status = "BOS ALCISTA 🟢 (Ruptura de Techo)"
    elif recent_lows and last_close < recent_lows[-1]:
        bos_status = "BOS BAJISTA 🔴 (Ruptura de Piso)"
        
    return pivots, bos_status

# ==========================================
# 6. MONITOREO Y SEGUIMIENTO DE OPERACIÓN ACTIVA
# ==========================================
def check_active_trade(state, current_high, current_low, current_close):
    trade = state["active_trade"]
    if not trade:
        return None

    trade_type = trade["type"]
    entry = trade["entry"]
    sl = trade["sl"]
    tp1 = trade["tp1"]
    tp2 = trade["tp2"]
    size = trade["size_btc"]
    tp1_hit = trade.get("tp1_hit", False)

    event_msg = None

    if trade_type == "LONG":
        # 1. Check Stop Loss
        if current_low <= sl:
            pnl = (sl - entry) * (size * 0.5 if tp1_hit else size)
            state["balance"] += pnl
            state["history"].append({"type": "LONG", "result": "SL", "pnl": pnl, "entry": entry, "exit": sl})
            event_msg = f"🔴 *PAPER TRADE CERRADO EN STOP LOSS*\n• PnL: `${pnl:,.2f} USD`\n• Saldo Virtual: `${state['balance']:,.2f} USD`"
            state["active_trade"] = None

        # 2. Check TP1 (Si no se tocó antes)
        elif not tp1_hit and current_high >= tp1:
            pnl_half = (tp1 - entry) * (size * 0.5)
            state["balance"] += pnl_half
            trade["tp1_hit"] = True
            trade["sl"] = entry  # Mover Stop Loss a Break-Even
            event_msg = f"🎯 *PAPER TRADE: TP1 ALCANZADO (50% asegurado)*\n• Ganancia Parcial: `${pnl_half:,.2f} USD`\n• SL movido a Break-Even (${entry:,.2f})"

        # 3. Check TP2
        elif tp1_hit and current_high >= tp2:
            pnl_remaining = (tp2 - entry) * (size * 0.5)
            state["balance"] += pnl_remaining
            state["history"].append({"type": "LONG", "result": "TP2_FULL", "pnl": pnl_remaining, "entry": entry, "exit": tp2})
            event_msg = f"🚀 *PAPER TRADE CERRADO EN TP2 (100% EXITO)*\n• Ganancia Segunda Mitad: `${pnl_remaining:,.2f} USD`\n• Saldo Virtual: `${state['balance']:,.2f} USD`"
            state["active_trade"] = None

    elif trade_type == "SHORT":
        # 1. Check Stop Loss
        if current_high >= sl:
            pnl = (entry - sl) * (size * 0.5 if tp1_hit else size)
            state["balance"] += pnl
            state["history"].append({"type": "SHORT", "result": "SL", "pnl": pnl, "entry": entry, "exit": sl})
            event_msg = f"🔴 *PAPER TRADE CERRADO EN STOP LOSS*\n• PnL: `${pnl:,.2f} USD`\n• Saldo Virtual: `${state['balance']:,.2f} USD`"
            state["active_trade"] = None

        # 2. Check TP1
        elif not tp1_hit and current_low <= tp1:
            pnl_half = (entry - tp1) * (size * 0.5)
            state["balance"] += pnl_half
            trade["tp1_hit"] = True
            trade["sl"] = entry  # Move SL to Break-Even
            event_msg = f"🎯 *PAPER TRADE: TP1 ALCANZADO (50% asegurado)*\n• Ganancia Parcial: `${pnl_half:,.2f} USD`\n• SL movido a Break-Even (${entry:,.2f})"

        # 3. Check TP2
        elif tp1_hit and current_low <= tp2:
            pnl_remaining = (entry - tp2) * (size * 0.5)
            state["balance"] += pnl_remaining
            state["history"].append({"type": "SHORT", "result": "TP2_FULL", "pnl": pnl_remaining, "entry": entry, "exit": tp2})
            event_msg = f"🚀 *PAPER TRADE CERRADO EN TP2 (100% EXITO)*\n• Ganancia Segunda Mitad: `${pnl_remaining:,.2f} USD`\n• Saldo Virtual: `${state['balance']:,.2f} USD`"
            state["active_trade"] = None

    save_state(state)
    return event_msg

# ==========================================
# 7. MOTOR CUANTITATIVO Y SIMULADOR
# ==========================================
def analyze_market_pro():
    state = load_state()
    df_1h, df_4h = get_multiframe_data()
    
    current_high = df_1h['high'].iloc[-1]
    current_low = df_1h['low'].iloc[-1]
    close_p = df_1h['close'].iloc[-1]
    
    # 1. REVISAR SI HAY OPERACIÓN VIRTUAL ABIERTA
    trade_event = check_active_trade(state, current_high, current_low, close_p)
    if trade_event:
        send_telegram_message(trade_event)

    # Indicadores Técnicos
    df_4h['ema_200'] = df_4h['close'].ewm(span=200, adjust=False).mean()
    macro_trend = "ALCISTA 🟢" if df_4h['close'].iloc[-1] > df_4h['ema_200'].iloc[-1] else "BAJISTA 🔴"
    
    df_1h['ema_20'] = df_1h['close'].ewm(span=20, adjust=False).mean()
    df_1h['ema_50'] = df_1h['close'].ewm(span=50, adjust=False).mean()
    df_1h['ema_200'] = df_1h['close'].ewm(span=200, adjust=False).mean()
    
    tr = pd.concat([
        df_1h['high'] - df_1h['low'],
        np.abs(df_1h['high'] - df_1h['close'].shift(1)),
        np.abs(df_1h['low'] - df_1h['close'].shift(1))
    ], axis=1).max(axis=1)
    df_1h['atr'] = tr.rolling(14).mean()
    atr_p = df_1h['atr'].iloc[-1]
    
    ema12 = df_1h['close'].ewm(span=12, adjust=False).mean()
    ema26 = df_1h['close'].ewm(span=26, adjust=False).mean()
    df_1h['macd'] = ema12 - ema26
    df_1h['macd_sig'] = df_1h['macd'].ewm(span=9, adjust=False).mean()
    df_1h['macd_hist'] = df_1h['macd'] - df_1h['macd_sig']

    df_1h['vol_ma'] = df_1h['volume'].rolling(20).mean()
    rvol = df_1h['volume'].iloc[-1] / df_1h['vol_ma'].iloc[-1]
    
    poc_price = calculate_poc(df_1h)
    pivots, bos_status = detect_structure_and_bos(df_1h)
    
    # Evaluar Filtros Estrictos
    pro_score = 0
    if macro_trend == "ALCISTA 🟢": pro_score += 2
    else: pro_score -= 2

    if close_p > poc_price: pro_score += 1
    else: pro_score -= 1

    if "ALCISTA" in bos_status: pro_score += 3
    elif "BAJISTA" in bos_status: pro_score -= 3

    # Decisión
    is_low_volume = rvol < 1.0
    is_in_range = "RANGO" in bos_status

    if is_low_volume or (is_in_range and rvol < 1.4):
        signal = "⚪ NEUTRAL - ESPERANDO LIQUIDEZ"
        bias = "NEUTRAL"
    elif pro_score >= 4.0 and rvol >= 1.3 and "ALCISTA" in bos_status and macro_trend == "ALCISTA 🟢":
        signal = "🚀 LONG INSTITUCIONAL"
        bias = "LONG"
    elif pro_score <= -4.0 and rvol >= 1.3 and "BAJISTA" in bos_status and macro_trend == "BAJISTA 🔴":
        signal = "🔻 SHORT INSTITUCIONAL"
        bias = "SHORT"
    else:
        signal = "⚪ NEUTRAL - SIN CONFLUENCIA"
        bias = "NEUTRAL"

    # Riesgo
    dist_sl = max(atr_p * 1.8, close_p * 0.01)
    risk_dollars = state["balance"] * MAX_RISK_PER_TRADE_PCT
    btc_position_size = (risk_dollars / dist_sl) if dist_sl > 0 else 0

    if bias == "LONG":
        sl_price = close_p - dist_sl
        tp1_price = close_p + (dist_sl * 1.5)
        tp2_price = close_p + (dist_sl * 3.0)
    elif bias == "SHORT":
        sl_price = close_p + dist_sl
        tp1_price = close_p - (dist_sl * 1.5)
        tp2_price = close_p - (dist_sl * 3.0)
    else:
        sl_price, tp1_price, tp2_price = 0, 0, 0

    # 2. APERTURA DE TRADE SIMULADO (Si no hay trade activo)
    if state["active_trade"] is None and bias in ["LONG", "SHORT"]:
        state["active_trade"] = {
            "type": bias,
            "entry": close_p,
            "sl": sl_price,
            "tp1": tp1_price,
            "tp2": tp2_price,
            "size_btc": btc_position_size,
            "tp1_hit": False
        }
        save_state(state)
        send_telegram_message(
            f"⚡ *NUEVO PAPER TRADE ABIERTO (SIMULACIÓN)*\n"
            f"• *Tipo:* `{bias}` a `${close_p:,.2f}`\n"
            f"• *Tamaño:* `{btc_position_size:.4f} BTC`\n"
            f"• *SL:* `${sl_price:,.2f}` | *TP1:* `${tp1_price:,.2f}` | *TP2:* `${tp2_price:,.2f}`"
        )

    # ESTADÍSTICAS DE SIMULACIÓN
    history = state["history"]
    total_trades = len(history)
    wins = len([t for t in history if t["pnl"] > 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    total_pnl = state["balance"] - ACCOUNT_CAPITAL_USD

    active_info = "Ninguna"
    if state["active_trade"]:
        t = state["active_trade"]
        active_info = f"{t['type']} en ${t['entry']:,.2f} (SL:${t['sl']:,.2f})"

    # FORMATO MENSAJE
    msg = (
        f"🏛 *INFORME Y MONITOREO DE SIMULACIÓN*\n"
        f"🪙 *BTC/USDT* | *Precio:* `${close_p:,.2f}`\n\n"
        f"🎯 *Estado Mercado:* `{signal}`\n"
        f"⚡ *RVOL:* `{rvol:.2f}x` | *Estructura:* `{bos_status}`\n\n"
        f"📊 *ESTADÍSTICAS DE EFECTIVIDAD VIRTUAL:*\n"
        f"• *Saldo Virtual Actual:* `${state['balance']:,.2f} USD` (PnL: `${total_pnl:+,.2f} USD`)\n"
        f"• *Trades Ejecutados:* `{total_trades}` | *Win Rate:* `{win_rate:.1f}%`\n"
        f"• *Posición Activa Simulación:* `{active_info}`"
    )

    # GRÁFICO
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)

    ax1.plot(df_1h.index, df_1h['close'], label='BTC/USDT (1H)', color='#FFFFFF', linewidth=1.2)
    ax1.plot(df_1h.index, df_1h['ema_20'], color='#00E5FF', linewidth=1, alpha=0.7)
    ax1.plot(df_1h.index, df_1h['ema_50'], color='#00E676', linewidth=1, alpha=0.7)
    ax1.plot(df_1h.index, df_1h['ema_200'], color='#FF1744', linewidth=1.2, alpha=0.8)
    
    ax1.axhline(poc_price, color='#FFD700', linestyle='-', linewidth=1.5, alpha=0.8, label=f'POC (${poc_price:,.0f})')
    
    if state["active_trade"]:
        tr = state["active_trade"]
        ax1.axhline(tr['sl'], color='#FF2A6D', linestyle='--', linewidth=1.2, label=f"SL Active (${tr['sl']:,.0f})")
        ax1.axhline(tr['tp1'], color='#05FFA1', linestyle='--', linewidth=1.2, label=f"TP1 Active (${tr['tp1']:,.0f})")
        ax1.axhline(tr['tp2'], color='#00FF66', linestyle=':', linewidth=1.2, label=f"TP2 Active (${tr['tp2']:,.0f})")

    ax1.set_title(f"BTC/USDT - Paper Trading Active | Win Rate: {win_rate:.1f}%", fontsize=11, color='white')
    ax1.legend(loc='upper left', fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.12)

    v_colors = ['#00E676' if df_1h['close'].iloc[i] >= df_1h['open'].iloc[i] else '#FF1744' for i in range(len(df_1h))]
    ax2.bar(df_1h.index, df_1h['volume'], color=v_colors, alpha=0.6)
    ax2.plot(df_1h.index, df_1h['vol_ma'], color='#FFD700', linewidth=1.2)
    ax2.grid(True, alpha=0.12)

    m_colors = ['#00E676' if v >= 0 else '#FF1744' for v in df_1h['macd_hist']]
    ax3.bar(df_1h.index, df_1h['macd_hist'], color=m_colors, alpha=0.4)
    ax3.plot(df_1h.index, df_1h['macd'], color='#00E5FF', linewidth=1.1)
    ax3.plot(df_1h.index, df_1h['macd_sig'], color='#FF9100', linewidth=1.1)
    ax3.grid(True, alpha=0.12)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=130)
    buf.seek(0)
    plt.close()

    send_telegram_photo(buf, caption=msg)

# ==========================================
# 8. BUCLE PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("Iniciando Servidor Web Health Check...")
    threading.Thread(target=run_health_server, daemon=True).start()
    
    send_telegram_message("🧪 *Bot de Paper Trading Iniciado (Medición de Efectividad en Vivo)*")
    
    while True:
        try:
            print(f"[{time.strftime('%H:%M:%S')}] Evaluando simulación y mercado...")
            analyze_market_pro()
            print(f"[{time.strftime('%H:%M:%S')}] Análisis y estado de simulación enviados.")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error: {e}")
            send_telegram_message(f"⚠ Reintento en el próximo ciclo: `{e}`")
        
        time.sleep(CHECK_INTERVAL)
