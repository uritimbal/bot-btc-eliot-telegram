import os
import time
import threading
import io
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
        self.wfile.write(b"Bot Profesional Cuantitativo Operativo")

    def log_message(self, format, *args):
        return

def run_health_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    server.serve_forever()

# ==========================================
# 2. CONFIGURACIÓN, PARÁMETROS Y TELEGRAM
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
CHECK_INTERVAL = 900  # Evaluación cada 15 minutos

ACCOUNT_CAPITAL_USD = 1000.0
MAX_RISK_PER_TRADE_PCT = 0.015 

HTTP_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
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
# 3. PROVEEDORES RESISTENTES A BLOQUEO CLOUD
# ==========================================
def fetch_cryptocompare_candles(symbol="BTC", convert="USD", limit=150, aggregate=1):
    """CryptoCompare: No bloquea IPs de datacenters como Render"""
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
    """Kraken API pública: Excelente tolerancia a peticiones desde la nube"""
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
        # CryptoCompare 1 Hora
        df = fetch_cryptocompare_candles(aggregate=1)
        if df is not None and not df.empty: return df
        # Kraken 60 min
        df = fetch_kraken_candles(interval=60)
        if df is not None and not df.empty: return df
    elif timeframe == "4H":
        # CryptoCompare 4 Horas (aggregate=4)
        df = fetch_cryptocompare_candles(aggregate=4)
        if df is not None and not df.empty: return df
        # Kraken 240 min
        df = fetch_kraken_candles(interval=240)
        if df is not None and not df.empty: return df
    return None

def get_multiframe_data():
    df_1h = fetch_candles_with_fallback("1H")
    df_4h = fetch_candles_with_fallback("4H")
    
    if df_1h is None or df_4h is None or df_1h.empty or df_4h.empty:
        raise ValueError("No se pudieron obtener datos de las API institucionales libre de bloqueos.")
        
    return df_1h, df_4h

# ==========================================
# 4. MOTOR DE ANÁLISIS TÉCNICO ESTRUCTURAL
# ==========================================
def calculate_poc(df, bins=20):
    price_min = df['low'].min()
    price_max = df['high'].max()
    counts, bin_edges = np.histogram(df['close'], bins=bins, weights=df['volume'], range=(price_min, price_max))
    max_idx = np.argmax(counts)
    poc_price = (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2.0
    return poc_price

def detect_structure_and_bos(df, window=4):
    pivots = []
    for i in range(window, len(df) - window):
        high_r = df['high'].iloc[i-window:i+window+1]
        low_r = df['low'].iloc[i-window:i+window+1]
        
        is_high = df['high'].iloc[i] == high_r.max()
        is_low = df['low'].iloc[i] == low_r.min()
        
        if is_high and not is_low:
            pivots.append(('HIGH', i, df['high'].iloc[i]))
        elif is_low and not is_high:
            pivots.append(('LOW', i, df['low'].iloc[i]))
            
    recent_highs = [p[2] for p in pivots if p[0] == 'HIGH']
    recent_lows = [p[2] for p in pivots if p[0] == 'LOW']
    
    last_close = df['close'].iloc[-1]
    bos_status = "RANGO / SIN RUPTURA"
    
    if recent_highs and last_close > recent_highs[-1]:
        bos_status = "BOS ALCISTA 🟢 (Break of Structure High)"
    elif recent_lows and last_close < recent_lows[-1]:
        bos_status = "BOS BAJISTA 🔴 (Break of Structure Low)"
        
    return pivots, bos_status

# ==========================================
# 5. MOTOR CUANTITATIVO MULTITEMPORAL
# ==========================================
def analyze_market_pro():
    df_1h, df_4h = get_multiframe_data()
    
    # --- ANÁLISIS MACRO 4H ---
    df_4h['ema_200'] = df_4h['close'].ewm(span=200, adjust=False).mean()
    macro_close = df_4h['close'].iloc[-1]
    macro_ema200 = df_4h['ema_200'].iloc[-1]
    
    macro_trend = "ALCISTA 🟢" if macro_close > macro_ema200 else "BAJISTA 🔴"
    
    # --- ANÁLISIS MICRO 1H ---
    df_1h['ema_20'] = df_1h['close'].ewm(span=20, adjust=False).mean()
    df_1h['ema_50'] = df_1h['close'].ewm(span=50, adjust=False).mean()
    df_1h['ema_200'] = df_1h['close'].ewm(span=200, adjust=False).mean()
    
    # RSI (14)
    delta = df_1h['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df_1h['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    ema12 = df_1h['close'].ewm(span=12, adjust=False).mean()
    ema26 = df_1h['close'].ewm(span=26, adjust=False).mean()
    df_1h['macd'] = ema12 - ema26
    df_1h['macd_sig'] = df_1h['macd'].ewm(span=9, adjust=False).mean()
    df_1h['macd_hist'] = df_1h['macd'] - df_1h['macd_sig']
    
    # ATR (14)
    tr = pd.concat([
        df_1h['high'] - df_1h['low'],
        np.abs(df_1h['high'] - df_1h['close'].shift(1)),
        np.abs(df_1h['low'] - df_1h['close'].shift(1))
    ], axis=1).max(axis=1)
    df_1h['atr'] = tr.rolling(14).mean()
    
    # RVOL
    df_1h['vol_ma'] = df_1h['volume'].rolling(20).mean()
    rvol = df_1h['volume'].iloc[-1] / df_1h['vol_ma'].iloc[-1]
    
    # POC y Estructura
    poc_price = calculate_poc(df_1h)
    pivots, bos_status = detect_structure_and_bos(df_1h)
    
    close_p = df_1h['close'].iloc[-1]
    atr_p = df_1h['atr'].iloc[-1]
    
    # SCORE CUANTITATIVO
    pro_score = 0
    factors = []
    
    if macro_trend == "ALCISTA 🟢":
        pro_score += 3
        factors.append("• Macro 4H Alcista (Sobre EMA 200 en 4H)")
    else:
        pro_score -= 3
        factors.append("• Macro 4H Bajista (Bajo EMA 200 en 4H)")
        
    if rvol >= 1.5:
        factors.append(f"• Volumen Inusualmente Alto (RVOL {rvol:.2f}x)")
        if close_p > df_1h['open'].iloc[-1]: pro_score += 2
        else: pro_score -= 2
    else:
        factors.append(f"• Volumen Promedio (RVOL {rvol:.2f}x)")
        
    if close_p > poc_price:
        pro_score += 1.5
        factors.append(f"• Precio por encima del POC (${poc_price:,.0f})")
    else:
        pro_score -= 1.5
        factors.append(f"• Precio por debajo del POC (${poc_price:,.0f})")
        
    if "ALCISTA" in bos_status: pro_score += 2
    elif "BAJISTA" in bos_status: pro_score -= 2
    factors.append(f"• Estructura: {bos_status}")

    # DETERMINACIÓN DE SESGO
    if pro_score >= 5.0:
        signal = "🚀 LONG INSTITUCIONAL (Compra Fuerte)"
        bias = "LONG"
    elif 2.0 <= pro_score < 5.0:
        signal = "🟢 LONG MODERADO"
        bias = "LONG"
    elif -2.0 < pro_score < 2.0:
        signal = "⚪ NEUTRAL / SIN CONFLUENCIA"
        bias = "NEUTRAL"
    elif -5.0 < pro_score <= -2.0:
        signal = "🔴 SHORT MODERADO"
        bias = "SHORT"
    else:
        signal = "🔻 SHORT INSTITUCIONAL (Venta Fuerte)"
        bias = "SHORT"

    # GESTIÓN DE RIESGO
    risk_dollars = ACCOUNT_CAPITAL_USD * MAX_RISK_PER_TRADE_PCT
    dist_sl = max(atr_p * 1.8, close_p * 0.01)
    
    if bias == "LONG":
        sl_price = close_p - dist_sl
        tp1_price = close_p + (dist_sl * 1.5)
        tp2_price = close_p + (dist_sl * 3.0)
    else:
        sl_price = close_p + dist_sl
        tp1_price = close_p - (dist_sl * 1.5)
        tp2_price = close_p - (dist_sl * 3.0)

    btc_position_size = risk_dollars / dist_sl
    pos_usd_val = btc_position_size * close_p

    # MENSAJE TELEGRAM
    factors_text = "\n".join(factors)
    msg = (
        f"🏛 *INFORME CUANTITATIVO PROFESIONAL*\n"
        f"🪙 *BTC/USDT* | *Capital Base:* `${ACCOUNT_CAPITAL_USD:,.0f} USD`\n\n"
        f"🎯 *SEÑAL:* `{signal}`\n"
        f"📊 *Confluencia Cuantitativa:* `{pro_score:+.1f} / 10`\n"
        f"🌍 *Tendencia Dominante Macro (4H):* {macro_trend}\n\n"
        f"💰 *Precio Actual:* `${close_p:,.2f}`\n"
        f"📌 *POC (Perfil de Vol.):* `${poc_price:,.2f}`\n"
        f"⚡ *RVOL (Volumen):* `{rvol:.2f}x` | *ATR (1H):* `${atr_p:,.0f}`\n\n"
        f"🔍 *Factores Institucionales:*\n{factors_text}\n\n"
        f"📐 *PLAN DE EJECUCIÓN Y TAMAÑO DE POSICIÓN:*\n"
        f"• *Riesgo Máximo por Trade:* `${risk_dollars:,.2f} USD` ({MAX_RISK_PER_TRADE_PCT*100}%)\n"
        f"• *Tamaño Sugerido Posición:* `{btc_position_size:.4f} BTC` (~${pos_usd_val:,.2f} USD)\n"
        f"• *Precio Entrada:* `${close_p:,.2f}`\n"
        f"• *Stop Loss (SL):* `${sl_price:,.2f}`\n"
        f"• *Take Profit 1 (R:R 1:1.5):* `${tp1_price:,.2f}`\n"
        f"• *Take Profit 2 (R:R 1:3.0):* `${tp2_price:,.2f}`\n"
    )

    # GRAFICACIÓN
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)

    ax1.plot(df_1h.index, df_1h['close'], label='BTC/USDT (1H)', color='#FFFFFF', linewidth=1.2)
    ax1.plot(df_1h.index, df_1h['ema_20'], label='EMA 20', color='#00E5FF', linewidth=1, alpha=0.7)
    ax1.plot(df_1h.index, df_1h['ema_50'], label='EMA 50', color='#00E676', linewidth=1, alpha=0.7)
    ax1.plot(df_1h.index, df_1h['ema_200'], label='EMA 200', color='#FF1744', linewidth=1.2, alpha=0.8)
    
    ax1.axhline(poc_price, color='#FFD700', linestyle='-', linewidth=1.5, alpha=0.8, label=f'POC Vol (${poc_price:,.0f})')
    ax1.axhline(sl_price, color='#FF2A6D', linestyle='--', linewidth=1.2, label=f'SL (${sl_price:,.0f})')
    ax1.axhline(tp1_price, color='#05FFA1', linestyle='--', linewidth=1.2, label=f'TP1 (${tp1_price:,.0f})')
    ax1.axhline(tp2_price, color='#00FF66', linestyle=':', linewidth=1.2, label=f'TP2 (${tp2_price:,.0f})')

    ax1.set_title("BTC/USDT - Análisis Cuantitativo de Grado Profesional", fontsize=11, color='white')
    ax1.legend(loc='upper left', fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.12)

    v_colors = ['#00E676' if df_1h['close'].iloc[i] >= df_1h['open'].iloc[i] else '#FF1744' for i in range(len(df_1h))]
    ax2.bar(df_1h.index, df_1h['volume'], color=v_colors, alpha=0.6, label='Volumen')
    ax2.plot(df_1h.index, df_1h['vol_ma'], color='#FFD700', linewidth=1.2, label='Vol MA 20')
    ax2.set_ylabel("Volumen", fontsize=8, color='white')
    ax2.legend(loc='upper left', fontsize=7)
    ax2.grid(True, alpha=0.12)

    m_colors = ['#00E676' if v >= 0 else '#FF1744' for v in df_1h['macd_hist']]
    ax3.bar(df_1h.index, df_1h['macd_hist'], color=m_colors, alpha=0.4, label='Hist MACD')
    ax3.plot(df_1h.index, df_1h['macd'], color='#00E5FF', linewidth=1.1, label='MACD')
    ax3.plot(df_1h.index, df_1h['macd_sig'], color='#FF9100', linewidth=1.1, label='Señal')
    ax3.set_ylabel("MACD", fontsize=8, color='white')
    ax3.legend(loc='upper left', fontsize=7)
    ax3.grid(True, alpha=0.12)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=130)
    buf.seek(0)
    plt.close()

    send_telegram_photo(buf, caption=msg)

# ==========================================
# 6. BUCLE PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("Iniciando Servidor Web Health Check...")
    threading.Thread(target=run_health_server, daemon=True).start()
    
    send_telegram_message("🚀 *Sistema Cuantitativo sin Bloqueos Conectado*")
    
    while True:
        try:
            print(f"[{time.strftime('%H:%M:%S')}] Ejecutando análisis profesional multitemporal...")
            analyze_market_pro()
            print(f"[{time.strftime('%H:%M:%S')}] Análisis y gráfico enviados correctamente.")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error en análisis: {e}")
            send_telegram_message(f"⚠ Reintento en el próximo ciclo: `{e}`")
        
        time.sleep(CHECK_INTERVAL)
