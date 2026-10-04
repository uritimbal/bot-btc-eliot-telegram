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
# 1. SERVIDOR WEB PARA PLAN GRATUITO DE RENDER
# ==========================================
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot activo y analizando mercado")

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

HTTP_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json'
}

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Error: Variables de entorno no configuradas.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Error enviando texto a Telegram: {e}")

def send_telegram_photo(image_bytes, caption=""):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {'photo': ('chart.png', image_bytes, 'image/png')}
    data = {
        'chat_id': TELEGRAM_CHAT_ID,
        'caption': caption,
        'parse_mode': 'Markdown'
    }
    try:
        requests.post(url, data=data, files=files, timeout=15)
    except Exception as e:
        print(f"Error enviando gráfico a Telegram: {e}")

# ==========================================
# 3. OBTENCIÓN DE DATOS ANTIBLOQUEO MULTIPROVEEDOR
# ==========================================
def get_market_data_coinbase(symbol="BTC-USD", limit=150):
    try:
        url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity=3600"
        res = requests.get(url, headers=HTTP_HEADERS, timeout=10)
        if res.status_code == 200:
            data = res.json()
            df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df[['open', 'high', 'low', 'close', 'volume']].tail(limit)
    except Exception:
        pass
    return None

def get_market_data_bitfinex(symbol="tBTCUSD", limit=150):
    try:
        url = f"https://api-pub.bitfinex.com/v2/candles/trade:1h:{symbol}/hist?limit={limit}"
        res = requests.get(url, headers=HTTP_HEADERS, timeout=10)
        if res.status_code == 200:
            data = res.json()
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception:
        pass
    return None

def get_market_data_bybit(symbol="BTCUSDT", limit=150):
    try:
        url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval=60&limit={limit}"
        res = requests.get(url, headers=HTTP_HEADERS, timeout=10)
        if res.status_code == 200:
            res_json = res.json()
            if res_json.get('retCode') == 0:
                list_data = res_json.get('result', {}).get('list', [])
                if list_data:
                    df = pd.DataFrame(list_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception:
        pass
    return None

def get_market_data_binance_vision(symbol="BTCUSDT", limit=150):
    try:
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1h&limit={limit}"
        res = requests.get(url, headers=HTTP_HEADERS, timeout=10)
        if res.status_code == 200:
            data = res.json()
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
            ])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception:
        pass
    return None

def get_market_data():
    providers = [
        ("Coinbase", get_market_data_coinbase),
        ("Bitfinex", get_market_data_bitfinex),
        ("Bybit", get_market_data_bybit),
        ("Binance Vision", get_market_data_binance_vision)
    ]
    for name, provider_func in providers:
        df = provider_func()
        if df is not None and not df.empty and len(df) >= 50:
            print(f"Datos obtenidos con éxito desde {name}.")
            return df
    raise ValueError("No se pudieron obtener datos de ningún proveedor de mercado.")

# ==========================================
# 4. ANÁLISIS DE ESTRUCTURA Y ZIGZAG REAL
# ==========================================
def get_clean_pivots(df, window=4):
    """Detecta pivotes y garantiza alternancia estricta (HIGH -> LOW -> HIGH)."""
    raw_pivots = []
    for i in range(window, len(df) - window):
        high_range = df['high'].iloc[i-window:i+window+1]
        low_range = df['low'].iloc[i-window:i+window+1]
        
        is_high = df['high'].iloc[i] == high_range.max()
        is_low = df['low'].iloc[i] == low_range.min()
        
        if is_high and not is_low:
            raw_pivots.append(('HIGH', i, df['high'].iloc[i]))
        elif is_low and not is_high:
            raw_pivots.append(('LOW', i, df['low'].iloc[i]))
            
    # Filtrar consecutivos del mismo tipo (manteniendo el máximo más alto o mínimo más bajo)
    clean_pivots = []
    for p in raw_pivots:
        if not clean_pivots:
            clean_pivots.append(p)
        else:
            last_type = clean_pivots[-1][0]
            if p[0] == last_type:
                if p[0] == 'HIGH' and p[2] > clean_pivots[-1][2]:
                    clean_pivots[-1] = p
                elif p[0] == 'LOW' and p[2] < clean_pivots[-1][2]:
                    clean_pivots[-1] = p
            else:
                clean_pivots.append(p)
    return clean_pivots

def label_market_structure(pivots):
    """Asigna etiquetas de Estructura de Mercado reales (HH, HL, LH, LL)."""
    labeled = []
    last_high = None
    last_low = None
    
    for p_type, idx, price in pivots:
        label = ""
        if p_type == 'HIGH':
            if last_high is None:
                label = "H"
            elif price > last_high:
                label = "HH"  # Higher High
            else:
                label = "LH"  # Lower High
            last_high = price
        else:
            if last_low is None:
                label = "L"
            elif price > last_low:
                label = "HL"  # Higher Low
            else:
                label = "LL"  # Lower Low
            last_low = price
        labeled.append((p_type, idx, price, label))
    return labeled

# ==========================================
# 5. ANÁLISIS TÉCNICO Y GENERACIÓN DE GRÁFICO
# ==========================================
def analyze_and_send():
    df = get_market_data()
    if df is None or df.empty:
        raise ValueError("DataFrame vacío recibido.")
        
    pivots = get_clean_pivots(df, window=4)
    structured_pivots = label_market_structure(pivots)
    
    # EMAs 50 y 200
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    current_price = df['close'].iloc[-1]
    latest_rsi = round(df['rsi'].iloc[-1], 2)
    ema_200_val = df['ema_200'].iloc[-1]
    trend = "BULLISH 🟢 (Sobre EMA 200)" if current_price > ema_200_val else "BEARISH 🔴 (Bajo EMA 200)"
    
    # Fibonacci sobre el último swing relevante
    recent_pivots = structured_pivots[-6:] if len(structured_pivots) >= 6 else structured_pivots
    high_prices = [p[2] for p in recent_pivots if p[0] == 'HIGH']
    low_prices = [p[2] for p in recent_pivots if p[0] == 'LOW']
    
    swing_high = max(high_prices) if high_prices else df['high'].max()
    swing_low = min(low_prices) if low_prices else df['low'].min()
    diff = swing_high - swing_low
    
    fib_382 = swing_high - 0.382 * diff
    fib_500 = swing_high - 0.500 * diff
    fib_618 = swing_high - 0.618 * diff
    
    # Construcción de mensaje Telegram
    msg = (
        f"📊 *ANÁLISIS DE ESTRUCTURA DE MERCADO*\n\n"
        f"🪙 *BTC/USDT (Velas 1H)*\n"
        f"💰 *Precio Actual:* ${current_price:,.2f}\n"
        f"📈 *Tendencia Macro:* {trend}\n"
        f"📊 *RSI (14):* {latest_rsi}\n\n"
        f"📐 *Niveles Fibonacci (Swing Reciente):*\n"
        f"• 0.382: ${fib_382:,.2f}\n"
        f"• 0.500: ${fib_500:,.2f}\n"
        f"• 0.618 (Golden Zone): ${fib_618:,.2f}\n"
    )

    # Gráfico
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Panel 1: Precio, EMAs, ZigZag y Estructura
    ax1.plot(df.index, df['close'], label='BTC/USDT', color='#F7931A', linewidth=1.5, alpha=0.9)
    ax1.plot(df.index, df['ema_50'], label='EMA 50', color='#00E676', linewidth=1, alpha=0.7)
    ax1.plot(df.index, df['ema_200'], label='EMA 200', color='#FF1744', linewidth=1.2, alpha=0.8)
    
    # Líneas Fibonacci
    ax1.axhline(fib_618, color='#FFD700', linestyle='--', alpha=0.8, label=f'Fib 0.618 (${fib_618:,.0f})')
    ax1.axhline(fib_382, color='#00FFFF', linestyle='--', alpha=0.5, label=f'Fib 0.382 (${fib_382:,.0f})')
    
    # Dibujar ZigZag y Etiquetas de Estructura (HH, HL, LH, LL)
    if recent_pivots:
        px = [p[1] for p in recent_pivots]
        py = [p[2] for p in recent_pivots]
        ax1.plot(px, py, color='#E040FB', linestyle='-', linewidth=1.8, marker='o', markersize=5, label='Estructura ZigZag')
        
        for p_type, idx, price, label in recent_pivots:
            offset = 12 if p_type == 'HIGH' else -18
            color_lbl = '#00FF7F' if 'H' in label else '#FF5252'
            ax1.annotate(f"{label}\n${price:,.0f}", (idx, price), 
                         textcoords="offset points", xytext=(0, offset),
                         ha='center', fontsize=8, color=color_lbl, weight='bold')

    ax1.set_title("BTC/USDT - Estructura de Mercado (ZigZag, Pivotes & Fibonacci)", fontsize=12, color='white')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.15)

    # Panel 2: RSI
    ax2.plot(df.index, df['rsi'], color='#9b59b6', linewidth=1.5, label='RSI (14)')
    ax2.axhline(70, color='#FF5252', linestyle='--', alpha=0.6)
    ax2.axhline(30, color='#4CAF50', linestyle='--', alpha=0.6)
    ax2.set_ylabel("RSI", color='white')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.15)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    plt.close()

    send_telegram_photo(buf, caption=msg)

# ==========================================
# 6. BUCLE PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("Iniciando servidor Web para Render Free...")
    threading.Thread(target=run_health_server, daemon=True).start()
    
    send_telegram_message("🚀 *Bot de Estructura de Mercado Conectado*")
    
    while True:
        try:
            print(f"[{time.strftime('%H:%M:%S')}] Ejecutando análisis técnico...")
            analyze_and_send()
            print(f"[{time.strftime('%H:%M:%S')}] Gráfico de estructura enviado exitosamente.")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error detectado: {e}")
            send_telegram_message(f"⚠ Reintento en el próximo ciclo: `{e}`")
        
        time.sleep(CHECK_INTERVAL)
