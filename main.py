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
# 2. CONFIGURACIÓN Y VARIABLES DE ENTORNO
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

SYMBOL = "BTCUSDT"
INTERVAL = "1h"        # Velas de 1 hora
CHECK_INTERVAL = 900   # 15 minutos (900 segundos)

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Error: Variables TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configuradas.")
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
        print(f"Error enviando mensaje a Telegram: {e}")

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
# 3. OBTENCIÓN Y PROCESAMIENTO DE DATOS
# ==========================================
def get_binance_data(symbol=SYMBOL, interval=INTERVAL, limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    res = requests.get(url, timeout=10)
    data = res.json()
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df

def find_pivots(df, window=5):
    df['pivot_high'] = False
    df['pivot_low'] = False
    
    for i in range(window, len(df) - window):
        high_range = df['high'].iloc[i-window:i+window+1]
        low_range = df['low'].iloc[i-window:i+window+1]
        
        if df['high'].iloc[i] == high_range.max():
            df.at[df.index[i], 'pivot_high'] = True
        if df['low'].iloc[i] == low_range.min():
            df.at[df.index[i], 'pivot_low'] = True
    return df

# ==========================================
# 4. ANÁLISIS COMPLETO Y GENERACIÓN DE GRÁFICO
# ==========================================
def analyze_and_send():
    df = get_binance_data()
    df = find_pivots(df, window=5)
    
    # Medias Móviles Exponenciales (EMAs)
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
    
    # Niveles de Fibonacci
    high_price = df['high'].max()
    low_price = df['low'].min()
    diff = high_price - low_price
    
    fib_382 = high_price - 0.382 * diff
    fib_500 = high_price - 0.500 * diff
    fib_618 = high_price - 0.618 * diff
    
    # Detección de Pivotes (Ondas de Elliott)
    pivots = []
    for idx, row in df.iterrows():
        if row['pivot_high']:
            pivots.append(('HIGH', idx, row['high']))
        elif row['pivot_low']:
            pivots.append(('LOW', idx, row['low']))
            
    recent_pivots = pivots[-5:] if len(pivots) >= 5 else pivots

    # Mensaje de Reporte
    msg = (
        f"📊 *REPORTE PROFESIONAL DE MERCADO*\n\n"
        f"🪙 *BTC/USDT (Velas 1H)*\n"
        f"💰 *Precio Actual:* ${current_price:,.2f}\n"
        f"📈 *Tendencia Macro:* {trend}\n"
        f"📊 *RSI (14):* {latest_rsi}\n\n"
        f"📐 *Niveles Fibonacci Clave:*\n"
        f"• 0.382: ${fib_382:,.2f}\n"
        f"• 0.500: ${fib_500:,.2f}\n"
        f"• 0.618 (Golden Pocket): ${fib_618:,.2f}\n"
    )

    # Generación de Gráfico Doble Panel (Precio + RSI)
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Panel 1: Precio, EMAs, Fibonacci y Elliott
    ax1.plot(df.index, df['close'], label='BTC/USDT', color='#F7931A', linewidth=1.8)
    ax1.plot(df.index, df['ema_50'], label='EMA 50', color='#00E676', linewidth=1, alpha=0.7)
    ax1.plot(df.index, df['ema_200'], label='EMA 200', color='#FF1744', linewidth=1.2, alpha=0.8)
    
    ax1.axhline(fib_618, color='#FFD700', linestyle='--', alpha=0.9, label=f'Fib 0.618 (${fib_618:,.0f})')
    ax1.axhline(fib_382, color='#00FFFF', linestyle='--', alpha=0.6, label=f'Fib 0.382 (${fib_382:,.0f})')
    
    pivot_x = [p[1] for p in recent_pivots]
    pivot_y = [p[2] for p in recent_pivots]
    if pivot_x:
        ax1.plot(pivot_x, pivot_y, color='#E040FB', linestyle='-', linewidth=1.5, marker='o', label='Ondas Elliott')
        wave_names = ['(1)', '(2)', '(3)', '(4)', '(5)']
        for i, p in enumerate(recent_pivots):
            label_text = wave_names[i] if i < len(wave_names) else ""
            ax1.annotate(label_text, (p[1], p[2]), textcoords="offset points", xytext=(0, 10),
                         ha='center', fontsize=10, color='#FFFF00', weight='bold')

    ax1.set_title("BTC/USDT - Análisis Elliott, Fibonacci & EMAs", fontsize=12, color='white')
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
# 5. EJECUCIÓN CONTINUA E INDESTRUCTIBLE
# ==========================================
if __name__ == "__main__":
    print("Iniciando servidor Web para Render Free...")
    threading.Thread(target=run_health_server, daemon=True).start()
    
    send_telegram_message("🚀 *Bot Profesional Activado*\nMonitoreando BTC/USDT en Render con análisis completo...")
    
    while True:
        try:
            print(f"[{time.strftime('%H:%M:%S')}] Ejecutando análisis completo...")
            analyze_and_send()
            print(f"[{time.strftime('%H:%M:%S')}] Reporte completo y gráfico enviando con éxito.")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error detectado y aislado: {e}")
            send_telegram_message(f"⚠️️ Reintento automático por micro-corte: `{e}`")
        
        time.sleep(CHECK_INTERVAL)
