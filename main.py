import time
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# ==========================================
# SERVIDOR WEB PARA PLAN GRATUITO DE RENDER
# ==========================================
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot activo")

    def log_message(self, format, *args):
        return

def run_health_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    server.serve_forever()

# ==========================================
# CONFIGURACIÓN TELEGRAM
# ==========================================
# Reemplazá con tu Token nuevo y tu Chat ID
TELEGRAM_BOT_TOKEN = "8628814558:AAHHOkbClziVXGfZSMXfR2NGs5kCjwAu_nU"
TELEGRAM_CHAT_ID = "6826848469"

SYMBOL = "BTCUSDT"
INTERVAL = "1h"        # Velas de 1 hora
CHECK_INTERVAL = 900   # Analizar cada 15 minutos (900 segundos)

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Error enviando mensaje: {e}")

def send_telegram_photo(image_bytes, caption=""):
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
        print(f"Error enviando gráfico: {e}")

def get_binance_data(symbol=SYMBOL, interval=INTERVAL, limit=120):
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

def find_pivots(df, window=4):
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

def analyze_elliott_and_fibonacci():
    df = get_binance_data()
    df = find_pivots(df, window=4)
    
    current_price = df['close'].iloc[-1]
    
    pivots = []
    for idx, row in df.iterrows():
        if row['pivot_high']:
            pivots.append(('HIGH', idx, row['high']))
        elif row['pivot_low']:
            pivots.append(('LOW', idx, row['low']))
    
    high_price = df['high'].max()
    low_price = df['low'].min()
    diff = high_price - low_price
    
    fib_236 = high_price - 0.236 * diff
    fib_382 = high_price - 0.382 * diff
    fib_500 = high_price - 0.500 * diff
    fib_618 = high_price - 0.618 * diff
    fib_786 = high_price - 0.786 * diff

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    latest_rsi = round(rsi.iloc[-1], 2)
    estado_rsi = "Sobrecompra ⚠️" if latest_rsi > 70 else ("Sobreventa 🟢" if latest_rsi < 30 else "Neutral ⚖️")

    recent_pivots = pivots[-5:] if len(pivots) >= 5 else pivots
    wave_labels = []
    
    if len(recent_pivots) >= 5:
        wave_names = ['(1)', '(2)', '(3)', '(4)', '(5)']
        for i, p in enumerate(recent_pivots):
            wave_labels.append((p[1], p[2], wave_names[i]))
        estado_elliott = f"Secuencia de 5 Ondas detectada (Último pivote en {recent_pivots[-1][0]})"
    else:
        estado_elliott = "Estructura de ondas en formación..."

    msg = (
        f"🌊 *ANÁLISIS ONDAS DE ELLIOTT & FIBONACCI*\n\n"
        f"🪙 *Par:* BTC/USDT (1H)\n"
        f"💰 *Precio Actual:* ${current_price:,.2f}\n"
        f"📊 *RSI (14):* {latest_rsi} ({estado_rsi})\n"
        f"🔍 *Estado Elliott:* {estado_elliott}\n\n"
        f"📐 *Niveles Fibonacci Clave:*\n"
        f"• 0.236: ${fib_236:,.2f}\n"
        f"• 0.382: ${fib_382:,.2f}\n"
        f"• 0.500: ${fib_500:,.2f}\n"
        f"• 0.618 (Nivel de Oro): ${fib_618:,.2f}\n"
        f"• 0.786: ${fib_786:,.2f}\n\n"
        f"🏔️ *Máximo Reciente:* ${high_price:,.2f}\n"
        f"📉 *Mínimo Reciente:* ${low_price:,.2f}"
    )

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    ax1.plot(df.index, df['close'], label='BTC/USDT', color='#F7931A', linewidth=1.8)
    ax1.axhline(fib_618, color='#FFD700', linestyle='--', alpha=0.8, label=f'Fib 0.618 (${fib_618:,.0f})')
    ax1.axhline(fib_382, color='#00FFFF', linestyle='--', alpha=0.8, label=f'Fib 0.382 (${fib_382:,.0f})')
    ax1.axhline(fib_500, color='#00FF00', linestyle=':', alpha=0.6, label=f'Fib 0.500 (${fib_500:,.0f})')
    
    pivot_x = [p[1] for p in recent_pivots]
    pivot_y = [p[2] for p in recent_pivots]
    if pivot_x:
        ax1.plot(pivot_x, pivot_y, color='#E040FB', linestyle='-', linewidth=1.5, marker='o', label='Ondas Elliott')
        for p_idx, p_val, label_text in wave_labels:
            ax1.annotate(label_text, (p_idx, p_val), textcoords="offset points", xytext=(0, 10),
                         ha='center', fontsize=11, color='#FFFF00', weight='bold')

    ax1.set_title("BTC/USDT - Análisis de Ondas de Elliott & Fibonacci", fontsize=13, color='white')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.15)

    ax2.plot(df.index, rsi, color='#9b59b6', linewidth=1.5, label='RSI (14)')
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
    print("Análisis Elliott + Fibonacci enviado correctamente.")

if __name__ == "__main__":
    print("Iniciando servidor de puerto para Render Free...")
    threading.Thread(target=run_health_server, daemon=True).start()
    
    print("Iniciando Bot Crypto Elliott + Fibonacci en Render...")
    send_telegram_message("🤖 *Bot de Ondas de Elliott & Fibonacci iniciado en Render*")
    while True:
        try:
            analyze_elliott_and_fibonacci()
        except Exception as e:
            print(f"Error en el ciclo de análisis: {e}")
        time.sleep(CHECK_INTERVAL)
