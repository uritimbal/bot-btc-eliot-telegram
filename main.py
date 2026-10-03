import os
import io
import time
import requests
import numpy as np
import pandas as pd

# CONFIGURACIÓN OBLIGATORIA PARA SERVIDORES EN LA NUBE (HEADLESS)
import matplotlib
matplotlib.use('Agg')  # Evita errores de pantalla/GUI en servidores Linux
import matplotlib.pyplot as plt

from datetime import datetime

# ==========================================
# 1. LECTURA DE VARIABLES DE ENTORNO (NUBE)
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("8628814558:AAHxYgYFZS5tgD19ZTkhhRm1FkA1XLsUFE4")
TELEGRAM_CHAT_ID = os.getenv("6826848469")

SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
HTF = os.getenv("TIMEFRAME_HTF", "4h")   # Trend Macro
LTF = os.getenv("TIMEFRAME_LTF", "1h")   # Gatillo de entrada

ACCOUNT_CAPITAL = float(os.getenv("ACCOUNT_CAPITAL", "10000.0"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
MIN_RRR = float(os.getenv("MIN_RRR", "2.0"))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "300"))  # 5 minutos


# ==========================================
# 2. CONEXIÓN A API BINANCE
# ==========================================
def get_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    res = requests.get(url, timeout=10)
    res.raise_for_status()
    data = res.json()

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'
    ])
    
    cols = ['open', 'high', 'low', 'close', 'volume']
    df[cols] = df[cols].apply(pd.to_numeric, axis=1)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


# ==========================================
# 3. INDICADORES TÉCNICOS
# ==========================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR (14)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['atr'] = np.max(ranges, axis=1).rolling(14).mean()

    return df


# ==========================================
# 4. PIVOTES Y FIBONACCI
# ==========================================
def find_pivots(df: pd.DataFrame, window: int = 4) -> list:
    pivots = []
    for i in range(window, len(df) - window):
        high_win = df['high'].iloc[i - window:i + window + 1]
        low_win = df['low'].iloc[i - window:i + window + 1]

        if df['high'].iloc[i] == high_win.max():
            pivots.append({
                'type': 'HIGH',
                'index': i,
                'price': df['high'].iloc[i],
                'timestamp': df['timestamp'].iloc[i],
                'rsi': df['rsi'].iloc[i]
            })
        elif df['low'].iloc[i] == low_win.min():
            pivots.append({
                'type': 'LOW',
                'index': i,
                'price': df['low'].iloc[i],
                'timestamp': df['timestamp'].iloc[i],
                'rsi': df['rsi'].iloc[i]
            })
    return pivots


def calculate_fib_levels(start_price: float, end_price: float) -> dict:
    diff = end_price - start_price
    return {
        'fib_0500': end_price - (diff * 0.500),
        'gp_0618': end_price - (diff * 0.618),
        'gp_0650': end_price - (diff * 0.650),
        'fib_0786': end_price - (diff * 0.786),
        'ext_1000': end_price + diff,
        'ext_1272': end_price + (diff * 1.272),
        'ext_1618': end_price + (diff * 1.618),
    }


# ==========================================
# 5. ELLIOTT WAVE EVALUATOR
# ==========================================
def evaluate_elliott(df_ltf: pd.DataFrame, df_htf: pd.DataFrame):
    htf_bullish = df_htf['ema_50'].iloc[-1] > df_htf['ema_200'].iloc[-1]
    if not htf_bullish:
        return None

    pivots = find_pivots(df_ltf, window=4)
    if len(pivots) < 5:
        return None

    p0, p1, p2, p3, p4 = pivots[-5:]
    curr_price = df_ltf['close'].iloc[-1]
    curr_atr = df_ltf['atr'].iloc[-1]

    # Patrón Impulsivo Alcista (0-1-2-3-4)
    if p0['type'] == 'LOW' and p1['type'] == 'HIGH' and p2['type'] == 'LOW' and p3['type'] == 'HIGH' and p4['type'] == 'LOW':
        rule1 = p2['price'] > p0['price']
        rule2 = (p3['price'] - p2['price']) > (p1['price'] - p0['price']) * 0.7
        rule3 = p4['price'] > p1['price']

        if rule1 and rule2 and rule3:
            fibs_w3 = calculate_fib_levels(p2['price'], p3['price'])
            in_golden_pocket = (p4['price'] <= fibs_w3['fib_0500']) and (p4['price'] >= fibs_w3['fib_0786'])

            if in_golden_pocket:
                confluences = []
                if p4['rsi'] > p2['rsi'] and p4['price'] < p2['price']:
                    confluences.append("Divergencia Alcista RSI")
                elif p4['rsi'] < 42:
                    confluences.append("RSI en Zona de Soporte/Descompresión")

                if curr_price > df_ltf['ema_50'].iloc[-1]:
                    confluences.append("Precio sobre EMA 50 LTF")

                stop_loss = round(min(p1['price'], p4['price'] - (1.5 * curr_atr)), 2)
                tp1 = round(fibs_w3['ext_1000'], 2)
                tp2 = round(fibs_w3['ext_1272'], 2)
                tp3 = round(fibs_w3['ext_1618'], 2)

                risk = curr_price - stop_loss
                reward = tp2 - curr_price
                rrr = reward / risk if risk > 0 else 0

                if rrr >= MIN_RRR:
                    position_usd = (ACCOUNT_CAPITAL * RISK_PER_TRADE) / (risk / curr_price)
                    return {
                        'price': curr_price,
                        'stop_loss': stop_loss,
                        'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
                        'rrr': round(rrr, 2),
                        'position_usd': round(position_usd, 2),
                        'confluences': confluences,
                        'pivots': [p0, p1, p2, p3, p4],
                        'fibs': fibs_w3
                    }
    return None


# ==========================================
# 6. GRÁFICOS Y NOTIFICACIONES TELEGRAM
# ==========================================
def generate_chart(df: pd.DataFrame, setup: dict) -> io.BytesIO:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.patch.set_facecolor('#121212')
    ax1.set_facecolor('#121212')
    ax2.set_facecolor('#121212')

    df_tail = df.iloc[-80:].copy()
    ax1.plot(df_tail['timestamp'], df_tail['close'], color='#00fff0', label='BTC Price', linewidth=1.5)
    ax1.plot(df_tail['timestamp'], df_tail['ema_50'], color='#ff9900', label='EMA 50', linestyle='--')

    pivots = setup['pivots']
    p_x = [p['timestamp'] for p in pivots]
    p_y = [p['price'] for p in pivots]
    labels = ['(0)', '(1)', '(2)', '(3)', '(4)']
    
    ax1.plot(p_x, p_y, color='#ffff00', linestyle='-', linewidth=2, marker='o')
    for i, txt in enumerate(labels):
        ax1.annotate(txt, (p_x[i], p_y[i]), textcoords="offset points", xytext=(0,8),
                     ha='center', color='#ffff00', fontweight='bold')

    ax1.axhline(setup['fibs']['gp_0618'], color='#00ff00', linestyle=':', label='Golden Pocket 0.618')
    ax1.axhline(setup['stop_loss'], color='#ff0000', linestyle='-', label='Stop Loss')
    ax1.axhline(setup['tp2'], color='#00ffcc', linestyle='-', label='Take Profit 2')

    ax1.set_title(f"BTC/USDT - Elliott Wave + Golden Pocket ({LTF})", color='white')
    ax1.legend(loc='upper left', facecolor='#222222', labelcolor='white')
    ax1.grid(True, color='#2a2a2a')
    ax1.tick_params(colors='white')

    ax2.plot(df_tail['timestamp'], df_tail['rsi'], color='#ab47bc')
    ax2.axhline(70, color='#ff0055', linestyle=':')
    ax2.axhline(30, color='#00ff00', linestyle=':')
    ax2.grid(True, color='#2a2a2a')
    ax2.tick_params(colors='white')

    plt.xticks(rotation=25)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    return buf


def send_telegram(msg: str, buf: io.BytesIO):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'caption': msg, 'parse_mode': 'HTML'}
    files = {'photo': ('chart.png', buf, 'image/png')}
    try:
        r = requests.post(url, data=payload, files=files, timeout=15)
        r.raise_for_status()
        print(f"[{datetime.now()}] Alerta enviada exitosamente a Telegram.")
    except Exception as e:
        print(f"Error enviando a Telegram: {e}")


# ==========================================
# 7. BUCLE PRINCIPAL
# ==========================================
def main():
    print("Iniciando Bot Crypto en la Nube...")
    last_time = None

    while True:
        try:
            df_htf = add_indicators(get_klines(SYMBOL, HTF, 200))
            df_ltf = add_indicators(get_klines(SYMBOL, LTF, 300))

            setup = evaluate_elliott(df_ltf, df_htf)

            if setup:
                p_time = setup['pivots'][-1]['timestamp']
                if last_time != p_time:
                    msg = f"""
🚀 <b>ALERTA ELLIOTT WAVE BTC/USDT</b>

<b>Entrada (Market):</b> ${setup['price']:,.2f}

🎯 <b>Niveles de Trading:</b>
• 🛑 <b>Stop Loss:</b> ${setup['stop_loss']:,.2f}
• 🎯 <b>Take Profit 1:</b> ${setup['tp1']:,.2f}
• 🚀 <b>Take Profit 2:</b> ${setup['tp2']:,.2f}
• 💣 <b>Take Profit 3:</b> ${setup['tp3']:,.2f}

📊 <b>Gestión de Riesgo:</b>
• <b>RRR:</b> 1:{setup['rrr']}
• <b>Posición recomendada:</b> ${setup['position_usd']:,.2f} USDT

💡 <b>Confluencias:</b>
{chr(10).join([f"• {c}" for c in setup['confluences']])}
"""
                    chart_buf = generate_chart(df_ltf, setup)
                    send_telegram(msg, chart_buf)
                    last_time = p_time
                else:
                    print(f"[{datetime.now()}] Patrón ya notificado. Esperando...")
            else:
                print(f"[{datetime.now()}] Analizando mercado... Sin patrones en {LTF}.")

        except Exception as e:
            print(f"Error en bucle: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
