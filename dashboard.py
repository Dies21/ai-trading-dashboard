import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import json
import time

# Безопасная обёртка для перезапуска Streamlit (учитывает разные верси и отсутствия API)
def safe_rerun():
    try:
        if hasattr(st, 'experimental_rerun'):
            try:
                st.experimental_rerun()
                return
            except Exception:
                pass
        if hasattr(st, 'rerun'):
            try:
                st.rerun()
                return
            except Exception:
                pass
    except Exception:
        pass

# Налаштування сторінки
st.set_page_config(
    page_title="AI Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive {
        color: #00cc00;
        font-weight: bold;
    }
    .negative {
        color: #ff0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.title("📈 AI Trading Bot Dashboard")
st.markdown("---")

# Бічне меню
page = st.sidebar.radio(
    "Навігація",
    ["📊 Огляд", "📉 Логи прогнозів", "📋 Статистика", "⚙️ Керування", "📖 Інструкція"]
)

# Кнопка оновлення даних (очищає кеш і перезавантажує сторінку)
if st.sidebar.button("🔄 Оновити дані"):
    try:
        if hasattr(st, 'cache_data') and hasattr(st.cache_data, 'clear'):
            st.cache_data.clear()
    except Exception:
        pass
    safe_rerun()

# Автооновлення: чекбокс і інтервал (секунди)
auto_refresh = st.sidebar.checkbox("⏱️ Автооновлення", value=False)
if auto_refresh:
    interval = st.sidebar.slider(
        "Інтервал оновлення (сек)",
        min_value=5,
        max_value=600,
        value=60,
        step=5,
        help="Автоматично оновлює сторінку кожні N секунд (буде виконано time.sleep)."
    )
    st.sidebar.caption(f"Оновлення кожні {interval} с")
    try:
        time.sleep(interval)
        safe_rerun()
    except Exception:
        pass

# Загрузка данных из логов
@st.cache_data
def load_predictions():
    log_file = Path("logs/predictions.csv")
    if not log_file.exists():
        # Попытка загрузки файла логов из репозитория GitHub (raw URL).
        # Это полезно для Streamlit Cloud, где локальные логи могут отсутствовать.
        try:
            import requests
            from io import StringIO
            # Попробуем стандартный путь к raw в GitHub (владелец/репо может отличаться)
            raw_urls = [
                "https://raw.githubusercontent.com/Dies21/ai-trading-dashboard/main/logs/predictions.csv",
                "https://raw.githubusercontent.com/" + ("${GITHUB_REPOSITORY}" if "GITHUB_REPOSITORY" in globals() else "Dies21/ai-trading-dashboard") + "/main/logs/predictions.csv"
            ]
            for url in raw_urls:
                try:
                    r = requests.get(url, timeout=5)
                    if r.status_code == 200 and r.text.strip():
                        return pd.read_csv(StringIO(r.text))
                except Exception:
                    continue
        except Exception:
            pass
        return pd.DataFrame()

    df = pd.read_csv(log_file)

    # Безопасное приведение типов и очистка полей
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        except Exception:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    numeric_cols = ['confidence', 'accuracy', 'close_price', 'volume', 'p_and_l', 'balance_simulated']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # win_rate может приходить как '52.2%'; очистим и приведём к float (52.2)
    if 'win_rate' in df.columns:
        def parse_win_rate(x):
            try:
                if isinstance(x, str) and x.strip().endswith('%'):
                    return float(x.strip().replace('%', ''))
                return float(x)
            except Exception:
                return 0.0
        df['win_rate'] = df['win_rate'].apply(parse_win_rate)

    return df

def load_statistics():
    from logger import PredictionLogger
    logger = PredictionLogger()
    return logger.get_statistics()

# ==================== СТОРІНКА 1: ОГЛЯД ====================
if page == "📊 Огляд":
    st.header("Загальний огляд системи")

    df = load_predictions()

    if len(df) == 0:
        st.warning("📭 Немає даних у логах. Запустіть main.py для накопичення даних.")
    else:
        # Ключевые метрики
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Усього прогнозів",
                len(df),
                delta=None,
                delta_color="off"
            )
        
        with col2:
            unique_symbols = df['symbol'].nunique()
            st.metric(
                "Аналізованих активів",
                unique_symbols,
                delta=None,
                delta_color="off"
            )
        
        with col3:
            up_count = len(df[df['prediction'] == 'UP'])
            st.metric(
                "🟢 Сигналів на зростання",
                up_count,
                delta=None,
                delta_color="off"
            )
        
        with col4:
            down_count = len(df[df['prediction'] == 'DOWN'])
            st.metric(
                "🔴 Сигналів на падіння",
                down_count,
                delta=None,
                delta_color="off"
            )
        
        with col5:
            avg_confidence = df['confidence'].astype(float).mean()
            st.metric(
                "Середня впевненість",
                f"{avg_confidence:.2%}",
                delta=None,
                delta_color="off"
            )
        
        st.markdown("---")
        
        # Останні 10 прогнозів
        st.subheader("🔔 Останні прогнози")
        latest = df.tail(10)[['timestamp', 'symbol', 'prediction', 'confidence', 'close_price', 'accuracy']].copy()
        latest['confidence'] = latest['confidence'].astype(float)
        latest['accuracy'] = latest['accuracy'].astype(float)
        
        # Форматування з цветовыми індикаторами
        latest['Час'] = pd.to_datetime(latest['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        latest['Актив'] = latest['symbol']
        latest['Прогноз'] = latest['prediction'].apply(lambda x: '🟢 UP' if x == 'UP' else '🔴 DOWN' if x == 'DOWN' else '⚪ UNSURE')
        latest['Впевненість'] = latest['confidence'].apply(lambda x: f"{x:.2%}")
        latest['Ціна'] = latest['close_price'].astype(float).apply(lambda x: f"${x:.2f}")
        latest['Точність'] = latest['accuracy'].astype(float).apply(lambda x: f"{x:.2%}")
        
        st.dataframe(
            latest[['Час', 'Актив', 'Прогноз', 'Впевненість', 'Ціна', 'Точність']],
            width='stretch',
            hide_index=True
        )
        
        # Додатково: активні сигнали на падіння
        down_signals = df[df['prediction'] == 'DOWN'].tail(5)
        if len(down_signals) > 0:
            st.markdown("---")
            st.subheader("🔴 Останні сигнали на ПАДІННЯ")
            down_display = down_signals[['timestamp', 'symbol', 'confidence', 'close_price']].copy()
            down_display['Час'] = pd.to_datetime(down_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            down_display['Актив'] = down_display['symbol']
            down_display['Впевненість'] = down_display['confidence'].astype(float).apply(lambda x: f"{x:.2%}")
            down_display['Ціна'] = down_display['close_price'].astype(float).apply(lambda x: f"${x:.2f}")
            
            st.dataframe(
                down_display[['Час', 'Актив', 'Впевненість', 'Ціна']],
                width='stretch',
                hide_index=True
            )
        
        st.markdown("---")
        
        # Графики
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Розподіл прогнозів")
            prediction_counts = df['prediction'].value_counts()
            fig = px.pie(
                values=prediction_counts.values,
                names=prediction_counts.index,
                color_discrete_map={'UP': '#00cc00', 'DOWN': '#ff0000', 'UNSURE': '#ffa500'},
                hole=0.3
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("📈 Точність у часі")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['accuracy_float'] = df['accuracy'].astype(float)
            
            fig = px.line(
                df,
                x='timestamp',
                y='accuracy_float',
                title='Зміна точності',
                color='symbol',
                markers=True
            )
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        
        # По активах
        st.subheader("💼 Статистика по активах")
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(f"{symbol} - Прогнозів", len(symbol_df))
            with col2:
                avg_conf = symbol_df['confidence'].astype(float).mean()
                st.metric(f"{symbol} - Впевненість", f"{avg_conf:.2%}")
            with col3:
                avg_acc = symbol_df['accuracy'].astype(float).mean()
                st.metric(f"{symbol} - Точність", f"{avg_acc:.2%}")

# ==================== СТОРІНКА 2: ЛОГИ ====================
elif page == "📉 Логи прогнозів":
    st.header("Логи всіх прогнозів")

    df = load_predictions()

    if len(df) == 0:
        st.warning("📭 Немає даних у логах.")
    else:
        # Фильтры
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_symbol = st.multiselect(
                "Оберіть активи",
                df['symbol'].unique(),
                default=df['symbol'].unique()
            )
        
        with col2:
            selected_prediction = st.multiselect(
                "Оберіть прогнози",
                df['prediction'].unique(),
                default=df['prediction'].unique()
            )
        
        with col3:
            date_range = st.date_input(
                "Діапазон дат",
                value=(
                    pd.to_datetime(df['timestamp']).min().date(),
                    pd.to_datetime(df['timestamp']).max().date()
                ),
                key="date_range"
            )
        
        # Фильтрация
        filtered_df = df[
            (df['symbol'].isin(selected_symbol)) &
            (df['prediction'].isin(selected_prediction))
        ]
        
        # Форматирование для отображения
        display_df = filtered_df.copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['confidence'] = display_df['confidence'].astype(float).apply(lambda x: f"{x:.2%}")
        display_df['close_price'] = display_df['close_price'].astype(float).apply(lambda x: f"${x:.2f}")
        display_df['p_and_l'] = display_df['p_and_l'].astype(float).apply(lambda x: f"${x:+.2f}")
        display_df['accuracy'] = display_df['accuracy'].astype(float).apply(lambda x: f"{x:.2%}")
        display_df['win_rate'] = display_df['win_rate'].astype(float).apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            display_df[['timestamp', 'symbol', 'prediction', 'confidence', 'close_price', 'volume', 'p_and_l', 'accuracy']],
            width='stretch',
            height=600,
            hide_index=True
        )
        
        # Скачивание данных
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Завантажити CSV",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ==================== СТОРІНКА 3: СТАТИСТИКА ====================
elif page == "📋 Статистика":
    st.header("Детальна статистика")

    df = load_predictions()

    if len(df) == 0:
        st.warning("📭 Немає даних для аналізу.")
    else:
        stats = load_statistics()
        
        if stats:
            # Загальна статистика
            st.subheader("🎯 Загальна статистика")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Усього записів", stats['total_predictions'])
            with col2:
                st.metric("Активів", stats['symbols'])
            with col3:
                avg_conf = float(stats['avg_confidence'].split('%')[0]) if '%' in str(stats['avg_confidence']) else stats['avg_confidence']
                st.metric("Середня впевненість", f"{avg_conf:.2%}" if isinstance(avg_conf, float) else stats['avg_confidence'])
            with col4:
                st.metric("UP / DOWN / UNSURE", f"{stats['up_predictions']} / {stats['down_predictions']} / {stats['unsure_predictions']}")
        
        st.markdown("---")
        
        # Статистика по активах
        st.subheader("💼 Детальна статистика по активах")
        
        for symbol in df['symbol'].unique():
            with st.expander(f"📊 {symbol}"):
                symbol_df = df[df['symbol'] == symbol]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Прогнозів", len(symbol_df))
                with col2:
                    avg_conf = symbol_df['confidence'].astype(float).mean()
                    st.metric("Середня впевненість", f"{avg_conf:.2%}")
                with col3:
                    avg_acc = symbol_df['accuracy'].astype(float).mean()
                    st.metric("Середня точність", f"{avg_acc:.2%}")
                with col4:
                    total_pnl = symbol_df['p_and_l'].astype(float).sum()
                    st.metric("Загальний P&L", f"${total_pnl:+.2f}")

                # Графік впевненості
                fig = px.line(
                    symbol_df,
                    x='timestamp',
                    y='confidence',
                    title=f'Впевненість для {symbol}',
                    markers=True
                )
                st.plotly_chart(fig, width='stretch')

# ==================== СТОРІНКА 4: КЕРУВАННЯ ====================
elif page == "⚙️ Керування":
    st.header("Керування системою")

    st.subheader("➕ Додавання нових активів")
    st.info("📝 Додайте новий актив для аналізу (наприклад: BNB/USDT, XRP/USDT, ADA/USDT)")

    new_symbol = st.text_input("Введіть назву активу", placeholder="BNB/USDT")

    if st.button("✅ Додати актив"):
        if new_symbol:
            try:
                from data_loader import CryptoDataLoader
                loader = CryptoDataLoader()
                loader.add_symbol(new_symbol)
                st.success(f"✅ Актив {new_symbol} додано!")
            except Exception as e:
                st.error(f"❌ Помилка: {e}")
        else:
            st.warning("⚠️ Будь ласка, введіть назву активу")
    
    st.markdown("---")
    
    st.subheader("📊 Текущие активы")
    try:
        from data_loader import CryptoDataLoader
        loader = CryptoDataLoader()
        st.write("Аналізовані активи:")
        for symbol in loader.symbols:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"• {symbol}")
            with col2:
                if st.button(f"🗑️", key=f"remove_{symbol}"):
                    loader.remove_symbol(symbol)
                    st.success(f"✅ {symbol} видалено")
                    safe_rerun()
    except Exception as e:
        st.error(f"❌ Ошибка загрузки активов: {e}")
    
    st.markdown("---")
    
    st.subheader("⚡ Параметри моделі")
    
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider(
            "Поріг впевненості",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.05
        )
        st.caption(f"Модель даватиме сигнали тільки якщо впевненість > {confidence_threshold:.0%}")
    
    with col2:
        n_estimators = st.slider(
            "Кількість дерев XGBoost",
            min_value=50,
            max_value=500,
            value=150,
            step=50
        )
        st.caption(f"Більше дерев = точніше, але повільніше")
    
    st.markdown("---")
    
    st.subheader("🧹 Очищення даних")

    if st.checkbox("Я розумію, що це видалить всі логи"):
        if st.button("❌ Очистити всі логи"):
            log_file = Path("logs/predictions.csv")
            if log_file.exists():
                log_file.unlink()
                st.success("✅ Логи очищено")
                safe_rerun()

# ==================== СТОРІНКА 5: ІНСТРУКЦІЯ ====================
elif page == "📖 Інструкція":
     st.header("📖 Інструкція з використання")

     st.markdown("""
     ### 🚀 Швидкий старт

     1. **Запустіть основний скрипт:**
         ```bash
         python main.py
         ```

     2. **Відкрийте цей dashboard у браузері:**
         ```bash
         streamlit run dashboard.py
         ```

     3. **Дивіться прогнози в реальному часі**

     ---

     ### 📊 Огляд вкладок

     **📊 Огляд** - Основні метрики та останні прогнози
     - Усього прогнозів
     - Аналізованих активів
     - UP/DOWN прогнози
     - Середня впевненість моделі

     **📉 Логи прогнозів** - Повна історія всіх прогнозів
     - Фільтр по активах
     - Фільтр за типом прогнозу
     - Завантаження у CSV

     **📋 Статистика** - Детальний аналіз
     - Статистика по активах
     - Графіки впевненості
     - Аналіз P&L

     **⚙️ Керування** - Налаштування системи
     - Додавання/видалення активів
     - Параметри моделі
     - Очищення логів

     ---

     ### 📈 Розуміння метрик

     **Prediction (Прогноз)**
     - ⬆ UP - модель очікує зростання ціни
     - ⬇ DOWN - модель очікує зниження ціни
     - ❓ UNSURE - модель не впевнена

     **Confidence (Впевненість)**
     - 0-50% - низька впевненість
     - 50-70% - середня впевненість
     - 70%+ - висока впевненість

     **Accuracy (Точність)**
     - Відсоток правильних прогнозів на тестовій вибірці
     - Понад 50% = краще випадкового вгадування

     **Win Rate**
     - Відсоток прибуткових угод у симуляції

     ---

     ### 💡 Поради

     1. **Використовуйте декілька активів** - знижує ризик
     2. **Слідкуйте за впевненістю** - висока впевненість = надійніше
     3. **Аналізуйте логи** - дивіться, як модель працює на різних активах
     4. **Регулярно перевіряйте статистику** - оцініть якість прогнозів
     5. **Експортуйте дані** - для подальшого аналізу в Excel/Python

     ---

     ### 🔧 Контакти та підтримка

     Якщо виникають питання або помилки - перевірте:
     - Інтернет-з'єднання (потрібне для CCXT)
     - Папка `logs/` існує
     - Всі залежності встановлені (`pip install -r requirements.txt`)

     """)

st.markdown("---")
st.caption(f"AI Trading Bot v1.0 | Останнє оновлення: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
