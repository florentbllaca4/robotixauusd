import os
import re
import time
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

import MetaTrader5 as mt5
from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv

# ===================== CONFIG & SETUP =====================

APP_NAME = "AI Smart Trader v10.0"

# --- Path bazë ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "ai_brain.db")
LOG_FILE = os.path.join(BASE_DIR, "ai_smart_trader.log")

os.makedirs(BASE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ],
)
log = logging.getLogger("AIv10")

# --- Load .env ---
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
    log.info(f"[ENV] Loading .env from: {env_path}")
else:
    log.warning(f"[ENV] .env file not found at: {env_path}")

MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
SYMBOL = os.getenv("SYMBOL", "XAUUSD")
MAGIC = int(os.getenv("MAGIC", "90009000"))

BASE_LOT = float(os.getenv("BASE_LOT", "0.01"))
RISK_PER_TRADE_USD = float(os.getenv("RISK_PER_TRADE_USD", "3"))
MIN_PROBABILITY = float(os.getenv("MIN_PROBABILITY", "0.55"))

CHECK_INTERVAL_SEC = 5
AI_UPDATE_EVERY_SEC = 60

log.info(f"[CONFIG] MT5_LOGIN={MT5_LOGIN}, SERVER={MT5_SERVER}")
log.info(f"[CONFIG] SYMBOL={SYMBOL}, MAGIC={MAGIC}")
log.info(f"[CONFIG] BOT_TOKEN_LEN={len(BOT_TOKEN)}")

# ===================== DATABASE LAYER =====================

class AIBrainDB:
    def __init__(self, db_path: str = DB_FILE):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket INTEGER,
            symbol TEXT,
            direction TEXT,
            entry_type TEXT,
            entry_price REAL,
            sl REAL,
            tp REAL,
            result_pips REAL,
            result_usd REAL,
            opened_at TEXT,
            closed_at TEXT,
            source TEXT,
            comment TEXT,
            label INTEGER
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS ai_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            total_trades INTEGER,
            win_trades INTEGER,
            loss_trades INTEGER,
            winrate REAL
        )
        """)

        self.conn.commit()

    def log_new_trade(self, ticket: int, symbol: str, direction: str,
                      entry_type: str, entry_price: float, sl: float,
                      tp: float, source: str, comment: str):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO trades(ticket, symbol, direction, entry_type, entry_price, sl, tp,
                           result_pips, result_usd, opened_at, closed_at,
                           source, comment, label)
        VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, NULL, ?, ?, NULL)
        """, (
            ticket, symbol, direction, entry_type, entry_price, sl, tp,
            datetime.utcnow().isoformat(), source, comment
        ))
        self.conn.commit()

    def update_trade_result(self, ticket: int, result_pips: float,
                            result_usd: float, closed_at: datetime):
        label = 1 if result_usd > 0 else 0
        cur = self.conn.cursor()
        cur.execute("""
        UPDATE trades
        SET result_pips = ?, result_usd = ?, closed_at = ?, label = ?
        WHERE ticket = ?
        """, (result_pips, result_usd, closed_at.isoformat(), label, ticket))
        self.conn.commit()

    def get_basic_stats(self):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) AS c FROM trades WHERE label IS NOT NULL")
        total = cur.fetchone()["c"]

        cur.execute("SELECT COUNT(*) AS c FROM trades WHERE label = 1")
        wins = cur.fetchone()["c"]

        cur.execute("SELECT COUNT(*) AS c FROM trades WHERE label = 0")
        losses = cur.fetchone()["c"]

        winrate = (wins / total) if total > 0 else 0.0
        return total, wins, losses, winrate

    def record_ai_snapshot(self):
        total, wins, losses, winrate = self.get_basic_stats()
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO ai_stats(created_at, total_trades, win_trades, loss_trades, winrate)
        VALUES (?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), total, wins, losses, winrate))
        self.conn.commit()

    def get_direction_stats(self, direction: str) -> Tuple[int, int, int, float]:
        cur = self.conn.cursor()
        cur.execute("""
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS losses
        FROM trades
        WHERE label IS NOT NULL AND direction = ?
        """, (direction,))
        row = cur.fetchone()
        total = row["total"] or 0
        wins = row["wins"] or 0
        losses = row["losses"] or 0
        winrate = (wins / total) if total > 0 else 0.0
        return total, wins, losses, winrate

    def get_summary_text(self) -> str:
        total, wins, losses, winrate = self.get_basic_stats()
        text = f"📊 AI Stats\n" \
               f"Total trades: {total}\n" \
               f"Wins: {wins}\nLosses: {losses}\n" \
               f"Winrate: {winrate*100:.2f}%\n\n"

        for d in ("BUY", "SELL"):
            t, w, l, wr = self.get_direction_stats(d)
            text += f"{d}: {t} trades | wins={w} | losses={l} | winrate={wr*100:.2f}%\n"

        return text.strip()

# ===================== MT5 LAYER =====================

class MT5Client:
    def __init__(self, symbol: str):
        self.symbol = symbol

    def init(self):
        if not mt5.initialize():
            log.error("[FATAL] MT5 nuk mund të inicializohet")
            raise RuntimeError("MT5 init failed")

        authorized = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
        if not authorized:
            log.error("[FATAL] MT5 login dështoi")
            raise RuntimeError("MT5 login failed")

        info = mt5.account_info()
        log.info(f"[MT5] Logged in -> {info.name}, balance={info.balance}, equity={info.equity}")

        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            log.error(f"[FATAL] Simboli {self.symbol} nuk egziston te brokeri.")
            raise RuntimeError("Symbol not found")

        if not symbol_info.visible:
            mt5.symbol_select(self.symbol, True)

    def get_symbol_info(self):
        return mt5.symbol_info(self.symbol)

    def get_price(self) -> Tuple[float, float]:
        tick = mt5.symbol_info_tick(self.symbol)
        return tick.bid, tick.ask

    def calc_lot_by_risk(self, sl_points: float) -> float:
        if sl_points <= 0:
            return BASE_LOT

        try:
            money_per_point = 10.0
            money_per_lot = sl_points * money_per_point
            
            if money_per_lot <= 0:
                return BASE_LOT

            lot = RISK_PER_TRADE_USD / money_per_lot
            lot = max(BASE_LOT, min(lot, 100.0))
            lot = round(lot, 2)
            
            log.info(f"[LOT] sl_points={sl_points}, risk={RISK_PER_TRADE_USD}, lot={lot}")
            return lot
            
        except Exception as e:
            log.error(f"[LOT] Error calculating lot: {e}")
            return BASE_LOT

    def open_single_position(self, direction: str, entry_type: str,
                           price: Optional[float], sl: float, tp: float,
                           lot_size: float, comment: str) -> Optional[int]:
        """
        Hap një pozitë të vetme me lot specifik
        """
        bid, ask = self.get_price()
        info = self.get_symbol_info()

        if direction == "BUY":
            trade_type = mt5.ORDER_TYPE_BUY
            price_market = ask
        else:
            trade_type = mt5.ORDER_TYPE_SELL
            price_market = bid

        if entry_type == "MARKET" or price is None:
            order_type = trade_type
            price_final = price_market
        elif entry_type == "LIMIT":
            order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT
            price_final = price
        elif entry_type == "STOP":
            order_type = mt5.ORDER_TYPE_BUY_STOP if direction == "BUY" else mt5.ORDER_TYPE_SELL_STOP
            price_final = price
        else:
            order_type = trade_type
            price_final = price_market

        request = {
            "action": mt5.TRADE_ACTION_DEAL if entry_type == "MARKET" else mt5.TRADE_ACTION_PENDING,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price_final,
            "sl": sl,
            "tp": tp,
            "deviation": 50,
            "magic": MAGIC,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        log.info(f"[ORDER] {direction} {entry_type} | lot={lot_size} | price={price_final} | SL={sl} | TP={tp}")

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"[ERROR] Order failed | retcode={result.retcode}")
            return None

        ticket = result.order if entry_type != "MARKET" else result.deal
        log.info(f"[ORDER] Success | ticket={ticket}")
        return ticket

    def open_multi_tp_position(self, direction: str, entry_type: str,
                              price: Optional[float], sl: float, 
                              tp_levels: List[float], 
                              lot_progression: List[Tuple[float, int]],
                              comment: str) -> List[Optional[int]]:
        """
        Hap multiple pozita sipas progression tënd
        """
        tickets = []
        tp_index = 0
        total_trades = 0
        
        for lot_size, trade_count in lot_progression:
            for i in range(trade_count):
                if tp_index >= len(tp_levels):
                    # Nëse mbaruan TP levels, përdor TP të fundit
                    tp = tp_levels[-1]
                else:
                    tp = tp_levels[tp_index]
                    
                ticket = self.open_single_position(
                    direction=direction,
                    entry_type=entry_type,
                    price=price,
                    sl=sl,
                    tp=tp,
                    lot_size=lot_size,
                    comment=f"{comment} | Lot:{lot_size} TP{tp_index+1}"
                )
                
                if ticket:
                    tickets.append(ticket)
                    total_trades += 1
                    log.info(f"✅ Trade {total_trades}: Lot {lot_size} -> TP{tp_index+1} = {tp}")
                
                # Kaloj në TP tjetër pas çdo 2 trades
                if (i + 1) % 2 == 0:
                    tp_index += 1
                    if tp_index >= len(tp_levels):
                        # Nëse mbaruan TP levels, dal nga loop
                        break
        
        log.info(f"🎯 Total trades opened: {len(tickets)}")
        return tickets

    def sync_closed_trades_for_ticket(self, db: AIBrainDB, ticket: int):
        try:
            to_time = datetime.now()
            from_time = to_time - timedelta(days=10)

            from_timestamp = int(from_time.timestamp())
            to_timestamp = int(to_time.timestamp())

            deals = mt5.history_deals_get(from_timestamp, to_timestamp)
            if deals is None:
                return

            for d in deals:
                if d.position_id == ticket:
                    result_usd = d.profit
                    result_pips = result_usd / 10
                    closed = datetime.fromtimestamp(d.time)
                    db.update_trade_result(ticket, result_pips, result_usd, closed)
                    log.info(f"[RESULT] ticket={ticket} | profit={result_usd:.2f} USD")
                    break
        except Exception as e:
            log.error(f"[ERROR] Failed to sync closed trades for ticket {ticket}: {e}")

# ===================== SIGNAL PARSER + AI ENGINE =====================

class Signal:
    def __init__(
        self,
        symbol: str,
        direction: str,
        entry_type: str,
        price: Optional[float],
        sl: float,
        tp_levels: List[float],
        lot_progression: List[Tuple[float, int]],
        comment: str = "",
        source: str = "telegram"
    ):
        self.symbol = symbol
        self.direction = direction
        self.entry_type = entry_type
        self.price = price
        self.sl = sl
        self.tp_levels = tp_levels
        self.lot_progression = lot_progression
        self.comment = comment
        self.source = source

class SignalParser:
    @staticmethod
    def parse(text: str) -> Optional[Signal]:
        txt = text.upper()

        m_sym = re.search(r"(XAUUSD|GOLD|[A-Z]{3,6})", txt)
        if not m_sym:
            return None
        symbol = m_sym.group(1)
        if symbol == "GOLD":
            symbol = "XAUUSD"

        m_dir = re.search(r"\b(BUY|SELL)\b", txt)
        if not m_dir:
            return None
        direction = m_dir.group(1)

        m_type = re.search(r"\b(LIMIT|STOP|MARKET)\b", txt)
        entry_type = m_type.group(1) if m_type else "MARKET"

        m_price = re.search(r"@?\s*(\d{3,5}\.?\d*)", txt)
        price = float(m_price.group(1)) if m_price else None

        m_sl = re.search(r"SL\s*:?[\s]*(\d{3,5}\.?\d*)", txt)
        sl = float(m_sl.group(1)) if m_sl else None
        
        # Marr të gjitha TP levels
        tp_matches = re.findall(r"TP\d*\s*:?[\s]*(\d{3,5}\.?\d*)", txt)
        tp_levels = [float(tp) for tp in tp_matches] if tp_matches else []
        
        # Marr lot progression nga format specifik
        lot_progression = SignalParser.parse_lot_progression(text)
        
        if not sl or not tp_levels:
            return None

        return Signal(symbol, direction, entry_type, price, sl, tp_levels, lot_progression, comment=text)

    @staticmethod
    def parse_lot_progression(text: str) -> List[Tuple[float, int]]:
        """
        Parse format: 0.01 x 2 -> 100$ +
        Kthen: [(0.01, 2), (0.01, 3), (0.02, 3), ...]
        """
        progression = []
        pattern = r"(\d+\.\d+)\s*x\s*(\d+)"
        matches = re.findall(pattern, text)
        
        for lot_str, count_str in matches:
            try:
                lot = float(lot_str)
                count = int(count_str)
                progression.append((lot, count))
            except ValueError:
                continue
                
        # Nëse nuk gjen pattern, përdor default progression tënd
        if not progression:
            progression = [
                (0.01, 2), (0.01, 3), (0.02, 3), 
                (0.05, 3), (0.08, 3), (0.10, 3), 
                (0.15, 3)
            ]
            
        return progression

class AITeacher:
    def __init__(self, db: AIBrainDB, mt5_client: MT5Client):
        self.db = db
        self.mt5 = mt5_client
        self.last_snapshot = 0
        self.last_auto_signal_time = 0
        self.auto_signal_interval = 3600  # 1 orë ndërmjet sugjerimeve

    def update_model_if_needed(self):
        now = time.time()
        if now - self.last_snapshot > AI_UPDATE_EVERY_SEC:
            self.db.record_ai_snapshot()
            total, wins, losses, winrate = self.db.get_basic_stats()
            log.info(f"[AI] Snapshot -> total={total} | wins={wins} | losses={losses} | winrate={winrate*100:.2f}%")
            self.last_snapshot = now

    def evaluate_signal(self, signal: Signal) -> float:
        total, wins, losses, winrate_global = self.db.get_basic_stats()
        t_dir, w_dir, l_dir, wr_dir = self.db.get_direction_stats(signal.direction)

        if total == 0:
            prob = 0.5
        else:
            dir_weight = wr_dir if t_dir > 0 else winrate_global
            prob = 0.3 * winrate_global + 0.7 * dir_weight

        log.info(
            f"[AI] Eval -> dir={signal.direction} | prob={prob:.3f} | "
            f"global_wr={winrate_global:.3f} | dir_wr={wr_dir:.3f}"
        )
        return prob

    def should_generate_auto_signal(self) -> bool:
        """Kontrollo nëse duhet të gjenerojë sinjal automatik"""
        total, wins, losses, winrate = self.db.get_basic_stats()
        
        # Kushtet për sinjal automatik:
        # 1. Të paktën 10 trade të suksesshëm
        # 2. Winrate > 60%
        # 3. Jo më shumë se 1 sinjal në orë
        if (wins >= 10 and 
            winrate >= 0.60 and 
            (time.time() - self.last_auto_signal_time) > self.auto_signal_interval):
            return True
        return False

    def generate_auto_signal(self) -> Optional[Signal]:
        """Gjenero sinjal automatik bazuar në performance"""
        if not self.should_generate_auto_signal():
            return None

        # Marr performance për BUY/SELL
        buy_total, buy_wins, buy_losses, buy_wr = self.db.get_direction_stats("BUY")
        sell_total, sell_wins, sell_losses, sell_wr = self.db.get_direction_stats("SELL")
        
        # Zgjidh direction me winrate më të lartë
        if buy_wr >= sell_wr and buy_wr > 0.60:
            direction = "BUY"
            current_wr = buy_wr
        elif sell_wr > 0.60:
            direction = "SELL" 
            current_wr = sell_wr
        else:
            return None

        # Marr çmimet aktuale
        try:
            bid, ask = self.mt5.get_price()
            if direction == "BUY":
                entry_price = ask
                sl = entry_price - 15.0  # 15 pips SL
                tp1 = entry_price + 18.0  # 18 pips TP1
                tp2 = entry_price + 32.0  # 32 pips TP2  
                tp3 = entry_price + 40.0  # 40 pips TP3
            else:
                entry_price = bid
                sl = entry_price + 15.0
                tp1 = entry_price - 18.0
                tp2 = entry_price - 32.0
                tp3 = entry_price - 40.0
        except:
            return None

        # Progression standard
        lot_progression = [
            (0.01, 2), (0.01, 3), (0.02, 3), 
            (0.05, 3), (0.08, 3), (0.10, 3), 
            (0.15, 3)
        ]

        signal = Signal(
            symbol=SYMBOL,
            direction=direction,
            entry_type="MARKET",
            price=None,
            sl=sl,
            tp_levels=[tp1, tp2, tp3],
            lot_progression=lot_progression,
            comment=f"🤖 AI AUTO SIGNAL | Winrate: {current_wr*100:.1f}%",
            source="ai_auto"
        )

        self.last_auto_signal_time = time.time()
        log.info(f"🎯 AI Generated Auto Signal: {direction} | WR: {current_wr*100:.1f}%")
        
        return signal

# ===================== CORE APP (TEACHER + TRADER) =====================

class AISmartTraderApp:
    def __init__(self):
        self.db = AIBrainDB(DB_FILE)
        self.mt5 = MT5Client(SYMBOL)
        self.ai = AITeacher(self.db, self.mt5)
        self.bot: Optional[Bot] = None
        self.dp: Optional[Dispatcher] = None
        self.last_signal_text = "None"
        self.auto_trade_enabled = True

    def init_mt5(self):
        self.mt5.init()

    def handle_signal(self, text: str, source: str = "telegram") -> str:
        sig = SignalParser.parse(text)
        if not sig:
            return "❌ Nuk munda me e lexu sinjalin. (symbol/SL/TP/format)"

        if sig.symbol != SYMBOL:
            return f"⚠️ Sinjali është për {sig.symbol}, bot-i është set për {SYMBOL}."

        prob = self.ai.evaluate_signal(sig)
        self.ai.update_model_if_needed()

        take_trade = prob >= MIN_PROBABILITY
        decision_text = "✅ ACCEPT" if take_trade else "⛔ SKIP"

        if not take_trade:
            log.info(f"[AI] SKIP signal prob={prob:.3f}")
            reply = f"{decision_text} – probabiliteti={prob:.2f} është nën pragun {MIN_PROBABILITY:.2f}"
            self.last_signal_text = reply
            return reply

        # Hap multiple pozita për Multi-TP
        tickets = self.mt5.open_multi_tp_position(
            direction=sig.direction,
            entry_type=sig.entry_type,
            price=sig.price,
            sl=sig.sl,
            tp_levels=sig.tp_levels,
            lot_progression=sig.lot_progression,
            comment=f"AIv10-MultiTP | {sig.comment}"
        )

        successful_tickets = [t for t in tickets if t is not None]
        
        if not successful_tickets:
            reply = f"{decision_text} – por order-i dështoi nga MT5."
            self.last_signal_text = reply
            return reply

        # Log të gjitha trades në database
        for i, ticket in enumerate(successful_tickets):
            if ticket:
                self.db.log_new_trade(
                    ticket=ticket,
                    symbol=sig.symbol,
                    direction=sig.direction,
                    entry_type=sig.entry_type,
                    entry_price=sig.price or 0.0,
                    sl=sig.sl,
                    tp=sig.tp_levels[i] if i < len(sig.tp_levels) else sig.tp_levels[-1],
                    source=source,
                    comment=f"{sig.comment} | TP{i+1}"
                )

        reply = f"{decision_text} – u hap {len(successful_tickets)} trades ✅ | prob={prob:.2f}"
        self.last_signal_text = reply
        return reply

    def check_and_execute_auto_signals(self):
        """Kontrollo dhe ekzekuto sinjale automatikë"""
        if not self.auto_trade_enabled:
            return
            
        auto_signal = self.ai.generate_auto_signal()
        if auto_signal:
            log.info(f"🚀 Executing AI Auto Signal: {auto_signal.direction}")
            
            tickets = self.mt5.open_multi_tp_position(
                direction=auto_signal.direction,
                entry_type=auto_signal.entry_type,
                price=auto_signal.price,
                sl=auto_signal.sl,
                tp_levels=auto_signal.tp_levels,
                lot_progression=auto_signal.lot_progression,
                comment=auto_signal.comment
            )
            
            successful_tickets = [t for t in tickets if t is not None]
            
            # Log trades
            for i, ticket in enumerate(successful_tickets):
                if ticket:
                    self.db.log_new_trade(
                        ticket=ticket,
                        symbol=auto_signal.symbol,
                        direction=auto_signal.direction,
                        entry_type=auto_signal.entry_type,
                        entry_price=auto_signal.price or 0.0,
                        sl=auto_signal.sl,
                        tp=auto_signal.tp_levels[i] if i < len(auto_signal.tp_levels) else auto_signal.tp_levels[-1],
                        source="ai_auto",
                        comment=auto_signal.comment
                    )

            # Dërgo njoftim në Telegram
            if self.bot and successful_tickets:
                self.send_auto_signal_notification(auto_signal, len(successful_tickets))

    def send_auto_signal_notification(self, signal: Signal, trade_count: int):
        """Dërgo njoftim në Telegram për sinjal automatik"""
        try:
            message = (
                f"🎯 *AI AUTO TRADE EXECUTED*\n"
                f"Direction: {signal.direction}\n"
                f"Trades: {trade_count}\n"
                f"SL: {signal.sl:.1f}\n"
                f"TPs: {', '.join(f'{tp:.1f}' for tp in signal.tp_levels)}\n"
                f"Comment: {signal.comment}\n"
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            )
            
            # Për tani, logjo vetëm - mund të shtosh dërgim mesazhesh më vonë
            log.info(f"📢 Auto Signal Notification: {message}")
            
        except Exception as e:
            log.error(f"Error sending auto signal notification: {e}")

    def init_telegram(self):
        if not BOT_TOKEN or BOT_TOKEN.strip() == "":
            log.warning("[TG] BOT_TOKEN nuk është set. Telegram bot nuk do të startohet.")
            return False

        try:
            log.info(f"[TG] Trying to initialize bot with token length: {len(BOT_TOKEN)}")
            self.bot = Bot(token=BOT_TOKEN)
            self.dp = Dispatcher(self.bot)
            
            bot_info = self.bot.get_me()
            log.info(f"[TG] ✅ Bot initialized successfully: @{bot_info.username}")
            
            self._setup_handlers()
            return True
            
        except Exception as e:
            log.error(f"[TG] ❌ Failed to initialize Telegram Bot: {e}")
            self.bot = None
            self.dp = None
            return False

    def _setup_handlers(self):
        @self.dp.message_handler(commands=["start", "help"])
        async def cmd_start(message: types.Message):
            text = (
                f"🤖 AI Smart Trader v10.0\n\n"
                "Dergo sinjal p.sh.:\n"
                "`BUY GOLD SL 4152 TP1 4170 TP2 4184 TP3 4192`\n"
                "`0.01 x 2 -> 100$ +`\n"
                "`0.01 x 3 -> 200$ +`\n\n"
                "Komanda:\n"
                "/stats – statistikat e AI\n"
                "/suggest – sugjerimet e AI\n"
                "/auto – njoftim për auto trading"
            )
            await message.reply(text, parse_mode="Markdown")

        @self.dp.message_handler(commands=["stats"])
        async def cmd_stats(message: types.Message):
            try:
                text = self.db.get_summary_text()
                total, wins, losses, winrate = self.db.get_basic_stats()
                
                # Shto info për auto trading
                if wins >= 10 and winrate >= 0.60:
                    text += f"\n\n🎯 AI Auto Trading: AKTIV (Winrate: {winrate*100:.1f}%)"
                else:
                    text += f"\n\n🎯 AI Auto Trading: JO AKTIV (Nevojiten 10 wins + 60% WR)"
                    
                await message.reply(text)
            except Exception as e:
                await message.reply(f"❌ Gabim gjatë marrjes së statistikave: {e}")

        @self.dp.message_handler(commands=["suggest"])
        async def cmd_suggest(message: types.Message):
            try:
                total, wins, losses, winrate = self.db.get_basic_stats()
                text = "📌 Sugjerimet e AI bazuar në rezultatet e deritanishme:\n\n"
                text += f"- Winrate total: {winrate*100:.2f}%\n"
                
                if total < 20:
                    text += "- Ende pak data, AI po mbledh përvojë...\n"
                elif winrate > 0.65:
                    text += "- Strategjia aktuale po punon shumë mirë! 🚀\n"
                elif winrate < 0.5:
                    text += "- Kujdes! Winrate nën 50%.\n"

                for d in ("BUY", "SELL"):
                    t, w, l, wr = self.db.get_direction_stats(d)
                    text += f"\n{d}: {t} trades | winrate={wr*100:.2f}%"

                await message.reply(text)
            except Exception as e:
                await message.reply(f"❌ Gabim gjatë gjenerimit të sugjerimeve: {e}")

        @self.dp.message_handler(commands=["auto"])
        async def cmd_auto(message: types.Message):
            total, wins, losses, winrate = self.db.get_basic_stats()
            
            if wins >= 10 and winrate >= 0.60:
                text = f"✅ AI Auto Trading është AKTIV\n\n"
                text += f"Wins: {wins}/10 ✅\n"
                text += f"Winrate: {winrate*100:.1f}% ✅\n\n"
                text += "AI do të hapë automatikisht tregti kur të gjejë mundësi të mira!"
            else:
                text = f"❌ AI Auto Trading JO AKTIV\n\n"
                text += f"Wins: {wins}/10 ❌\n"
                text += f"Winrate: {winrate*100:.1f}% {'✅' if winrate >= 0.60 else '❌'}\n\n"
                text += "Nevojiten 10 trades të fituara dhe winrate > 60%"
                
            await message.reply(text)

        @self.dp.message_handler()
        async def handle_any_message(message: types.Message):
            text = message.text or ""
            log.info(f"[TG] Mesazh nga {message.from_user.id}: {text}")
            try:
                reply = self.handle_signal(text, source=f"telegram:{message.from_user.id}")
                await message.reply(reply)
            except Exception as e:
                error_msg = f"❌ Gabim i brendshëm: {e}"
                log.error(f"[TG] Error processing message: {e}")
                await message.reply(error_msg)

    def auto_signal_loop(self):
        """Loop për kontrollimin e sinjaleve automatikë"""
        while True:
            try:
                self.ai.update_model_if_needed()
                self.check_and_execute_auto_signals()
                time.sleep(300)  # Kontrollo çdo 5 minuta
            except Exception as e:
                log.exception(f"[AUTO] Monitor loop error: {e}")
                time.sleep(60)

    def run(self):
        log.info(f"🚀 Start AI Smart Trader Bot v10.0")
        
        try:
            self.init_mt5()
            log.info("[APP] MT5 initialized successfully")
        except Exception as e:
            log.error(f"[APP] MT5 initialization failed: {e}")
            return
        
        telegram_initialized = self.init_telegram()

        if telegram_initialized:
            log.info("[APP] Starting Telegram bot polling...")
            # Nis thread për auto signals
            import threading
            auto_signal_thread = threading.Thread(target=self.auto_signal_loop, daemon=True)
            auto_signal_thread.start()
            
            try:
                executor.start_polling(self.dp, skip_updates=True, relax=0.1)
            except Exception as e:
                log.exception(f"[APP] Telegram polling failed: {e}")
        else:
            log.warning("[APP] Telegram bot not available, running in monitor mode only")
            self.auto_signal_loop()

# ===================== MAIN =====================

if __name__ == "__main__":
    print("🚀 Starting AI Smart Trader Bot v10.0...")
    print("✅ Multi-TP with Lot Progression")
    print("✅ AI Auto Trading after 10 wins")
    print("✅ 24/7 Hosting Ready")
    
    # Setup logging to console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    def main():
        try:
            app = AISmartTraderApp()
            print("✅ App initialized, starting main loop...")
            app.run()
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🔄 Restarting in 10 seconds...")
            time.sleep(10)
            main()

    main()