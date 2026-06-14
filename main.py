from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import pytesseract
import statistics
import re
import numpy as np
import psycopg2
import json
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from difflib import get_close_matches

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DATABASE_URL = os.environ["DATABASE_URL"]

@contextmanager
def get_db():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    id SERIAL PRIMARY KEY,
                    item TEXT NOT NULL,
                    date TIMESTAMPTZ DEFAULT NOW(),
                    min_price INTEGER,
                    median_price INTEGER,
                    average_price INTEGER
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_prices_item ON prices(item)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)")
        conn.commit()

init_db()

try:
    with open("items.json", "r", encoding="utf-8") as f:
        KNOWN_ITEMS = json.load(f)
except:
    KNOWN_ITEMS = []

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nos-tracker.xyz", "https://www.nos-tracker.xyz"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "NosTale Market Tracker API"}


@app.get("/items")
def get_items():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT ON (item)
                    item, min_price, median_price, average_price,
                    to_char(date, 'YYYY-MM-DD HH24:MI') as date
                FROM prices
                ORDER BY item, date DESC
            """)
            rows = cur.fetchall()
    return [
        {
            "name": r["item"],
            "minimum": r["min_price"],
            "median": r["median_price"],
            "average": r["average_price"],
            "date": r["date"],
        }
        for r in rows
    ]


@app.get("/dashboard")
def dashboard():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(DISTINCT item) FROM prices")
            items = cur.fetchone()["count"]
            cur.execute("SELECT COUNT(*) FROM prices")
            measurements = cur.fetchone()["count"]
            cur.execute("SELECT to_char(MAX(date), 'YYYY-MM-DD HH24:MI') FROM prices")
            last_update = cur.fetchone()["to_char"]
    return {"items": items, "measurements": measurements, "last_update": last_update}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)
    text = pytesseract.image_to_string(gray, config="--psm 6")

    prices = re.findall(r"\d{1,3}(?:,\d{3})+", text)
    numbers = [int(p.replace(",", "")) for p in prices]
    if not numbers:
        return {"error": "No prices found"}
    minimum = min(numbers)
    median = int(statistics.median(numbers))
    average = int(sum(numbers) / len(numbers))

    item_name = None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    raw_name = None
    for line in lines:
        m = re.match(r"^([A-Za-zÀ-žА-я][^\d]{3,}?)\s+\d", line)
        if m:
            raw_name = m.group(1).strip().rstrip(".")
            break

    if raw_name:
        raw_lower = raw_name.lower()

        matches = get_close_matches(raw_name, KNOWN_ITEMS, n=1, cutoff=0.5)
        if matches:
            item_name = matches[0]
        else:
            for k in KNOWN_ITEMS:
                if k.lower().startswith(raw_lower[:10]):
                    item_name = k
                    break

        if not item_name:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT DISTINCT item FROM prices")
                    db_known = [r["item"] for r in cur.fetchall()]
            matches = get_close_matches(raw_name, db_known, n=1, cutoff=0.5)
            if matches:
                item_name = matches[0]
            else:
                for k in db_known:
                    if k.lower().startswith(raw_lower[:10]):
                        item_name = k
                        break

    return {"minimum": minimum, "median": median, "average": average, "count": len(numbers), "item_name": item_name}


@app.post("/save")
def save_result(data: dict):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO prices (item, min_price, median_price, average_price)
                VALUES (%s, %s, %s, %s)
            """, (data["item"], data["minimum"], data["median"], data["average"]))
        conn.commit()
    return {"success": True}


@app.get("/history/{item}")
def get_history(item: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    to_char(date, 'YYYY-MM-DD HH24:MI') as date,
                    min_price, median_price, average_price
                FROM prices
                WHERE item = %s
                ORDER BY date DESC
            """, (item,))
            rows = cur.fetchall()
    return [
        {"date": r["date"], "min_price": r["min_price"], "median": r["median_price"], "average": r["average_price"]}
        for r in rows
    ]


@app.get("/analyze_price/{item}")
def analyze_price(item: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    to_char(date, 'YYYY-MM-DD HH24:MI') as date,
                    min_price, median_price, average_price
                FROM prices
                WHERE item = %s
                ORDER BY date ASC
            """, (item,))
            rows = cur.fetchall()

    if not rows:
        return {"error": f"Žádná data pro item '{item}'"}
    if len(rows) < 2:
        return {"error": "Pro analýzu jsou potřeba alespoň 2 měření."}

    medians = [r["median_price"] for r in rows]
    current_median = medians[-1]
    hist_median = int(statistics.median(medians))
    diff_pct = (current_median - hist_median) / hist_median * 100
    last = rows[-1]
    first = rows[0]

    return {
        "item": item,
        "current_median": current_median,
        "hist_median": hist_median,
        "diff_pct": round(diff_pct, 2),
        "count": len(rows),
        "first_date": first["date"],
        "last_date": last["date"],
        "last_min": last["min_price"],
        "last_avg": last["average_price"],
        "history": medians,
        "dates": [r["date"] for r in rows],
    }
