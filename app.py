import os
from dotenv import load_dotenv
from openai import OpenAI
import yfinance as yf

from finance.extract_debt import extract_debt_total_musd
from finance.montecarlo import get_annual_vol, montecarlo_bankruptcy_proxy
from finance.merton_pd import merton_pd
from rag.qa_engine import search

load_dotenv()  # carga variables desde .env

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY. Revisa tu archivo .env en la raíz del proyecto.")

client = OpenAI(api_key=API_KEY)

def main():
    ticker = "TSLA"

    # 1) Precio live
    price_now = float(yf.Ticker(ticker).fast_info["lastPrice"])

    # 2) Volatilidad anual (histórica)
    vol = get_annual_vol(ticker, period="1y")

    # 3) Deuda automática desde PDF (con evidencia)
    debt_out = extract_debt_total_musd()
    debt_total_musd = debt_out["debt_total_musd"]
    if debt_total_musd is None:
        raise RuntimeError(f"No se pudo extraer deuda: {debt_out['missing']}")

    # 4) Montecarlo proxy (precio + vol + deuda)
    mc = montecarlo_bankruptcy_proxy(
        price_now=price_now,
        debt_total_musd=debt_total_musd,
        years=2.0,
        n_sims=20000,
        vol_annual=vol,
    )

    # 5) Merton PD (probabilidad de default 2 años)
    fi = yf.Ticker(ticker).fast_info
    market_cap = float(fi["marketCap"])  # USD
    D_usd = debt_total_musd * 1_000_000   # MUSD -> USD
    r = 0.04
    T = 2.0

    merton = merton_pd(V0=market_cap, D=D_usd, r=r, sigma_V=vol, T=T)
    if merton is None:
        raise RuntimeError("Merton PD no pudo calcularse (inputs inválidos).")
    pd_merton, d1, d2 = merton

    # 6) Evidencia con páginas (de extractor)
    ev_lines = []
    for page, tag, snippet in debt_out["evidence"]:
        ev_lines.append(f"- Página {page} ({tag}): {snippet}")

    # Extra evidencia opcional
    extra = search("debt and finance leases total debt", k=2)
    for d in extra:
        ev_lines.append(f"- Página {d['meta']['page_number']} (extra): {d['text'][:600]}")

    ev_text = "\n\n".join(ev_lines)

    prompt = f"""
Responde corto y directo. Incluye:
- Precio actual ({ticker})
- Deuda total (USD millones) y componentes
- Montecarlo (probabilidad proxy 2 años)
- Merton (probabilidad de default 2 años)
- Citas de páginas exactas del PDF

Datos:
Precio actual: {price_now:.2f} USD
Market Cap aprox: {market_cap:,.0f} USD
Volatilidad anual estimada: {vol:.3f}

Deuda total (USD millones): {debt_total_musd}
Componentes: {debt_out["components"]}

Montecarlo (proxy):
prob_bankruptcy_proxy={mc['prob_bankruptcy_proxy']:.4f}
barrera={mc['barrier_price']:.2f} (frac={mc['barrier_frac']:.3f})

Merton (default real):
PD_2y={pd_merton:.4f}
d1={d1:.4f}  d2={d2:.4f}

Evidencia PDF:
{ev_text}

Pregunta: "¿Es seguro invertir?"
Responde en español en 6-10 líneas y termina con: "No es asesoría financiera."
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    print(resp.choices[0].message.content)

if __name__ == "__main__":
    main()
