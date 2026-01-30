import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import re
from rag.qa_engine import search


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extract_two_numbers_after_label(text: str, label_regex: str, window: int = 220):
    t = norm(text)
    m = re.search(label_regex, t, flags=re.IGNORECASE)
    if not m:
        return None

    tail = t[m.end(): m.end() + window]
    nums = re.findall(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\b", tail)

    vals = []
    for n in nums:
        v = float(n.replace(",", ""))
        if 100 <= v <= 20000:
            vals.append(v)
        if len(vals) >= 2:
            return vals[0], vals[1]
    return None


def extract_one_number_after_label(text: str, label_regex: str, window: int = 220):
    t = norm(text)
    m = re.search(label_regex, t, flags=re.IGNORECASE)
    if not m:
        return None

    tail = t[m.end(): m.end() + window]
    nums = re.findall(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\b", tail)
    for n in nums:
        v = float(n.replace(",", ""))
        if 100 <= v <= 20000:
            return v
    return None


def extract_debt_total_musd():
    docs = search(
        "Total debt and finance leases $ 2,456 $ 5,757 debt and finance leases net of current portion "
        "VIEs current portion of debt and finance leases 2,114 debt and finance leases net of current portion 1,834",
        k=30
    )

    cons_chunk = None
    vie_chunk = None

    for d in docs:
        t = norm(d["text"])
        # consolidado: tiene la frase exacta y NO menciona VIEs
        if cons_chunk is None and re.search(r"Total debt and finance leases", t, re.IGNORECASE) and "VIE" not in t:
            cons_chunk = d
        # VIEs: menciona VIEs y tiene current portion + net of current portion
        if vie_chunk is None and re.search(r"\bVIEs\b|\bVariable Interest Entity\b", t, re.IGNORECASE) and re.search(r"Current portion of debt and finance leases", t, re.IGNORECASE):
            vie_chunk = d

    evidence = []

    cons_total = cons_net = None
    if cons_chunk:
        cons_text = cons_chunk["text"]
        cons_page = cons_chunk["meta"]["page_number"]

        pair = extract_two_numbers_after_label(cons_text, r"Total\s+debt\s+and\s+finance\s+leases")
        if pair:
            cons_total, cons_net = pair

        evidence.append((cons_page, "CONSOLIDADO", norm(cons_text)[:900]))

    vie_cur = vie_long = None
    if vie_chunk:
        vie_text = vie_chunk["text"]
        vie_page = vie_chunk["meta"]["page_number"]

        vie_cur = extract_one_number_after_label(vie_text, r"Current\s+portion\s+of\s+debt\s+and\s+finance\s+leases")
        vie_long = extract_one_number_after_label(vie_text, r"Debt\s+and\s+finance\s+leases\s*,?\s*net\s+of\s+(?:the\s+)?current\s+portion")

        evidence.append((vie_page, "VIEs", norm(vie_text)[:900]))

    missing = []
    if cons_total is None: missing.append("Consolidado total (2456)")
    if cons_net is None: missing.append("Consolidado net (5757)")
    if vie_cur is None: missing.append("VIE current (2114)")
    if vie_long is None: missing.append("VIE long (1834)")

    debt_total = None
    if not missing:
        debt_total = float(cons_total + cons_net + vie_cur + vie_long)

    return {
        "debt_total_musd": debt_total,
        "components": {
            "consolidated_total_musd": cons_total,
            "consolidated_net_musd": cons_net,
            "vie_current_musd": vie_cur,
            "vie_long_musd": vie_long,
        },
        "missing": missing,
        "evidence": evidence,
    }


if __name__ == "__main__":
    out = extract_debt_total_musd()
    print(out)
