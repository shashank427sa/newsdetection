"""
News Detector — claim and headline verification.

Design rules this version follows:
  * No hardcoded "ground truth" facts. Facts change; code does not.
  * No invented confidence numbers. The score is computed from evidence found.
  * The writing-style classifier is a weak signal, reported as such — never
    used on its own to call something true or false.
"""

import difflib
import html
import re
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import pandas as pd
import requests
import streamlit as st
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="News Detector", page_icon="📰", layout="wide")

UA = {"User-Agent": "Mozilla/5.0 (compatible; NewsDetector/2.0)"}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "of", "to", "for", "and", "or", "but", "if", "as",
    "by", "with", "from", "that", "this", "these", "those", "it", "its",
    "has", "have", "had", "will", "would", "can", "could", "should", "may",
    "says", "said", "new", "after", "over", "amid", "into", "about",
}

# Words worth spell-correcting toward. Extend freely.
VOCAB = {
    "tomorrow", "yesterday", "today", "announced", "announcement", "week",
    "month", "holiday", "cancelled", "postponed", "minister", "prime",
    "president", "government", "election", "police", "arrested", "court",
    "supreme", "school", "college", "university", "students", "curfew",
    "lockdown", "shutdown", "closed", "strike", "protest", "budget",
    "vaccine", "hospital", "earthquake", "flood", "cyclone", "rainfall",
}

# Patterns typical of viral misinformation, not proof of falsehood.
SENSATIONAL_PATTERNS = [
    (r"\bshocking\s+truth\b", "sensational framing"),
    (r"\bthey\s+don'?t\s+want\s+you\s+to\s+know\b", "conspiracy framing"),
    (r"\bdoctors?\s+hate\b", "clickbait health framing"),
    (r"\bmiracle\s+(cure|drug|remedy)\b", "miracle-cure claim"),
    (r"\bcures?\s+(cancer|diabetes|aids|hiv)\b", "unsupported cure claim"),
    (r"\b100%\s+(guaranteed|effective|proven)\b", "absolute guarantee"),
    (r"\bsecretly\b|\bsecret\s+(plan|microchip|agenda)\b", "secrecy claim"),
    (r"\bforward\s+(this|to)\s+(all|everyone|10)\b", "chain-message marker"),
    (r"\bwhistleblower\s+(confesses|reveals)\b", "unnamed-source framing"),
    (r"!{2,}", "excessive punctuation"),
]

URGENCY_WORDS = {
    "tomorrow", "today", "tonight", "holiday", "curfew", "lockdown",
    "shutdown", "closed", "cancelled", "postponed", "strike", "evacuate",
}


# ---------------------------------------------------------------------------
# Text handling
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, and fuzzy-correct obvious typos."""
    text = re.sub(r"\s+", " ", text).strip()
    out = []
    for token in text.lower().split():
        core = re.sub(r"[^\w]", "", token)
        if len(core) > 3 and core not in VOCAB:
            match = difflib.get_close_matches(core, VOCAB, n=1, cutoff=0.86)
            if match:
                token = token.replace(core, match[0])
        out.append(token)
    return " ".join(out)


def content_tokens(text: str) -> list:
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def proper_nouns(text: str) -> set:
    """Capitalised words from the ORIGINAL text — usually the key entities."""
    return {
        w.lower()
        for w in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
        if w.lower() not in STOPWORDS
    }


# ---------------------------------------------------------------------------
# Live news retrieval
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_news(query: str, limit: int = 15) -> list:
    clean = re.sub(r"[^\w\s]", " ", query).strip()
    if not clean:
        return []
    url = (
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote(clean)
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )
    items = []
    try:
        resp = requests.get(url, timeout=8, headers=UA)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as exc:  # network down, bad XML, rate limit
        st.session_state["fetch_error"] = str(exc)
        return []

    now = datetime.now(timezone.utc)
    for node in root.findall("./channel/item")[:limit]:
        title = node.findtext("title") or ""
        if not title.strip():
            continue
        title = html.unescape(title)
        # Google News appends " - Publisher" to each headline.
        source = node.findtext("source") or (
            title.rsplit(" - ", 1)[-1] if " - " in title else "Unknown"
        )
        headline = title.rsplit(" - ", 1)[0] if " - " in title else title

        pub_raw = node.findtext("pubDate") or ""
        age_days = None
        when = None
        if pub_raw:
            try:
                when = parsedate_to_datetime(pub_raw)
                age_days = (now - when).days
            except Exception:
                pass

        items.append({
            "headline": headline,
            "source": source.strip(),
            "link": node.findtext("link") or "",
            "published": when,
            "age_days": age_days,
        })
    return items


def _bigrams(tokens: list) -> set:
    return {f"{a} {b}" for a, b in zip(tokens, tokens[1:])}


def score_overlap(claim_tokens: list, entities: set, headline: str) -> float:
    """How much of the claim a headline supports.

    Half the score is single words, half is adjacent word PAIRS. The pair half
    is what stops 'donald trump is pm of india' from matching a headline where
    Trump and India and PM all appear but never in that relationship.
    """
    if not claim_tokens:
        return 0.0
    hl_tokens = [w for w in re.findall(r"\b\w+\b", headline.lower()) if w not in STOPWORDS]
    hl = set(hl_tokens)

    total = hit = 0.0
    for tok in set(claim_tokens):
        weight = 2.0 if tok in entities else 1.0
        total += weight
        if tok in hl or (len(tok) >= 6 and any(h.startswith(tok[:5]) for h in hl)):
            hit += weight
    unigram = hit / total if total else 0.0

    claim_pairs = _bigrams(claim_tokens)
    if not claim_pairs:
        return unigram
    bigram = len(claim_pairs & _bigrams(hl_tokens)) / len(claim_pairs)

    return 0.5 * unigram + 0.5 * bigram


# ---------------------------------------------------------------------------
# Role claims ("X is the PM of Y") — headline overlap cannot judge these
# ---------------------------------------------------------------------------
ROLE_ALIASES = {
    "pm": "prime minister",
    "cm": "chief minister",
    "potus": "president",
    "chairperson": "chairman",
}
ROLES = (
    "prime minister|chief minister|chief justice|president|governor|mayor"
    "|chairman|chairperson|captain|ceo|king|queen|pm|cm|potus"
)

ROLE_PATTERNS = [
    re.compile(rf"^(?P<name>.+?)\s+is\s+(?:the\s+)?(?P<role>{ROLES})\s+of\s+(?P<place>.+)$", re.I),
    re.compile(rf"^(?:the\s+)?(?P<role>{ROLES})\s+of\s+(?P<place>.+?)\s+is\s+(?P<name>.+)$", re.I),
    re.compile(rf"^(?P<place>.+?)'?s?\s+(?:new\s+)?(?P<role>{ROLES})\s+is\s+(?P<name>.+)$", re.I),
]


def parse_role_claim(text: str):
    for pat in ROLE_PATTERNS:
        m = pat.match(text.strip().rstrip(".!?"))
        if not m:
            continue
        role = m.group("role").lower()
        role = ROLE_ALIASES.get(role, role)
        name = re.sub(r"^(the|mr|mrs|ms|dr|shri)\s+", "", m.group("name").strip(), flags=re.I)
        place = m.group("place").strip()
        if len(name.split()) > 4 or not name:
            return None
        return {"role": role, "place": place, "name": name}
    return None


def incumbents_from_headlines(role: str, place: str) -> Counter:
    """Find which names headlines actually attach to this role."""
    abbrevs = [role] + [k for k, v in ROLE_ALIASES.items() if v == role]
    role_re = "|".join(re.escape(a) for a in abbrevs)
    name_re = r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2}"

    after = re.compile(rf"(?:{role_re})\s+(?P<n>{name_re})", re.I)
    before = re.compile(rf"(?P<n>{name_re}),?\s+(?:the\s+)?(?:{role_re})\b", re.I)

    found = Counter()
    for art in fetch_news(f"{role} of {place}", limit=20):
        for rx in (after, before):
            for m in rx.finditer(art["headline"]):
                cand = m.group("n").strip()
                if cand.lower() in {"of", "said", "says", "india", place.lower()}:
                    continue
                found[cand] += 1
    return found


def _same_person(a: str, b: str) -> bool:
    a_parts = [p for p in a.lower().split() if len(p) > 2]
    b_parts = [p for p in b.lower().split() if len(p) > 2]
    if not a_parts or not b_parts:
        return False
    # Surname match is enough; "Modi" == "Narendra Modi".
    return a_parts[-1] == b_parts[-1] or set(a_parts) <= set(b_parts) or set(b_parts) <= set(a_parts)


# ---------------------------------------------------------------------------
# Wikipedia (background context only, never a verdict)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_wikipedia(query: str):
    terms = " ".join(content_tokens(query)[:6])
    if not terms:
        return None
    try:
        hits = wikipedia.search(terms, results=3)
        for hit in hits:
            try:
                page = wikipedia.page(hit, auto_suggest=False)
                return {
                    "title": page.title,
                    "summary": page.summary.split("\n")[0][:400],
                    "url": page.url,
                }
            except Exception:
                continue
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Style classifier — weak signal, explicitly labelled
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_style_model():
    rows = [
        ("The ministry announced a revision to industrial taxation rates effective next quarter.", "REPORTED"),
        ("Researchers published peer-reviewed findings on cardiovascular outcomes in a cohort study.", "REPORTED"),
        ("The summit concluded with three bilateral agreements signed by the visiting delegations.", "REPORTED"),
        ("Police registered an FIR and said an investigation is underway.", "REPORTED"),
        ("The court issued directions to the state on compliance timelines.", "REPORTED"),
        ("The company reported quarterly revenue of 412 crore, up 8 percent year on year.", "REPORTED"),
        ("Officials said the repair work would close one lane until Friday morning.", "REPORTED"),
        ("The central bank held the benchmark rate steady, citing inflation data.", "REPORTED"),
        ("Election officials confirmed polling will be held in three phases.", "REPORTED"),
        ("A spokesperson declined to comment on the ongoing negotiations.", "REPORTED"),
        ("SHOCKING TRUTH! Boiled garlic water cures cancer overnight, doctors hate this!!", "VIRAL"),
        ("Government secretly replaced all city birds with surveillance drones!!!", "VIRAL"),
        ("Whistleblower confesses the moon is hollow and full of aliens, share before deleted!", "VIRAL"),
        ("Miracle magnetic bracelet completely reverses diabetes and aging, 100% guaranteed!", "VIRAL"),
        ("Secret microchips found inside municipal water pipelines, forward this to everyone!", "VIRAL"),
        ("BREAKING!! Holiday declared tomorrow for all schools, forward to all groups fast!!", "VIRAL"),
        ("They don't want you to know this one trick that banks are hiding!", "VIRAL"),
        ("URGENT: Currency notes banned from midnight, withdraw all cash now!!!", "VIRAL"),
        ("Drinking this at 4am burns belly fat instantly, no exercise needed!", "VIRAL"),
        ("Alien spacecraft landed last night, media ordered to stay silent!", "VIRAL"),
    ]
    df = pd.DataFrame(rows, columns=["text", "label"])
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X = vec.fit_transform(df["text"])
    clf = LogisticRegression(C=2.0, max_iter=1000, random_state=42).fit(X, df["label"])
    return clf, vec


def style_signal(text: str) -> dict:
    clf, vec = load_style_model()
    probs = clf.predict_proba(vec.transform([text]))[0]
    classes = list(clf.classes_)
    viral = probs[classes.index("VIRAL")] * 100
    return {"viral_pct": viral, "label": "viral/clickbait" if viral > 60 else "neutral/journalistic"}


def pattern_flags(original: str) -> list:
    flags = []
    for pattern, label in SENSATIONAL_PATTERNS:
        if re.search(pattern, original, flags=re.I):
            flags.append(label)
    letters = [c for c in original if c.isalpha()]
    if len(letters) > 15 and sum(c.isupper() for c in letters) / len(letters) > 0.5:
        flags.append("mostly uppercase")
    return flags


# ---------------------------------------------------------------------------
# Verdict assembly
# ---------------------------------------------------------------------------
def analyze_role_claim(claim: dict, normalized: str, raw: str) -> dict:
    role, place, name = claim["role"], claim["place"], claim["name"]
    found = incumbents_from_headlines(role, place)
    top = [n for n, _ in found.most_common(3)]
    match = next((n for n in found if _same_person(name, n)), None)

    base = {
        "normalized": normalized,
        "flags": pattern_flags(raw),
        "style": style_signal(normalized),
        "wiki": fetch_wikipedia(f"{role} of {place}"),
        "checked": sum(found.values()),
        "nearby": [],
        "matches": [],
        "role_note": (
            f"Read as a role claim: **{name}** holds the office of "
            f"**{role} of {place}**. Checked against who news reports "
            f"currently attach to that office."
        ),
    }

    if match:
        base.update(
            verdict="CONSISTENT WITH CURRENT COVERAGE",
            tone="success",
            reason=f"News coverage associates this office with {match}, matching the claim.",
            strength=min(60 + found[match] * 8, 100),
        )
    elif top:
        base.update(
            verdict="CONTRADICTED BY COVERAGE",
            tone="error",
            reason=(
                f"News coverage associates {role} of {place} with "
                f"{', '.join(top)} — not {name}."
            ),
            strength=0,
        )
    else:
        base.update(
            verdict="COULD NOT VERIFY OFFICE HOLDER",
            tone="warning",
            reason=(
                "No headline clearly attached a name to this office. "
                "Check an official source rather than trusting this result."
            ),
            strength=0,
        )
    return base


def analyze(raw: str) -> dict:
    normalized = normalize_text(raw)
    tokens = content_tokens(normalized)
    entities = proper_nouns(raw) or set(tokens[:3])

    role_claim = parse_role_claim(normalized)
    if role_claim:
        return analyze_role_claim(role_claim, normalized, raw)

    articles = fetch_news(normalized)
    scored = []
    for art in articles:
        art["overlap"] = score_overlap(tokens, entities, art["headline"])
        scored.append(art)
    scored.sort(key=lambda a: a["overlap"], reverse=True)

    strong = [a for a in scored if a["overlap"] >= 0.65]
    partial = [a for a in scored if 0.40 <= a["overlap"] < 0.65]
    fresh = [a for a in strong if a["age_days"] is not None and a["age_days"] <= 7]
    outlets = {a["source"] for a in strong}

    flags = pattern_flags(raw)
    style = style_signal(normalized)
    time_sensitive = bool(URGENCY_WORDS & set(tokens))
    wiki = fetch_wikipedia(normalized) if not strong else None

    # ---- verdict logic: evidence in, verdict out -------------------------
    fresh_outlets = {a["source"] for a in fresh}
    if len(fresh_outlets) >= 2:
        verdict = "CORROBORATED"
        tone = "success"
        reason = (
            f"Matched by {len(fresh)} report(s) from the last 7 days "
            f"across {len(fresh_outlets)} outlet(s)."
        )
    elif strong:
        verdict = "SINGLE-SOURCE MATCH"
        tone = "warning"
        reason = "Found in news coverage, but from too few outlets to treat as confirmed."
    elif partial:
        verdict = "PARTIAL MATCH"
        tone = "warning"
        reason = "Related coverage exists, but no headline supports the specific claim."
    elif time_sensitive:
        verdict = "NO CONFIRMATION — TREAT AS RUMOR"
        tone = "error"
        reason = (
            "Time-sensitive announcements (holidays, closures, curfews) are always "
            "covered by news outlets. Nothing matching was found in the last 7 days."
        )
    elif flags or style["viral_pct"] > 75:
        verdict = "NO CONFIRMATION — VIRAL STYLE"
        tone = "error"
        reason = "No supporting coverage, and the wording follows known viral-misinformation patterns."
    else:
        verdict = "NO CONFIRMATION FOUND"
        tone = "warning"
        reason = "Nothing in the live news index matched this. Absence of coverage is not proof of falsehood."

    # ---- evidence strength, derived not invented -------------------------
    strength = 0
    strength += min(len(outlets), 4) * 15          # up to 60
    strength += 20 if fresh else 0
    strength += min(len(partial), 4) * 5           # up to 20
    strength = min(strength, 100)

    return {
        "verdict": verdict,
        "tone": tone,
        "reason": reason,
        "strength": strength,
        "normalized": normalized,
        "matches": (strong or partial)[:5],
        "nearby": scored[:5] if not (strong or partial) else [],
        "flags": flags,
        "style": style,
        "wiki": wiki,
        "checked": len(articles),
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📰 News Detector")
st.caption(
    "Checks a claim against live news coverage. It reports what evidence exists — "
    "it does not decide what is true."
)

claim = st.text_area(
    "Paste a headline, forwarded message, or claim:",
    placeholder="write here",
    height=130,
)

col_run, col_opt = st.columns([1, 2])
with col_run:
    run = st.button("Check", type="primary", use_container_width=True)
with col_opt:
    show_raw = st.checkbox("Show all retrieved headlines", value=False)

if run:
    if not claim.strip():
        st.warning("Enter something to check.")
        st.stop()

    with st.spinner("Searching live news index…"):
        st.session_state.pop("fetch_error", None)
        res = analyze(claim)

    if st.session_state.get("fetch_error"):
        st.warning(
            "Could not reach the news feed, so this verdict rests on wording alone. "
            f"({st.session_state['fetch_error']})"
        )

    st.divider()

    if res["normalized"] != claim.lower().strip():
        st.info(f"Interpreted as: *{res['normalized']}*")

    if res.get("role_note"):
        st.info(res["role_note"])

    banner = {"success": st.success, "warning": st.warning, "error": st.error}[res["tone"]]
    banner(f"### {res['verdict']}\n{res['reason']}")

    st.progress(res["strength"], text=f"Supporting evidence found: {res['strength']}/100")
    st.caption(f"Compared against {res['checked']} headlines from the live index.")

    if res["matches"]:
        st.subheader("Matching coverage")
        for m in res["matches"]:
            age = f"{m['age_days']}d ago" if m["age_days"] is not None else "date unknown"
            line = f"**{m['headline']}** — {m['source']} · {age} · overlap {m['overlap']:.0%}"
            if m["link"]:
                line += f" · [open]({m['link']})"
            st.markdown("- " + line)

    if res["flags"]:
        st.subheader("Wording flags")
        st.markdown("\n".join(f"- {f}" for f in res["flags"]))
        st.caption(
            "These describe how the message is written, not whether it is false. "
            "Real news is occasionally written this way; rumors usually are."
        )

    st.caption(
        f"Style classifier: {res['style']['label']} "
        f"({res['style']['viral_pct']:.0f}% viral-like). Weak signal — ignore it "
        "when coverage evidence exists."
    )

    if res["wiki"]:
        with st.expander("Background from Wikipedia"):
            st.markdown(f"**{res['wiki']['title']}** — {res['wiki']['summary']}")
            st.markdown(f"[Read more]({res['wiki']['url']})")
            st.caption("Context about the subject. Not evidence for or against the claim.")

    if show_raw and res["nearby"]:
        with st.expander("All retrieved headlines"):
            for n in res["nearby"]:
                st.markdown(f"- {n['headline']} — {n['source']} (overlap {n['overlap']:.0%})")

    st.divider()
    st.caption(
        "Limits: covers only what Google News indexes, skews toward English and India, "
        "and cannot verify claims that news outlets never write about. "
        "For time-sensitive announcements, confirm with the official source directly."
    )