"""NHL web page: docs/index.html (one self-contained file). Built at the end of every nhl_predict.py run.

Shows the upcoming games (from output/picks_<date>.csv) as cards, the pick history (output/ledger.csv), and a Key tab.
"""
from __future__ import annotations

import colorsys
import json
import os
from datetime import date, datetime
from html import escape
from pathlib import Path
from string import Template
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT, SITE = Path("output"), Path("docs")
ET = ZoneInfo("America/New_York")


def F(fmt):
    return fmt.replace("%-", "%#") if os.name == "nt" else fmt


def num(x):
    try:
        x = float(x)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def has(x):
    return x is not None and str(x) not in ("", "nan", "None", "NaN")


def odds(o):
    o = num(o)
    return "" if o is None else f"{o:+.0f}".replace("-", "−")


def pct(p, d=0):
    p = num(p)
    return "—" if p is None else f"{p * 100:.{d}f}%"


def breakeven(o):
    o = num(o)
    if o is None:
        return None
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def mark(res):
    return {"Win": '<span class="mk win">✓</span>', "Loss": '<span class="mk loss">✗</span>',
            "Push": '<span class="mk push">P</span>'}.get(str(res), "")


# ------------------------------------------------------------------------------------------ team colours
def _rgb(h):
    h = str(h or "").lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)) if len(h) == 6 else None


def _hex(c):
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in c)


def _visible(c, dark):
    lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    if (dark and lum < 0.12) or (not dark and lum > 0.82):
        h, l, s = colorsys.rgb_to_hls(*c)
        return colorsys.hls_to_rgb(h, 0.62 if dark else 0.35, max(s, 0.25))
    return c


def team_colors(r):
    out = []
    for dark, suf in ((False, ""), (True, "-d")):
        a, h = _rgb(r.get("away_color")), _rgb(r.get("home_color"))
        a = _visible(a, dark) if a else None
        h = _visible(h, dark) if h else None
        if a and h and sum((x - y) ** 2 for x, y in zip(a, h)) < 0.06:
            hh, ll, ss = colorsys.rgb_to_hls(*h)
            h = colorsys.hls_to_rgb(hh, 0.7 if ll < 0.5 else 0.3, ss)
        out.append(f"--ca{suf}:{_hex(a) if a else 'var(--ink)'};--ch{suf}:{_hex(h) if h else 'var(--faint)'}")
    return ";".join(out)


# ------------------------------------------------------------------------------------------ data
def upcoming() -> pd.DataFrame:
    today = date.today().isoformat()
    files = sorted(f for f in OUT.glob("picks_*.csv") if f.stem.split("_", 1)[1] >= today)
    if not files:
        return pd.DataFrame()
    p = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates("game_id", keep="last")
    led = pd.read_csv(OUT / "ledger.csv") if (OUT / "ledger.csv").exists() else pd.DataFrame()
    if not led.empty:   # games that already started show the pick logged before puck drop, plus the result
        led = led.set_index("game_id")
        for c in ("home_score", "away_score", "last_period", "pick_result", "value_result", "value_units"):
            p[c] = p["game_id"].map(led[c]) if c in led else np.nan
    p["start"] = pd.to_datetime(p["start_utc"], utc=True).dt.tz_convert(ET)
    return p.sort_values(["start", "game_id"])


def ledger() -> pd.DataFrame:
    return pd.read_csv(OUT / "ledger.csv") if (OUT / "ledger.csv").exists() else pd.DataFrame()


# ------------------------------------------------------------------------------------------ html pieces
def ml_text(r):
    h, a = num(r.get("home_ml")), num(r.get("away_ml"))
    if h is None or a is None:
        return "—"
    first, second = ((r["home"], h), (r["away"], a)) if h <= a else ((r["away"], a), (r["home"], h))
    return f"{first[0]} {odds(first[1])} · {second[0]} {odds(second[1])}"


def confidence(r):
    bets = []
    pick = r["pick"]
    price = r["home_ml"] if pick == r["home"] else r["away_ml"]
    bets.append({"bet": "Moneyline", "pick": f"{pick} {odds(price)}", "p": num(r["pick_prob"]), "be": breakeven(price),
                 "value": has(r.get("value_side")) and r.get("value_side") == pick, "tested": True})
    if has(r.get("value_side")) and r["value_side"] != pick:          # value on the underdog
        vp = 1 - num(r["pick_prob"])
        bets.append({"bet": "Moneyline", "pick": f"{r['value_side']} {odds(r['value_ml'])}", "p": vp,
                     "be": breakeven(r["value_ml"]), "value": True, "tested": True})
    if has(r.get("pl_lean")):
        home_side = str(r["pl_lean"]).startswith(r["home"])
        p = num(r["p_home_pl"]) if home_side else (None if num(r["p_home_pl"]) is None else 1 - num(r["p_home_pl"]))
        o = r.get("home_pl_odds") if home_side else r.get("away_pl_odds")
        bets.append({"bet": "Puck line", "pick": f"{r['pl_lean']} {odds(o)}".strip(), "p": p,
                     "be": breakeven(o) or breakeven(-110), "value": False, "tested": False})
    if has(r.get("total_lean")) and num(r.get("total_line")) is not None:
        over = r["total_lean"] == "Over"
        p = num(r.get("p_over")) if over else num(r.get("p_under"))
        o = r.get("over_odds") if over else r.get("under_odds")
        if p is not None:
            push = 1 - (num(r.get("p_over")) or 0) - (num(r.get("p_under")) or 0)
            p = p / (1 - push) if push < 1 else p
        bets.append({"bet": "Total", "pick": f"{r['total_lean']} {num(r['total_line']):g} {odds(o)}".strip(), "p": p,
                     "be": breakeven(o) or breakeven(-110), "value": False, "tested": False})
    bets = [b for b in bets if b["p"] is not None]
    rows = []
    likely = max(bets, key=lambda b: b["p"]) if bets else None
    for b in bets:
        cush = None if b["be"] is None else b["p"] - b["be"]
        tags = ("<em>likeliest</em>" if b is likely else "") + ("<em class=best>value</em>" if b["value"] else "") + \
               ("" if b["tested"] else "<em>lean only</em>")
        tick = "" if b["be"] is None else f'<s style="left:{b["be"] * 100:.1f}%"></s>'
        cu = "" if cush is None else f'<span class="cu {"up" if cush > 0 else "dn"}">{cush * 100:+.1f}</span>'
        rows.append(f'<div class="cf"><span class="cb">{b["bet"]}</span><span class="cp">{escape(b["pick"])}{tags}</span>'
                    f'<span class="cm"><span class="meter"><i style="width:{b["p"] * 100:.0f}%"></i>{tick}</span><b>{pct(b["p"])}</b></span>'
                    f'<span class="cn">needs {pct(b["be"], 1)}</span>{cu}</div>')
    val = [b for b in bets if b["value"]]
    verdict = (f'Value bet: <b>{escape(val[0]["pick"])}</b>. The model gives it {pct(val[0]["p"])} vs the {pct(val[0]["be"], 1)} '
               f'the price needs. In testing these paid when bet <b>early</b> (near the opening line).') if val else \
        "No value bet: the model and the sportsbook roughly agree, so the pick is a lean."
    edge = num(r.get("value_edge"))
    if val and edge is not None and edge >= 0.12:
        verdict += (' <span class="warn">Very large edge: in the NHL that usually means news the model doesn&#39;t have '
                    '(injury, lineup or roster change). Check before betting.</span>')
    return (f'<div class="conf"><div class="conf-h"><span class="lbl">Confidence</span>'
            f'<span class="lbl">hit chance · needs to break even · cushion</span></div>{"".join(rows)}'
            f'<p class="verdict">{verdict}</p></div>')


def stats_rows(r):
    raw = r.get("matchup_stats")
    if not has(raw):
        return []
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return []


def stats_html(r) -> str:
    rows = stats_rows(r)
    if not rows:
        return ""
    A, H = r["away"], r["home"]
    n_a, n_h = sum(x["edge"] == "away" for x in rows), sum(x["edge"] == "home" for x in rows)
    n_e = sum(x["edge"] == "even" for x in rows)
    lead = A if n_a > n_h else H if n_h > n_a else None
    body, group = [], None
    for x in rows:
        if x["group"] != group:
            group = x["group"]
            body.append(f'<tr class="grp"><th colspan="3">{escape(group)}</th></tr>')
        ca = ' class="better"' if x["edge"] == "away" else ""
        ch = ' class="better"' if x["edge"] == "home" else ""
        body.append(f'<tr><td{ca}>{x["away"]}</td><td class="st">{escape(x["label"])}</td><td{ch}>{x["home"]}</td></tr>')
    summary = f'Stat edges: <b>{A} {n_a}</b> · <b>{H} {n_h}</b> · even {n_e}' + (f' <span class="lead">{lead} leads</span>' if lead else "")
    return (f'<details class="stats"><summary><span class="lbl">Team stats</span><span class="sum">{summary}</span></summary>'
            f'<div class="stats-in"><table class="mt"><thead><tr><th>{A}</th><th></th><th>{H}</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table><p class="note">Recency-weighted team ratings (last season still counts early on). '
            f'Highlighted = better; small gaps count as even.</p></div></details>')


def bet_chart(r) -> str:
    rows = []
    ph = num(r["p_home_win"])
    for team, p, o in ((r["away"], 1 - ph, r.get("away_ml")), (r["home"], ph, r.get("home_ml"))):
        rows.append(("Moneyline", f"{team} {odds(o)}".strip(), p, breakeven(o)))
    pf = num(r.get("p_fav_pl"))
    if pf is not None:
        home_fav = str(r.get("home_fav")).lower() == "true"
        fav, dog = (r["home"], r["away"]) if home_fav else (r["away"], r["home"])
        fo = r.get("home_pl_odds") if home_fav else r.get("away_pl_odds")
        do = r.get("away_pl_odds") if home_fav else r.get("home_pl_odds")
        rows.append(("Puck line", f"{fav} −1.5 {odds(fo)}".strip(), pf, breakeven(fo) or breakeven(-110)))
        rows.append(("Puck line", f"{dog} +1.5 {odds(do)}".strip(), 1 - pf, breakeven(do) or breakeven(-110)))
    po, pu = num(r.get("p_over")), num(r.get("p_under"))
    tl = num(r.get("total_line"))
    if po is not None and pu is not None and tl is not None:
        share = po / (po + pu)
        rows.append(("Total", f"Over {tl:g} {odds(r.get('over_odds'))}".strip(), share, breakeven(r.get("over_odds")) or breakeven(-110)))
        rows.append(("Total", f"Under {tl:g} {odds(r.get('under_odds'))}".strip(), 1 - share, breakeven(r.get("under_odds")) or breakeven(-110)))
    best = max((x for x in rows if x[3] is not None), key=lambda x: x[2] - x[3], default=None)
    out, group = [], None
    for x in rows:
        g, bet, p, be = x
        if g != group:
            group = g
            note = "" if g == "Moneyline" else ' <span class="lo">lean only</span>'
            out.append(f'<div class="bc-g">{g}{note}</div>')
        cush = None if be is None else p - be
        tick = "" if be is None else f'<s style="left:{be * 100:.1f}%"></s>'
        top = " top" if x is best and cush is not None and cush > 0 else ""
        cu = "" if cush is None else f'<span class="cu {"up" if cush > 0 else "dn"}">{cush * 100:+.1f}</span>'
        out.append(f'<div class="bc{top}"><span class="bc-b">{escape(bet)}</span><span class="meter big"><i style="width:{p * 100:.1f}%"></i>{tick}</span>'
                   f'<b>{pct(p, 1)}</b><span class="cn">needs {pct(be, 1)}</span>{cu}</div>')
    return ('<p class="note">Dark bar = hit chance (pushes excluded), blue tick = what the price needs to break even. '
            'Only the moneyline showed an edge in testing (when bet early).</p>' + "".join(out))


def stats_chart(r) -> str:
    rows = stats_rows(r)
    if not rows:
        return ""
    out, group = [], None
    for x in rows:
        if x["group"] != group:
            group = x["group"]
            out.append(f'<div class="bf-g">{escape(group)}</div>')
        ap, hp = x.get("ap"), x.get("hp")
        fa = " lose" if x["edge"] == "home" else ""
        fh = " lose" if x["edge"] == "away" else ""
        out.append(f'<div class="bf"><span class="bf-v">{x["away"]}</span>'
                   f'<span class="bf-l"><i class="{fa}" style="width:{0 if ap is None else max(ap, 2)}%"></i></span>'
                   f'<span class="bf-n">{escape(x["label"])}</span>'
                   f'<span class="bf-r"><i class="{fh}" style="width:{0 if hp is None else max(hp, 2)}%"></i></span>'
                   f'<span class="bf-v h">{x["home"]}</span></div>')
    return (f'<div class="bf-head"><span>{r["away"]}</span><span>{r["home"]}</span></div>{"".join(out)}'
            '<p class="note">Bar length = league rank (percentile) for that stat: longer is better, the middle line is league average. '
            'Faded bar = the other team has the edge.</p>')


def matchup_view(r) -> str:
    k = r["start"]

    def team(s):
        logo = r.get(f"{s}_logo")
        img = f'<img src="{escape(str(logo))}" alt="">' if has(logo) else ""
        name = escape(str(r.get(f"{s}_name"))) if has(r.get(f"{s}_name")) else r[s]
        goalie = escape(str(r.get(f"{s}_goalie"))) if has(r.get(f"{s}_goalie")) else "goalie TBD"
        return (f'<div class="mt-team">{img}<span class="mt-name">{name}</span><span class="rec">🥅 {goalie}</span>'
                f'<b class="sc">{num(r[f"proj_{s}"]):.1f}</b></div>')
    return (f'<div class="m-head"><span class="when"><b>{k.strftime(F("%a %b %-d · %-I:%M %p"))} ET</b></span>'
            f'<div class="mt-teams">{team("away")}<span class="mt-at">@</span>{team("home")}</div></div>'
            f'<section class="m-sec"><h3>Chances of each bet hitting</h3>{bet_chart(r)}</section>'
            f'<section class="m-sec"><h3>Team stats</h3>{stats_chart(r)}</section>')


def card(r):
    final = num(r.get("home_score")) is not None
    ph = num(r["p_home_win"]) or 0.5
    k = r["start"]
    tv = escape(str(r["tv"])) if has(r.get("tv")) else ""

    def side(s):
        t = r[s]
        name = escape(str(r.get(f"{s}_name"))) if has(r.get(f"{s}_name")) else t
        logo = r.get(f"{s}_logo")
        img = f'<img src="{escape(str(logo))}" alt="" loading="lazy">' if has(logo) else '<span class="nologo"></span>'
        goalie = escape(str(r.get(f"{s}_goalie"))) if has(r.get(f"{s}_goalie")) else "goalie TBD"
        proj = num(r[f"proj_{s}"])
        score = num(r.get(f"{s}_score"))
        right = (f'<span class="lbl">Final</span><b class="sc">{score:.0f}</b><span class="was">proj {proj:.1f}</span>' if final
                 else f'<span class="lbl">Projected</span><b class="sc">{proj:.1f}</b>')
        fav = " fav" if r["pick"] == t else ""
        return (f'<div class="side{fav}">{img}<div class="tn"><span class="name">{name}</span>'
                f'<span class="rec">🥅 {goalie}</span></div><div class="scw">{right}</div></div>')

    pa = 1 - ph
    bar = (f'<div class="bar"><span class="a" style="flex:{pa:.4f}"></span><span class="h" style="flex:{ph:.4f}"></span></div>'
           f'<div class="barlbl"><span>{r["away"]} {pct(pa)}</span><span>win chance</span><span>{pct(ph)} {r["home"]}</span></div>')
    pl = "—"
    if num(r.get("pl_home_line")) is not None:
        fav_home = num(r["pl_home_line"]) < 0
        fav, o = (r["home"], r.get("home_pl_odds")) if fav_home else (r["away"], r.get("away_pl_odds"))
        pl = f"{fav} −1.5 {odds(o)}".strip()
    tot = "—" if num(r.get("total_line")) is None else f"{num(r['total_line']):g} (o{odds(r.get('over_odds')) or ''})"
    fair = r["fair_home_ml"] if r["pick"] == r["home"] else r["fair_away_ml"]
    pick_res = mark(r.get("pick_result"))
    value = ""
    if has(r.get("value_side")):
        value = (f'<div class="pickrow"><span class="lbl">Value bet</span><span class="val">{r["value_side"]} {odds(r["value_ml"])} '
                 f'<i>{num(r["value_edge"]) * 100:+.1f}% edge</i> {mark(r.get("value_result"))}<span class="tag">Value</span></span></div>')
    status = "confirmed goalies" if r.get("goalie_status") == "confirmed" else "projected goalies"
    mkt = num(r.get("p_mkt_home"))
    detail = f"""<details class="more"><summary>Details</summary><div class="more-in"><dl>
      <div><dt>Opening moneyline</dt><dd>{r['home']} {odds(r.get('open_home_ml')) or '—'} · {r['away']} {odds(r.get('open_away_ml')) or '—'}</dd></div>
      <div><dt>Sportsbook win chance</dt><dd>{'—' if mkt is None else f"{r['home']} {pct(mkt, 1)}"}</dd></div>
      <div><dt>Model win chance</dt><dd>{r['home']} {pct(ph, 1)}</dd></div>
      <div><dt>Goalies</dt><dd>{status}</dd></div></dl></div></details>"""
    return f"""
<article class="game{' is-final' if final else ''}" data-value="{int(has(r.get('value_side')))}" data-kick="{k.isoformat()}" data-final="{int(final)}" style="{team_colors(r)}">
  <header><span class="when"><b>{k.strftime(F('%-I:%M %p'))} ET</b>{' · ' + tv if tv else ''}</span><span class="chip" data-chip></span></header>
  <div class="venue">{escape(str(r['venue'])) if has(r.get('venue')) else ''}</div>
  <div class="teams">{side('away')}{side('home')}{bar}</div>
  <div class="grid3">
    <div class="cell"><span class="lbl">Moneyline</span><span class="val">{ml_text(r)}</span></div>
    <div class="cell"><span class="lbl">Puck line</span><span class="val">{pl}</span></div>
    <div class="cell"><span class="lbl">Market total</span><span class="val">{tot}</span></div>
    <div class="cell mod"><span class="lbl">Projected winner</span><span class="val">{r['pick']} <i>{pct(r['pick_prob'])}</i> {pick_res}</span></div>
    <div class="cell mod"><span class="lbl">Fair moneyline</span><span class="val">{r['pick']} {odds(fair)}</span></div>
    <div class="cell mod"><span class="lbl">Model total</span><span class="val">{num(r['proj_total']):.1f}</span></div>
  </div>
  <div class="picksbox">
    <div class="pickrow"><span class="lbl">Model pick</span><span class="val">{r['pick']} ML {odds(r.get('pick_ml'))} <i>{pct(r['pick_prob'])}</i> {pick_res}</span></div>
    {value}
  </div>
  {confidence(r)}
  {stats_html(r)}
  <button class="open-mx" type="button">Open matchup view <span aria-hidden="true">↗</span></button>
  <template class="mx">{matchup_view(r)}</template>
  {detail}
</article>"""


def history_html(led: pd.DataFrame) -> str:
    g = led[led["pick_result"].notna()] if not led.empty else led
    if g.empty:
        return ('<div class="empty"><b>No graded picks yet.</b><p>Every game is logged before puck drop. Run it again the next '
                'day and last night\'s picks show up here with ✓ or ✗.</p></div>')
    out = []
    for d, sub in sorted(g.groupby("date"), key=lambda x: x[0], reverse=True):
        rows = []
        for _, r in sub.iterrows():
            clv = num(r.get("value_clv"))
            tl = num(r.get("total_line"))
            tot = f"{r['total_lean']} {tl:g} {mark(r.get('total_result'))}" if has(r.get("total_lean")) and tl is not None else "—"
            rows.append(f"<tr><td>{r['away']} <span class=at>@</span> {r['home']}</td>"
                        f"<td class=n>{num(r['away_score']):.0f}–{num(r['home_score']):.0f}{' ' + str(r['last_period']) if str(r.get('last_period')) in ('OT', 'SO') else ''}</td>"
                        f"<td>{r['pick']} {mark(r['pick_result'])}</td>"
                        f"<td>{(str(r['value_side']) + ' ' + odds(r['value_ml']) + ' ' + mark(r['value_result'])) if has(r.get('value_side')) else '—'}</td>"
                        f"<td>{tot}</td>"
                        f"<td>{(str(r['pl_lean']) + ' ' + mark(r.get('pl_result'))) if has(r.get('pl_lean')) else '—'}</td>"
                        f"<td class=n>{'' if clv is None else ('✓' if clv > 0 else '✗')}</td></tr>")
        w = (sub["pick_result"] == "Win").sum()
        l = (sub["pick_result"] == "Loss").sum()
        u = pd.to_numeric(sub["value_units"], errors="coerce").sum()
        out.append(f'<section class="wk"><header><h3>{datetime.fromisoformat(str(d)).strftime(F("%A, %B %-d"))}</h3>'
                   f'<p>Winners {w}–{l} · value bets {u:+.2f} units</p></header><div class="scroll"><table>'
                   f'<thead><tr><th>Game</th><th>Final</th><th>Pick</th><th>Value bet</th><th>Total</th><th>Puck line</th>'
                   f'<th title="value bet beat the closing line">Beat close</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>')
    return "".join(out)


def ribbon(led: pd.DataFrame) -> str:
    g = led[led["pick_result"].notna()] if not led.empty else led
    if g.empty:
        bt = json.loads((OUT / "backtest_summary.json").read_text()) if (OUT / "backtest_summary.json").exists() else {}
        return ('<div class="ribbon"><span class="lbl">Backtest 2019–2026</span>'
                f'<div class="ri"><span>Winners</span><b>{bt.get("winner_pct", 0) * 100:.1f}%</b><em>{bt.get("games", 0):,} games</em></div>'
                '<div class="ri"><span>Value bets at the open</span><b>+2.6% to +5%</b><em>ROI, 2024–26</em></div>'
                '<div class="ri"><span>Beat the closing line</span><b>~70%</b><em>of value bets</em></div></div>'
                '<p class="note">Out-of-sample test. Live tracking starts with today\'s picks.</p>')
    w, l = (g["pick_result"] == "Win").sum(), (g["pick_result"] == "Loss").sum()
    v = g[g["value_result"].notna()]
    vw, vl = (v["value_result"] == "Win").sum(), (v["value_result"] == "Loss").sum()
    u = pd.to_numeric(v["value_units"], errors="coerce").sum()
    clv = pd.to_numeric(g["value_clv"], errors="coerce").dropna()
    return ('<div class="ribbon"><span class="lbl">Live record</span>'
            f'<div class="ri"><span>Winners</span><b>{w}–{l}</b><em>{w / max(w + l, 1):.1%}</em></div>'
            f'<div class="ri"><span>Value bets</span><b>{vw}–{vl}</b><em>{u:+.2f} units</em></div>'
            f'<div class="ri"><span>Beat the close</span><b>{"—" if clv.empty else f"{(clv > 0).mean():.0%}"}</b><em>{len(clv)} value bets</em></div></div>'
            f'<p class="note">{len(g)} games graded, all picks logged before puck drop.</p>')


def build() -> Path:
    SITE.mkdir(exist_ok=True)
    up = upcoming()
    led = ledger()
    days = []
    if not up.empty:
        for d, grp in up.groupby(up["start"].dt.date, sort=True):
            days.append(f'<h2 class="day"><span>{grp["start"].iloc[0].strftime("%A")}</span> {grp["start"].iloc[0].strftime(F("%B %-d"))}'
                        f'<em>{len(grp)} game{"s" if len(grp) > 1 else ""}</em></h2><div class="cards">'
                        + "".join(card(r) for _, r in grp.iterrows()) + "</div>")
    first = up["start"].iloc[0].strftime(F("%b %-d")) if not up.empty else ""
    last = up["start"].iloc[-1].strftime(F("%b %-d")) if not up.empty else ""
    html = PAGE.substitute(
        span=first if first == last else f"{first} – {last}", n_games=len(up),
        n_value=int(up["value_side"].apply(has).sum()) if not up.empty else 0,
        n_hist=int(led["pick_result"].notna().sum()) if not led.empty else 0,
        updated=datetime.now(ET).strftime(F("%a %b %-d, %-I:%M %p ET")), ribbon=ribbon(led),
        rows="".join(days) or '<div class="empty"><b>No games in the next few days.</b></div>', history=history_html(led))
    path = SITE / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


PAGE = Template(r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NHL Picks</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#f4f6f8;--card:#fdfefe;--ink:#12161b;--muted:#6c737c;--faint:#b4bac1;--rule:#e1e6ea;
--accent:#0b72d9;--accent-soft:#e3f0fd;--win:#2b8a52;--loss:#c9382a;--shadow:0 1px 2px rgba(18,22,27,.04),0 8px 24px -12px rgba(18,22,27,.14)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0c0f13;--card:#141920;--ink:#e9eef3;--muted:#8d96a0;--faint:#4b535c;--rule:#232a33;
--accent:#4ea3ff;--accent-soft:#10233a;--win:#62c98c;--loss:#f2806f;--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px -14px rgba(0,0,0,.6)}}
:root[data-theme="dark"]{--bg:#0c0f13;--card:#141920;--ink:#e9eef3;--muted:#8d96a0;--faint:#4b535c;--rule:#232a33;
--accent:#4ea3ff;--accent-soft:#10233a;--win:#62c98c;--loss:#f2806f;--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px -14px rgba(0,0,0,.6)}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1100px;margin:0 auto;padding:0 16px}
.sc,.when,.cell .val,.pickrow .val,table,.ri b,.count,.cf,.barlbl,.chip{font-family:"IBM Plex Mono",ui-monospace,monospace}
button{font:inherit;color:inherit}
.top{position:sticky;top:0;z-index:10;background:color-mix(in srgb,var(--bg) 88%,transparent);backdrop-filter:blur(10px);border-bottom:1px solid var(--rule)}
.top .wrap{display:flex;align-items:center;gap:14px;height:56px}
.brand{font-weight:600;display:flex;align-items:center;gap:8px} .brand i{width:8px;height:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 4px var(--accent-soft)}
.sp{flex:1} .icon{background:none;border:1px solid var(--rule);border-radius:999px;width:34px;height:34px;cursor:pointer;color:var(--muted)}
.mast{padding:52px 0 26px;display:grid;grid-template-columns:auto 1fr;gap:32px;align-items:end}
.mast h1{font-family:"Instrument Serif",Georgia,serif;font-weight:400;font-size:clamp(64px,12vw,128px);line-height:.85;margin:0;letter-spacing:-.02em}
.mast h1 em{color:var(--accent)} .mast h1 small{display:block;font-family:Inter,sans-serif;font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}
.kick{justify-self:end;text-align:right;display:flex;flex-direction:column;gap:4px;color:var(--muted);font-size:13px}
.count{font-size:28px;color:var(--ink)} .count small{font-size:13px;color:var(--muted);margin:0 6px 0 2px}
.ribbon{display:flex;flex-wrap:wrap;align-items:center;gap:10px 30px;padding:16px 0 8px;border-top:1px solid var(--rule)}
.ribbon .lbl{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--accent);font-weight:600}
.ri{display:flex;gap:8px;align-items:baseline} .ri span{color:var(--muted);font-size:13px} .ri b{font-weight:500;font-size:18px} .ri em{font-style:normal;color:var(--muted);font-size:12.5px}
.note{color:var(--muted);font-size:12.5px;margin:0 0 6px}
.tabs{display:flex;gap:6px;margin:30px 0 0;overflow-x:auto;overflow-y:hidden;scrollbar-width:none;border-bottom:1px solid var(--rule)}
.tabs button{background:none;border:0;padding:10px 12px;font-size:14px;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px;white-space:nowrap}
.tabs button[aria-pressed="true"]{color:var(--ink);border-color:var(--accent)} .tabs sup{font-family:"IBM Plex Mono",monospace;font-size:10px;margin-left:3px;color:var(--muted)}
.day{font-family:"Instrument Serif",Georgia,serif;font-weight:400;font-size:30px;margin:34px 0 10px;display:flex;align-items:baseline;gap:10px}
.day span{color:var(--accent);font-style:italic} .day em{font-family:Inter,sans-serif;font-style:normal;font-size:12px;color:var(--muted);margin-left:auto}
.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.game{background:var(--card);border:1px solid var(--rule);border-radius:16px;box-shadow:var(--shadow);display:flex;flex-direction:column;overflow:hidden;transition:transform .18s,border-color .18s}
.game:hover{transform:translateY(-2px);border-color:color-mix(in srgb,var(--accent) 35%,var(--rule))}
.game>header{display:flex;justify-content:space-between;align-items:center;padding:14px 18px 0}
.when{font-size:12px;color:var(--muted)} .when b{color:var(--ink);font-weight:500}
.venue{padding:2px 18px 0;font-size:12px;color:var(--muted);min-height:18px}
.chip{font-size:10px;letter-spacing:.06em;text-transform:uppercase;padding:3px 8px;border-radius:999px;background:var(--accent-soft);color:var(--accent)}
.chip:empty{display:none} .chip.locked{background:var(--rule);color:var(--muted)} .chip.final{background:var(--ink);color:var(--bg)}
.teams{padding:10px 18px 14px}
.side{display:grid;grid-template-columns:34px 1fr auto;gap:12px;align-items:center;padding:6px 0}
.side img,.nologo{width:34px;height:34px;object-fit:contain}
.tn{display:flex;flex-direction:column;line-height:1.25;min-width:0} .name{font-size:17px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rec{color:var(--muted);font-size:12px} .side.fav .name{color:var(--accent);font-weight:600}
.scw{display:flex;flex-direction:column;align-items:flex-end;line-height:1.1} .sc{font-size:28px;font-weight:500} .was{font-size:10.5px;color:var(--muted);font-family:"IBM Plex Mono",monospace}
.bar{display:flex;gap:3px;height:6px;margin-top:10px} .bar span{border-radius:3px;min-width:4px}
.bar .a{background:var(--ca)} .bar .h{background:var(--ch)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .bar .a{background:var(--ca-d)}:root:not([data-theme="light"]) .bar .h{background:var(--ch-d)}}
:root[data-theme="dark"] .bar .a{background:var(--ca-d)} :root[data-theme="dark"] .bar .h{background:var(--ch-d)}
.barlbl{display:flex;justify-content:space-between;font-size:10.5px;color:var(--muted);margin-top:4px}
.lbl{font-family:Inter,sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));border-top:1px solid var(--rule)}
.cell{padding:10px 18px;display:flex;flex-direction:column;gap:3px;border-right:1px solid var(--rule);min-width:0}
.cell:nth-child(3n){border-right:0} .cell:nth-child(n+4){border-top:1px solid var(--rule)}
.cell .val{font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cell.mod{background:color-mix(in srgb,var(--accent-soft) 55%,transparent)} .cell.mod .val{color:var(--accent);font-weight:500}
.cell i,.pickrow i{font-style:normal;color:var(--muted);font-weight:400}
.picksbox{border-top:1px solid var(--rule);padding:10px 18px;display:flex;flex-direction:column;gap:6px}
.pickrow{display:flex;justify-content:space-between;align-items:center;gap:10px} .pickrow .val{font-size:14px;font-weight:600;text-align:right}
.tag{font-family:Inter,sans-serif;font-size:9.5px;text-transform:uppercase;letter-spacing:.06em;background:var(--accent);color:#fff;padding:2px 6px;border-radius:4px;margin-left:8px;vertical-align:2px}
.mk{font-weight:600} .mk.win{color:var(--win)} .mk.loss{color:var(--loss)} .mk.push{color:var(--muted)}
.conf{border-top:1px solid var(--rule);padding:10px 18px 12px;display:flex;flex-direction:column;gap:7px}
.conf-h{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}
.cf{display:grid;grid-template-columns:78px minmax(0,1fr) 110px 86px 44px;gap:8px;align-items:center;font-size:12.5px}
.cb{font-family:Inter,sans-serif;font-size:11px;color:var(--muted)} .cp{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cp em{font-style:normal;font-family:Inter,sans-serif;font-size:9.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);border:1px solid var(--rule);border-radius:4px;padding:1px 4px;margin-left:6px}
.cp em.best{color:#fff;background:var(--accent);border-color:var(--accent)}
.cm{display:flex;align-items:center;gap:6px} .meter{position:relative;flex:1;height:6px;border-radius:3px;background:var(--rule)}
.meter i{position:absolute;inset:0 auto 0 0;border-radius:3px;background:var(--ink)} .meter s{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--accent)}
.cn{color:var(--muted);font-size:11px} .cu{font-size:11.5px;text-align:right} .cu.up{color:var(--win)} .cu.dn{color:var(--loss)}
.verdict{margin:2px 0 0;font-size:12.5px;color:var(--muted)} .verdict b{color:var(--ink)}
.warn{display:block;margin-top:4px;color:#b7791f}
.more{border-top:1px solid var(--rule);margin-top:auto}
.more>summary{list-style:none;cursor:pointer;padding:10px 18px;font-size:12px;color:var(--muted);display:flex;justify-content:space-between}
.more>summary::-webkit-details-marker{display:none} .more>summary::after{content:"+"} .more[open]>summary::after{content:"−"}
.more-in{padding:0 18px 16px} .more dl{margin:0;display:grid;grid-template-columns:1fr 1fr;gap:10px 22px}
.more dt{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em} .more dd{margin:2px 0 0;font-size:13.5px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th{font-family:Inter,sans-serif;font-weight:500;font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;text-align:left;padding:6px 10px 6px 0;border-bottom:1px solid var(--rule)}
td{padding:8px 10px 8px 0;border-bottom:1px solid var(--rule);white-space:nowrap} td.n{text-align:right} .at{color:var(--faint)}
.wk{background:var(--card);border:1px solid var(--rule);border-radius:14px;padding:6px 18px 10px;margin:18px 0;box-shadow:var(--shadow)}
.wk header{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;padding:10px 0}
.wk h3{font-family:"Instrument Serif",Georgia,serif;font-weight:400;font-size:26px;margin:0} .wk p{margin:0;color:var(--muted);font-size:13px} .scroll{overflow-x:auto}
.empty{background:var(--card);border:1px dashed var(--rule);border-radius:14px;padding:26px;margin:24px 0;color:var(--muted);max-width:620px} .empty b{color:var(--ink)} .empty p{margin:6px 0 0}
.key{max-width:760px;color:var(--muted);line-height:1.7;padding:10px 0} .key h2{font-family:"Instrument Serif",Georgia,serif;font-weight:400;font-size:32px;color:var(--ink);margin:26px 0 6px} .key b{color:var(--ink)}
.hidden{display:none!important}
.game{cursor:pointer}
.stats{border-top:1px solid var(--rule)}
.stats>summary{list-style:none;cursor:pointer;padding:10px 18px;display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.stats>summary::-webkit-details-marker{display:none} .stats .sum{font-size:12.5px;color:var(--muted)} .stats .sum b{color:var(--ink)}
.stats .lead{font-size:9.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--accent);border:1px solid var(--accent);border-radius:4px;padding:1px 5px;margin-left:6px}
.stats[open]>summary .sum::after{content:" −"} .stats:not([open])>summary .sum::after{content:" +"}
.stats-in{padding:0 18px 14px}
table.mt td,table.mt th{text-align:center;padding:5px 6px} table.mt td:first-child,table.mt td:last-child{width:26%}
table.mt td.st{font-family:Inter,sans-serif;font-size:12px;color:var(--muted);white-space:normal}
table.mt td.better{color:var(--accent);font-weight:600} table.mt td.better::after{content:" ●";font-size:8px;vertical-align:2px}
table.mt tr.grp th{text-align:left;font-size:10px;letter-spacing:.1em;color:var(--ink);padding-top:12px;border-bottom:1px solid var(--rule)}
.open-mx{border:0;border-top:1px solid var(--rule);background:none;color:var(--accent);font:inherit;font-size:12.5px;padding:10px 18px;text-align:left;cursor:pointer}
.open-mx:hover{background:var(--accent-soft)}
.modal{position:fixed;inset:0;z-index:50;display:none;align-items:flex-start;justify-content:center;padding:40px 16px;background:color-mix(in srgb,#000 45%,transparent);backdrop-filter:blur(4px);overflow-y:auto}
.modal.on{display:flex}
.m-panel{position:relative;width:100%;max-width:780px;background:var(--card);border:1px solid var(--rule);border-radius:18px;box-shadow:0 30px 80px -20px rgba(0,0,0,.45);padding:22px 24px 26px}
.m-close{position:absolute;top:12px;right:12px;border:1px solid var(--rule);background:var(--bg);color:var(--muted);border-radius:999px;width:34px;height:34px;cursor:pointer;font-size:16px}
.m-head{border-bottom:1px solid var(--rule);padding-bottom:14px}
.mt-teams{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:12px;margin-top:10px}
.mt-team{display:flex;flex-direction:column;align-items:center;gap:3px;text-align:center} .mt-team img{width:54px;height:54px;object-fit:contain}
.mt-name{font-weight:600} .mt-at{color:var(--muted)}
.m-sec h3{font-family:"Instrument Serif",Georgia,serif;font-weight:400;font-size:26px;margin:22px 0 4px}
.bc-g,.bf-g{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--ink);font-weight:600;margin:12px 0 4px}
.bc-g .lo{font-weight:400;color:var(--muted);text-transform:none;letter-spacing:0;margin-left:6px}
.bc{display:grid;grid-template-columns:150px 1fr 56px 90px 48px;gap:10px;align-items:center;font-family:"IBM Plex Mono",monospace;font-size:13px;padding:4px 6px;border-radius:8px}
.bc.top{background:var(--accent-soft)} .bc b{text-align:right} .meter.big{height:10px;border-radius:5px} .meter.big i{border-radius:5px}
.bf-head{display:flex;justify-content:space-between;font-weight:600;margin:6px 0}
.bf{display:grid;grid-template-columns:70px 1fr 160px 1fr 70px;gap:8px;align-items:center;padding:3px 0;font-size:12.5px}
.bf-v{font-family:"IBM Plex Mono",monospace} .bf-v.h{text-align:right} .bf-n{text-align:center;color:var(--muted);font-size:11.5px;line-height:1.2}
.bf-l,.bf-r{position:relative;height:12px;background:var(--rule);border-radius:6px;overflow:hidden}
.bf-l i{position:absolute;right:0;top:0;bottom:0;background:var(--ca);border-radius:6px} .bf-r i{position:absolute;left:0;top:0;bottom:0;background:var(--ch);border-radius:6px}
.bf-l::after,.bf-r::after{content:"";position:absolute;top:0;bottom:0;width:1px;background:var(--muted);opacity:.5} .bf-l::after{right:50%} .bf-r::after{left:50%}
.bf i.lose{opacity:.3}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .bf-l i{background:var(--ca-d)}:root:not([data-theme="light"]) .bf-r i{background:var(--ch-d)}}
:root[data-theme="dark"] .bf-l i{background:var(--ca-d)} :root[data-theme="dark"] .bf-r i{background:var(--ch-d)}
@media (max-width:600px){.modal{padding:0}.m-panel{border-radius:0;min-height:100%;padding:18px 14px}
  .bc{grid-template-columns:1fr 70px 50px}.bc .meter,.bc .cn{display:none}
  .bf{grid-template-columns:52px 1fr 100px 1fr 52px;gap:5px;font-size:11.5px}.bf-n{font-size:10.5px}}
footer{margin:70px 0 0;padding:22px 0 44px;border-top:1px solid var(--rule);color:var(--muted);font-size:12.5px}
@media (max-width:860px){.mast{grid-template-columns:1fr;gap:18px;padding-top:36px}.kick{justify-self:start;text-align:left}.cards{grid-template-columns:1fr}}
@media (max-width:520px){.cf{grid-template-columns:62px minmax(0,1fr) 72px 40px}.cf .cn{display:none}.conf-h .lbl:last-child{display:none}.cell{padding:9px 10px}.cell .val{font-size:12px;white-space:normal}.name{font-size:15px}.sc{font-size:24px}}
</style></head>
<body>
<header class="top"><div class="wrap"><div class="brand"><i></i><span>NHL model</span></div><span class="sp"></span>
<button class="icon" id="theme" aria-label="Toggle dark mode">◐</button></div></header>
<div class="wrap">
<section class="mast">
  <h1><small>NHL · 2026–27</small>Game <em>day</em></h1>
  <div class="kick"><span>$span · $n_games games</span><span class="count" id="countdown">—</span><span id="countlbl">until next puck drop</span>
  <span>Updated $updated</span></div>
</section>
$ribbon
<nav class="tabs" id="tabs">
  <button data-t="all" aria-pressed="true">All games<sup>$n_games</sup></button>
  <button data-t="value" aria-pressed="false">Value bets<sup>$n_value</sup></button>
  <button data-t="history" aria-pressed="false">Pick history<sup>$n_hist</sup></button>
  <button data-t="key" aria-pressed="false">Key</button>
</nav>
<main>
<div id="board">$rows
  <div id="novalue" class="empty hidden"><b>No value bets right now.</b><p>A value bet needs the model's win chance to beat the sportsbook's by 6+ points.</p></div>
</div>
<div id="history" class="hidden">$history</div>
<div id="key" class="key hidden">
  <h2>Reading a game card</h2>
  <p><b>Teams and goalies.</b> 🥅 is the starting goalie: projected from current rosters and recent starts (on the second night
  of a back-to-back, the goalie who didn't play last night) until you confirm starters in <b>goalie_overrides.csv</b>.
  <b>Projected</b> is each team's expected goals; after the game it shows the final score.</p>
  <p><b>Win-chance bar</b>, in team colors. The team in blue is the model's pick to win.</p>
  <p><b>Top row = the sportsbook</b> (DraftKings via ESPN): moneyline for each team, the puck line (the favorite gives 1.5 goals),
  and the total with its over price. <b>Blue row = the model</b>: projected winner and win chance, the <b>fair moneyline</b>
  (the price the model thinks is right), and the model's total.</p>
  <p><b>Value bet</b>: the model's win chance beats the sportsbook's (after removing its margin) by 6+ points. In the 2024–26 test those bets
  returned about +2.6% to +5% when placed at the <b>opening</b> line and beat the closing line ~70% of the time, but lost at the closing line.
  So run it in the morning and bet value picks early.</p>
  <p><b>Confidence</b>: each bet's hit chance (dark bar), the win rate the price needs to break even (tick), and the cushion between them.
  All three were tested against 2024–26 closing prices and corrected so the percentages are honest. Only the <b>moneyline</b> showed an edge
  (value bets placed early). <b>Puck line and total are marked "lean only"</b>: betting the model's side there came out about break-even.</p>
  <h2>Pick history</h2>
  <p>Every pick is logged before puck drop and locked when the game starts; the next run grades it. <b>Beat close</b> shows whether a value
  bet's price was better than the closing price, the fastest honest sign that the edge is real.</p>
  <h2>How the model works</h2>
  <p>Built the way Rob Pizzola lays out a hockey model: team ratings from 5-on-5 expected goals, shot share, power play and penalty kill
  (recency-weighted, with last season fading as a stabiliser), a goalie model of goals saved above expected (shrunk toward average for
  small samples), and game-day factors (starting goalies, home ice, back-to-backs). A Poisson model turns those into each team's expected goals,
  then win chance, puck line and total. Backtest 2019–26: 59.3% winners with calibrated probabilities.</p>
</div>
</main>
<div class="modal" id="modal" role="dialog" aria-modal="true" aria-label="Matchup view"><div class="m-panel">
  <button class="m-close" id="m-close" aria-label="Close">✕</button><div id="m-body"></div></div></div>
<footer>Research and entertainment only. No model guarantees profit; bet responsibly. Data: NHL API, MoneyPuck.com, ESPN / DraftKings.</footer>
</div>
<script>
(function(){
  var root=document.documentElement;
  function store(k,v){try{localStorage.setItem(k,v)}catch(e){}} function load(k){try{return localStorage.getItem(k)}catch(e){return null}}
  var tabs=document.querySelectorAll('#tabs button'),board=document.getElementById('board'),hist=document.getElementById('history'),
      key=document.getElementById('key'),none=document.getElementById('novalue');
  function tab(t){
    tabs.forEach(function(b){b.setAttribute('aria-pressed',b.dataset.t===t?'true':'false')});
    board.classList.toggle('hidden',!(t==='all'||t==='value'));hist.classList.toggle('hidden',t!=='history');key.classList.toggle('hidden',t!=='key');
    var shown=0;board.querySelectorAll('.game').forEach(function(g){var h=t==='value'&&g.dataset.value!=='1';g.classList.toggle('hidden',h);if(!h)shown++});
    board.querySelectorAll('.day').forEach(function(d){var grid=d.nextElementSibling,any=[].some.call(grid.children,function(g){return !g.classList.contains('hidden')});
      d.classList.toggle('hidden',!any);grid.classList.toggle('hidden',!any)});
    none.classList.toggle('hidden',!(t==='value'&&shown===0));store('nhl-tab',t);
  }
  tabs.forEach(function(b){b.addEventListener('click',function(){tab(b.dataset.t)})}); tab(load('nhl-tab')||'all');
  var th=load('nhl-theme'); if(th) root.dataset.theme=th;
  document.getElementById('theme').addEventListener('click',function(){var dark=root.dataset.theme?root.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme=dark?'light':'dark';store('nhl-theme',root.dataset.theme)});
  var modal=document.getElementById('modal'),mbody=document.getElementById('m-body');
  function openMx(card){var t=card.querySelector('template.mx');if(!t)return;mbody.innerHTML='';mbody.appendChild(t.content.cloneNode(true));
    modal.querySelector('.m-panel').setAttribute('style',card.getAttribute('style')||'');modal.classList.add('on');document.body.style.overflow='hidden';modal.scrollTop=0}
  function closeMx(){modal.classList.remove('on');document.body.style.overflow=''}
  document.querySelectorAll('.game').forEach(function(card){card.addEventListener('click',function(e){
    if(e.target.closest('details,summary,a,button:not(.open-mx)'))return;openMx(card)})});
  document.getElementById('m-close').addEventListener('click',closeMx);
  modal.addEventListener('click',function(e){if(e.target===modal)closeMx()});
  document.addEventListener('keydown',function(e){if(e.key==='Escape')closeMx()});
  var games=[].slice.call(document.querySelectorAll('.game'));
  function fmt(ms){var m=Math.floor(ms/6e4),d=Math.floor(m/1440),h=Math.floor(m%1440/60),mm=m%60;return d?d+'d '+h+'h':h?h+'h '+mm+'m':mm+'m'}
  function tick(){var now=Date.now(),next=null;
    games.forEach(function(g){var t=Date.parse(g.dataset.kick),c=g.querySelector('[data-chip]');
      if(g.dataset.final==='1'){c.textContent='Final';c.className='chip final'} else if(t<=now){c.textContent='Locked';c.className='chip locked'}
      else{c.textContent='in '+fmt(t-now);c.className='chip';if(next===null||t<next)next=t}});
    var el=document.getElementById('countdown'),lbl=document.getElementById('countlbl');
    if(next===null){el.textContent='All games locked';lbl.textContent=''}
    else{var m=Math.floor((next-now)/6e4),d=Math.floor(m/1440),h=Math.floor(m%1440/60),mm=m%60;
      el.innerHTML=(d?d+'<small>d</small>':'')+h+'<small>h</small>'+(mm<10?'0':'')+mm+'<small>m</small>'}}
  tick(); setInterval(tick,30000);
})();
</script>
</body></html>
""")
