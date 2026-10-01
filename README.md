# NHL model

Daily NHL picks built the way Rob Pizzola lays out a hockey model (team ratings → goalie model → game-day
adjustments → prices), using today's public data. Separate from the football project.

## Running it

- **First time:** install Python 3.11+ (tick "Add python.exe to PATH"), then double-click **`setup.bat`**.
- **Game days:** double-click **`run_daily.bat`**. It grades last night's picks, refreshes the data, and makes picks for
  today and the next two days.
- **Starting goalies** are projected from current rosters and recent starts; on the second night of a back-to-back
  it assumes the goalie who didn't play last night. Once starters are confirmed (usually late morning or afternoon),
  add them to **`goalie_overrides.csv`**, one line per team, for example `2026-10-01,TOR,Stolarz`, and run it again
  before puck drop. Games that have started are locked in the history.

## What you get

| file | what it is |
|---|---|
| `output/picks_<date>.csv` | that day's board: goalies, projected score, pick and win %, fair vs sportsbook moneyline, value flag, total and puck-line leans |
| `output/ledger.csv` | full pick history, logged before puck drop, graded after |
| `output/history.csv` | the same history in a simple one-row-per-game layout for Excel / Google Sheets |
| `output/backtest_report.txt` | the honest walk-forward test (`python nhl_backtest.py`) |

**Value bet:** the model's win chance beats the sportsbook's (after removing its margin) by 6+ percentage points.
In the 2024-26 backtest those bets returned about +2.6% to +5% **at the opening line** and beat the closing line about 70%
of the time, but **lost** at the closing line. So run it in the morning and bet value picks early. The history records
whether each value bet beat the closing line (`valueBeatClose`), which is the fastest sign the edge is real.
Puck-line and total leans are the side with the better expected value at the listed price. They're leans only, since
the model's goal-count probabilities haven't been tested against those markets.

## How it works

1. **Team ratings** (Rob's Corsi / shooting / save framework, modernised): 5-on-5 expected goals for and against per 60
   minutes (score and venue adjusted), shot-attempt share, power-play and penalty-kill expected goals, penalty
   tendencies, and finishing (goals minus expected goals, shrunk heavily because shooting luck regresses).
   Recency-weighted; last season carries over as a stabiliser and fades as the new season's games pile up.
2. **Goalie model:** goals saved above expected per expected goal faced, weighted toward recent seasons and pulled
   hard toward league average until a goalie has faced a real sample.
3. **Game day:** starting goalies, home ice, back-to-backs.
4. **Prices:** a Poisson regression gives each team's expected **regular** (non-empty-net) goals. An **empty-net step**
   then adds late empty-netters the way they actually happen (measured on 13,170 games: a team up 1 adds one 29% of the
   time, up 2 about 59%). Those become win chance (ties go to OT/shootout), puck line and total, calibrated on past
   seasons and corrected against 2024–26 closing prices.

## Known limits

- **Early season:** ratings lean on last season and don't know about summer trades and signings beyond goalies (which
  come from current rosters). In the backtest, early-season value bets still did fine (122 bets in the first 30 days,
  small sample), but treat very large edges (10%+) with suspicion: they often mean the model is missing news.
- **No skater injuries yet** (Rob's player-value-above-replacement adjustment). That's the natural next step.

Data: NHL API, MoneyPuck.com (free for non-commercial use, with credit), ESPN / DraftKings lines.
