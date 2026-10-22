# fun/

Side projects for fun, outside the Big Data Bowl analysis. Competition data stays out of git as in the main project. Public data (nflverse) is allowed here only, on the
user's go-ahead (2026-10-22), downloaded to `data/external/` and never committed.

- [`week1_explosion/`](week1_explosion/): which team put the most explosion on the field in regular-season week 1
  (2023–2025)? Starters' top-speed kinetic energy in the game (joules, from game tracking) and their Combine
  explosive axis, summed per team. See [`week1_explosion/ranking.md`](week1_explosion/ranking.md).
- [`eight_week_explosion/`](eight_week_explosion/): the same measures averaged per game over weeks 1–8
  (2023–2025). See [`eight_week_explosion/ranking.md`](eight_week_explosion/ranking.md).
- [`team_explosion_wins/`](team_explosion_wins/): does a team's (young-starter) explosion predict its season,
  and how much TNT is it? Results from the public nflverse schedule. See
  [`team_explosion_wins/report.md`](team_explosion_wins/report.md).
- [`week1_2026/`](week1_2026/): the 2026 rookie class in joules (public Combine results through the project's
  physics) and week 1 of 2026 (public Next Gen Stats tables, published top speeds). Uses public nflverse data,
  downloaded to `data/external/` (not committed). See [`week1_2026/report.md`](week1_2026/report.md).
