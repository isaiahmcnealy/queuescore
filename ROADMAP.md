# Project Radar — Roadmap

Bite-sized chunks for the Candid hackathon (Sat 8/8/2026, ~5 hrs, 2 people).
See [CHARTER.md](CHARTER.md) for the goal and [SOURCES.md](SOURCES.md) for the data.

## How to read this
Each milestone adds something we can actually show on screen. We get a working
map with real data up first, then make it smarter. If we run out of time, we
still have something that runs and looks great.

**Two rules that settle every trade-off:**
1. **Always have a working demo.** Never be more than one milestone away from
   something we can show.
2. **A few right answers beat a long messy list.** Two sources done well wins —
   that's how the judges score it.

**Who does what:** **You** = data pipeline, the map, the project detail view,
putting it all together, deploying. **Camille** = pulling and cleaning the TCEQ
permit data, matching projects across sources, working out each project's stage.

## The bar we must hit (~T+3.5h)
A live map of Texas power projects where clicking one shows its full story pulled
from both ERCOT and TCEQ, with a label for how far along it is. Everything after
that is making it smarter and prettier.

---

## Milestones

| # | What you can show at the end | Owner | ~Time |
|---|---|---|---|
| M0 | The new project set up and running on fake data | You | 20m |
| M1 | Real data flowing in from both ERCOT and TCEQ | Camille + You | 45m |
| M2 | An interactive map of Texas with every project on it | You | 45m |
| M3 | The same project matched across both sources | Camille + You | 60m |
| M4 | A label for how far along each project is | Camille | 45m |
| M5 | Click a project → its full story on one screen | You | 45m |
| M6 | Focused on the projects Candid cares about, and polished | Both | 30m |
| M7 | Proof it stays current + README + demo script | Both | 20m |

### M0 — Set up the project
Clear out the old prediction-model code, create the empty files for the new app,
and agree on the **one table format** both of us will fill data into (so our work
fits together later).
**What you can show:** the app runs on fake data, and the shared table format is
locked in.

### M1 — Get the real data in
Pull the ERCOT connection queue (already working) and the TCEQ permit list (from
their public data feed). Put both into the same table format and save a copy so
the app works offline.
**What you can show:** one command returns a combined table of Texas power
projects — each with its location, name, county, and current status.

### M2 — Put it on a map
Show every project as a dot on an interactive map of Texas, colored by where it
came from, with the name popping up on hover. This is the part people will want
to keep looking at, so we build it early and never risk cutting it.
**What you can show:** the map is live with real data and is fun to pan and zoom.

### M3 — Match the same project across both sources
The same power plant often shows up in ERCOT under one company name and in TCEQ
under a different one. Automatically spot that they're the same project — compare
names within the same county, and let Claude settle the tricky cases — then link
their records together and keep the reason they matched.
**What you can show:** clicking a project pulls up all of its records from both
sources, with a note on why they were linked.

### M4 — Work out how far along each project is
Place each project on its journey: **idea → early planning → engineering studies
→ grid-connection agreement signed → under construction → up and running.** We
read this from clues in the data (ERCOT's study phase, whether the connection
agreement is signed, the permit's status). Show how sure we are and which
documents back it up.
**What you can show:** each project has a stage label, a confidence level, and the
filings that prove it.

### M5 — The project detail view
Click a project on the map and see its whole story on one screen: what stage it's
at, a timeline of what happened across both sources, the source documents, and
which company is behind it.
**What you can show:** the "one project, whole story in one place" view, working
end to end.

### M6 — Focus and polish
Filter and rank down to what Candid cares about most (early-stage gas-to-power and
data-center power). Apply the color palette, tidy the layout, and add a short
ranked list of "worth reaching out about right now."
**What you can show:** the focused view is the default and looks intentional.

### M7 — Prove it stays current, and write it up
Add a "last updated" time and a refresh button (shows it's not a one-time
download). Write the README (what we built, our sources, what's next) and a
2-minute demo script.

---

## What must ship vs. what can slip
- **Must ship:** M0 → M1 → M2 → M3 → M5. That alone is the whole story: live,
  pulled from two sources, matched, and shown on a map.
- **Very important:** M4 (how far along each project is) — the second thing the
  judges reward.
- **Cut these first if behind:** the ranked list in M6, and the auto-refresh in
  M7 (a manual refresh + a "last pulled" time is enough to prove it's live).

## Suggested timeline
| Clock | Focus |
|---|---|
| T+0:00 | M0 set up (together, fast) |
| T+0:20 | Split up: Camille → TCEQ data + matching; You → ERCOT data + the map |
| T+1:30 | Put M1 + M2 together — first live-map check-in |
| T+2:00 | You → M5 detail view; Camille → M4 how-far-along |
| T+3:00 | Put M3 + M4 + M5 together — the must-ship demo works |
| T+3:30 | **Stop adding features.** Polish and focus (M6) |
| T+4:15 | README + demo script (M7); rehearse the 2-minute demo |
| T+5:00 | Submit |

## What could go wrong, and the backup plan
- **The TCEQ data feed is stubborn** → fall back to their bulk spreadsheet
  download (in SOURCES.md). Don't lose more than 30 minutes here.
- **Matching is noisy** → only keep the confident matches. A short correct list
  beats a long messy one — on purpose.
- **Behind at T+3:30** → cut the M6 ranked list and the M7 auto-refresh. Never
  cut the map or the cross-source matching — those are the demo.
