# QueueScore — Charter

Origination intelligence for the Texas power pipeline. Built for the Candid
Intelligence hackathon (Sat 8/8/2026), Track 1.

## 1. Target users
**Deal originators** — the people whose job is finding deals early, at the companies that develop or 
own-and-run power projects.

## 2. Key use cases
- **Pipeline at a glance** — every Texas power project and where it sits in the
  funnel, right now.
- **One project, whole story** — click a project to see it stitched across ERCOT
  + TCEQ on one screen, instead of hopping between five sites.
- **On-thesis filtering** — surface early-stage gas-to-power / behind-the-meter
  data-center power specifically.
- **Timing signal** — a project advancing a stage (permit filed, IA signed) is
  the cue to reach out before the RFP.
- **Who to call** — resolve the company actually driving a project so BD can
  start the relationship early.

## 3. Problem it solves
There is no single source of truth for capital projects in the energy pipeline.
The data is scattered across a dozen disconnected systems, and the same project
appears under a different LLC or name in each one. Today, building the picture of
one project means a human cross-referencing five sites and guessing at the
connections — so you learn about projects late, when they're already a
price-driven bid instead of a relationship built at the concept stage.

## 4. How it solves it
- **Aggregate** ERCOT queue + TCEQ air permits into one clean dataset, kept
  **live** — not a one-time scrape.
- **Entity resolution** — recognize that an ERCOT interconnection request and a
  TCEQ air permit under a holding company are the same project. Fuzzy match on
  company + county to propose; Claude to adjudicate the hard cases; source
  records kept as evidence.
- **Stage inference** — place each project on the funnel (concept → feasibility →
  study → IA → construction → COD) with a confidence level and the filings that
  justify the call.
- **Addictive presentation** — a live, explorable Texas map; click any project
  for its full cross-source story on one screen.

## 5. Deliverable
A working demo + short README. Concretely: a live map/dashboard of Texas power
projects, entity-resolved across ERCOT + TCEQ, each showing an inferred funnel
stage + confidence + linked source filings, filterable to the gas-to-power /
data-center lens, with click-through to a single project's cross-source story.

**Scope (2 people, ~5 hours):** two sources done well, entity resolution and
stage inference that work on real cases, and a genuinely addictive map. Precision
over volume — a short list of right answers, exactly how the hackathon is judged.

---
Sources and API recon: see [SOURCES.md](SOURCES.md).
