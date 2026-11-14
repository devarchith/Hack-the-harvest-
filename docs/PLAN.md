# Team plan (draft for review)

Status: **draft, Sat 11:45 AM.** Nothing below the "Facts" section is final. Each assumption and proposed
decision has an ID so we can discuss it by number. Mark each **Agree**, **Change** or **Drop**, and add your
initials.

## What the new information changed

The organizers' starter kit (Keeboh) and decks arrived after our earlier planning. They change three things:

1. **We have real labels, but very few.** There are 44 labeled field-years from one USDA research farm in
   Maryland. The Technical score is measured on them.
2. **The organizers say what they value.** Clear hypotheses, transparent baselines, honest validation and
   uncertainty "matter as much as predictive accuracy". The USDA deck asks for "earlier than January" and
   "explainable to a skeptic".
3. **Presentation is 40% of the score**, and ties go to Presentation.

## Facts (from the organizers; settled)

| ID | Fact |
|---|---|
| F1 | The evaluation file `04_Evaluation/BARC_Corn.2021-2023.gpkg` has 83 field polygons at BARC (Beltsville, MD). There are 44 labeled field-years: 2021 has 14 Grain and 4 Silage, 2022 has 3 and 4, 2023 has 15 and 4. Blank labels are not negatives. |
| F2 | 11 of the 12 Silage cases are on Central Farm (which also has 14 Grain). Silage fields are larger: median 20 acres against 14 for Grain. |
| F3 | We must report a confusion matrix, precision and recall for both classes, counts by year and class, the split design, and error sources and limits. Random field splits are discouraged. |
| F4 | Rubric (25 points): Technical Quality 20%, scored on precision, recall, false positive rate, response time and modeling strategy. Innovation 20%, for a new data source or signal. Feasibility and Mission 20%. Presentation 40%. |
| F5 | Only public data. Document every source: URL, dates, query, processing. |
| F6 | Compute: a B200 GPU, 256 CPU cores and about 1.5 TB of RAM. The catalogs (Planetary Computer, Earth Search) are reachable. The team gets 10 GB of storage in total. |
| F7 | Submission: only Brandon (the lead) submits, in Keeboh, by **9:00 AM Sunday**. Submitting is final, ends access for everyone, and copies the whole workspace: notebooks with saved outputs, plus the presentation file. |
| F8 | Everyone checks in individually Sunday between 9:00 and 9:30 AM. At least 5 members must be checked in on both days to stay eligible for every prize. |
| F9 | Aggie Square closes to us at 3 PM. Then the hotel workspace (Courtyard by Marriott, 4422 Y St) runs from 3 PM to 9 AM. |

## Assumptions to check

| ID | Assumption | Basis | How and when we find out |
|---|---|---|---|
| A1 | At BARC, silage is cut weeks before grain is harvested (late Aug to Sep against Oct to Nov) | USDA deck; general agronomy | Plot all field curves **with labels hidden** once preprocessing is done (about 4 PM) |
| A2 | Post-harvest residue (NDTI) separates silage (bare soil) from grain (stalks) | USDA deck; our pre-event tests in CA and WI (weaker in NY and PA) | One blind score of the frozen rule (step D2) |
| A3 | There are enough clear images around harvest | Partly checked: gaps of 16 to 28 days inside the silage window in 2022 and 2023 | Count usable readings per field after per-field cloud masking |
| A4 | Sentinel-2 values need a correction between 2021 and 2022 to 2023 | Processing versions differ between those years (checked) | Compare one field across years |
| A5 | **The judges score us on the same 44 labels we have, not a hidden set** | The challenge statement says "validate against the provided evaluation data" | **Ask the organizers on Discord** |
| A6 | **"Response time" means how soon after harvest we can make the call** | Not defined in the rubric | **Ask the organizers on Discord** |
| A7 | Judges prefer an interpretable rule to a complex model | The challenge text and the USDA deck say so | Not testable; it is a judgment call |
| A8 | Cover crops planted after silage harvest are common at BARC and may confuse the residue signal | Maryland's cover-crop program; the included papers | Look for fall green-up in the curves |
| A9 | Any model tuned on 44 labels will overfit | Small-sample reasoning | Leave-one-year-out results (step D3) |
| A10 | The GPU is not needed for the core pipeline | 83 fields × 3 years is CPU-sized work | Time the first pull |

## Proposed decisions

| ID | Proposal | Why | Alternative |
|---|---|---|---|
| D1 | **Core method: an interpretable rule** combining harvest timing (the greenness cliff) and post-harvest residue (NDTI), with "no call" when the images are too cloudy | Few labels; organizers value explainability (A7) | A machine-learning classifier on the same features, kept as a comparison |
| D2 | **Freeze the rule's thresholds before seeing the labels**, then score once. One person holds the labels and the scoring notebook | Keeps 44 labels honest | Tune openly, and report it as tuned |
| D3 | **Leave one year out** as the main split. There is no spatial holdout, because 11 of 12 silage cases are on one farm; say so | The challenge asks for a year or spatial holdout (F3) | — |
| D4 | **Report baselines on the same rows:** always grain; field size; farm; drop date only; NDTI only; cliff only | Shows the signal is not just "which farm" or "big field" (F2) | — |
| D5 | **Data from STAC catalogs inside Keeboh**, not Earth Engine | Reachable (F6); judges can rerun it; no quota | Earth Engine |
| D6 | **Preprocessing first, finished by about 4 PM**, labels hidden ([PREPROCESSING.md](PREPROCESSING.md)) | Data quality decides accuracy with this few labels | — |
| D7 | **Deliverables:** pipeline notebook; results table with error ranges; a "how early can we call it" chart (A6); one evidence card per field; failure analysis; deck | Matches F3 and F4 | — |
| D8 | **Extras chosen at 7 PM:** (a) county check against NASS totals for 1 or 2 Maryland dairy counties; (b) Nemotron agent, only with one dedicated person from the start; (c) GPU foundation model, low priority; (d) radar, only if clouds cause the misses | Core first | — |
| D9 | **Freeze the numbers at 11 PM.** Brandon submits at about **8:30 AM** after a checklist (save notebooks with outputs, stop kernels, check the file list) | Submitting is final (F7) | — |
| D10 | **The Keeboh workspace is the source of truth**; GitHub mirrors it. Keys only in `.env` | F7; everyone can read workspace files | — |
| D11 | Mention AI coding tools and pre-event research in one line of the deck | Transparency | Leave it out |

## Open questions for the team

1. **Roles:** who owns data, signal, validation, deck and visuals, plus the Nemotron agent and the county check
   if we do them?
2. **Nemotron prize:** go for it (one person, full time) or skip?
3. **Name and story** for the product and deck.
4. **Discord questions** (A5, A6). Who posts them?

## Timeline (proposed)

| When | What |
|---|---|
| By 12:15 | Review this page, assign roles, post the Discord questions |
| 12:15 to 4 PM | Preprocessing (D6); deck skeleton in parallel. Move to the hotel at 3 PM; save first |
| 4 PM | **Data checkpoint:** look at the curves with labels hidden; confirm or change D1 |
| 4 to 7 PM | Frozen rule scored once; baselines; evidence cards; timeliness chart |
| 7 PM | **Extras checkpoint** (D8) |
| 7 to 11 PM | Leave-one-year-out; extras; failure analysis |
| 11 PM | **Freeze numbers.** Then final notebook, deck, rehearsals, sleep in shifts |
| About 8:30 AM | Brandon submits (deadline 9:00). Everyone checks in by 9:30 |
