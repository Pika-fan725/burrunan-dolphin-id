# Burrunan Dolphin Identification Program

Identifies individual dolphins from a photo of their dorsal fin, by matching the
pattern of nicks and notches along the fin's trailing edge.

This is Part 1 of the Gippsland Lakes project — the part that goes in the final
display. The monitoring buoys and the sightings database are separate stages and
are not built here.

Originally built as a school project, and still the answer to that brief; it has
since grown well past what was needed for it.

---

## Background

### The animal

The Burrunan dolphin (*Tursiops australis*) was only recognised as a separate
species in 2011, when Kate Charlton-Robb of Monash University and her colleagues
described it. The name comes from the Boonwurrung, Woiwurrung and Taungurung
languages and means roughly "large sea fish of the porpoise kind".

It is one of the rarest dolphins in the world. Only **two resident populations
are known, both in Victoria**: one in Port Phillip, and one in the **Gippsland
Lakes**. Estimates for the Gippsland Lakes population run to somewhere around
50–65 animals, depending on the survey and the year. That is the entire
population — not a sample of it.

Since October 2021 the species has been listed as **Critically Endangered**
under Victoria's *Flora and Fauna Guarantee Act 1988*, upgraded from Endangered
in 2013. It has no federal listing, partly because there is not enough data on
it. The main pressures are its very small and genetically isolated population,
and the fact that both groups live right alongside towns, farmland and busy
boat traffic.

### The problem this project addresses

When a population is this small, every individual matters, and researchers need
to know which dolphins are still around, where they go, and whether the
population is holding steady. The way that is done is **photo-identification**:
photograph a dolphin's dorsal fin, then match it against a catalogue of known
animals.

This works because a dolphin is born with a smooth fin and collects nicks and
notches through its life — from fights, boat strikes, fishing line and ordinary
wear. The pattern that builds up is unique. The Marine Mammal Foundation, who
run the Burrunan surveys, describe it as a "FINgerprint".

The catch is that photo-ID is slow and inconvenient. It means getting a boat
out, finding the dolphins, taking thousands of photos, and then a person sitting
down and matching them to the catalogue by eye, one at a time. Surveys therefore
happen only now and then, and the data has gaps.

### The proposition

Two things that work together:

1. **Monitoring buoys** stationed in the Lakes, photographing dolphins as they
   pass — so sightings are collected continuously instead of only when a survey
   boat goes out.
2. **A program** that takes each photo, works out which dolphin it is, and saves
   the sighting to a database.

**This repository is the identification half of point 2.** Given a fin photo, it
returns the most likely dolphin, a confidence score, and a flag when a person
needs to check the answer. It does not attempt the buoys, and it does not
attempt the database.

It is worth being clear about what it is and is not: this is a working prototype
that demonstrates the method, not a replacement for a researcher. Professional
systems for this task (CurvRank, finFindR) use trained neural networks and years
of catalogue data. What this program shows is that the underlying idea — turn
the fin edge into a curve, compare the curves — genuinely works, and it is
honest about where it falls down. See [Results](#results).

### Sources

- Charlton-Robb et al. (2011), describing the species — *PLOS ONE*
- [Marine Mammal Foundation — Burrunan dolphin](https://www.marinemammal.org.au/burrunan-dolphin)
- [Victorian Government — Burrunan dolphin, threatened species](https://www.environment.vic.gov.au/conserving-threatened-species/threatened-species/burrunan-dolphin)
- [Wikipedia — Burrunan dolphin](https://en.wikipedia.org/wiki/Burrunan_dolphin)

---

## Quick start

Double-click **`Run Dolphin Identifier.command`** to open the window.

Or from a terminal:

```bash
cd "School Project"
python3 dolphin_interface.py
```

To run it without the window and just print results:

```bash
python3 identify.py --no-display
```

Needs Python 3 with **OpenCV** and **NumPy** installed (`pip3 install opencv-python numpy`).
Tested on Python 3.14, OpenCV 5.0, NumPy 2.3. Tkinter, used for the window,
comes with Python.

---

## How it works

Dolphins are told apart by the **shape of the dorsal fin seen side on** —
specifically the nicks and notches cut into its trailing edge over the animal's
life. The front edge is smooth on almost every dolphin and carries no identity,
so the program ignores it.

Each photo goes through five steps:

1. **Separate the fin from the water.** Several ways of splitting the picture
   are tried (brightness, blue-vs-grey colour, and colour saturation), because a
   fin can be darker *or* lighter than the water depending on the light. Each
   resulting blob is scored on how fin-like it is, and the best one is tidied up
   with GrabCut.
2. **Find the tip and trace down the trailing edge** to the base of the fin.
3. **Measure curvature along that edge.** A small circle is placed at each point
   and the program checks how much of it falls inside the fin — a straight edge
   gives about half, a notch gives less. This turns the fin edge into a graph
   where every notch is a dip. Three circle sizes are used at once, so both
   small nicks and large missing chunks show up.
4. **Compare the graph against every catalogue fin**, allowing a little stretch
   to line them up (dynamic time warping), since no two photos are taken from
   exactly the same distance or angle.
5. **Highest similarity wins** — unless it is below the threshold, in which case
   the program says it does not recognise the dolphin.

This is the same approach used by professional systems such as CurvRank and
finFindR.

### Photos from the dolphin's other side

A dolphin photographed from its left gives a mirror image of the same fin from
its right. Because the trace always runs from the tip down the trailing edge,
the notch sequence comes out the same either way, so a mirrored photo is not a
problem. Every catalogue photo is *also* traced mirrored and the better result
kept, because the fin-finding step is not perfectly even-handed. The program
still reports which side the dolphin was seen from.

---

## Using it

Put photos in these folders. The names before the last underscore are the
dolphin's ID, so `NOAA0087_01.jpg` and `NOAA0087_02.jpg` are the same animal.

| Folder | What goes in it |
| --- | --- |
| `catalogue/` | Side-on fin photos of dolphins you already know. More photos per dolphin gives a better result. |
| `sightings/` | New photos waiting to be identified. |
| `human_review/` | Written by the program. Rejected and uncertain results are copied here for a person to check. |
| `temp_edges/` | Written by the program. Temporary edge images from step 3. Safe to delete. |

**Photos must be side on.** A fin photographed from the front or behind shows its
flat face instead of its outline and cannot be identified by any method, human
or computer. That is a rejected sighting, not a difficult one.

### The window

- **Add photos** / **Remove selected** / **Reload sightings folder** — manage the list.
- **Identify** — run the matching.
- Results are colour coded: green for a confident match, amber for one a person
  should double check, grey for no confident match.
- Selecting a result shows the sighting and its best match with the traced edge
  drawn on, plus both notch graphs so you can see whether the dips line up.
- **Show traced outline on photos** — turn the yellow line off to see the real
  fin edge clearly.
- The score table shows *every* dolphin, not just the winner, so a close second
  is easy to spot.
- **Compare** — put any two photos side by side with their score.
- **Browse dataset** — look through every photo in the three folders.
- **Export review CSV** — save results for the database stage.

### Command line options

```
python3 identify.py [photos...] [options]
```

| Option | What it does |
| --- | --- |
| *(no photos given)* | Identifies everything in `sightings/` |
| `--no-display` | Print results only, no windows |
| `--catalogue FOLDER` | Use a different catalogue folder |
| `--method notches\|shape` | `notches` (default) is the real method; `shape` is the older, weaker one, kept for comparison |
| `--results-csv FILE` | Append results to a spreadsheet |
| `--no-review` | Do not copy anything to `human_review/` |
| `--self-test` | Mirror every catalogue photo and check each still recognises itself |

---

## Results

Showcase set: 10 catalogued dolphins, each with a second photo taken on a
different day (some years apart), plus 2 dolphins deliberately left out of the
catalogue.

**11 of 12 correct.**

- Every true match scored between **64% and 85%**.
- Both uncatalogued dolphins were correctly refused (0% and 50%).
- The match threshold is **55%**, which sits in the gap between those two groups.
- Self-test: **10 of 10** fins recognise themselves when mirrored.

The one miss is `NOAA0423`, a pair of photos seven years apart where the fin
gained a large new notch in between. The program scores it 35% and refuses it.
That is the honest answer — the fin genuinely changed.

Two showcase results show the "check by hand" flag because their runner-up is
close. Both are still the right answer; the flag is doing its job.

### Where the method is weak

Measured on **36 unfiltered NOAA dolphins with no quality check on the photos,
the correct fin came first only 31% of the time** (guessing would be about 3%).
So the program is roughly eleven times better than chance, but it needs
reasonably clear, side-on photos to be trusted. Specifically:

- Separating fin from water uses colour and brightness rules. It works on clear
  photos and struggles on murky, backlit or splashy ones. Professional systems
  use trained neural networks for this step.
- **Dolphins with very few notches are hard.** One (`PS1372`) was tried and
  removed from the showcase because its nearly smooth fin scored moderately
  against everything and caused false matches. Poorly marked animals are a known
  hard case for human researchers too.
- The 55% threshold was measured on a small set. With many more photos it should
  be measured again.
- Fins change over years as new notches appear, so old catalogue photos can stop
  matching.
- **Any result near the threshold, or with a close runner-up, must be checked by
  a person** before it is saved as a confirmed identity.

---

## Test photos

There is no free set of Burrunan dolphin photos showing the same animal more
than once, which is what testing this properly requires. So the test photos are
**bottlenose dolphins from North Carolina**, from a public-domain NOAA Fisheries
catalogue (1,011 photos of 199 individuals, 2001–2018).

The ~20 photos actually used are in `catalogue/` and `sightings/`. The full
308 MB archive is not included in this repository, but it is public domain and
can be downloaded from
[NOAA InPort item 67547](https://www.fisheries.noaa.gov/inport/item/67547);
unzip it into a folder called `noaa_raw/` to reproduce the selection. Every
original filename used is listed in `CREDITS.txt`.

**These are stand-in data and must be described that way in the presentation.**
The identification method is the same for both species; only the animals differ.

Tagged, freeze-branded, overlaid and front-on photos were deliberately left out,
so the program is never matching a tag or a piece of text instead of a fin.

Full attribution for every photo used is in **`CREDITS.txt`**. Some photos are
Creative Commons and require credit if they go on a poster or slide.

---

## The files

| File | What it is |
| --- | --- |
| `dolphin_interface.py` | The window — what runs in the display |
| `identify.py` | Matching, ranking, review queue, CSV export; also runs on its own from the terminal |
| `fin_curvature.py` | The hard part: separating fin from water, tracing the trailing edge, measuring curvature, comparing |
| `Run Dolphin Identifier.command` | Double-click launcher |
| `CREDITS.txt` | Photo sources and licences |
| `noaa_raw/` | The NOAA archive the test photos were picked from. **Not included here** — it is 308 MB. See [Test photos](#test-photos) to re-download it. |

---

## Licence

The **code** in this repository is released under the MIT Licence — see
[`LICENSE`](LICENSE). You may use, modify and share it, provided the copyright
notice stays with it.

The **photographs** are not covered by that licence. They come from a
public-domain NOAA Fisheries catalogue and are free to reuse, but `CREDITS.txt`
records where every one of them came from and should travel with them.

---

## Still to do

- The **database stage**: saving each confirmed sighting (dolphin, date, place,
  photo) so the population can be tracked over time. The program currently hands
  off at the right point — it produces an ID, a score, and a flag when a result
  needs checking.
- Real **Burrunan** fin photos, to build a genuine Gippsland Lakes catalogue.
- The **monitoring buoys** themselves.
