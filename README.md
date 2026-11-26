# Orbital time game

A time-blocking game for one working day, told as orbital mechanics.
The working day is the flight of a small stealth vessel that leaves a bigger "home ship", orbits a planet and docks again at the end of the shift. The hours you want undisturbed ("communication silences") become the intervals in which the planet hides the vessel from the home ship. Python fits the two orbits to your schedule with Monte Carlo sampling (emcee), and you pick the solution you like on a slider.

![Day view of the example agenda](docs/example_day_view.png)

> **Status:** v0.1, a working prototype for personal use. The input format and the outputs may still change between versions.

## Contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [The day file](#the-day-file)
- [Outputs](#outputs)
- [Project layout](#project-layout)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Privacy](#privacy)
- [License](#license)
- [Author](#author)

## What it does

- Reads a plain-text agenda, `day_YYYY-MM-DD-N.txt`, with your day's time blocks, silences, tasks ("emeralds"), prizes and reminders.
- **MC1** fits two Keplerian orbits (vessel and home ship) plus the planet's density, so that:
  - the two ships coincide at the start and end of the day;
  - the planet blocks the line of sight between them during your silence intervals.
- **MC2** fits the orbit of a "junky station" (an outdoor or exercise block) that the vessel meets at the block's start and end times.
- Shows the candidate solutions sorted by posterior probability on a slider. **Save** writes the chosen parameters back into the day file.
- Draws a day view, re-read every 30 s:
  - orbits coloured by local time, the current positions, the real silences;
  - container deadlines, prophet times and reminders;
  - the day's earnings (emeralds, sapphires, diamonds, Lian balls, Martian coins).
- Speaks reminders with a Windows voice (default: Zira) at randomised intervals.

## How it works

Everything is dimensionless. Time is $\tau=t/T$, where $T$ is the length of the day, and lengths are in planet radii, $u=r/R$.

Each orbit has three parameters:
- energy $\eta=\frac{E}{\mu}\frac{T^2}{R^2}$;
- angular momentum $\lambda=\frac{l}{\mu}\frac{T}{R^2}$;
- orientation $\theta_0$.

The planet enters through $\kappa=\frac{4}{3}\pi G\rho T^2$.

Two kinds of condition are imposed:
- **Silence:** the planet is between the ships when the segment joining them passes within one radius of the centre.
- **Docking:** the ships meet at the end of the day. Docking and the station encounters are solved exactly as two-dimensional Lambert problems.

Each mode found this way is refined to its maximum-posterior point. emcee then samples the posterior around it.

The derivations, including corrections to the first version of the notes, are in [`docs/`](docs/) (planned).

## Requirements

- Python 3.10 or newer
- numpy, scipy, matplotlib, emcee (listed in [`requirements.txt`](requirements.txt))
- Windows, for spoken reminders. On other systems reminders are printed instead.

## Installation

```powershell
git clone https://github.com/11529922+lrrimulo@users.noreply.github.com/orbital-time-game.git
cd orbital-time-game
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick start

```powershell
python orbital.py examples\day_2026-01-01-1.txt
```

1. MC1 runs (about two minutes) and opens a window with a slider. Move it (or use the arrow keys), then press **Save this model**.
2. MC2 looks for a junky-station orbit through the vessel's positions at the station's start and end times.
   - If one exists, a second slider opens; save again.
   - If none exists, the program says so and continues without the station. With the example day this is the usual outcome; see [Known limitations](#known-limitations).
3. The day view opens and refreshes every 30 s. Edit and save the day file while it runs: new tasks and prizes appear on the next refresh.

Useful options:

| Option | Effect |
|---|---|
| `--once` | one pass, then show the day view |
| `--auto` | take the highest-posterior model without opening a slider |
| `--interval 60` | re-read the file every 60 s |
| `--no-voice` | print reminders instead of speaking them |
| `--snapshot day.png` | save the day view as an image and exit |

To recompute the orbits, delete the `MC1_` lines (or the `MC2_` lines) inside `SAVED PARAMETERS`.

## The day file

One keyword per line, keywords in capitals. Everything after `END DAY` is ignored.

| Line | Meaning |
|---|---|
| `SAVED PARAMETERS` … `END SAVED PARAMETERS` | written by the program; leave empty in a new day |
| `DAY [8:00 - 20:00]` | start and end of the day (local time) |
| `SILENCE [10:30 - 13:00], [15:30 - 18:00]` | intervals with no communication |
| `PROPHETS [8:00], [12:30]` | single instants marked on the orbit |
| `JUNKY STATION <name> [a - b]` | the station block (fitted by MC2) |
| `CONTAINER <name> [a - b]` | a work block; it ends when its container arrives |
| `*E text` / `*Ed text` / `*Ed08:40 text` | task to do / done / done at 08:40 |
| `... ===> SAPPHIRE + 3 MARTIANS` | prizes earned when the task is done |
| `WHY` followed by `* ...` lines | reasons for the block (for you) |
| `REPORT` followed by `"""..."""` | report written after the block |
| `REMINDER <name> [a - b], NORMAL [10,1], VOICE "Zira"` | spoken every N(10, 1) minutes, from the `* "..."` lines, until `END REMINDER` |

See [`examples/day_2026-01-01-1.txt`](examples/day_2026-01-01-1.txt) for a complete day.

## Outputs

Written next to the day file and not tracked by Git:

| File | Content |
|---|---|
| `day_….txt` (updated) | the `SAVED PARAMETERS` block: orbits, planet, chosen models, real silence times |
| `day_….txt.bak` | the previous version of the day file |
| `day_….mc1.npz`, `day_….mc2.npz` | cached posterior samples, so a slider can be reopened without re-running |

## Project layout

```
orbital.py          the program (single file)
requirements.txt    Python dependencies
examples/           an anonymised example day
sounds/<event>/     folders for event sounds (audio files are not distributed)
docs/               derivations and images (planned)
```

## Known limitations

- **MC2 often finds no station orbit for short station blocks.** The orbits MC1 favours often lie around low-density planets, where the circular period at the surface is several hours. Between the vessel's positions at the block's start and end, the only orbit that keeps its pericentre above the surface is usually the vessel's own. In the example day, a 1¾-hour block has no solution; blocks of about five hours or more do.
  - Workaround: on the MC1 slider, pick a model whose info line shows `junky station: N possible orbit(s)` with N ≥ 1.
- The reminder voice is available only on Windows.
- One silence must not end exactly at the end of the day: the ships dock then, so the planet cannot be between them.

## Roadmap

- [x] v0.1: parser, MC1, MC2, sliders, day view, reminders
- [ ] MC2 feasibility for short station blocks
- [ ] Event sounds (`sounds/<event>/`): silences, containers, gems, game over
- [ ] MC3: container launch and return orbits between planet outposts and the vessel
- [ ] HTML view with ECharts, for sharing and for streaming in OBS
- [ ] Enemy probes triggered by real earthquakes (stand-up breaks)
- [ ] "Beauty" functions to rank orbits by aesthetic criteria

## Privacy

Day files hold a real personal schedule. The repository's `.gitignore` excludes them, together with `notes/`, `past/`, the caches and all audio files. Only `examples/` contains an agenda, and it is invented.

## License

To be chosen. Until a `LICENSE` file is added, all rights are reserved by the author.

## Author

Leandro Rímulo ([@<github-username>](https://github.com/11529922+lrrimulo@users.noreply.github.com)).
Code developed with AI assistance (Claude, Anthropic) and reviewed by the author.
