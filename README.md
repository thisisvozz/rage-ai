# Rage·RL — *I Gave An AI Anger Issues*

A reinforcement learning agent with an anger meter and a quit button, playing a platformer that cheats. The angrier it gets, the less it listens to its own plan — and that turns out to be the only reason it ever finishes.

**[▶ Watch the video](https://youtu.be/0XTSb80FI2o)** · **[🎮 Try the simulation](https://rage-ai-game.vercel.app/)**

> One HTML file. No dependencies, no build step, no framework. Open it and it runs.

---

## The premise

The agent spends the first **600 attempts** learning a fair platformer: spikes, a pit, a crumbling platform, a checkpoint, a flag. It gets good at it. Then the simulation switches on three cheats, by itself, and every habit it built turns into a liability.

1. **The checkpoint burns out** after four touches. Every death after that sends it back to the start.
2. **The platform crumbles** the moment you land on it. The jump that worked a second ago drops you into a pit.
3. **The flag runs away.** Get near the finish line and it hops onto the cliff, out of reach.

The third one is the interesting one, and not because it's hard. Chasing the flag is never *punished* — there's no spike, no pit, nothing. You just run into a wall and nothing happens, forever. Nothing in the game ever tells the agent it's doing the wrong thing.

The way out is to **let go of the controls**. Stand perfectly still for about a second and the flag walks back to you. A human works this out in one attempt.

The agent has spent its entire life being told that standing still is the worst possible move — that's literally the number it starts with. So it runs at that wall a hundred times instead, its anger meter climbing, until the meter gets high enough that it stops following its own plan and does something stupid for a full second. On seed `rage-5` the stupid thing it happens to do is nothing at all, and it wins with **16 deaths left** before it would have quit.

The control run is the point of the whole thing. Same brain, same level, frustration set to zero: it reaches **100% of the level on every single attempt** and never finishes it. Not once in four thousand tries. It isn't worse at the game. It's better. It just has no reason to try anything else.

---

## Run it

It's a single static file, so just open it:

```bash
open index.html          # macOS
xdg-open index.html      # Linux
start index.html         # Windows
```

Or serve it if you prefer a real origin:

```bash
python3 -m http.server 8000
# → http://localhost:8000
```

Everything runs client-side on `<canvas>`. The only network request is Google Fonts; without it the page falls back to system fonts and works fine.

---

## Reproducing the numbers from the video

The RNG is seeded, so the same seed reproduces the same run every time, regardless of what ran before it. Defaults are the video's settings.

| Setting | Value |
|---|---|
| Seed | `rage-5` |
| Cheats switch on after | **600** attempts |
| Anger per death | 1.0 · calm-down per record 6.0 · cool-down per attempt 0.15 |
| Max exploration | **60%** · anger curve **9** · tantrum length **10** |
| Calm agent exploration | **0%** |
| Attempt time limit | 600 frames |

Measured results on that configuration:

| Milestone | Attempt |
|---|---|
| Stops walking into the first spikes | 4 |
| Clears the pit, finds the checkpoint | 45 |
| First win on the fair level | **48** |
| …and cannot repeat it for another 180 attempts | — |
| Wins ten in a row | 227 |
| Cheats switch on | 600 |
| Anger passes 25 / 50 / 75 | +31 / +60 / +90 |
| **Solves the unfair level** | **+101**, anger **84**, 16 deaths from quitting |
| Calm agent, same brain | never, in 4000 attempts, at 100% every time |

Not every seed ends well, and that's honest — sometimes it just runs out of patience:

| Seed | 250 attempts of training | 600 | 1000 |
|---|---|---|---|
| `rage-5` | solves, anger 88 | **solves, anger 84** | solves, anger 86 |
| `rage-6` | **quits at 119** | **quits at 119** | **quits at 119** |
| `rage-4` | quits at 140 | quits at 140 | solves, anger 64 |
| `rage-9` | solves, anger 82 | solves, anger 67 | solves, anger 67 |

Across ten seeds trained to 600: six solve it, four quit. The calm agent solved it on **none** of them.

### Four settings that will ruin the result

- **Calm agent exploration must be 0%.** Above roughly 0.1% the "calm" agent eventually stumbles onto standing still too, given three thousand attempts, and there is no control experiment left.
- **Tantrum length is not a cosmetic.** At 1 the agent's random moves last a sixth of a second and nothing in this level can be discovered in a sixth of a second — nobody ever solves it. The commitment is the mechanism, not the randomness.
- **Don't push "stand still before it returns" past ~48 frames.** The longest tantrum is `tantrum length × 6` frames; ask for more stillness than that and the answer becomes literally unreachable.
- **A low anger curve makes it solve too early**, at anger 20 or 30, and the whole arc collapses into "it got a bit annoyed and then won".

---

## How the AI works

No neural network, no training data, no backpropagation. It's **tabular Q-learning**: a table of *how well has this move worked, from this spot*.

**The state** is three things: which 40×40 cell of the level it's in, whether it's on the ground or rising or falling, and roughly how long it has been holding perfectly still. It is deliberately **not** told what the flag is doing — so when the cheats switch on, it walks into them carrying every habit it built, instead of a blank slate.

**The actions** are six: idle, left, right, jump, jump-left, jump-right. It decides once every 6 frames.

**The reward** is the change in distance to the flag, plus −12 for dying and +120 for finishing. Running out of time pays nothing and costs nothing.

**The policy** is ε-greedy: play the best-known move most of the time, a random one the rest of the time. That "rest of the time" is the exploration rate — and the anger meter is wired straight into it:

```
epsilon = 0.002 + 0.598 × (anger / 100) ^ 9
```

Deaths push the anger up, new records pull it down, and at 100 the agent quits: the run ends, the table is thrown away, nothing is saved.

Three details do the actual work, and all three are one line each:

**Anger doesn't only make it random, it makes it commit.** When it explores, it picks a random move *and a random duration*, and holds that move for up to ten decisions. A calm agent that ignores its plan ignores it for a sixth of a second and goes straight back — useless, because nothing here can be found in one move. An angry one holds the bad idea for a full second. Same amount of randomness, completely different creature. (This is temporally-extended ε-greedy, from Dabney et al.)

**Every move starts out rated slightly bad, and doing nothing starts out rated worst of all.** `Q0 = −1`, `Q0(idle) = −2.5`. Without that, an untried action looks *better* than a tried-and-disappointing one, and the agent finds the answer by process of elimination on its second attempt. With it, standing still is the last thing it will ever consider — which is exactly the trap.

**Wasting time is free.** If the time penalty were nonzero, an endless futile chase would converge to a value *below* the starting value, and "stand still" — never tried, still sitting at its starting value — would quietly become the best-looking move on the board. At zero, the chase converges to exactly zero, comfortably above it, and the agent runs at that wall until something makes it stop.

---

## Controls

**Run** — speed (1× / 4× / 16× / turbo), seed, learning rate, attempt time limit, pause, restart, and **Play it yourself** (← → and space; same physics, same cheats — go to the flag, take your hands off the keyboard, and watch what happens).

**Anger** — anger per death, calm-down per record, cool-down per attempt, max exploration, the shape of the anger→exploration curve, tantrum length, and whether hitting 100 actually quits.

**The three cheats** — when they switch on, and the parameters of each one. Set "cheats switch on after" to 0 to start unfair, or drag it past the current attempt to stay fair forever.

**Angry vs calm** — split screen, both running the same level. The calm agent is cloned from the brain *as it was when the cheats came on*, not as it is now, so it can't inherit an answer the angry one has already found.

**Decision map** — draws the move the agent would make from every cell of the level, coloured by how good it thinks that move is. This is the whole brain, on one screen.

| Key | Action |
|---|---|
| `R` | Fullscreen view — hides the panels, fills the window, fades the cursor |
| `P` | Pause |
| `S` | Cycle speed |
| `V` | Toggle angry vs calm |
| `M` | Decision map |
| `G` | Record-per-attempt graph |
| `H` | Hide the HUD |
| `?` | Pin the shortcut list |

### URL flags

Handy for a kiosk or an OBS browser source:

```
index.html#big&seed=rage-6&speed=220&cheatsat=300&graph
```

`big` · `graph` · `seed=NAME` · `speed=1|4|16|220` · `cheatsat=N`

---

## Deploying to Vercel

It's a static site, so there's nothing to configure. Push the repo, import it at [vercel.com/new](https://vercel.com/new) — Framework Preset: **Other**, no build command, no output directory. Or:

```bash
npm i -g vercel
vercel --prod
```

---

## Repo contents

```
.
├── index.html    # the entire simulation
└── README.md
```

---

## Notes

The level is one screen with no scrolling camera, on purpose: the agent and the flag have to be visible at the same time or the third cheat doesn't read. The frame is wider than the level is tall, so the leftover space is filled with ridge lines and a star field rather than a scrolling camera.

The physics are deliberately generous — a random flailing agent has to be able to stumble across the *fair* version within a few hundred attempts, or there's no skill to take away later.

## License

MIT — do whatever you like with it. If you make something interesting, I'd like to see it.
