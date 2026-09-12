# SALAAR — Brand Guidelines v1.0

## Quick reference

| | |
|---|---|
| **Name** | SALAAR |
| **Line** | You think. You drop the ideas. The agent connects the dots. |
| **Wedge** | Every edge is **typed, explained, and approved by you.** Other AI canvases group ideas; we commit to *how* two ideas relate, quote the words that justify it, and let you reject it. |
| **Category** | Idea intelligence — not a mind map, not a notes app |
| **Proof** | A stranger's idea, dropped live, connected in seconds with a rationale you can read |

## 1. Colour palette

Dark by default — a graph is light on dark, like every instrument that shows signal.

| Token | Hex | Use |
|---|---|---|
| `--ink` | `#0B0E14` | Canvas ground |
| `--surface` | `#141926` | Nodes, panel |
| `--raise` | `#1A2030` | Cards, buttons |
| `--line` | `#232B3C` | Borders, dot grid |
| `--text` | `#E6E9EF` | Body |
| `--muted` | `#8A93A8` | Meta, labels |
| `--signal` | `#4F7CFF` | **Primary.** Structural edges (`depends_on`, `causes`, `solves`, `implements`), selection, primary action |
| `--pulse` | `#22D3A8` | **Agent.** Anything the AI did: combinative edges (`expands`, `specializes`, `can_combine_with`), live badge, connected dots |
| `--friction` | `#F5A524` | `contradicts`, warnings, offline mode |
| `--stop` | `#FF5C5C` | `duplicates`, reject |

Rule: **colour carries meaning, never decoration.** If an edge is blue it is structural; if something pulses teal the agent produced it. Never use `--pulse` for a human action, or `--signal` for AI output — the user must always be able to see which thoughts are theirs.

Contrast: `--text` on `--ink` is 14.8:1, `--muted` on `--surface` 5.2:1 — both pass WCAG AA. Edge labels are never the only signal; type is always spelled out in words.

## 2. Typography

- **UI:** Inter, falling back to the system stack. One weight for body (400), 600 for emphasis. No display face — the ideas are the content.
- **Meta:** `ui-monospace` at 10–11px for edge types, canvas id, status. Machine output looks like machine output.
- **Scale:** 11 (meta) / 13 (body, nodes) / 15 (headers). Line height 1.4–1.5. Nothing larger — a canvas full of headlines is a poster, not a workspace.

## 3. Logo

The mark is three dots on a baseline: one teal (the agent), one blue (the idea), one grey (the one not connected yet). Built in CSS, no asset file. Clear space = one dot diameter. Minimum size 16px. Don't: add a brain, a lightbulb, a network cliché, or a gradient.

## 4. Voice

**Personality:** a sharp colleague who read your notes before the meeting. Direct, specific, never impressed with itself.

| We are | We are not |
|---|---|
| Concrete — "both mention *waiting time*" | Vague — "these seem related" |
| Plain — "what's missing" | Corporate — "actionable gap analysis" |
| Honest about doubt — "offline mode" | Confident when guessing |
| Short | Chatty |

**Tone by context**
- *Empty canvas:* instructive, one sentence, then get out of the way.
- *AI output:* declarative but provisional — it is a proposal until the user accepts it. Suggestions are phrased as suggestions.
- *Failure:* say what broke and what still works. "Model unreachable — canvas still saved."

**Never say:** revolutionary, seamless, unleash, supercharge, magic, effortless, AI-powered (say what it does instead), "I think you might want to consider".

**Microcopy that ships:** `Connect the dots` · `Synthesize` · `What's missing` · `Ask the canvas…` · `offline mode`.

## 5. Motion & imagery

No illustration, no stock photography, no hero art. The product's only image is the user's own graph. Motion is limited to state changes the user caused — a new edge appears, it does not animate for applause.
