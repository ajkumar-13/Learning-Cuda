# Contributing

Thanks for reading the series closely enough to want to improve it. Corrections and clarifications are very welcome.

## What is most useful

- **Technical corrections.** If a number, claim, or explanation is wrong, open an issue with the post number and the specific line. Every performance number in the series is meant to be *real* (measured, or derived by a committed CPU model), so a wrong number is a bug.
- **Clarity fixes.** A sentence that made you re-read it twice is worth reporting. The house style favors one idea per paragraph and short worked examples over dense prose.
- **Reproducibility.** If a CPU model prints different numbers on your machine, or a `.cu` file will not build with the documented `nvcc` command, that is a defect. Include your OS, CUDA version, and GPU.

## House conventions

If you send a change to a post, keep it consistent with the rest of the series:

- **Structure.** Each post is `> **TL;DR.**` + an "After reading" list → a hero diagram → numbered **picture → intuition → math → code** sections → `## Common pitfalls` → `## Further reading` (cite [REFERENCES.md](REFERENCES.md)) → `## What to read next`.
- **Real numbers only.** Do not invent benchmarks. Add a runnable model (Python stdlib or NumPy, no GPU required) that prints the number, and paste its true output. GPU wall-clock timings are labeled with the card they were measured on.
- **Diagrams.** SVG, light + dark mode, restrained: at most one structural color plus one highlight per figure, one idea per diagram. See the existing `posts/*/diagrams/` for the palette.
- **Notation.** Concrete matrix operands are code-spans `A`, `B`, `C`, never `\mathbf{}`. Keep em-dashes sparse.
- **Prose.** American spelling; define an acronym on first use.

## Running the checks

Before opening a pull request, make sure the series still reads and links cleanly. The setup and per-post run recipes are in [SETUP.md](SETUP.md) and [RUNNING.md](RUNNING.md); when a build fails, [TROUBLESHOOTING.md](TROUBLESHOOTING.md) has the cross-OS catalog.
