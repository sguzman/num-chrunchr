# num-chrunchr

`num-chrunchr` is a Rust toolkit for factoring and structurally analyzing very large integers, including representations that do not fit comfortably in RAM.

## Intent

Treat giant-number analysis as a problem of representations and capabilities rather than assuming every workflow should materialize a `BigInt` in memory.

## Ambition

The current docs make the broader ambition clear: support multiple number representations and analysis strategies, potentially spanning disk-backed, symbolic, and other capability-oriented approaches.

## Current Status

The repository already contains an MVP, config/docs/scripts, reports, and a detailed README. It looks like an active exploratory research tool with a clear architecture thesis.

## Core Capabilities Or Focus Areas

- Large-number structural analysis and factoring workflows.
- Disk-backed representation ideas for oversized integers.
- Config and script support for experiment workflows.
- Reports and resource folders for research outputs.
- Explicit architectural direction captured in documentation.

## Project Layout

- `config/`: checked-in runtime configuration and configuration examples.
- `docs/`: project documentation, reference material, and roadmap notes.
- `res/`: bundled resources used by the application.
- `scripts/`: helper scripts for development, validation, or release workflows.
- `src/`: Rust source for the main crate or application entrypoint.
- `Cargo.toml`: crate or workspace manifest and the first place to check for package structure.

## Setup And Requirements

- Rust toolchain.
- Input representations or datasets appropriate to the current analysis pipeline.
- Enough local storage for disk-backed workflows when using large inputs.

## Build / Run / Test Commands

```bash
cargo build
cargo test
cargo run -- --help
```

## Notes, Limitations, Or Known Gaps

- This repo is research-oriented, so some interfaces may still be evolving around the representation model.
- Performance characteristics are likely input-shape dependent rather than uniform.

## Next Steps Or Roadmap Hints

- Keep representation contracts explicit as new backends or capabilities are added.
- Add more benchmark and regression coverage around large-input behavior.
