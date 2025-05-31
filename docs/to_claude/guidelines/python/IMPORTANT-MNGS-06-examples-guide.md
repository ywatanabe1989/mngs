<!-- ---
!-- Timestamp: 2025-05-31 06:14:23
!-- Author: ywatanabe
!-- File: /home/ywatanabe/.dotfiles/.claude/to_claude/guidelines/python/IMPORTANT-MNGS-06-examples-guide_.md
!-- --- -->

## Example Guidelines

- All example files MUST use `mngs` and follow the `mngs` framework
  Understand all the `mngs` guidelines in this directory
- MUST use `./examples/sync_examples_with_source.sh` to make `./examples` mirror:
  `./src` for pip packages or 
  `./scripts` for scientific projects

- ALL EXAMPLE FILES MUST HAVE THE CORRESPONDING OUTPUT DIRECTORY
  This is handled by `mngs.gen.start` and `mngs.gen.clode`
  - If output directory is not created, that means:
    1. That script does not follow the `mngs` framework
    2. That script is not run yet
    3. The `mngs` package has problems


### Running Examples
```bash
./examples/run_examples.sh             # Run all examples
./examples/path/to/example_filename.py # Direct
```
## Your Understanding Check
Did you understand the guideline? If yes, please say:
`CLAUDE UNDERSTOOD: <THIS FILE PATH HERE>`

<!-- EOF -->