<!-- ---
!-- Timestamp: 2025-05-29 20:32:47
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/.dotfiles/.claude/to_claude/guidelines/python/IMPORTANT-general.md
!-- --- -->

## MNGS Rule
**!!! IMPORATANT !!!**
**ANY PYTHON SCRIPTS MUST BE WRITTEN IN THE MNGS FORMAT EXPLAINED BELOW.**
THE EXCEPTIONS ARE:
    - Pacakges authored by others
    - Source (`./src`) of pip packages to reduce dependency
THUS, MNGS MUST BE USED IN:
- `./scripts`
- `./tests`
- `./examples`
For details, see the mngs guideline (`./TOP-MOST-IMPORTANT-mngs.md`).

## Python General Rules
Do not forget to add the executable permission (`chmod +x`) for `*.py` files.
Use `Agg` for matplotlib backend. Do not show images but save to file.
    ``` python
    import matplotlib
    matplotlib.use("Agg")
    ```

## Follow MNGS Framework
See `./docs/to_claude/guidelines/IMPORTANT_guidelines_programming_python_MNGS_rules.md`

## Lint with Black
IMPORTANT: LINT ALL .PY SCRIPTS USING `black` (`~/.env/bin/black`)

## Run scripts
When writing scripts, please run them without hesitation as long as they are destructive.

## Run with CUDA
When a GPU can accelerate processing, we prioritize GPU over CPU. If GPU is not available and CPU processing will take a long time, please let us know.

## Your Understanding Check
Did you understand the guideline? If yes, please say:
`CLAUDE UNDERSTOOD: <THIS FILE PATH HERE>`

<!-- EOF -->