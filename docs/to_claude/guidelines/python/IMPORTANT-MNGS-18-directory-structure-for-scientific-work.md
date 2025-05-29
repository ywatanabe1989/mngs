<!-- ---
!-- Timestamp: 2025-05-29 20:36:01
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/.dotfiles/.claude/to_claude/guidelines/python/MNGS-18-mngs-directory-structure-for-scientific-work.md
!-- --- -->

## Directory Structure of Scientific Project

```
<project root>
│
├── config/                 # Configuration files
│   └── *.yaml              # YAML config files (PATH.yaml, etc.)
│
├── data/                   # Centralized data storage
│   └── <dir_name>/         # Organized by category
│        └── file.ext → ../../scripts/<script>_out/file.ext  # Symlinks to script outputs
│
└── scripts/                # Script files and outputs
    └── <category>/
        ├── script.py       # Python script
        └── script_out/     # Output directory for this script
            ├── file.ext    # Output files
            └── logs/       # Logging directory for each run (managed by `mngs.gen.start` and `mngs.gen.close`)
                ├── RUNNING
                ├── FINISHED_SUCCESS
                └── FINISHED_FAILURE
└── examples/
└── tests/
└── .playground/
```


**IMPORTANT**: 
- DO NOT CREATE DIRECTORIES IN PROJECT ROOT  
- Create child directories under predefined directories instead

## Temporal Working Space: `./.playground`
- For your temporally work, use `./.playground`
  - Organize playground with categoris: 
    `./.playground/category-name-1/...`
    `./.playground/category-name-2/...`

## Your Understanding Check
Did you understand the guideline? If yes, please say:
`CLAUDE UNDERSTOOD: <THIS FILE PATH HERE>`

<!-- EOF -->