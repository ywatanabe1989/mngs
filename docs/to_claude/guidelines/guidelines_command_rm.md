<!-- ---
!-- Timestamp: 2025-05-18 02:47:16
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/.claude/to_claude/guidelines/guidelines_command_rm.md
!-- --- -->

# `rm` Command

- `rm` command is not allowed. 
- Instead, use the `./docs/to_claude/bin/safe_rm.sh` script, which is designed to keep files under the corresponding `.old` directory:

## Usage
`safe_rm.sh [-h|--help] file_or_directory [file_or_directory...]`

## Examples:
`$(basename $0) file1.txt dir1`
`$(basename $0) *.txt`

<!-- EOF -->