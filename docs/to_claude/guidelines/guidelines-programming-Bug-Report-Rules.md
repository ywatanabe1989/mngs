<!-- ---
!-- Timestamp: 2025-05-20 05:51:52
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/.claude/to_claude/guidelines/guidelines-programming-Bug-Report-Rules.md
!-- --- -->

# Local Bug Report Rules

## Locations
- Bug report MUST be written as:
  `./project_management/bug_reports/bug-report-<title>.md`
- Note that bug report should be as much simple as possible
- Once solved, bug reports should be moved to:
  `./project_management/bug_reports/solved/bug-report-<title>.md`

## How to solve bug reports
1. Identify the root cause
2. List your opinions, priorities, and reasons
3. Make a plan to fix the problem
4. If fixation will be simple, just fix there
5. Otherwise, create a dedicated `feature/bug-fix-<title>` feature branch from
6. Once bug report is solved, merge the `feature/bug-fix-<title>` branch back to the original branch

## Format
- Add progress section in `./project_management/bug_reports/bug-report-<title>.md` as follows:
  ```
  ## Bug Fix Progress
  - [x] Identify root cause
  - [ ] Fix XXX
  ```


- Once merge succeeded, delete the merged feature branch

<!-- EOF -->