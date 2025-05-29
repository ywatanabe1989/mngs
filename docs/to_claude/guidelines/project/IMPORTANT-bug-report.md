<!-- ---
!-- Timestamp: 2025-05-29 02:26:08
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/.dotfiles/.claude/to_claude/guidelines/project/bug-report.md
!-- --- -->

# Local Bug Report Rules

## Locations
- Bug report MUST be written as:
  `PROJECT_ROOT/project_management/bug_reports/bug-report-<title>.md`
- Here, `PROJECT_ROOT` is the project root where the bug included.
  - For example, bug comes from outside of the current project, write bug-report to the `BUG_INCLUDED_ROOT/project_management/bug_reports/bug-reoprt-<title>.md`
- Bug report MUST be as much simple as possible
- Once solved, bug reports MUST be moved to:
  `PROJECT_ROOT/project_management/bug_reports/solved/bug-report-<title>.md`

## How to solve bug reports
1. Think potential causes and make a plan to troubleshoot
   2. Identify the root cause with simple temporal testing
   3. If route cause is not identified, THINK MORE DEEPLY and restart from step 1.
4. Add debugging code to potential code
5. List your opinions, priorities, and reasons
6. Make a plan to fix the problem
7. If fixation will be simple, just fix there
8. Otherwise, create a dedicated `feature/bug-fix-<title>` feature branch from
9. Once bug report is solved, merge the `feature/bug-fix-<title>` branch back to the original branch

## When solving problem is difficult
Consider reverting to the latest commit which did not raise the problem. We sometimes make mistakes but retry with experiences and updated ideas.

## Format
- Add progress section in `PROJECT_ROOT/project_management/bug_reports/bug-report-<title>.md` as follows:
  ```
  ## Bug Fix Progress
  - [x] Identify root cause
  - [ ] Fix XXX
  ```


- Once merge succeeded, delete the merged feature branch

<!-- EOF -->