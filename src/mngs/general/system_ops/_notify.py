#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-10 20:20:28 (ywatanabe)"
# /home/ywatanabe/proj/_mngs_repo_openhands/src/mngs/general/system_ops/_notify.py

"""This script does XYZ."""

import inspect
import os
import pwd
import socket
import subprocess
import sys

import mngs

from ._email import send_gmail


def get_username():
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except:
        return os.getenv("USER") or os.getenv("LOGNAME") or "unknown"


def get_hostname():
    return socket.gethostname()


def get_git_branch():
    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=mngs.__path__[0],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return branch
    except Exception as e:
        return "main"


def notify(
    subject="",
    message=":)",
    ID="auto",
    recipient_email=None,
    recipient_name="there",
    cc=None,
    log_paths=None,
    verbose=False,
):
    sender_gmail = os.getenv("MNGS_SENDER_GMAIL")
    sender_password = os.getenv("MNGS_SENDER_GMAIL_PASSWORD")
    recipient_email = recipient_email or os.getenv("MNGS_RECIPIENT_GMAIL")

    if sys.argv[0]:
        script_name = os.path.basename(sys.argv[0])
    else:
        frames = inspect.stack()
        script_name = (
            os.path.basename(frames[-1].filename) if frames else "(Not found)"
        )
    if (script_name == "-c") or (script_name.endswith(".py")):
        script_name = "`$ python -c ...`"

    sender = f"{get_username()}@{get_hostname()}"
    header = f"Hi {recipient_name} 👋\n\n"
    branch = get_git_branch()
    footer = f"""

Best regards,
{sender}

{'-'*20}
Sent via
- Script: {script_name}
- Source: mngs v{mngs.__version__} (github.com/ywatanabe1989/mngs/blob/{branch}/src/mngs/general/system_ops/_notify.py)
{'-'*20}"""

    full_message = header + message + footer
    full_subject = f"{subject}"

    if sender_gmail is None or sender_password is None:
        print(
            f"""
        Please set environmental variables to use this function (f"{inspect.stack()[0][3]}"):\n\n
        $ export MNGS_SENDER_GMAIL="mngs.notification@gmail.com"
        $ export MNGS_SENDER_GMAIL_PASSWORD="YOUR_APP_PASSWORD"
        $ export MNGS_RECIPIENT_GMAIL="YOUR_GMAIL_ADDRESS"
        """
        )

    send_gmail(
        sender_gmail,
        sender_password,
        recipient_email,
        full_subject,
        full_message,
        cc=cc,
        ID=ID,
        log_paths=log_paths,
        verbose=verbose,
    )


if __name__ == "__main__":
    notify(verbose=True)
