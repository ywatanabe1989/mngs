#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-10 21:38:44 (ywatanabe)"
# /home/ywatanabe/proj/_mngs_repo_openhands/src/mngs/general/system_ops/_email.py

import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import mngs


def send_gmail(
    sender_gmail,
    sender_password,
    recipient_email,
    subject,
    message,
    sender_name=None,
    cc=None,
    ID=None,
    log_paths=None,
    verbose=True,
):
    if ID == "auto":
        ID = mngs.gen.gen_ID()
        subject = f"{subject} (ID: {ID})"

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_gmail, sender_password)

        gmail = MIMEMultipart()
        gmail["Subject"] = subject
        gmail["To"] = recipient_email
        if cc:
            if isinstance(cc, str):
                gmail["Cc"] = cc
            elif isinstance(cc, list):
                gmail["Cc"] = ", ".join(cc)
        if sender_name:
            gmail["From"] = f"{sender_name} <{sender_gmail}>"
        else:
            gmail["From"] = sender_gmail
        gmail_body = MIMEText(message, "plain")
        gmail.attach(gmail_body)

        if log_paths:
            for path in log_paths:
                with open(path, "rb") as file:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(file.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename= {os.path.basename(path)}",
                    )
                    gmail.attach(part)

        recipients = [recipient_email]
        if cc:
            if isinstance(cc, str):
                recipients.append(cc)
            elif isinstance(cc, list):
                recipients.extend(cc)
        server.send_message(gmail, to_addrs=recipients)

        server.quit()

        if verbose:
            cc_info = f" (CC: {cc})" if cc else ""
            print(
                f"\nEmail was sent:\n\t{sender_gmail} -> {recipient_email}{cc_info}\n\t(ID: {ID})"
            )

    except Exception as e:
        print(f"Email was not sent: {e}")


# EOF
