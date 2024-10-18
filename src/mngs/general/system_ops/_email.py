#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-18 21:14:37 (ywatanabe)"
# /home/ywatanabe/proj/_mngs_repo_openhands/src/mngs/general/system_ops/_email.py

import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import mimetypes
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
    attachment_file_paths=None,
    verbose=True,
):
    if ID == "auto":
        ID = mngs.gen.gen_ID()

    if ID:
        if subject:
            subject = f"{subject} (ID: {ID})"
        else:
            subject = f"ID: {ID}"

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

        if attachment_file_paths:
            for path in attachment_file_paths:
                mime_type, _ = mimetypes.guess_type(path)
                if mime_type is None:
                    mime_type = 'application/octet-stream'

                main_type, sub_type = mime_type.split('/', 1)

                with open(path, "rb") as file:
                    part = MIMEBase(main_type, sub_type)
                    part.set_payload(file.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename= {os.path.basename(path)}",
                    )
                    gmail.attach(part)

        # if attachment_file_paths:
        #     for path in attachment_file_paths:
        #         with open(path, "rb") as file:
        #             part = MIMEBase("application", "octet-stream")
        #             part.set_payload(file.read())
        #             encoders.encode_base64(part)
        #             part.add_header(
        #                 "Content-Disposition",
        #                 f"attachment; filename= {os.path.basename(path)}",
        #             )
        #             gmail.attach(part)

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
