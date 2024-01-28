#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-26 16:13:56 (ywatanabe)"

import inspect
import os
import smtplib
import sys
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import StringIO

import mngs

# # Function to capture stdout and stderr
# def capture_stdout_stderr():
#     sys.stdout = StringIO()  # Capture stdout
#     sys.stderr = StringIO()  # Capture stderr


# def release_stdout_stderr():
#     output = sys.stdout.getvalue()
#     error = sys.stderr.getvalue()
#     sys.stdout = sys.__stdout__  # Release stdout
#     sys.stderr = sys.__stderr__  # Release stderr
#     return output, error


def notify(subject="", message=":)", ID=None, log_paths=None, show=False):
    """
    Usage:
        notify("mngs.gen.notify()", "Hello world from mngs.")

    Note:
        This function operates correctly when a Gmail address is configured as follows:

        Step 1: Set up 2-Step Verification (if not already active):
            1. Navigate to your Google Account.
            2. Click on "Security".
            3. Under "Signing in to Google", select "2-Step Verification".
            4. Follow the on-screen instructions.

        Step 2: Generate an App Password
            1. Return to the "Security" section of your Google Account.
            2. Under "Signing in to Google", choose "App Passwords". You may need to sign in again.
            3. At the bottom, click "Select app" and pick the app you are using (select "Mail" for general purposes).
            4. Click "Select device" and select the device you are using (or choose "Other" and label it "Python Script" or similar).
            5. Click "Generate".
            6. Use the 16-digit App Password in your script as directed. The App Password will be shown only once, so remember to copy it.

        Step 3: Configure Gmail as environment variables
               ```bash
               export MNGS_SENDER_GMAIL="mngs.notification@gmail.com"
               export MNGS_SENDER_GMAIL_PASSWORD="YOUR_APP_PASSWORD"
               export MNGS_RECIPIENT_GMAIL="YOUR_GMAIL_ADDRESS"
               ```
    """
    # Get the script name from sys.argv or inspect the stack
    if sys.argv[0]:
        script_name = os.path.basename(sys.argv[0])
    else:
        frames = inspect.stack()
        script_name = (
            os.path.basename(frames[-1].filename)
            if frames
            else "No script name available."
        )

    full_message = f"{message}"
    full_subject = f"{script_name} from {os.getenv('USER')}@{os.getenv('HOSTNAME')} | {subject}"

    # Environmental variables
    mngs_sender_gmail = os.getenv("MNGS_SENDER_GMAIL")
    mngs_sender_password = os.getenv("MNGS_SENDER_GMAIL_PASSWORD")
    mngs_recipient_gmail = os.getenv("MNGS_RECIPIENT_GMAIL")

    if mngs_sender_gmail is None or mngs_sender_password is None:
        print(
            f"""
        Please set environmental variables to use this function:
        
        $ export MNGS_SENDER_GMAIL="mngs.notification@gmail.com"
        $ export MNGS_SENDER_GMAIL_PASSWORD="YOUR_APP_PASSWORD"
        $ export MNGS_RECIPIENT_GMAIL="YOUR_GMAIL_ADDRESS"
        """
        )
        # raise ValueError(
        #     "Sender email or password not set in environment variables."
        # )

    send_gmail(
        mngs_sender_gmail,
        mngs_sender_password,
        mngs_recipient_gmail,
        full_subject,
        full_message,
        ID=ID,
        log_paths=log_paths,
        show=show,
    )


def send_gmail(
    sender_gmail,
    sender_password,
    recipient_gmail,
    subject,
    message,
    ID=None,
    log_paths=None,
    show=True,
):

    if ID is None:
        ID = mngs.gen.gen_ID()  # .split("_")[-1]

    try:
        # Set up the gmail server
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()

        # Login to the gmail account
        server.login(
            sender_gmail, sender_password
        )  # Corrected variable name here

        # Create the gmail
        gmail = MIMEMultipart()
        gmail["From"] = sender_gmail
        gmail["To"] = recipient_gmail
        gmail["Subject"] = f"{subject} (ID: {ID})"
        gmail_body = MIMEText(message, "plain")
        gmail.attach(gmail_body)

        # Attach log files if provided
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

        # Send the gmail
        server.send_message(gmail)

        # Quit the server
        server.quit()

        if show:
            print(
                f"\nEmail was sent: {sender_gmail} -> {recipient_gmail} (ID: {ID})"
            )

    except Exception as e:
        print(f"Email was not sent: {e}")
