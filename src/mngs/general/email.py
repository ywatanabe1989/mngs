#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-25 18:26:49 (ywatanabe)"

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def notify(
    subject="mngs notification", message="Notification sent from your script"
):
    """
    Usage:
        notify(subject="Hello world from mngs", message="This is a test email")

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
    # Environmental variables
    mngs_sender_gmail = os.getenv("MNGS_SENDER_GMAIL")
    mngs_sender_password = os.getenv("MNGS_SENDER_GMAIL_PASSWORD")
    mngs_recipient_gmail = os.getenv("MNGS_RECIPIENT_GMAIL")
    send_gmail(
        mngs_sender_gmail,
        mngs_sender_password,
        mngs_recipient_gmail,
        subject,
        message,
    )


def send_gmail(
    sender_gmail, sender_password, recipient_gmail, subject, message
):

    # Set up the gmail server
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    # SMTPAuthenticationError: (535, b'5.7.8 Username and Password not accepted. For more information, go to\n5.7.8  https://support.google.com/mail/?p=BadCredentials l2-20020a056a00140200b006db0c82959asm14956248pfu.43 - gsmtp')

    # Login to the gmail account
    server.login(sender_gmail, sender_password)  # Corrected variable name here

    # Create the gmail
    gmail = MIMEMultipart()
    gmail["From"] = sender_gmail
    gmail["To"] = recipient_gmail
    gmail["Subject"] = subject
    gmail_body = MIMEText(message, "plain")
    gmail.attach(gmail_body)

    # Send the gmail
    server.send_message(gmail)

    # Quit the server
    server.quit()
