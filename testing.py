import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Generate OTP
def generate_otp():
    return str(random.randint(100000, 999999))

# Send OTP Email
def send_otp(receiver_email):
    sender_email = "swaths246@gmail.com"
    app_password = "gfyrtymnvnlkvcxx"

    otp = generate_otp()

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "OTP Verification"

    body = f"""
Hello,

Your OTP for verification is:

{otp}

This OTP is valid for 10 minutes.

Regards,
Verification Team
"""

    message.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(
            sender_email,
            receiver_email,
            message.as_string()
        )
        server.quit()

        print(f"OTP sent to {receiver_email}")
        return otp

    except Exception as e:
        print("Error:", e)
        return None

# Example Usage
email = input("Enter email: ")

sent_otp = send_otp(email)

if sent_otp:
    user_otp = input("Enter OTP received: ")

    if user_otp == sent_otp:
        print("✅ Verification Successful")
    else:
        print("❌ Invalid OTP")