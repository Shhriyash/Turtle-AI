"""
Configuration management for email tools.

Handles environment variable loading and configuration validation
for email sending functionality.
"""

import os
from typing import Optional
from core.env import load_env

from .models import EmailConfig


def load_email_config_from_env() -> Optional[EmailConfig]:
    """
    Load email configuration from environment variables.
    
    Expected environment variables:
    - TURTLE_EMAIL_NAME: Display name for sender
    - TURTLE_EMAIL_ADDRESS: Gmail address
    - TURTLE_EMAIL_PASSKEY: Gmail App Password
    
    Returns:
        EmailConfig if all required variables are present, None otherwise
    """
    # Load environment variables
    load_env()
    
    sender_name = os.getenv("TURTLE_EMAIL_NAME")
    sender_email = os.getenv("TURTLE_EMAIL_ADDRESS")
    sender_passkey = os.getenv("TURTLE_EMAIL_PASSKEY")
    
    if not all([sender_name, sender_email, sender_passkey]):
        return None
    
    return EmailConfig(
        sender_name=sender_name,
        sender_email=sender_email,
        sender_passkey=sender_passkey
    )


def create_email_tool_from_env():
    """
    Create an EmailTool instance using environment variables.
    
    Returns:
        EmailTool instance or None if configuration is missing
    """
    config = load_email_config_from_env()
    if not config:
        return None
    
    from .email_toolkit import EmailTool
    
    return EmailTool(
        sender_name=config.sender_name,
        sender_email=config.sender_email,
        sender_passkey=config.sender_passkey
    )


def validate_email_config() -> tuple[bool, str]:
    """
    Validate that email configuration is properly set up.
    
    Returns:
        Tuple of (is_valid, message)
    """
    config = load_email_config_from_env()
    
    if not config:
        missing_vars = []
        if not os.getenv("TURTLE_EMAIL_NAME"):
            missing_vars.append("TURTLE_EMAIL_NAME")
        if not os.getenv("TURTLE_EMAIL_ADDRESS"):
            missing_vars.append("TURTLE_EMAIL_ADDRESS")
        if not os.getenv("TURTLE_EMAIL_PASSKEY"):
            missing_vars.append("TURTLE_EMAIL_PASSKEY")
        
        return False, f"Missing environment variables: {', '.join(missing_vars)}"
    
    # Basic validation
    if "@" not in config.sender_email:
        return False, "Invalid email address format"
    
    if len(config.sender_passkey) < 10:
        return False, "App password seems too short (should be 16 characters)"
    
    return True, "Email configuration is valid"


# Configuration constants
GMAIL_SMTP_SERVER = "smtp.gmail.com"
GMAIL_SMTP_PORT = 465

# Gmail App Password setup instructions
GMAIL_SETUP_INSTRUCTIONS = """
Gmail App Password Setup Instructions:

1. Enable 2-Factor Authentication on your Gmail account
2. Go to Google Account settings > Security > 2-Step Verification
3. Scroll down to "App passwords"
4. Select "Mail" and your device
5. Generate a 16-character app password
6. Use this app password (not your regular Gmail password)

Environment Variables to Set:
TURTLE_EMAIL_NAME="Your Display Name"
TURTLE_EMAIL_ADDRESS="your.email@gmail.com"
TURTLE_EMAIL_PASSKEY="your_16_character_app_password"
"""
