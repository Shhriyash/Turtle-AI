"""
Email Tools Package

A comprehensive package for email sending functionality using Gmail SMTP.

This package provides tools for sending emails through Gmail with proper
authentication, validation, and error handling.

Example Usage:
    # Basic usage with manual configuration
    from email_tools import EmailTool
    
    email_tool = EmailTool(
        sender_name="Turtle Assistant",
        sender_email="turtle.assistant@gmail.com",
        sender_passkey="your_app_password"
    )
    
    result = email_tool.send_email(
        receiver="user@example.com",
        subject="Hello from Turtle",
        body="This is a test email from Turtle assistant."
    )
    print(result)
    
    # Usage with environment variables
    from email_tools import create_email_tool_from_env, validate_email_config
    
    # Check configuration
    is_valid, message = validate_email_config()
    if not is_valid:
        print(f"Configuration error: {message}")
        exit(1)
    
    # Create tool from environment
    email_tool = create_email_tool_from_env()
    if email_tool:
        result = email_tool.send_email(
            receiver="user@example.com",
            subject="Hello",
            body="Test message"
        )
        print(result)
    
    # For use with Pydantic AI agents
    from email_tools import EmailTool
    from pydantic_ai import Agent
    
    email_tool = EmailTool(...)
    
    agent = Agent(
        model,
        builtin_tools=[email_tool],
        system_prompt="You are an email assistant..."
    )

Setup Requirements:
    1. Gmail account with 2-Factor Authentication enabled
    2. Gmail App Password generated (not regular password)
    3. Environment variables set (optional but recommended):
       - TURTLE_EMAIL_NAME="Your Display Name"
       - TURTLE_EMAIL_ADDRESS="your.email@gmail.com" 
       - TURTLE_EMAIL_PASSKEY="your_16_character_app_password"
"""

__version__ = "1.0.0"
__author__ = "Turtle AI Assistant"
__description__ = "Email sending tools for Gmail SMTP integration"

# Import main functionality
from .models import EmailConfig, EmailRequest, EmailResult
from .email_toolkit import EmailTool
from .config import (
    load_email_config_from_env,
    create_email_tool_from_env,
    validate_email_config,
    GMAIL_SETUP_INSTRUCTIONS
)

# Define what gets imported with "from email_tools import *"
__all__ = [
    'EmailConfig',
    'EmailRequest', 
    'EmailResult',
    'EmailTool',
    'load_email_config_from_env',
    'create_email_tool_from_env',
    'validate_email_config',
    'GMAIL_SETUP_INSTRUCTIONS'
]
