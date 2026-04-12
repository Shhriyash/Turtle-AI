"""
Data models for email tools package.
"""

from dataclasses import dataclass
from typing import Optional, Literal
import re


@dataclass
class EmailConfig:
    """Configuration for email sending"""
    sender_name: str
    sender_email: str
    sender_passkey: str  # Gmail App Password
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 465


@dataclass
class EmailRequest:
    """Email request data structure"""
    receiver: str
    subject: str
    body: str
    cc: str = ""
    bcc: str = ""
    content_type: Literal["plain", "html"] = "plain"
    
    def __post_init__(self):
        """Validate email request data"""
        if not self.receiver:
            raise ValueError("Receiver email is required")
        receivers = self.parse_recipients(self.receiver)
        if not receivers:
            raise ValueError("Receiver email is required")
        invalid_receivers = [email for email in receivers if not self.is_valid_email(email)]
        if invalid_receivers:
            raise ValueError(f"Invalid receiver email format: {', '.join(invalid_receivers)}")
        invalid_cc = [email for email in self.parse_recipients(self.cc) if not self.is_valid_email(email)]
        if invalid_cc:
            raise ValueError(f"Invalid cc email format: {', '.join(invalid_cc)}")
        invalid_bcc = [email for email in self.parse_recipients(self.bcc) if not self.is_valid_email(email)]
        if invalid_bcc:
            raise ValueError(f"Invalid bcc email format: {', '.join(invalid_bcc)}")
        if not self.subject:
            raise ValueError("Email subject is required")
        if not self.body:
            raise ValueError("Email body is required")

    @staticmethod
    def parse_recipients(recipients: str) -> list[str]:
        return [email.strip() for email in recipients.split(",") if email.strip()]
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None


@dataclass
class EmailResult:
    """Result of email sending operation"""
    success: bool
    message: str
    receiver: Optional[str] = None
    subject: Optional[str] = None
    error_details: Optional[str] = None
    
    def to_string(self) -> str:
        """Convert result to formatted string"""
        if self.success:
            return f" Email sent successfully to {self.receiver}\nSubject: {self.subject}\nStatus: {self.message}"
        else:
            return f"Failed to send email\nError: {self.message}\nDetails: {self.error_details or 'None'}"
