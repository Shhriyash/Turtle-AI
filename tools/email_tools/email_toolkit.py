"""
Email toolkit implementation for Pydantic AI.

Based on the user's example EmailTool class with enhancements for better error handling,
validation, and integration with Pydantic AI framework.
"""

import smtplib
import logging
from email.message import EmailMessage
from typing import Optional
from pydantic_ai import Tool

from .models import EmailConfig, EmailRequest, EmailResult


logger = logging.getLogger(__name__)


class EmailTool:
    """
    Email toolkit for sending emails via Gmail SMTP.
    
    This tool provides email sending capabilities through Gmail's SMTP server.
    Requires Gmail App Password for authentication (not regular password).
    """
    
    def __init__(
        self,
        sender_name: Optional[str] = None,
        sender_email: Optional[str] = None,
        sender_passkey: Optional[str] = None,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 465,
    ):
        """
        Initialize EmailTool with Gmail configuration.
        
        Args:
            sender_name: Display name for the sender
            sender_email: Gmail address for sending emails
            sender_passkey: Gmail App Password (not regular password)
            smtp_server: SMTP server address (default: Gmail)
            smtp_port: SMTP server port (default: 465 for SSL)
        """        
        self.config = EmailConfig(
            sender_name=sender_name or "",
            sender_email=sender_email or "",
            sender_passkey=sender_passkey or "",
            smtp_server=smtp_server,
            smtp_port=smtp_port
        )
    
    def send_email(self, receiver: str, subject: str, body: str, content_type: str = "plain") -> str:
        """
        Send an email to the specified recipient(s).
        
        Args:
            receiver: Recipient email address(es) - can be single email or comma-separated list
            subject: Email subject line
            body: Email content/body
            content_type: Content type - "plain" or "html"
            
        Returns:
            Success message or error description
        """
        try:
            # Parse multiple recipients
            recipients = [email.strip() for email in receiver.split(',') if email.strip()]
            
            if not recipients:
                return "error: No valid recipient email addresses provided"
            
            # Validate all recipients
            invalid_emails = []
            for email in recipients:
                if not EmailRequest.is_valid_email(email):
                    invalid_emails.append(email)
            
            if invalid_emails:
                return f"error: Invalid email addresses: {', '.join(invalid_emails)}"
            
            # Create email request for validation
            email_request = EmailRequest(
                receiver=recipients[0],  # Use first recipient for validation
                subject=subject,
                body=body,
                content_type=content_type
            )
            
            # Check configuration
            if not self.config.sender_name:
                return "error: No sender name configured"
            if not self.config.sender_email:
                return "error: No sender email configured"
            if not self.config.sender_passkey:
                return "error: No sender passkey configured"
            
            # Send email to all recipients
            result = self._send_email_internal_multiple(recipients, email_request)
            return result.to_string()
            
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return f"error: {e}"
        except Exception as e:
            logger.error(f"Unexpected error in send_email: {e}")
            return f"error: Unexpected error occurred: {e}"
    
    def _send_email_internal_multiple(self, recipients: list[str], email_request: EmailRequest) -> EmailResult:
        """
        Internal method to handle email sending to multiple recipients.
        
        Args:
            recipients: List of recipient email addresses
            email_request: Validated email request object
            
        Returns:
            EmailResult with success/failure information
        """
        try:
            # Create email message
            msg = EmailMessage()
            msg["Subject"] = email_request.subject
            msg["From"] = f"{self.config.sender_name} <{self.config.sender_email}>"
            msg["To"] = ", ".join(recipients)  # Multiple recipients in To field
            
            # Set content based on type
            if email_request.content_type == "html":
                msg.set_content(email_request.body, subtype="html")
            else:
                msg.set_content(email_request.body)
            
            logger.info(f"Attempting to send email to {len(recipients)} recipients: {', '.join(recipients)}")
            
            # Send email via Gmail SMTP
            with smtplib.SMTP_SSL(self.config.smtp_server, self.config.smtp_port) as smtp:
                smtp.login(self.config.sender_email, self.config.sender_passkey)
                smtp.send_message(msg)
            
            logger.info(f"Email sent successfully to {len(recipients)} recipients")
            
            recipient_list = ", ".join(recipients) if len(recipients) <= 3 else f"{', '.join(recipients[:3])} and {len(recipients)-3} more"
            
            return EmailResult(
                success=True,
                message=f"Email sent successfully to {len(recipients)} recipient(s)",
                receiver=recipient_list,
                subject=email_request.subject
            )
            
        except smtplib.SMTPAuthenticationError as e:
            error_msg = "Authentication failed - check email and app password"
            logger.error(f"SMTP Authentication error: {e}")
            return EmailResult(
                success=False,
                message=error_msg,
                error_details=str(e)
            )
            
        except smtplib.SMTPRecipientsRefused as e:
            error_msg = f"One or more recipients refused: {e}"
            logger.error(f"SMTP Recipients refused: {e}")
            return EmailResult(
                success=False,
                message=error_msg,
                error_details=str(e)
            )
            
        except smtplib.SMTPException as e:
            error_msg = f"SMTP error occurred"
            logger.error(f"SMTP error: {e}")
            return EmailResult(
                success=False,
                message=error_msg,
                error_details=str(e)
            )
            
        except Exception as e:
            error_msg = f"Unexpected error sending email"
            logger.error(f"Unexpected error in _send_email_internal_multiple: {e}")
            return EmailResult(
                success=False,
                message=error_msg,
                error_details=str(e)
            )

    def _send_email_internal(self, email_request: EmailRequest) -> EmailResult:
        """
        Internal method to handle the actual email sending.
        
        Args:
            email_request: Validated email request object
            
        Returns:
            EmailResult with success/failure information
        """
        try:
            # Create email message
            msg = EmailMessage()
            msg["Subject"] = email_request.subject
            msg["From"] = f"{self.config.sender_name} <{self.config.sender_email}>"
            msg["To"] = email_request.receiver
            
            # Set content based on type
            if email_request.content_type == "html":
                msg.set_content(email_request.body, subtype="html")
            else:
                msg.set_content(email_request.body)
            
            logger.info(f"Attempting to send email to {email_request.receiver}")
            
            # Send email via Gmail SMTP
            with smtplib.SMTP_SSL(self.config.smtp_server, self.config.smtp_port) as smtp:
                smtp.login(self.config.sender_email, self.config.sender_passkey)
                smtp.send_message(msg)
            
            logger.info(f"Email sent successfully to {email_request.receiver}")
            
            return EmailResult(
                success=True,
                message="Email sent successfully",
                receiver=email_request.receiver,
                subject=email_request.subject
            )
            
        except smtplib.SMTPAuthenticationError as e:
            error_msg = "Authentication failed - check email and app password"
            logger.error(f"SMTP Authentication error: {e}")
            return EmailResult(
                success=False,
                message=error_msg,
                error_details=str(e)
            )
            
        except smtplib.SMTPRecipientsRefused as e:
            error_msg = f"Recipient refused: {email_request.receiver}"
            logger.error(f"SMTP Recipients refused: {e}")
            return EmailResult(
                success=False,
                message=error_msg,
                error_details=str(e)
            )
            
        except smtplib.SMTPException as e:
            error_msg = f"SMTP error occurred"
            logger.error(f"SMTP error: {e}")
            return EmailResult(
                success=False,
                message=error_msg,
                error_details=str(e)
            )
            
        except Exception as e:
            error_msg = f"Unexpected error sending email"
            logger.error(f"Unexpected error in _send_email_internal: {e}")
            return EmailResult(
                success=False,
                message=error_msg,
                error_details=str(e)
            )
    
    def test_connection(self) -> EmailResult:
        """
        Test SMTP connection without sending an email.
        
        Returns:
            EmailResult indicating connection success/failure
        """
        try:
            if not all([self.config.sender_email, self.config.sender_passkey]):
                return EmailResult(
                    success=False,
                    message="Missing email configuration",
                    error_details="sender_email and sender_passkey are required"
                )
            
            logger.info("Testing SMTP connection...")
            
            with smtplib.SMTP_SSL(self.config.smtp_server, self.config.smtp_port) as smtp:
                smtp.login(self.config.sender_email, self.config.sender_passkey)
            
            logger.info("SMTP connection test successful")
            
            return EmailResult(
                success=True,
                message="SMTP connection successful"
            )
            
        except Exception as e:
            logger.error(f"SMTP connection test failed: {e}")
            return EmailResult(
                success=False,
                message="SMTP connection failed",
                error_details=str(e)
            )
