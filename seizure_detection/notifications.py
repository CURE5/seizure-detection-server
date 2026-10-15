"""
Notification system for seizure alerts.

Supports multiple notification backends:
- Console (always enabled)
- Email (SMTP)
- SMS (Twilio)
- Push Notifications (Firebase, Pushover, etc.)
- Webhooks (generic HTTP POST)
"""
import os
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List
import smtplib

from seizure_detection.utils.logger import setup_logger

logger = setup_logger(__name__)


class NotificationBackend(ABC):
    """Abstract base class for notification backends."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    @abstractmethod
    async def send(self, alert: Dict) -> bool:
        """
        Send notification.
        
        Args:
            alert: Alert data dictionary
        
        Returns:
            True if sent successfully
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get backend name."""
        pass


class ConsoleNotifier(NotificationBackend):
    """Console/log file notification."""
    
    def __init__(self, enabled: bool = True):
        super().__init__(enabled)
    
    async def send(self, alert: Dict) -> bool:
        """Log to console."""
        if not self.enabled:
            return False

        logger.warning("="*70)
        if alert.get("type") == "hr_alert":
            logger.warning("🚨 HR ALERT 🚨")
            logger.warning("="*70)
            logger.warning(f"Time: {alert['timestamp']}")
            logger.warning(f"BPM: {alert.get('bpm')}")
            baseline = alert.get('baseline_bpm')
            if baseline is not None:
                logger.warning(f"Baseline: {baseline:.1f}")
            delta = alert.get('delta')
            if delta is not None:
                logger.warning(f"Delta: +{delta:.1f}")
            logger.warning(
                f"Sustained: {alert.get('sustained_sec', 0.0):.1f}s "
                f"(trigger: {alert.get('trigger')})"
            )
        else:
            logger.warning("🚨 SEIZURE ALERT 🚨")
            logger.warning("="*70)
            logger.warning(f"Time: {alert['timestamp']}")
            logger.warning(f"Confidence: {alert['confidence']:.2%}")
            logger.warning(f"Stage: {alert['stage']}")
            logger.warning(f"Time Weight: {alert.get('time_weight', 1.0):.2f}x")
            if 'custom_probability' in alert:
                logger.warning(
                    f"Model Probability: {alert['custom_probability']:.2%}"
                )

        logger.warning("="*70)
        return True
    
    def get_name(self) -> str:
        return "Console"


class EmailNotifier(NotificationBackend):
    """Email notification via SMTP."""
    
    def __init__(
        self,
        enabled: bool = False,
        smtp_host: str = "",
        smtp_port: int = 587,
        smtp_user: str = "",
        smtp_password: str = "",
        from_email: str = "",
        to_emails: List[str] = None
    ):
        super().__init__(enabled)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_email = from_email
        self.to_emails = to_emails or []
        
        if enabled and not all([smtp_host, smtp_user, from_email, to_emails]):
            logger.warning("Email notifier misconfigured, disabling")
            self.enabled = False
    
    async def send(self, alert: Dict) -> bool:
        """Send email alert."""
        if not self.enabled:
            return False
        
        try:
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"🚨 Seizure Detected - {alert['timestamp']}"
            msg['From'] = self.from_email
            msg['To'] = ", ".join(self.to_emails)
            
            # HTML body
            html = f"""
            <html>
            <body style="font-family: Arial, sans-serif;">
                <h2 style="color: #d32f2f;">🚨 Seizure Alert</h2>
                <table style="border-collapse: collapse; width: 100%;">
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;">
                            <strong>Time:</strong>
                        </td>
                        <td style="padding: 8px; border: 1px solid #ddd;">
                            {alert['timestamp']}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;">
                            <strong>Confidence:</strong>
                        </td>
                        <td style="padding: 8px; border: 1px solid #ddd;">
                            {alert['confidence']:.1%}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;">
                            <strong>Detection Stage:</strong>
                        </td>
                        <td style="padding: 8px; border: 1px solid #ddd;">
                            {alert['stage']}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;">
                            <strong>Video Recording:</strong>
                        </td>
                        <td style="padding: 8px; border: 1px solid #ddd;">
                            {alert.get('recording_started', 'Unknown')}
                        </td>
                    </tr>
                </table>
                <p style="margin-top: 20px; color: #666;">
                    This is an automated alert from the seizure detection system.
                </p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html, 'html'))
            
            # Send via SMTP
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._send_smtp,
                msg
            )
            
            logger.info(f"Email sent to {len(self.to_emails)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def _send_smtp(self, msg):
        """Send message via SMTP (blocking, run in executor)."""
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
    
    def get_name(self) -> str:
        return "Email"


class TwilioSMSNotifier(NotificationBackend):
    """SMS notification via Twilio."""
    
    def __init__(
        self,
        enabled: bool = False,
        account_sid: str = "",
        auth_token: str = "",
        from_number: str = "",
        to_numbers: List[str] = None
    ):
        super().__init__(enabled)
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.to_numbers = to_numbers or []
        
        if enabled and not all([account_sid, auth_token, from_number, to_numbers]):
            logger.warning("Twilio SMS notifier misconfigured, disabling")
            self.enabled = False
    
    async def send(self, alert: Dict) -> bool:
        """Send SMS via Twilio."""
        if not self.enabled:
            return False
        
        try:
            from twilio.rest import Client
            
            client = Client(self.account_sid, self.auth_token)
            
            message_body = (
                f"🚨 SEIZURE ALERT\n"
                f"Time: {alert['timestamp']}\n"
                f"Confidence: {alert['confidence']:.0%}\n"
                f"Check monitoring system immediately."
            )
            
            # Send to all numbers
            for to_number in self.to_numbers:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.messages.create(
                        body=message_body,
                        from_=self.from_number,
                        to=to_number
                    )
                )
            
            logger.info(f"SMS sent to {len(self.to_numbers)} recipients")
            return True
            
        except ImportError:
            logger.error("Twilio not installed. Install: pip install twilio")
            self.enabled = False
            return False
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return False
    
    def get_name(self) -> str:
        return "Twilio SMS"


class PushoverNotifier(NotificationBackend):
    """Push notification via Pushover."""
    
    def __init__(
        self,
        enabled: bool = False,
        user_key: str = "",
        api_token: str = "",
        priority: int = 2  # Emergency priority
    ):
        super().__init__(enabled)
        self.user_key = user_key
        self.api_token = api_token
        self.priority = priority
        
        if enabled and not all([user_key, api_token]):
            logger.warning("Pushover notifier misconfigured, disabling")
            self.enabled = False
    
    async def send(self, alert: Dict) -> bool:
        """Send push notification via Pushover."""
        if not self.enabled:
            return False
        
        try:
            import aiohttp
            
            if alert.get("type") == "hr_alert":
                title = "💓 HR Alert — sustained elevated heart rate"
                bpm = alert.get("bpm")
                baseline = alert.get("baseline_bpm")
                delta = alert.get("delta")
                sustained = alert.get("sustained_sec", 0.0)
                trigger = alert.get("trigger", "?")
                lines = [f"BPM {bpm}"]
                if baseline is not None:
                    lines[0] += f" (baseline {baseline:.0f})"
                if delta is not None:
                    lines.append(f"+{delta:.0f} bpm above baseline")
                lines.append(f"sustained {sustained:.0f}s ({trigger} trigger)")
                lines.append(f"at {alert['timestamp']}")
                message = "\n".join(lines)
            else:
                title = "🚨 Seizure Detected"
                message = (
                    f"Time: {alert['timestamp']}\n"
                    f"Confidence: {alert['confidence']:.0%}"
                )
            data = {
                "token": self.api_token,
                "user": self.user_key,
                "title": title,
                "message": message,
                "priority": self.priority,
                "sound": "siren"
            }
            
            # Emergency priority requires retry/expire
            if self.priority == 2:
                data["retry"] = 30
                data["expire"] = 300
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.pushover.net/1/messages.json",
                    data=data
                ) as response:
                    if response.status == 200:
                        logger.info("Pushover notification sent")
                        return True
                    else:
                        logger.error(
                            f"Pushover failed: {response.status}"
                        )
                        return False
            
        except ImportError:
            logger.error("aiohttp not installed. Install: pip install aiohttp")
            self.enabled = False
            return False
        except Exception as e:
            logger.error(f"Failed to send Pushover: {e}")
            return False
    
    def get_name(self) -> str:
        return "Pushover"


class WebhookNotifier(NotificationBackend):
    """Generic webhook notification (HTTP POST)."""
    
    def __init__(
        self,
        enabled: bool = False,
        url: str = "",
        headers: Dict[str, str] = None,
        auth_token: str = ""
    ):
        super().__init__(enabled)
        self.url = url
        self.headers = headers or {}
        
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
        
        if enabled and not url:
            logger.warning("Webhook notifier misconfigured, disabling")
            self.enabled = False
    
    async def send(self, alert: Dict) -> bool:
        """Send webhook POST request."""
        if not self.enabled:
            return False
        
        try:
            import aiohttp
            
            # Prepare payload
            payload = {
                "event": "seizure_detected",
                "timestamp": alert['timestamp'],
                "confidence": alert['confidence'],
                "stage": alert['stage'],
                "time_weight": alert.get('time_weight', 1.0),
                "custom_probability": alert.get('custom_probability'),
                "recording_started": alert.get('recording_started')
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if 200 <= response.status < 300:
                        logger.info(f"Webhook sent to {self.url}")
                        return True
                    else:
                        logger.error(
                            f"Webhook failed: {response.status}"
                        )
                        return False
            
        except ImportError:
            logger.error("aiohttp not installed. Install: pip install aiohttp")
            self.enabled = False
            return False
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            return False
    
    def get_name(self) -> str:
        return f"Webhook ({self.url})"


class NotificationManager:
    """
    Manages multiple notification backends.
    
    Coordinates sending alerts through configured channels.
    """
    
    def __init__(self):
        self.backends: List[NotificationBackend] = []
        self.alert_history: List[Dict] = []
        self.max_history = 100
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load notification configuration from environment."""
        # Console (always enabled)
        self.add_backend(ConsoleNotifier(enabled=True))
        
        # Email
        if os.getenv('NOTIFY_EMAIL_ENABLED', 'false').lower() == 'true':
            self.add_backend(EmailNotifier(
                enabled=True,
                smtp_host=os.getenv('SMTP_HOST', 'smtp.gmail.com'),
                smtp_port=int(os.getenv('SMTP_PORT', '587')),
                smtp_user=os.getenv('SMTP_USER', ''),
                smtp_password=os.getenv('SMTP_PASSWORD', ''),
                from_email=os.getenv('SMTP_FROM', ''),
                to_emails=os.getenv('NOTIFY_EMAIL_TO', '').split(',')
            ))
        
        # Twilio SMS
        if os.getenv('NOTIFY_SMS_ENABLED', 'false').lower() == 'true':
            self.add_backend(TwilioSMSNotifier(
                enabled=True,
                account_sid=os.getenv('TWILIO_ACCOUNT_SID', ''),
                auth_token=os.getenv('TWILIO_AUTH_TOKEN', ''),
                from_number=os.getenv('TWILIO_FROM_NUMBER', ''),
                to_numbers=os.getenv('NOTIFY_SMS_TO', '').split(',')
            ))
        
        # Pushover
        if os.getenv('NOTIFY_PUSHOVER_ENABLED', 'false').lower() == 'true':
            self.add_backend(PushoverNotifier(
                enabled=True,
                user_key=os.getenv('PUSHOVER_USER_KEY', ''),
                api_token=os.getenv('PUSHOVER_API_TOKEN', ''),
                priority=int(os.getenv('PUSHOVER_PRIORITY', '2'))
            ))
        
        # Webhook
        if os.getenv('NOTIFY_WEBHOOK_ENABLED', 'false').lower() == 'true':
            self.add_backend(WebhookNotifier(
                enabled=True,
                url=os.getenv('WEBHOOK_URL', ''),
                auth_token=os.getenv('WEBHOOK_AUTH_TOKEN', '')
            ))
        
        enabled_backends = [b.get_name() for b in self.backends if b.enabled]
        logger.info(f"Notification backends: {', '.join(enabled_backends)}")
    
    def add_backend(self, backend: NotificationBackend):
        """Add a notification backend."""
        self.backends.append(backend)
    
    async def send_alert(self, alert: Dict) -> Dict[str, bool]:
        """
        Send alert through all enabled backends.
        
        Args:
            alert: Alert data dictionary
        
        Returns:
            Dictionary of backend names and success status
        """
        results = {}
        
        # Send to all backends concurrently
        tasks = []
        for backend in self.backends:
            if backend.enabled:
                tasks.append(self._send_to_backend(backend, alert))
        
        if tasks:
            backend_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for backend, result in zip(
                [b for b in self.backends if b.enabled],
                backend_results
            ):
                if isinstance(result, Exception):
                    logger.error(
                        f"{backend.get_name()} failed: {result}"
                    )
                    results[backend.get_name()] = False
                else:
                    results[backend.get_name()] = result
        
        # Store in history
        alert_with_results = alert.copy()
        alert_with_results['notification_results'] = results
        self.alert_history.append(alert_with_results)
        
        # Trim history
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
        
        return results
    
    async def _send_to_backend(
        self,
        backend: NotificationBackend,
        alert: Dict
    ) -> bool:
        """Send to a single backend with error handling."""
        try:
            return await backend.send(alert)
        except Exception as e:
            logger.error(
                f"Unhandled error in {backend.get_name()}: {e}"
            )
            return False
    
    def get_enabled_backends(self) -> List[str]:
        """Get list of enabled backend names."""
        return [b.get_name() for b in self.backends if b.enabled]
    
    def get_alert_history(self, limit: int = 10) -> List[Dict]:
        """Get recent alert history."""
        return self.alert_history[-limit:]
    
    def get_statistics(self) -> Dict:
        """Get notification statistics."""
        total_alerts = len(self.alert_history)
        
        # Count successes per backend
        backend_stats = {}
        for backend_name in self.get_enabled_backends():
            successes = sum(
                1 for alert in self.alert_history
                if alert.get('notification_results', {}).get(backend_name, False)
            )
            backend_stats[backend_name] = {
                'sent': successes,
                'total': total_alerts,
                'success_rate': successes / total_alerts if total_alerts > 0 else 0
            }
        
        return {
            'total_alerts': total_alerts,
            'backends': backend_stats
        }
