# Notification System Setup Guide

## Overview

The seizure detection system supports multiple notification backends:
- **Console** - Always enabled (logs to file)
- **Email** - SMTP-based email notifications
- **SMS** - Twilio text messages
- **Push** - Pushover mobile notifications
- **Webhook** - Generic HTTP POST (Discord, Slack, custom)

## Quick Start

### 1. Copy Environment Template

```bash
cp .env.example .env
```

### 2. Configure Notifications

Edit `.env` and enable desired notification methods.

### 3. Test Configuration

```bash
# Test without alerts (shadow mode)
SHADOW_MODE=true python server.py

# Enable alerts
SHADOW_MODE=false python server.py
```

## Configuration Options

### Email (Gmail Example)

**Prerequisites:**
1. Gmail account with 2FA enabled
2. App password generated

**Setup:**
```bash
# In .env file
NOTIFY_EMAIL_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=abcd efgh ijkl mnop  # 16-char app password
SMTP_FROM=your-email@gmail.com
NOTIFY_EMAIL_TO=parent1@email.com,parent2@email.com
```

**Generate Gmail App Password:**
1. Go to https://myaccount.google.com/security
2. Enable 2-Step Verification
3. Go to https://myaccount.google.com/apppasswords
4. Select "Mail" and generate
5. Use the 16-character password in `.env`

**Test:**
```python
# Test email notification
import asyncio
from seizure_detection.notifications import EmailNotifier

notifier = EmailNotifier(
    enabled=True,
    smtp_host='smtp.gmail.com',
    smtp_port=587,
    smtp_user='your-email@gmail.com',
    smtp_password='your-app-password',
    from_email='your-email@gmail.com',
    to_emails=['recipient@email.com']
)

alert = {
    'timestamp': '2025-12-03T15:30:00',
    'confidence': 0.95,
    'stage': 3
}

asyncio.run(notifier.send(alert))
```

### SMS (Twilio)

**Prerequisites:**
1. Twilio account (free trial available)
2. Verified phone numbers

**Setup:**
```bash
# In .env file
NOTIFY_SMS_ENABLED=true
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_FROM_NUMBER=+15551234567
NOTIFY_SMS_TO=+15559876543,+15551111111
```

**Get Twilio Credentials:**
1. Sign up at https://www.twilio.com/
2. Get free $15 trial credit
3. Copy Account SID from dashboard
4. Copy Auth Token from dashboard
5. Get a phone number from Twilio console
6. Verify recipient phone numbers (required for trial)

**Install Twilio SDK:**
```bash
source sz/bin/activate
uv pip install twilio
```

**Cost:**
- Trial: Free $15 credit
- Paid: $0.0075 per SMS (USA)
- ~1,333 messages per $10

### Push Notifications (Pushover)

**Prerequisites:**
1. Pushover account ($5 one-time fee)
2. Pushover app on phone

**Setup:**
```bash
# In .env file
NOTIFY_PUSHOVER_ENABLED=true
PUSHOVER_USER_KEY=uxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
PUSHOVER_API_TOKEN=axxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
PUSHOVER_PRIORITY=2  # Emergency (requires acknowledgment)
```

**Get Pushover Credentials:**
1. Sign up at https://pushover.net/ ($5 one-time)
2. Install Pushover app on phone
3. Create an application in dashboard
4. Copy API Token from application
5. Copy User Key from dashboard

**Priority Levels:**
- `-2`: Silent (no notification)
- `-1`: Quiet (no sound)
- `0`: Normal
- `1`: High priority
- `2`: Emergency (requires acknowledgment, retries)

**Cost:**
- One-time $5 per platform (iOS, Android)
- Unlimited notifications

### Webhook (Discord Example)

**Prerequisites:**
1. Discord server with webhook

**Setup:**
```bash
# In .env file
NOTIFY_WEBHOOK_ENABLED=true
WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_TOKEN
```

**Create Discord Webhook:**
1. Open Discord server settings
2. Go to Integrations → Webhooks
3. Click "New Webhook"
4. Copy webhook URL
5. Paste in `.env`

**Webhook Payload:**
```json
{
  "event": "seizure_detected",
  "timestamp": "2025-12-03T15:30:00",
  "confidence": 0.95,
  "stage": 3,
  "time_weight": 1.5,
  "recording_started": "Yes",
  "seizure_id": "sz_123"
}
```

### Webhook (Slack Example)

**Setup:**
```bash
# In .env file
NOTIFY_WEBHOOK_ENABLED=true
WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

**Create Slack Webhook:**
1. Go to https://api.slack.com/apps
2. Create new app
3. Enable Incoming Webhooks
4. Add webhook to workspace
5. Copy webhook URL

## Testing

### Test Individual Notifiers

```python
import asyncio
from seizure_detection.notifications import (
    ConsoleNotifier,
    EmailNotifier,
    TwilioSMSNotifier,
    PushoverNotifier,
    WebhookNotifier
)

async def test():
    alert = {
        'timestamp': '2025-12-03T15:30:00',
        'confidence': 0.95,
        'stage': 3,
        'custom_probability': 0.95,
        'time_weight': 1.5,
        'recording_started': 'Yes'
    }
    
    # Test console
    console = ConsoleNotifier()
    await console.send(alert)
    
    print("✓ Console notification sent")

asyncio.run(test())
```

### Test Full System

```bash
# Start server in shadow mode (logs only)
SHADOW_MODE=true python server.py

# Check logs
tail -f logs/seizure_detection.log

# Enable alerts
SHADOW_MODE=false python server.py
```

### Simulate Alert

```python
# Send test alert through API
import requests

# Trigger manual seizure event
response = requests.post(
    'http://localhost:8000/command',
    json={'action': 'seizure_started'}
)

print(response.json())
```

## API Endpoints

### Get Notification Statistics

```bash
curl http://localhost:8000/notifications/stats
```

**Response:**
```json
{
  "total_alerts": 15,
  "backends": {
    "Console": {"sent": 15, "total": 15, "success_rate": 1.0},
    "Email": {"sent": 14, "total": 15, "success_rate": 0.93},
    "Twilio SMS": {"sent": 15, "total": 15, "success_rate": 1.0}
  }
}
```

### Get Notification History

```bash
curl http://localhost:8000/notifications/history?limit=5
```

**Response:**
```json
{
  "history": [
    {
      "timestamp": "2025-12-03T15:30:00",
      "confidence": 0.95,
      "stage": 3,
      "notification_results": {
        "Console": true,
        "Email": true,
        "Twilio SMS": true
      }
    }
  ]
}
```

## Shadow Mode (Testing)

Shadow mode logs detections without sending notifications.

**Enable:**
```bash
# In .env
SHADOW_MODE=true
```

**Use Case:**
- Test detection without alerting caregivers
- Collect baseline data (1-2 weeks)
- Tune thresholds based on false positive rate

**What Happens:**
- ✓ Detections logged to console/file
- ✓ Video recording starts
- ✓ Seizure events created in logs
- ✗ No emails sent
- ✗ No SMS sent
- ✗ No push notifications
- ✗ No webhooks triggered

## Troubleshooting

### Email Not Sending

**Problem:** "Authentication failed" or "SMTP error"

**Solutions:**
1. Enable 2FA on Gmail
2. Generate app password (not regular password)
3. Check SMTP_HOST and SMTP_PORT
4. Verify SMTP_USER is full email address
5. Test with telnet:
   ```bash
   telnet smtp.gmail.com 587
   ```

### SMS Not Sending

**Problem:** "Unable to create record" or "Twilio error"

**Solutions:**
1. Verify Account SID and Auth Token
2. Check phone number format (+1234567890)
3. Verify recipient numbers (required for trial)
4. Check Twilio balance
5. View error logs in Twilio console

### Pushover Not Sending

**Problem:** "Invalid user key" or "HTTP 400"

**Solutions:**
1. Verify User Key (not Group Key)
2. Verify API Token
3. Check Pushover app is installed
4. Test with curl:
   ```bash
   curl -F "token=YOUR_TOKEN" \
        -F "user=YOUR_USER" \
        -F "message=test" \
        https://api.pushover.net/1/messages.json
   ```

### Webhook Not Sending

**Problem:** "Connection refused" or "HTTP error"

**Solutions:**
1. Verify webhook URL is correct
2. Check URL is publicly accessible
3. Test with curl:
   ```bash
   curl -X POST YOUR_WEBHOOK_URL \
        -H "Content-Type: application/json" \
        -d '{"test": "message"}'
   ```

## Best Practices

### Multiple Notification Methods

Use multiple methods for redundancy:
```bash
NOTIFY_EMAIL_ENABLED=true      # Primary
NOTIFY_SMS_ENABLED=true         # Backup
NOTIFY_PUSHOVER_ENABLED=true    # Emergency
```

### Alert Cooldown

Prevent notification spam:
```bash
ALERT_COOLDOWN_SECONDS=30  # Min 30s between alerts
```

### Time-Based Sensitivity

Increase sensitivity during high-risk hours:
```bash
ENABLE_TIME_WEIGHTING=true
HIGH_RISK_START_HOUR=5
HIGH_RISK_END_HOUR=8
HIGH_RISK_MULTIPLIER=1.5  # 50% more sensitive 5-8 AM
```

### Auto Recording

Automatically record video on detection:
```bash
AUTO_RECORD_ON_DETECTION=true
AUTO_RECORD_DURATION_SECONDS=300  # 5 minutes
```

## Cost Comparison

| Method | Setup Cost | Per Alert | Monthly (30 alerts) | Notes |
|--------|-----------|-----------|---------------------|-------|
| Console | Free | Free | Free | Always enabled |
| Email | Free | Free | Free | Requires email account |
| SMS (Twilio) | Free trial | $0.0075 | $0.23 | After trial: $0.23/month |
| Pushover | $5 once | Free | Free | One-time payment |
| Webhook | Free | Free | Free | Requires endpoint |

**Recommended Setup (Free):**
- Email (Gmail) - Primary
- Pushover ($5 one-time) - Backup
- Console - Always logging

**Budget Option (Free):**
- Email (Gmail)
- Webhook (Discord/Slack)
- Console

## Next Steps

1. **Week 1-2: Shadow Mode**
   - Enable shadow mode
   - Log all detections
   - Analyze false positive rate

2. **Week 3: Testing Mode**
   - Enable notifications
   - Test during high-risk hours only
   - Tune thresholds

3. **Week 4+: Production**
   - Enable 24/7 monitoring
   - Review weekly statistics
   - Adjust as needed

## Support

For issues or questions:
1. Check logs: `tail -f logs/seizure_detection.log`
2. Check notification stats: `curl localhost:8000/notifications/stats`
3. Review detection history: `curl localhost:8000/notifications/history`
