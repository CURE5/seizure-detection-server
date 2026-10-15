# AI-Detected Seizure Logging

## Overview

The system now automatically logs AI-detected seizures to the seizure log API with a special `aiDetected` flag. This allows you to distinguish between manually logged seizures and those detected by the multi-stage detection pipeline.

## Features

### 1. Automatic Seizure Logging

When the AI detection pipeline identifies a seizure:
- Automatically creates a seizure log entry via POST `/sz` API
- Sets `aiDetected: true` flag in the seizure data
- Includes detection confidence and pipeline stage information
- Tracks the seizure ID for correlation with recordings and notifications

### 2. Seizure Data Structure

```json
{
  "timestamp": "2025-12-03T10:30:00.000Z",
  "duration": 45,
  "severity": "moderate",
  "notes": "AI detected seizure (confidence: 0.85, stage: custom_model)",
  "aiDetected": true
}
```

**New Field:**
- `aiDetected` (boolean): `true` for AI-detected seizures, `false` or omitted for manually logged seizures

### 3. Status Endpoint Integration

The `/status` endpoint now includes information about the current seizure:

```json
{
  "is_seizing": "true",
  "seizure_started_at": "2025-12-03T10:30:00.000Z",
  "current_seizure_id": "1733221800000",
  ...
}
```

**New Field:**
- `current_seizure_id` (string): ID of the active seizure, or `null` when not seizing

## Detection Flow

```
1. Multi-Stage Pipeline detects seizure
   ↓
2. handle_seizure_alert() triggered
   ↓
3. Start video recording (if enabled)
   ↓
4. Create seizure log via POST /sz API
   - Sets aiDetected=true
   - Stores seizure_id
   ↓
5. Send notifications (unless shadow mode)
   ↓
6. Update status (is_seizing, current_seizure_id)
```

## API Endpoints

### POST `/sz` - Create Seizure Log
Creates a new seizure entry with optional `aiDetected` flag.

**Request Body:**
```json
{
  "timestamp": "2025-12-03T10:30:00.000Z",
  "duration": null,
  "severity": "moderate",
  "notes": "AI detected seizure (confidence: 0.85)",
  "aiDetected": true
}
```

**Response:**
```json
{
  "success": true,
  "id": "1733221800000"
}
```

### PUT `/sz/{seizure_id}` - Update Seizure Log
Updates an existing seizure (used to set end time and calculate duration).

**Used automatically when:**
- Seizure ends (manual or auto-detection)
- User manually ends seizure via `/command`

### GET `/status` - Server Status
Returns current server state including active seizure information.

**Response includes:**
```json
{
  "is_seizing": "true",
  "seizure_started_at": "2025-12-03T10:30:00.000Z",
  "seizure_ended_at": null,
  "current_seizure_id": "1733221800000"
}
```

## Usage Examples

### Query AI-Detected Seizures

```bash
# Get all seizures for date range
curl "http://localhost:8000/sz?start=2025-12-01T00:00:00Z&end=2025-12-31T23:59:59Z"

# Filter in application:
seizures = response.json()
ai_detected = [s for s in seizures if s.get('aiDetected') == True]
manual = [s for s in seizures if not s.get('aiDetected')]
```

### Check Current Seizure Status

```bash
# Check if currently in seizure
curl http://localhost:8000/status | jq '.is_seizing, .current_seizure_id'
```

### Manually End AI-Detected Seizure

```bash
# Send seizure_ended command
curl -X POST http://localhost:8000/command \
  -H "Content-Type: application/json" \
  -d '{"action": "seizure_ended"}'
```

## Configuration

### Environment Variables

```bash
# Enable/disable auto-recording when seizure detected
AUTO_RECORD=true

# Shadow mode - detect but don't send alerts (for testing)
SHADOW_MODE=false
```

### Detection Notes Format

AI-detected seizures automatically include detection metadata in the notes field:

```
AI detected seizure (confidence: 0.85, stage: custom_model)
```

Where:
- `confidence`: Detection confidence score (0.0 - 1.0)
- `stage`: Pipeline stage that triggered (`motion`, `activity`, or `custom_model`)

## Benefits

### 1. **Automated Logging**
- No manual intervention required
- Immediate logging when detected
- Consistent data format

### 2. **Clear Attribution**
- `aiDetected` flag distinguishes AI vs manual logs
- Helps evaluate AI detection accuracy
- Allows filtering for analysis

### 3. **Correlation**
- `current_seizure_id` links seizures to:
  - Video recordings
  - Notifications sent
  - Detection pipeline stages
  - Alert history

### 4. **Real-time Status**
- `/status` endpoint shows active seizure
- Client apps can track ongoing events
- Useful for live monitoring dashboards

## Testing

### Shadow Mode Testing

```bash
# In .env
SHADOW_MODE=true

# Start server
source sz/bin/activate
fastapi run server.py --port 8000

# Monitor logs for AI detections
tail -f logs/seizure-detection.log

# Check seizure logs were created
curl http://localhost:8000/sz?start=$(date -u +%Y-%m-01T00:00:00Z)
```

### Verify AI Detection Flag

```python
import requests
from datetime import datetime, timedelta

# Get today's seizures
start = datetime.now().replace(hour=0, minute=0, second=0)
end = start + timedelta(days=1)

response = requests.get(
    "http://localhost:8000/sz",
    params={
        "start": start.isoformat() + "Z",
        "end": end.isoformat() + "Z"
    }
)

seizures = response.json()
print(f"Total: {len(seizures)}")
print(f"AI-detected: {sum(1 for s in seizures if s.get('aiDetected'))}")
print(f"Manual: {sum(1 for s in seizures if not s.get('aiDetected'))}")
```

## Future Enhancements

Potential improvements:

1. **Detection Metadata**: Store full pipeline results with seizure log
2. **Confidence Tracking**: Separate field for detection confidence
3. **Multi-Stage History**: Track which stages triggered over time
4. **False Positive Marking**: Allow marking AI detections as false positives
5. **Training Feedback**: Use marked false positives to retrain model
6. **Analytics Dashboard**: Visualize AI vs manual detection rates

## Related Documentation

- `NOTIFICATION_SETUP.md` - Configure alerting for AI detections
- `PIPELINE_ARCHITECTURE.md` - How the multi-stage detection works
