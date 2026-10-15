"""
Tests for API endpoints including command, seizure, and medication CRUD operations.
"""
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

# Import the server app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from seizure_detection.logging.seizure_logger import SeizureLogger
from seizure_detection.logging.medication_logger import MedicationLogger


class TestSeizureLogger:
    """Test SeizureLogger CRUD operations."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test logs."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def logger(self, temp_dir):
        """Create a SeizureLogger instance with temp directory."""
        return SeizureLogger(log_dir=temp_dir)
    
    @pytest.mark.asyncio
    async def test_create_seizure(self, logger):
        """Test creating a seizure event."""
        now = datetime.now(timezone.utc)
        event_data = {
            "date": now.isoformat(),
            "type": "Epileptic spasm",
            "intensity": "3",
            "notes": "Test seizure",
            "video": "",
            "duration": "",
            "durationSeconds": 0
        }
        
        event = await logger.create(event_data)
        
        assert event is not None
        assert "id" in event
        assert event["type"] == "Epileptic spasm"
        assert event["intensity"] == "3"
        assert event["notes"] == "Test seizure"
    
    @pytest.mark.asyncio
    async def test_start_and_end_seizure(self, logger):
        """Test starting and ending a seizure event."""
        # Start seizure
        event_id = await logger.start(
            seizure_type="Tonic-clonic",
            intensity="5",
            notes="Started automatically",
            video=""
        )
        
        assert event_id is not None
        
        # Wait a bit to get some duration (at least 1 second for reliable measurement)
        await asyncio.sleep(1.1)
        
        # End seizure
        updated_event = await logger.end(event_id)
        
        assert updated_event is not None
        assert updated_event["id"] == event_id
        assert updated_event["durationSeconds"] >= 1
        assert updated_event["duration"] != ""
    
    @pytest.mark.asyncio
    async def test_update_seizure(self, logger):
        """Test updating a seizure event."""
        # Create event
        now = datetime.now(timezone.utc)
        event_data = {
            "date": now.isoformat(),
            "type": "Absence",
            "intensity": "2",
            "notes": "Original notes",
            "video": "",
            "duration": "",
            "durationSeconds": 0
        }
        
        event = await logger.create(event_data)
        event_id = event["id"]
        
        # Update event
        updates = {
            "notes": "Updated notes",
            "intensity": "3"
        }
        
        updated = await logger.update(event_id, updates)
        
        assert updated is not None
        assert updated["notes"] == "Updated notes"
        assert updated["intensity"] == "3"
        assert updated["type"] == "Absence"  # Unchanged
    
    @pytest.mark.asyncio
    async def test_delete_seizure(self, logger):
        """Test deleting a seizure event."""
        # Create event
        now = datetime.now(timezone.utc)
        event_data = {
            "date": now.isoformat(),
            "type": "Myoclonic",
            "intensity": "1",
            "notes": "To be deleted",
            "video": "",
            "duration": "",
            "durationSeconds": 0
        }
        
        event = await logger.create(event_data)
        event_id = event["id"]
        
        # Delete event
        deleted = await logger.delete(event_id)
        
        assert deleted is True
        
        # Verify deletion
        deleted_again = await logger.delete(event_id)
        assert deleted_again is False
    
    @pytest.mark.asyncio
    async def test_get_range(self, logger):
        """Test getting seizures in a date range."""
        now = datetime.now(timezone.utc)
        
        # Create multiple events
        for i in range(3):
            event_data = {
                "date": (now + timedelta(hours=i)).isoformat(),
                "type": f"Type {i}",
                "intensity": str(i),
                "notes": f"Event {i}",
                "video": "",
                "duration": "",
                "durationSeconds": 0
            }
            await logger.create(event_data)
        
        # Get events
        start = now - timedelta(hours=1)
        end = now + timedelta(hours=4)
        events = await logger.get_range(start, end)
        
        assert len(events) == 3
        assert events[0]["type"] == "Type 0"
        assert events[1]["type"] == "Type 1"
        assert events[2]["type"] == "Type 2"
    
    @pytest.mark.asyncio
    async def test_get_range_empty(self, logger):
        """Test getting seizures from empty date range."""
        start = datetime.now(timezone.utc) - timedelta(days=30)
        end = datetime.now(timezone.utc) - timedelta(days=29)
        
        events = await logger.get_range(start, end)
        
        assert len(events) == 0


class TestMedicationLogger:
    """Test MedicationLogger CRUD operations."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test logs."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def logger(self, temp_dir):
        """Create a MedicationLogger instance with temp directory."""
        return MedicationLogger(log_dir=temp_dir)
    
    @pytest.mark.asyncio
    async def test_create_medication(self, logger):
        """Test creating a medication event."""
        now = datetime.now(timezone.utc)
        event_data = {
            "date": now.isoformat(),
            "name": "Keppra",
            "dosage": "500mg",
            "notes": "Morning dose"
        }
        
        event = await logger.create(event_data)
        
        assert event is not None
        assert "id" in event
        assert event["name"] == "Keppra"
        assert event["dosage"] == "500mg"
        assert event["notes"] == "Morning dose"
    
    @pytest.mark.asyncio
    async def test_update_medication(self, logger):
        """Test updating a medication event."""
        # Create event
        now = datetime.now(timezone.utc)
        event_data = {
            "date": now.isoformat(),
            "name": "Diazepam",
            "dosage": "5mg",
            "notes": "Original notes"
        }
        
        event = await logger.create(event_data)
        event_id = event["id"]
        
        # Update event
        updates = {
            "notes": "Updated notes",
            "dosage": "10mg"
        }
        
        updated = await logger.update(event_id, updates)
        
        assert updated is not None
        assert updated["notes"] == "Updated notes"
        assert updated["dosage"] == "10mg"
        assert updated["name"] == "Diazepam"  # Unchanged
    
    @pytest.mark.asyncio
    async def test_delete_medication(self, logger):
        """Test deleting a medication event."""
        # Create event
        now = datetime.now(timezone.utc)
        event_data = {
            "date": now.isoformat(),
            "name": "Lamictal",
            "dosage": "100mg",
            "notes": "To be deleted"
        }
        
        event = await logger.create(event_data)
        event_id = event["id"]
        
        # Delete event
        deleted = await logger.delete(event_id)
        
        assert deleted is True
        
        # Verify deletion
        deleted_again = await logger.delete(event_id)
        assert deleted_again is False
    
    @pytest.mark.asyncio
    async def test_get_range(self, logger):
        """Test getting medications in a date range."""
        now = datetime.now(timezone.utc)
        
        # Create multiple events
        for i in range(3):
            event_data = {
                "date": (now + timedelta(hours=i)).isoformat(),
                "name": f"Med {i}",
                "dosage": f"{100 * (i+1)}mg",
                "notes": f"Note {i}"
            }
            await logger.create(event_data)
        
        # Get events
        start = now - timedelta(hours=1)
        end = now + timedelta(hours=4)
        events = await logger.get_range(start, end)
        
        assert len(events) == 3
        assert events[0]["name"] == "Med 0"
        assert events[1]["name"] == "Med 1"
        assert events[2]["name"] == "Med 2"
    
    @pytest.mark.asyncio
    async def test_get_range_empty(self, logger):
        """Test getting medications from empty date range."""
        start = datetime.now(timezone.utc) - timedelta(days=30)
        end = datetime.now(timezone.utc) - timedelta(days=29)
        
        events = await logger.get_range(start, end)
        
        assert len(events) == 0


class TestAPIEndpoints:
    """Test API endpoints using TestClient."""
    
    @pytest.fixture
    def client(self):
        """Create a test client with mocked dependencies."""
        # Import here to avoid issues
        with patch('server.camera_manager') as mock_camera, \
             patch('server.video_recorder') as mock_recorder, \
             patch('server.seizure_logger') as mock_sz_logger, \
             patch('server.medication_logger') as mock_med_logger:
            
            # Setup mocks
            mock_camera.is_running.return_value = False
            mock_camera.start = AsyncMock(return_value=True)
            mock_camera.stop = AsyncMock()
            mock_camera.switch_mode = AsyncMock()
            mock_camera.set_controls = AsyncMock(return_value=True)
            mock_camera.resolution = (640, 360)
            
            mock_recorder.is_recording.return_value = False
            mock_recorder.start = AsyncMock()
            mock_recorder.stop = AsyncMock(return_value=Path("test_video.mp4"))
            
            mock_sz_logger.get_range = AsyncMock(return_value=[])
            mock_sz_logger.create = AsyncMock(return_value={"id": "test-id"})
            mock_sz_logger.update = AsyncMock(return_value={"id": "test-id"})
            mock_sz_logger.delete = AsyncMock(return_value=True)
            mock_sz_logger.start = AsyncMock(return_value="seizure-id")
            mock_sz_logger.end = AsyncMock(return_value={"id": "seizure-id"})
            
            mock_med_logger.get_range = AsyncMock(return_value=[])
            mock_med_logger.create = AsyncMock(return_value={"id": "med-id"})
            mock_med_logger.update = AsyncMock(return_value={"id": "med-id"})
            mock_med_logger.delete = AsyncMock(return_value=True)
            
            from server import app
            yield TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        with patch('server.health_monitor') as mock_health:
            mock_health.get_system_info.return_value = {
                "cpu_percent": 25.0,
                "memory_percent": 50.0
            }
            mock_health.check_health.return_value = (True, "System healthy")
            
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["is_healthy"] is True
    
    def test_status_format(self, client):
        """Test that status endpoint returns original format for compatibility."""
        # Need to mock the motion_detector, camera_manager, and other global objects
        from unittest.mock import PropertyMock
        
        with patch('server.motion_detector') as mock_motion, \
             patch('server.camera_manager') as mock_camera, \
             patch('server.video_recorder') as mock_recorder, \
             patch('server.seizure_detector') as mock_seizure:
            
            # Setup motion detector mock
            mock_motion.motion_degree = 0.045
            mock_sleep_mode = MagicMock()
            mock_sleep_mode.name = "ACTIVE"
            type(mock_motion).sleep_mode = PropertyMock(return_value=mock_sleep_mode)
            mock_motion.motion_last_detected_at = datetime(2025, 11, 30, 12, 30, 0, 123456)
            mock_motion.get_time_since_last_motion.return_value = timedelta(seconds=135)
            
            # Setup camera manager mock
            mock_camera.is_running.return_value = True
            mock_camera_mode = MagicMock()
            mock_camera_mode.name = "DAY"
            type(mock_camera).mode = PropertyMock(return_value=mock_camera_mode)
            mock_camera.get_state.return_value = {
                "state": "RUNNING",
                "mode": "DAY",
                "source": "mac_cam",
                "resolution": (640, 360),
                "options": {}
            }
            
            # Setup video recorder mock
            mock_recorder.is_recording.return_value = False
            
            # Setup seizure detector mock
            mock_seizure.analyze.return_value = {}
            mock_seizure.get_statistics.return_value = {
                'motion_triggers': 0,
                'activity_triggers': 0,
                'custom_detections': 0,
                'frames_processed': 0
            }
            mock_seizure.persistence_detector.get_state.return_value = {}
            mock_seizure.in_seizure_state = False
            mock_seizure.seizure_start_time = None
            mock_seizure.motion_persistence_detector.get_state.return_value = {}
            
            response = client.get("/status")
            assert response.status_code == 200
            data = response.json()
            
            # Verify all required fields are present
            # Top-level keys of GET /status; update when the payload changes.
            required_fields = [
                "analyzing_video", "camera_options", "camera_src", "connected_clients",
                "current_seizure_id", "detection_notify_only", "detection_stats", "diagnostics",
                "hr", "is_capturing", "is_recording", "is_seizing",
                "last_person_detected", "motion_magnitude", "night_mode", "notification_backends",
                "notify_on_hr_spike", "notify_on_wakeup", "person_detection_window_minutes", "pipeline",
                "seizure_ended_at", "seizure_started_at", "shadow_mode", "show_overlay",
                "sleep_mode", "time_since_last_motion", "type", "use_pose_detection",
                "wakeup_activity_threshold", "wakeup_require_in_bed", "wakeup_sleep_threshold_minutes"
            ]
            
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            # Verify field types
            assert data["type"] == "status"
            for flag in ("is_capturing", "is_recording", "use_pose_detection",
                         "night_mode", "show_overlay", "is_seizing"):
                assert isinstance(data[flag], bool), flag
            assert isinstance(data["sleep_mode"], str)  # Enum name
            assert isinstance(data["connected_clients"], int)
            assert isinstance(data["motion_magnitude"], float)
            assert isinstance(data["time_since_last_motion"], float)
            assert isinstance(data["hr"], dict) and "enabled" in data["hr"]
    
    def test_get_seizures(self, client):
        """Test GET /sz endpoint."""
        response = client.get("/sz?start=2025-01-01T00:00:00Z&end=2025-12-31T23:59:59Z")
        assert response.status_code == 200
        data = response.json()
        assert "events" in data
    
    def test_create_seizure(self, client):
        """Test POST /sz endpoint."""
        seizure_data = {
            "date": "2025-11-30T12:00:00Z",
            "type": "Epileptic spasm",
            "intensity": "3",
            "notes": "Test seizure"
        }
        
        response = client.post("/sz", json=seizure_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] is True
        assert "event" in data
    
    def test_update_seizure(self, client):
        """Test PUT /sz/{event_id} endpoint."""
        updates = {
            "notes": "Updated notes",
            "intensity": "4"
        }
        
        response = client.put("/sz/test-event-id", json=updates)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] is True
    
    def test_delete_seizure(self, client):
        """Test DELETE /sz/{event_id} endpoint."""
        response = client.delete("/sz/test-event-id")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    def test_get_medications(self, client):
        """Test GET /medicine endpoint."""
        response = client.get("/medicine?start=2025-01-01T00:00:00Z&end=2025-12-31T23:59:59Z")
        assert response.status_code == 200
        data = response.json()
        assert "events" in data
    
    def test_create_medication(self, client):
        """Test POST /medicine endpoint."""
        med_data = {
            "date": "2025-11-30T12:00:00Z",
            "name": "Keppra",
            "dosage": "500mg",
            "notes": "Morning dose"
        }
        
        response = client.post("/medicine", json=med_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] is True
        assert "event" in data
    
    def test_update_medication(self, client):
        """Test PUT /medicine/{event_id} endpoint."""
        updates = {
            "notes": "Updated notes",
            "dosage": "1000mg"
        }
        
        response = client.put("/medicine/test-med-id", json=updates)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] is True
    
    def test_delete_medication(self, client):
        """Test DELETE /medicine/{event_id} endpoint."""
        response = client.delete("/medicine/test-med-id")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    def test_command_start_capturing(self, client):
        """Test POST /cmd with startCapturing action."""
        command = {"action": "startCapturing"}
        
        response = client.post("/cmd", json=command)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_command_stop_capturing(self, client):
        """Test POST /cmd with stopCapturing action."""
        command = {"action": "stopCapturing"}
        
        response = client.post("/cmd", json=command)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_command_start_recording(self, client):
        """Test POST /cmd with startRecording action."""
        with patch('server.motion_detector') as mock_motion:
            mock_motion.get_adaptive_fps.return_value = 5.0
            
            command = {"action": "startRecording"}
            
            response = client.post("/cmd", json=command)
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
    
    def test_command_stop_recording(self, client):
        """Test POST /cmd with stopRecording action."""
        command = {"action": "stopRecording"}
        
        response = client.post("/cmd", json=command)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_command_night_mode(self, client):
        """Test POST /cmd with night_mode action."""
        command = {"action": "night_mode", "value": True}
        
        response = client.post("/cmd", json=command)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_command_show_overlay(self, client):
        """Test POST /cmd with show_overlay action."""
        command = {"action": "show_overlay", "value": True}
        
        response = client.post("/cmd", json=command)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_command_seizure_started(self, client):
        """Test POST /cmd with seizure_started action."""
        with patch('server.motion_detector') as mock_motion:
            mock_motion.get_adaptive_fps.return_value = 5.0
            
            command = {
                "action": "seizure_started",
                "value": {
                    "type": "Tonic-clonic",
                    "intensity": "5",
                    "notes": "Emergency"
                }
            }
            
            response = client.post("/cmd", json=command)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "Seizure started"
            assert "id" in data
    
    def test_command_seizure_ended(self, client):
        """Test POST /cmd with seizure_ended action."""
        # First start a seizure
        with patch('server.current_seizure_id', 'test-seizure-id'):
            command = {"action": "seizure_ended"}
            
            response = client.post("/cmd", json=command)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "Seizure ended"
    
    def test_command_set_controls(self, client):
        """Test POST /cmd with setControls action."""
        command = {
            "action": "setControls",
            "options": {
                "Brightness": 0.5,
                "Contrast": 1.2
            }
        }
        
        response = client.post("/cmd", json=command)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_command_unknown_action(self, client):
        """Test POST /cmd with unknown action."""
        command = {"action": "unknownAction"}
        
        response = client.post("/cmd", json=command)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Unknown action"
        assert data["action"] == "unknownAction"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
