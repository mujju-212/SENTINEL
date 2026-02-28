"""
MODULE 8 - Alert System
Detects important events and triggers:
  - Sound alerts (via pygame)
  - On-screen banners (via Visualizer)
  - Screenshot saves
  - Console log entries

Alert types:
  UNKNOWN_PERSON  → unrecognized face detected
  KNOWN_PERSON    → known person appeared (first time this session)
  OBJECT_ALERT    → specific watched object appeared
  TEXT_DETECTED   → OCR found important text
"""

import cv2
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class AlertEvent:
    """Single alert event record."""
    def __init__(self, alert_type: str, message: str, data: dict = None):
        self.alert_type = alert_type
        self.message = message
        self.data = data or {}
        self.timestamp = datetime.now()

    def __repr__(self):
        return f"[{self.alert_type}] {self.message}"


class AlertSystem:
    """Manages all alert detection and notifications."""

    # Minimum seconds between same-type alerts (prevents spam)
    COOLDOWN = {
        'UNKNOWN_PERSON': 10.0,
        'KNOWN_PERSON':   30.0,
        'OBJECT_ALERT':   15.0,
        'TEXT_DETECTED':  20.0,
    }

    def __init__(self, config: dict):
        self.config = config
        self.enabled_unknown    = config.get('unknown_person', True)
        self.enabled_greeting   = config.get('known_person_greeting', True)
        self.sound_enabled      = config.get('sound_enabled', True)
        self.save_screenshot    = config.get('save_screenshot_on_alert', True)
        self.screenshot_dir     = Path(config.get('screenshot_dir', 'output/screenshots'))
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Track cooldowns: key = (alert_type, subject)
        self._last_trigger: dict = {}
        self._session_greeted: set = set()  # Avoid repeat greetings per session
        self._active_alert: Optional[AlertEvent] = None
        self._alert_display_until: float = 0.0

        self._sound_ready = self._init_sound()
        logger.info("[Alert] Alert system ready")

    # ── SOUND INIT ───────────────────────────────────────────────────
    def _init_sound(self) -> bool:
        if not self.sound_enabled:
            return False
        try:
            import pygame
            pygame.mixer.init()
            return True
        except Exception as e:
            logger.warning(f"[Alert] Sound unavailable: {e}")
            return False

    # ── MAIN: CHECK FRAME ────────────────────────────────────────────
    def check(self, detections: list, ocr_results: list = None,
              frame=None) -> Optional[AlertEvent]:
        """
        Analyze detections and OCR results for alert conditions.

        Args:
            detections:  List of detection dicts
            ocr_results: List of OCRResult objects (optional)
            frame:       Current frame for screenshot (optional)

        Returns:
            AlertEvent if triggered, else None
        """
        event = None

        # ── Check for persons ────────────────────────────────────────
        for det in detections:
            if det.get('class_name') != 'person':
                continue

            name = det.get('person_name') or 'Unknown'
            face_conf = det.get('face_confidence', 0.0)

            if name == 'Unknown' and face_conf == 0.0:
                # Unknown person with no face match
                if self.enabled_unknown and self._can_trigger('UNKNOWN_PERSON', 'generic'):
                    event = AlertEvent(
                        'UNKNOWN_PERSON',
                        '⚠  UNKNOWN PERSON DETECTED',
                        {'track_id': det.get('track_id')}
                    )
                    self._fire(event, frame)
                    break
            elif name != 'Unknown':
                # Known person appeared
                if self.enabled_greeting and name not in self._session_greeted:
                    if self._can_trigger('KNOWN_PERSON', name):
                        event = AlertEvent(
                            'KNOWN_PERSON',
                            f'✓  Welcome back, {name}!',
                            {'name': name, 'confidence': face_conf}
                        )
                        self._session_greeted.add(name)
                        self._fire(event, frame)
                        break

        # ── Check OCR for important text ─────────────────────────────
        if event is None and ocr_results:
            important_keywords = ['stop', 'exit', 'danger', 'warning', 'fire', 'alert']
            for r in ocr_results:
                if any(kw in r.text.lower() for kw in important_keywords):
                    if self._can_trigger('TEXT_DETECTED', r.text[:20]):
                        event = AlertEvent(
                            'TEXT_DETECTED',
                            f'📝  Text: "{r.text}"',
                            {'text': r.text, 'confidence': r.confidence}
                        )
                        self._fire(event, frame)
                        break

        return event

    # ── FIRE ALERT ───────────────────────────────────────────────────
    def _fire(self, event: AlertEvent, frame=None):
        """Execute all alert actions."""
        logger.info(f"[Alert] {event}")

        # Set display banner for 3 seconds
        self._active_alert = event
        self._alert_display_until = time.time() + 3.0

        # Save screenshot
        if self.save_screenshot and frame is not None:
            self._save_screenshot(event, frame)

        # Play sound in background
        if self._sound_ready:
            threading.Thread(target=self._play_beep,
                             args=(event.alert_type,), daemon=True).start()

    # ── COOLDOWN CHECK ───────────────────────────────────────────────
    def _can_trigger(self, alert_type: str, subject: str) -> bool:
        key = f"{alert_type}:{subject}"
        cooldown = self.COOLDOWN.get(alert_type, 10.0)
        last = self._last_trigger.get(key, 0.0)
        now = time.time()
        if now - last >= cooldown:
            self._last_trigger[key] = now
            return True
        return False

    # ── SCREENSHOT ───────────────────────────────────────────────────
    def _save_screenshot(self, event: AlertEvent, frame):
        try:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_type = event.alert_type.lower().replace(' ', '_')
            path = self.screenshot_dir / f"{safe_type}_{ts}.jpg"
            cv2.imwrite(str(path), frame)
            logger.info(f"[Alert] Screenshot: {path}")
        except Exception as e:
            logger.warning(f"[Alert] Screenshot failed: {e}")

    # ── SOUND ────────────────────────────────────────────────────────
    def _play_beep(self, alert_type: str):
        try:
            import pygame
            import numpy as np
            sample_rate = 44100
            freq = 880 if alert_type == 'UNKNOWN_PERSON' else 660
            dur  = 0.25 if alert_type == 'UNKNOWN_PERSON' else 0.15
            t = np.linspace(0, dur, int(sample_rate * dur), False)
            tone = (np.sin(freq * 2 * np.pi * t) * 32767).astype(np.int16)
            stereo = np.column_stack([tone, tone])
            sound = pygame.sndarray.make_sound(stereo)
            sound.play()
            pygame.time.wait(int(dur * 1200))
        except Exception as e:
            logger.debug(f"[Alert] Sound error: {e}")

    # ── GET ACTIVE ALERT ─────────────────────────────────────────────
    def get_active_alert(self) -> Optional[AlertEvent]:
        """Return current alert if still within display window."""
        if self._active_alert and time.time() < self._alert_display_until:
            return self._active_alert
        return None

    def reset_session(self):
        """Clear session-persistent state (call on restart)."""
        self._session_greeted.clear()
        self._last_trigger.clear()
        self._active_alert = None
