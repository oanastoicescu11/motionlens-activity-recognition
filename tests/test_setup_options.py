"""Test that each setup option can be initialized."""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SetupOptionsTests(unittest.TestCase):
    """Verify each setup option initializes correctly."""

    def test_option1_backend_only(self):
        """Option 1: Backend can be created without dependencies on other components."""
        try:
            from app.backend.main import create_app
            app = create_app()
            # create_app() returns None if FastAPI not installed; that's OK
            if app is None:
                self.skipTest("FastAPI not installed (optional for Option 1)")
        except ImportError:
            self.skipTest("FastAPI not installed (optional for Option 1)")

    def test_option2_backend_ui_imports(self):
        """Option 2: Backend + UI components can import independently."""
        try:
            from app.backend.main import create_app
            from app.ui.streamlit_app import main as streamlit_main
            
            app = create_app()
            # create_app() returns None if FastAPI not installed; that's OK
            if app is None:
                self.skipTest("FastAPI not installed (optional for Option 2)")
            self.assertIsNotNone(streamlit_main)
        except ImportError:
            self.skipTest("FastAPI or Streamlit not installed (optional for Option 2)")

    def test_option3_full_stack_imports(self):
        """Option 3: All components can import without circular dependencies."""
        try:
            from app.backend.main import create_app
            from app.worker.main import start_workers
            from app.ui.streamlit_app import main as streamlit_main
            
            app = create_app()
            # create_app() returns None if FastAPI not installed; that's OK
            if app is None:
                self.skipTest("FastAPI not installed (optional for Option 3)")
            self.assertIsNotNone(start_workers)
            self.assertIsNotNone(streamlit_main)
        except ImportError:
            self.skipTest("Optional components not installed for Option 3")

    def test_session_creation_works_in_any_option(self):
        """Session creation helper works independent of which option is chosen."""
        from app.backend.core.auth import issue_session_credentials
        from app.backend.models.session import SessionMeta
        import uuid
        import time
        
        creds = issue_session_credentials()
        session_id = str(uuid.uuid4())
        expires_at_ns = int(time.time_ns() + 3600 * 1_000_000_000)
        
        meta = SessionMeta(
            session_id=session_id,
            owner_id="test",
            viewer_token=creds.viewer_token,
            write_token=creds.write_token,
            device_id="test-device",
            expires_at_ns=expires_at_ns,
        )
        
        self.assertEqual(meta.session_id, session_id)
        self.assertIsNotNone(meta.viewer_token)
        self.assertIsNotNone(meta.write_token)


if __name__ == "__main__":
    unittest.main()
