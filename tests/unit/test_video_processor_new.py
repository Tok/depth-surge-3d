"""Tests for VideoProcessor thin orchestrator."""

from unittest.mock import Mock

from src.depth_surge_3d.processing.orchestration.video_processor import VideoProcessor


class TestVideoProcessorInit:
    """Test VideoProcessor initialization."""

    def test_init_creates_all_processors(self):
        """Test that init creates all specialized processor modules."""
        mock_estimator = Mock()

        processor = VideoProcessor(mock_estimator, verbose=False)

        assert processor.depth_processor is not None
        assert processor.stereo_generator is not None
        assert processor.distortion_processor is not None
        assert processor.upscaler is not None
        assert processor.vr_assembler is not None
        assert processor.video_encoder is not None
        assert processor.orchestrator is not None

    def test_init_with_verbose(self):
        """Test initialization with verbose enabled."""
        mock_estimator = Mock()

        processor = VideoProcessor(mock_estimator, verbose=True)

        assert processor.depth_processor.verbose is True
        assert processor.stereo_generator.verbose is True


class TestVideoProcessorProcess:
    """Test VideoProcessor.process delegation."""

    def test_process_delegates_to_orchestrator(self, tmp_path):
        """Test that process method delegates to orchestrator."""
        mock_estimator = Mock()
        processor = VideoProcessor(mock_estimator)

        # Mock the orchestrator's process method
        processor.orchestrator.process = Mock(return_value=True)

        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "output"
        video_properties = {"fps": 30}
        settings = {"vr_format": "side_by_side"}
        progress_callback = None

        result = processor.process(
            video_path, output_dir, video_properties, settings, progress_callback
        )

        assert result is True
        processor.orchestrator.process.assert_called_once()
