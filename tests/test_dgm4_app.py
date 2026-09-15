import io
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.domain.forensics import aggregate_frames, classification_result
from src.infrastructure.ml.dgm4 import result_from_scores
from src.infrastructure.ml.model_factory import ModelFactory
from src.shared.exceptions import ModelLoadError


class ApplicationTests(unittest.TestCase):
    def prediction(self, p):
        return classification_result(result_from_scores([1-p, p]))

    def test_video_uses_edited_score_not_real_confidence(self):
        result = aggregate_frames([self.prediction(.01), self.prediction(.8)], ['a', 'b'])
        self.assertEqual(result.coarse_label, 'AI_EDITED')
        self.assertEqual(result.meta['worst_frame'], 'b')
        self.assertAlmostEqual(sum(result.meta['scores'].values()), 1)
        with self.assertRaises(ValueError):
            aggregate_frames([], [])

    def test_factory_requires_trained_checkpoint(self):
        with self.assertRaises(ModelLoadError):
            ModelFactory.build('configs/dgm4_binary.yml', 'missing_test_checkpoint.pth')

    def test_factory_delegates_to_shared_predictor(self):
        predictor = Mock(class_names=['REAL', 'AI_EDITED'])
        predictor.predict.return_value = result_from_scores([.2, .8])
        with patch('src.infrastructure.ml.model_factory.DGM4Predictor.load', return_value=predictor):
            classifier = ModelFactory.build('config', 'weights')
            self.assertEqual(classifier.classify_image('image').coarse_label, 'AI_EDITED')

    def test_upload_and_video_api(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.prediction(.8)
            video_result = aggregate_frames([result], ['frame.jpg'])
            container = SimpleNamespace(settings=SimpleNamespace(tmp_dir=folder, frame_step=24),
                workers=SimpleNamespace(start=AsyncMock(), stop=AsyncMock()),
                classifier_service=Mock(), storage=Mock(), video=Mock())
            container.classifier_service.classify_image.return_value = result
            container.classifier_service.classify_video_frames.return_value = video_result
            container.video.extract_keyframes.return_value = ['frame.jpg']
            app = create_app(lambda settings: container)
            buffer = io.BytesIO()
            Image.new('RGB', (16, 16)).save(buffer, format='PNG')
            with TestClient(app) as client:
                response = client.post('/classify/image', files={'file': ('a.png', buffer.getvalue(), 'image/png')})
                self.assertEqual(response.status_code, 200)
                payload = response.json()['ai_result']
                self.assertEqual(payload['label'], 'AI_EDITED')
                self.assertNotIn('defects', payload)
                self.assertEqual(client.post('/classify/image', files={'file': ('bad', b'bad')}).status_code, 422)
                self.assertNotIn('/inspect/image', app.openapi()['paths'])
                video = client.post('/classify/video', json={'url': 'https://example.com/video.mp4'})
                self.assertEqual(video.status_code, 200)
                self.assertEqual(video.json()['ai_result']['worst_frame_index'], 0)
                self.assertEqual(video.json()['ai_result']['label'], 'AI_EDITED')
                self.assertEqual(client.post('/classify/video', json={'url': 'https://example.com/video.mp4', 'frame_step': 0}).status_code, 422)
