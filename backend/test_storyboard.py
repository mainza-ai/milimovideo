import asyncio
import os
import shutil
import unittest
from unittest.mock import MagicMock, patch
from backend.storyboard.manager import StoryboardManager

class TestStoryboardManager(unittest.TestCase):
    def setUp(self):
        self.job_id = "test_job"
        self.prompt = "A cinematic shot of a robot"
        self.output_dir = "test_output"
        self.params = {
            "num_frames": 241, # Should be 2 chunks of 121 (if overlap 8 -> wait, calculation is complex)
            "overlap_frames": 8
        }
        os.makedirs(self.output_dir, exist_ok=True)
        
    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_chunk_calculation(self):
        # 121 chunk size, 8 overlap.
        # Chunk 1: 0-120. (121 frames)
        # Chuck 2: starts at 121-8 = 113? 
        # Logic says: Effective new frames = 121 - 8 = 113.
        # Total 241.
        # 241 - 121 = 120 remaining.
        # 120 / 113 = 1.06 -> 2 more chunks? Total 3?
        
        manager = StoryboardManager(self.job_id, self.prompt, self.params, self.output_dir)
        num_chunks = manager.get_total_chunks()
        print(f"Total chunks for 241 frames: {num_chunks}")
        # manual: 
        # C1: 121. Covered 121.
        # C2: adds 113. Covered 234.
        # C3: adds 7. Covered 241.
        # So expects 3.
        self.assertEqual(num_chunks, 3)

    @patch("subprocess.run")
    def test_prepare_next_chunk(self, mock_subprocess):
        manager = StoryboardManager(self.job_id, self.prompt, self.params, self.output_dir)
        
        # Mock file existence for "prev_chunk"
        prev_chunk = os.path.join(self.output_dir, "chunk0.mp4")
        with open(prev_chunk, 'w') as f:
            f.write("dummy")
            
        # Mock ffmpeg extraction logic in _extract_last_n_frames
        # We need it to return list of files
        with patch.object(manager, '_extract_last_n_frames') as mock_extract:
            mock_extract.return_value = ["img1.png", "img2.png"]
            
            # Async run
            loop = asyncio.new_event_loop()
            config = loop.run_until_complete(manager.prepare_next_chunk(1, prev_chunk))
            
            self.assertEqual(config['prompt'], self.prompt)
            self.assertEqual(len(config['images']), 2)
            # Check tuple structure: (path, idx, strength)
            self.assertEqual(config['images'][0], ("img1.png", 0, 1.0))
            self.assertEqual(config['images'][1], ("img2.png", 1, 1.0))
            
if __name__ == "__main__":
    unittest.main()
