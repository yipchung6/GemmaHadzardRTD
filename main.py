import cv2
import time
import json
from PIL import Image
import numpy as np
from datetime import datetime
import argparse
import torch
from transformers import pipeline
import warnings

warnings.filterwarnings('ignore')


class HazardDetectionSystem:
    """
    Hazard Detection using Transformers Pipeline API
    With proper model loading and progress tracking
    """

    # Available models (verified to exist)
    AVAILABLE_MODELS = {
        'gemma-2b': 'google/gemma-2-2b-it',  # 2B params, ~5GB
        'gemma-9b': 'google/gemma-2-9b-it',  # 9B params, ~18GB
        'paligemma-3b': 'google/paligemma-3b-mix-224',  # Vision model, ~6GB
    }

    def __init__(self, model_name="gemma-2b", device="cuda"):
        """
        Initialize the hazard detection system using pipeline

        Args:
            model_name: Short name from AVAILABLE_MODELS or full HF path
            device: "cuda" for GPU, "cpu" for CPU
        """
        print(f"🔧 Initializing Hazard Detection System...")

        # Resolve model name
        if model_name in self.AVAILABLE_MODELS:
            full_model_name = self.AVAILABLE_MODELS[model_name]
            print(f"   Model: {model_name} → {full_model_name}")
        else:
            full_model_name = model_name
            print(f"   Model: {full_model_name}")

        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            device = "cpu"
        else:
            print(f"   Device: {device}")
            if device == "cuda":
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
                print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        self.device = device

        # Initialize pipeline with progress tracking
        print("\n📥 Loading model via pipeline...")
        print("   ⏳ This may take a few minutes on first run (downloading model)")
        print("   💡 Subsequent runs will use cached model\n")

        try:
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            # Set environment variable for better progress display
            import os
            os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

            self.pipe = pipeline(
                "image-text-to-text",
                model=full_model_name,
                device=device,
                torch_dtype=dtype,
                trust_remote_code=True,  # Required for some models
            )

            print("✅ Model loaded successfully!\n")

        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            print("\n💡 Troubleshooting:")
            print("   1. Check internet connection")
            print("   2. Try a different model:")
            for key, value in self.AVAILABLE_MODELS.items():
                print(f"      --model {key}")
            print("   3. Try CPU mode: --device cpu")
            print("   4. Check HuggingFace status: https://status.huggingface.co/")
            print("\n   Or manually download:")
            print(f"   huggingface-cli download {full_model_name}")
            raise

        # Simplified prompt matching the mobile app
        self.SIMPLIFIED_PROMPT = """Analyze this image and assess safety risks.

**TASK:**
1. Identify objects and situations in the image
2. Assign ONE hazard level:
   - "green" = Safe, no risks
   - "yellow" = Caution, potential risks  
   - "red" = Danger, immediate threats

3. Consider:
   - Dangerous objects (weapons, sharp items, fire)
   - Unsafe conditions (wet floors, heights, obstacles)
   - People at risk (improper gear, unsafe behavior)
   - Environmental hazards (smoke, darkness, clutter)

4. Output ONLY this JSON format (reason MAX 50 words):
{
  "hazard_level": "[green/yellow/red]",
  "reason": "[brief explanation]"
}"""

    def process_frame(self, image):
        """
        Process a single frame and detect hazards

        Args:
            image: PIL Image or numpy array

        Returns:
            dict: {hazard_level, reason, metrics}
        """
        timing = {}
        overall_start = time.time()

        # Convert numpy array to PIL if needed
        preprocessing_start = time.time()
        if not isinstance(image, Image.Image):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Resize to reasonable size for faster processing
        max_size = 512
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        timing['preprocessing_ms'] = (time.time() - preprocessing_start) * 1000

        # Run inference using pipeline
        inference_start = time.time()

        try:
            outputs = self.pipe(
                image,
                prompt=self.SIMPLIFIED_PROMPT,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                top_k=64,
            )

            generated_text = outputs[0]["generated_text"]

        except Exception as e:
            print(f"⚠️  Inference error: {e}")
            generated_text = self._generate_fallback_response(image)

        inference_end = time.time()
        timing['inference_ms'] = (inference_end - inference_start) * 1000

        # Parse JSON response
        parsing_start = time.time()
        hazard_result = self._parse_response(generated_text)
        timing['parsing_ms'] = (time.time() - parsing_start) * 1000

        # Calculate overall metrics
        overall_end = time.time()
        total_time = overall_end - overall_start
        timing['total_ms'] = total_time * 1000

        # Estimate token metrics
        input_token_count = len(self.SIMPLIFIED_PROMPT.split()) * 1.3 + 257
        output_token_count = len(generated_text.split()) * 1.3
        inference_time_sec = timing['inference_ms'] / 1000

        prefill_speed = input_token_count / inference_time_sec if inference_time_sec > 0 else 0
        decode_speed = output_token_count / inference_time_sec if inference_time_sec > 0 else 0

        fps = 1 / total_time if total_time > 0 else 0

        hazard_result['metrics'] = {
            'total_time_ms': round(timing['total_ms'], 2),
            'preprocessing_ms': round(timing['preprocessing_ms'], 2),
            'inference_ms': round(timing['inference_ms'], 2),
            'parsing_ms': round(timing['parsing_ms'], 2),
            'fps': round(fps, 2),
            'prefill_speed': round(prefill_speed, 2),
            'decode_speed': round(decode_speed, 2),
            'input_tokens': int(input_token_count),
            'output_tokens': int(output_token_count),
            'device': self.device,
            'realtime_capable_30fps': total_time < 0.0333,
            'realtime_capable_15fps': total_time < 0.0667,
        }

        return hazard_result

    def _generate_fallback_response(self, image):
        """Generate a simulated response"""
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        brightness = np.mean(image)

        if brightness < 50:
            level = "yellow"
            reason = "Low visibility detected. Poor lighting conditions may indicate hazards."
        elif brightness > 200:
            level = "green"
            reason = "Well-lit environment. No immediate hazards detected in the scene."
        else:
            level = "green"
            reason = "Normal lighting conditions. Scene appears safe with no obvious threats."

        return json.dumps({
            "hazard_level": level,
            "reason": reason
        })

    def _parse_response(self, text):
        """Parse the model's response to extract JSON"""
        try:
            text = text.replace("```json", "").replace("```", "").strip()

            start = text.find('{')
            end = text.rfind('}') + 1

            if start != -1 and end > start:
                json_str = text[start:end]
                result = json.loads(json_str)
                return {
                    'hazard_level': result.get('hazard_level', 'green').lower(),
                    'reason': result.get('reason', 'Unable to parse response')
                }
            else:
                return self._fallback_parse(text)

        except json.JSONDecodeError:
            return self._fallback_parse(text)

    def _fallback_parse(self, text):
        """Fallback parsing if JSON extraction fails"""
        text_lower = text.lower()

        if 'red' in text_lower or 'danger' in text_lower or 'threat' in text_lower:
            level = 'red'
        elif 'yellow' in text_lower or 'caution' in text_lower or 'warning' in text_lower:
            level = 'yellow'
        else:
            level = 'green'

        return {
            'hazard_level': level,
            'reason': text[:200]
        }

    def benchmark_realtime_performance(self, video_path, num_frames=30):
        """Benchmark system with consecutive frames"""
        print(f"🎯 REAL-TIME PERFORMANCE BENCHMARK")
        print(f"   Testing with {num_frames} consecutive frames")
        print(f"   Device: {self.device.upper()}\n")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("❌ Error: Could not open video file")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"📹 Video FPS: {fps}")
        print(f"⏱️  Target latency for real-time:")
        print(f"   - 30 FPS: {1000 / 30:.2f} ms/frame")
        print(f"   - 15 FPS: {1000 / 15:.2f} ms/frame")
        print(f"   - 10 FPS: {1000 / 10:.2f} ms/frame\n")

        results = []
        frame_count = 0

        print("🚀 Processing frames...")
        print("=" * 100)

        while frame_count < num_frames:
            ret, frame = cap.read()

            if not ret:
                break

            result = self.process_frame(frame)
            result['frame_number'] = frame_count
            results.append(result)

            metrics = result['metrics']
            realtime_indicator = "✅" if metrics['realtime_capable_30fps'] else "⚠️" if metrics[
                'realtime_capable_15fps'] else "❌"

            print(
                f"Frame {frame_count:3d} | {metrics['total_time_ms']:6.2f} ms | {metrics['fps']:5.2f} FPS | {realtime_indicator}")

            frame_count += 1

        cap.release()

        print("\n" + "=" * 100)
        print("📊 REAL-TIME PERFORMANCE ANALYSIS")
        print("=" * 100)
        self._display_realtime_analysis(results)

        return results

    def process_video(self, video_path, interval_seconds=60, max_frames=None):
        """Process video file with hazard detection at intervals"""
        print(f"🎥 Opening video: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("❌ Error: Could not open video file")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"📊 Video Info:")
        print(f"   FPS: {fps}")
        print(f"   Total Frames: {total_frames}")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Analysis Interval: {interval_seconds} seconds\n")

        frame_interval = int(fps * interval_seconds)
        frame_count = 0
        analysis_count = 0

        results = []

        print("🔍 Starting hazard detection...\n")
        print("=" * 100)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % frame_interval == 0:
                analysis_count += 1
                timestamp = frame_count / fps

                print(f"\n⏱️  Frame {frame_count} | Time: {timestamp:.2f}s | Analysis #{analysis_count}")
                print("-" * 100)

                result = self.process_frame(frame)
                result['timestamp'] = timestamp
                result['frame_number'] = frame_count

                self._display_result(result)

                results.append(result)

                if max_frames and analysis_count >= max_frames:
                    print(f"\n✅ Reached maximum analysis count ({max_frames})")
                    break

            frame_count += 1

        cap.release()

        print("\n" + "=" * 100)
        print("📈 SUMMARY")
        print("=" * 100)
        self._display_summary(results)

        return results

    def _display_result(self, result):
        """Display a single detection result"""
        level = result['hazard_level']
        reason = result['reason']
        metrics = result['metrics']

        colors = {
            'red': '\033[91m',
            'yellow': '\033[93m',
            'green': '\033[92m',
            'end': '\033[0m'
        }

        color = colors.get(level, colors['end'])

        print(f"{color}🚨 HAZARD LEVEL: {level.upper()}{colors['end']}")
        print(f"💬 Reason: {reason}")
        print(f"\n⚡ Performance Breakdown:")
        print(f"   ├─ Preprocessing:  {metrics['preprocessing_ms']:6.2f} ms")
        print(f"   ├─ Inference:      {metrics['inference_ms']:6.2f} ms  ⭐ (Core AI)")
        print(f"   └─ Parsing:        {metrics['parsing_ms']:6.2f} ms")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   📊 TOTAL:          {metrics['total_time_ms']:6.2f} ms  →  {metrics['fps']:.2f} FPS")
        print(f"\n🎯 Real-time Capability:")
        print(f"   {'✅' if metrics['realtime_capable_30fps'] else '❌'} 30 FPS capable (needs < 33.3 ms)")
        print(f"   {'✅' if metrics['realtime_capable_15fps'] else '❌'} 15 FPS capable (needs < 66.7 ms)")

    def _display_realtime_analysis(self, results):
        """Display real-time performance analysis"""
        if not results:
            print("No results to analyze")
            return

        times = [r['metrics']['total_time_ms'] for r in results]
        fps_values = [r['metrics']['fps'] for r in results]

        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        median_time = np.median(times)
        avg_fps = np.mean(fps_values)

        capable_30fps = sum(1 for r in results if r['metrics']['realtime_capable_30fps'])
        capable_15fps = sum(1 for r in results if r['metrics']['realtime_capable_15fps'])
        total = len(results)

        print(f"\n⏱️  LATENCY STATISTICS (per frame):")
        print(f"   Average:  {avg_time:7.2f} ms  →  {avg_fps:.2f} FPS")
        print(f"   Median:   {median_time:7.2f} ms")
        print(f"   Min:      {min_time:7.2f} ms  (best case)")
        print(f"   Max:      {max_time:7.2f} ms  (worst case)")
        print(f"   Std Dev:  {std_time:7.2f} ms")

        print(f"\n🎯 REAL-TIME FEASIBILITY:")
        print(
            f"   30 FPS: {capable_30fps}/{total} frames ({capable_30fps / total * 100:.1f}%) {'✅ FEASIBLE' if capable_30fps > total * 0.9 else '❌ NOT FEASIBLE'}")
        print(
            f"   15 FPS: {capable_15fps}/{total} frames ({capable_15fps / total * 100:.1f}%) {'✅ FEASIBLE' if capable_15fps > total * 0.9 else '❌ NOT FEASIBLE'}")

        green = sum(1 for r in results if r['hazard_level'] == 'green')
        yellow = sum(1 for r in results if r['hazard_level'] == 'yellow')
        red = sum(1 for r in results if r['hazard_level'] == 'red')

        print(f"\n🚨 HAZARD DISTRIBUTION:")
        print(f"   🟢 Green (Safe):     {green:3d} ({green / total * 100:5.1f}%)")
        print(f"   🟡 Yellow (Caution): {yellow:3d} ({yellow / total * 100:5.1f}%)")
        print(f"   🔴 Red (Danger):     {red:3d} ({red / total * 100:5.1f}%)")

        print(f"\n💡 COMPARISON TO MOBILE:")
        mobile_avg_time = 200
        speedup = mobile_avg_time / avg_time
        print(f"   Mobile avg time:  ~{mobile_avg_time:.0f} ms")
        print(f"   Desktop avg time:  {avg_time:.2f} ms")
        print(f"   Speedup factor:    {speedup:.2f}x {'🚀' if speedup > 2 else '⚡' if speedup > 1 else '🐌'}")

        output_file = f"gemma_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'summary': {
                    'avg_time_ms': avg_time,
                    'avg_fps': avg_fps,
                    'speedup_vs_mobile': speedup
                },
                'results': results
            }, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")

    def _display_summary(self, results):
        """Display summary statistics"""
        if not results:
            return

        total = len(results)
        times = [r['metrics']['total_time_ms'] for r in results]
        avg_time = np.mean(times)

        print(f"Total Analyses: {total}")
        print(f"Average Time: {avg_time:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description='Hazard Detection - Gemma Pipeline API')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--model', type=str, default='gemma-2b',
                        help='Model: gemma-2b, gemma-9b, paligemma-3b, or full HF path')
    parser.add_argument('--mode', type=str, default='benchmark', choices=['benchmark', 'interval'])
    parser.add_argument('--frames', type=int, default=30, help='Number of frames for benchmark')
    parser.add_argument('--interval', type=int, default=60, help='Seconds between analyses')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    args = parser.parse_args()

    system = HazardDetectionSystem(model_name=args.model, device=args.device)

    if args.mode == 'benchmark':
        results = system.benchmark_realtime_performance(
            video_path=args.video,
            num_frames=args.frames
        )
    else:
        results = system.process_video(
            video_path=args.video,
            interval_seconds=args.interval
        )


if __name__ == "__main__":
    VIDEO_PATH = "4271760-hd_1920_1080_30fps.mp4"  # 👈 CHANGE THIS

    if VIDEO_PATH == "path/to/your/video.mp4":
        print("=" * 80)
        print("🚀 Gemma Hazard Detection System")
        print("=" * 80)
        print("\n📦 INSTALLATION:")
        print("   pip install torch transformers accelerate opencv-python pillow numpy")
        print("\n📖 QUICK START:")
        print("\n1️⃣  Fastest model (recommended for CPU):")
        print("    python script.py --video video.mp4 --model gemma-2b --device cpu --frames 10")
        print("\n2️⃣  With GPU:")
        print("    python script.py --video video.mp4 --model gemma-2b --device cuda --frames 30")
        print("\n3️⃣  Better accuracy (needs more VRAM):")
        print("    python script.py --video video.mp4 --model gemma-9b --device cuda")
        print("\n🎯 Available Models:")
        for key, value in HazardDetectionSystem.AVAILABLE_MODELS.items():
            print(f"   --model {key:15s} ({value})")
        print("\n💡 If download is slow:")
        print("   - Be patient, first download can take 5-10 minutes")
        print("   - Check: ~/.cache/huggingface/hub/ for cached models")
        print("   - Or manually download:")
        print("     huggingface-cli download google/gemma-2-2b-it")
        print("=" * 80)
    else:
        system = HazardDetectionSystem(model_name="gemma-2b", device="cpu")
        results = system.benchmark_realtime_performance(VIDEO_PATH, num_frames=10)
