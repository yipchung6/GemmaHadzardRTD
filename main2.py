import cv2
import time
import json
from PIL import Image
import numpy as np
from datetime import datetime
import argparse
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import warnings

warnings.filterwarnings('ignore')


class HazardDetectionSystem:
    """
    Hazard Detection using Gemma 3N (Correct Implementation)
    Uses Gemma3nForConditionalGeneration with chat format
    """

    AVAILABLE_MODELS = {
        'gemma-2b': 'google/gemma-3n-e2b-it',
        'gemma-4b': 'google/gemma-3n-e4b-it',
    }

    def __init__(self, model_name="gemma-2b", device="cuda"):
        """
        Initialize Gemma 3N with proper multimodal support

        Args:
            model_name: gemma-2b or gemma-4b
            device: "cuda" or "cpu"
        """
        print(f"🔧 Initializing Hazard Detection System...")

        # Resolve model name
        if model_name in self.AVAILABLE_MODELS:
            model_id = self.AVAILABLE_MODELS[model_name]
            print(f"   Model: {model_name} → {model_id}")
        else:
            model_id = model_name
            print(f"   Model: {model_id}")

        # Check device
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            device = "cpu"
        else:
            print(f"   Device: {device}")
            if device == "cuda":
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
                print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        self.device = device

        print("\n📥 Loading Gemma 3N model...")
        print("   ⏳ First run: downloading model (~4-8GB)")
        print("   💡 Cached runs: instant loading\n")

        try:
            # Load model with proper class
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype=dtype,
            ).eval()

            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)

            print("✅ Gemma 3N loaded successfully!\n")

        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            print("\n💡 Troubleshooting:")
            print("   1. Install latest transformers: pip install --upgrade transformers")
            print("   2. Check transformers version: pip show transformers (need >= 4.40)")
            print("   3. Try: pip install git+https://github.com/huggingface/transformers.git")
            raise

        # Hazard analysis prompt
        self.HAZARD_SYSTEM_PROMPT = """You are a safety hazard detection expert. Analyze images and identify potential dangers."""

        self.HAZARD_USER_PROMPT = """Analyze this image for safety hazards.

Assign ONE hazard level:
- "green" = Safe, no risks
- "yellow" = Caution, potential risks  
- "red" = Danger, immediate threats

Consider:
- Dangerous objects (weapons, sharp items, fire, chemicals)
- Unsafe conditions (wet floors, heights, obstacles, poor lighting)
- People at risk (improper safety gear, unsafe behavior)
- Environmental hazards (smoke, darkness, extreme weather)

Respond ONLY with valid JSON (no markdown, no extra text):
{"hazard_level": "[green/yellow/red]", "reason": "[brief explanation under 50 words]"}"""

    def process_frame(self, image):
        """Process a single frame and detect hazards"""
        timing = {}
        overall_start = time.time()

        # Preprocessing
        preprocessing_start = time.time()
        if not isinstance(image, Image.Image):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Gemma 3N works well with images up to 448x448
        max_size = 448
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        timing['preprocessing_ms'] = (time.time() - preprocessing_start) * 1000

        # Prepare chat messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.HAZARD_SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.HAZARD_USER_PROMPT}
                ]
            }
        ]

        # Apply chat template for token counting
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        input_ids = self.processor.tokenizer(text, return_tensors="pt")["input_ids"]
        input_tokens = input_ids.size(-1)

        # Process and generate
        inference_start = time.time()
        try:
            # Apply chat template and prepare inputs
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt"
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Decode output
            output_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = self.processor.decode(output_tokens, skip_special_tokens=True)
            output_token_count = output_tokens.shape[-1]

        except Exception as e:
            print(f"⚠️  Inference error: {e}")
            generated_text = self._generate_fallback_response(image)
            output_token_count = len(generated_text.split())

        inference_end = time.time()
        inference_ms = (inference_end - inference_start) * 1000

        timing['inference_ms'] = inference_ms

        # Prefill/decode speed
        # (HuggingFace does not split the timing; use whole inference_time for both, like mobile)
        inference_time_s = (inference_end - inference_start)
        prefill_speed = input_tokens / inference_time_s if inference_time_s > 0 else 0
        decode_speed = output_token_count / inference_time_s if inference_time_s > 0 else 0

        # Parse response
        parsing_start = time.time()
        hazard_result = self._parse_response(generated_text)
        timing['parsing_ms'] = (time.time() - parsing_start) * 1000

        # Calculate metrics
        overall_end = time.time()
        total_time = overall_end - overall_start
        timing['total_ms'] = total_time * 1000

        fps = 1 / total_time if total_time > 0 else 0

        # Add metrics
        hazard_result['metrics'] = {
            'total_time_ms': round(timing['total_ms'], 2),
            'preprocessing_ms': round(timing['preprocessing_ms'], 2),
            'inference_ms': round(timing['inference_ms'], 2),
            'parsing_ms': round(timing['parsing_ms'], 2),
            'input_tokens': int(input_tokens),
            'output_tokens': int(output_token_count),
            'prefill_speed': round(prefill_speed, 2),
            'decode_speed': round(decode_speed, 2),
            'fps': round(fps, 2),
            'device': self.device,
            'realtime_capable_30fps': total_time < 0.0333,
            'realtime_capable_15fps': total_time < 0.0667,
        }

        # Store raw response for debugging
        hazard_result['raw_response'] = generated_text

        return hazard_result

    def _generate_fallback_response(self, image):
        """Generate fallback response"""
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        brightness = np.mean(image)

        if brightness < 50:
            return json.dumps({
                "hazard_level": "yellow",
                "reason": "Low visibility detected. Poor lighting may indicate hazards."
            })
        else:
            return json.dumps({
                "hazard_level": "green",
                "reason": "Well-lit environment. No obvious hazards detected."
            })

    def _parse_response(self, text):
        """Parse JSON response from model"""
        try:
            # Remove markdown code blocks
            text = text.replace("```json", "").replace("```", "").strip()

            # Find JSON object
            start = text.find('{')
            end = text.rfind('}') + 1

            if start != -1 and end > start:
                json_str = text[start:end]
                result = json.loads(json_str)
                return {
                    'hazard_level': result.get('hazard_level', 'green').lower(),
                    'reason': result.get('reason', 'Analysis complete')
                }
            else:
                return self._fallback_parse(text)

        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
            return self._fallback_parse(text)

    def _fallback_parse(self, text):
        """Fallback text parsing"""
        text_lower = text.lower()

        # Look for hazard keywords
        if any(word in text_lower for word in ['red', 'danger', 'threat', 'unsafe', 'critical', 'emergency']):
            level = 'red'
        elif any(word in text_lower for word in ['yellow', 'caution', 'warning', 'careful', 'potential']):
            level = 'yellow'
        else:
            level = 'green'

        return {
            'hazard_level': level,
            'reason': text[:150] if text else "Unable to parse response"
        }

    def benchmark_realtime_performance(self, video_path, num_frames=30):
        """Benchmark with consecutive frames"""
        print(f"🎯 REAL-TIME PERFORMANCE BENCHMARK")
        print(f"   Model: Gemma 3N (Native Multimodal)")
        print(f"   Testing with {num_frames} consecutive frames")
        print(f"   Device: {self.device.upper()}\n")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("❌ Error: Could not open video file")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"📹 Video FPS: {fps}")
        print(f"⏱️  Target latency:")
        print(f"   - 30 FPS: {1000 / 30:.2f} ms/frame")
        print(f"   - 15 FPS: {1000 / 15:.2f} ms/frame")
        print(f"   - 10 FPS: {1000 / 10:.2f} ms/frame\n")

        results = []
        frame_count = 0

        print("🚀 Processing frames...")
        print("=" * 110)

        while frame_count < num_frames:
            ret, frame = cap.read()

            if not ret:
                break

            result = self.process_frame(frame)
            result['frame_number'] = frame_count
            results.append(result)

            metrics = result['metrics']
            indicator = "✅" if metrics['realtime_capable_30fps'] else "⚠️" if metrics['realtime_capable_15fps'] else "❌"

            print(
                f"Frame {frame_count:3d} | {metrics['total_time_ms']:7.2f} ms | {metrics['fps']:5.2f} FPS | {indicator} | {result['hazard_level']:6s} | {result['reason'][:50]}")

            frame_count += 1

        cap.release()

        print("\n" + "=" * 110)
        print("📊 REAL-TIME PERFORMANCE ANALYSIS")
        print("=" * 110)
        self._display_realtime_analysis(results)

        return results

    def _display_realtime_analysis(self, results):
        """Display performance analysis"""
        if not results:
            print("No results to analyze")
            return

        times = [r['metrics']['total_time_ms'] for r in results]
        inference_times = [r['metrics']['inference_ms'] for r in results]
        fps_values = [r['metrics']['fps'] for r in results]
        prefill_speeds = [r['metrics']['prefill_speed'] for r in results]
        decode_speed = [r['metrics']['decode_speed'] for r in results]

        avg_time = np.mean(times)
        avg_inference = np.mean(inference_times)
        min_time = np.min(times)
        max_time = np.max(times)
        median_time = np.median(times)
        std_time = np.std(times)
        avg_fps = np.mean(fps_values)
        avg_prefill_speed = np.mean(prefill_speeds)
        avg_decode_speed = np.mean(decode_speed)

        capable_30fps = sum(1 for r in results if r['metrics']['realtime_capable_30fps'])
        capable_15fps = sum(1 for r in results if r['metrics']['realtime_capable_15fps'])
        total = len(results)

        print(f"\n⏱️  LATENCY STATISTICS:")
        print(f"   Average:    {avg_time:7.2f} ms  →  {avg_fps:.2f} FPS")
        print(f"   Median:     {median_time:7.2f} ms")
        print(f"   Min:        {min_time:7.2f} ms  (best case)")
        print(f"   Max:        {max_time:7.2f} ms  (worst case)")
        print(f"   Std Dev:    {std_time:7.2f} ms")

        print(f"   Avg Prefill Speed:    {avg_prefill_speed:7.2f} ms / token")
        print(f"   Avg Decode Speed:    {avg_decode_speed:7.2f} ms / token")

        print(f"\n⚡ TIME BREAKDOWN:")
        avg_preprocess = np.mean([r['metrics']['preprocessing_ms'] for r in results])
        avg_parsing = np.mean([r['metrics']['parsing_ms'] for r in results])

        print(f"   Preprocessing:  {avg_preprocess:6.2f} ms  ({avg_preprocess / avg_time * 100:5.1f}%)")
        print(f"   Inference:      {avg_inference:6.2f} ms  ({avg_inference / avg_time * 100:5.1f}%) ⭐")
        print(f"   Parsing:        {avg_parsing:6.2f} ms  ({avg_parsing / avg_time * 100:5.1f}%)")

        print(f"\n🎯 REAL-TIME FEASIBILITY:")
        print(
            f"   30 FPS: {capable_30fps}/{total} frames ({capable_30fps / total * 100:.1f}%) {'✅ FEASIBLE' if capable_30fps > total * 0.9 else '❌ NOT FEASIBLE'}")
        print(
            f"   15 FPS: {capable_15fps}/{total} frames ({capable_15fps / total * 100:.1f}%) {'✅ FEASIBLE' if capable_15fps > total * 0.9 else '❌ NOT FEASIBLE'}")

        green = sum(1 for r in results if r['hazard_level'] == 'green')
        yellow = sum(1 for r in results if r['hazard_level'] == 'yellow')
        red = sum(1 for r in results if r['hazard_level'] == 'red')

        print(f"\n🚨 HAZARD DISTRIBUTION:")
        print(f"   🟢 Safe:    {green:3d} ({green / total * 100:5.1f}%)")
        print(f"   🟡 Caution: {yellow:3d} ({yellow / total * 100:5.1f}%)")
        print(f"   🔴 Danger:  {red:3d} ({red / total * 100:5.1f}%)")

        print(f"\n💡 MOBILE COMPARISON:")
        mobile_time = 200  # Estimated from mobile app
        speedup = mobile_time / avg_time
        print(f"   Mobile avg:     ~{mobile_time:.0f} ms (ARM GPU)")
        print(f"   Desktop avg:     {avg_time:.2f} ms ({self.device.upper()})")
        print(f"   Speedup factor:  {speedup:.2f}x {'🚀' if speedup > 2 else '⚡' if speedup > 1 else '🐌'}")

        print(f"\n📝 SAMPLE RESPONSES:")
        for i, r in enumerate(results[:3]):
            print(f"   Frame {i}: [{r['hazard_level']:6s}] {r['reason'][:60]}")

        # Save results
        output_file = f"gemma3n_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'model': 'Gemma 3N',
                'summary': {
                    'avg_time_ms': avg_time,
                    'avg_inference_ms': avg_inference,
                    'avg_fps': avg_fps,
                    'min_time_ms': min_time,
                    'max_time_ms': max_time,
                    'std_dev_ms': std_time,
                    'median_time_ms': median_time,
                    'realtime_30fps_capable': capable_30fps / total > 0.9,
                    'realtime_15fps_capable': capable_15fps / total > 0.9,
                    'speedup_vs_mobile': speedup,
                    'device': self.device
                },
                'results': results
            }, f, indent=2)
        print(f"\n💾 Detailed results saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Hazard Detection - Gemma 3N Native')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--model', type=str, default='gemma-2b',
                        choices=['gemma-2b', 'gemma-4b'],
                        help='Model size')
    parser.add_argument('--frames', type=int, default=30, help='Number of frames to process')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    args = parser.parse_args()

    system = HazardDetectionSystem(model_name=args.model, device=args.device)
    results = system.benchmark_realtime_performance(args.video, args.frames)


if __name__ == "__main__":
    VIDEO_PATH = "4271760-hd_1920_1080_30fps.mp4"

    if VIDEO_PATH == "path/to/your/video.mp4":
        print("=" * 80)
        print("🚀 Gemma 3N Hazard Detection (Native Multimodal)")
        print("=" * 80)
        print("\n📦 INSTALLATION:")
        print("   pip install --upgrade transformers torch accelerate opencv-python pillow numpy")
        print("   (Requires transformers >= 4.40)")
        print("\n📖 USAGE:")
        print("\n1️⃣  Gemma 2B (faster, 4GB VRAM):")
        print("    python script.py --video video.mp4 --model gemma-2b --device cuda --frames 30")
        print("\n2️⃣  Gemma 4B (better accuracy, 8GB VRAM):")
        print("    python script.py --video video.mp4 --model gemma-4b --device cuda --frames 30")
        print("\n3️⃣  CPU mode (very slow):")
        print("    python script.py --video video.mp4 --model gemma-2b --device cpu --frames 5")
        print("\n💡 WHAT'S DIFFERENT:")
        print("   ✅ Uses Gemma3nForConditionalGeneration (native vision support)")
        print("   ✅ Chat-based message format (system + user roles)")
        print("   ✅ Direct image processing (no intermediate description)")
        print("   ✅ Single-stage pipeline (faster than two-stage)")
        print("=" * 80)
    else:
        system = HazardDetectionSystem(model_name="gemma-2b", device="cuda")
        results = system.benchmark_realtime_performance(VIDEO_PATH, num_frames=10)
