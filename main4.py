import cv2
import time
import json
from PIL import Image
import numpy as np
from datetime import datetime
import argparse
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import warnings

warnings.filterwarnings('ignore')


class HazardDetectionSystem:
    """
    Hazard Detection using Gemma 3 (Multimodal: Image-Text-to-Text)
    Generates safety reminders instead of reasons
    """

    AVAILABLE_MODELS = {
        'gemma-3-4b': 'google/gemma-3-4b-it',
        'gemma-3-12b': 'google/gemma-3-12b-it',
        'gemma-3-27b': 'google/gemma-3-27b-it',
    }

    def __init__(self, model_name="gemma-3-4b", device="cuda"):
        """
        Initialize Gemma 3 with multimodal support

        Args:
            model_name: gemma-3-4b, gemma-3-12b, or gemma-3-27b
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
        self.model_name = model_name

        print("\n📥 Loading Gemma 3 model...")
        print("   ⏳ First run: downloading model")
        print("      - Gemma 3-4B:  ~8GB")
        print("      - Gemma 3-12B: ~24GB")
        print("      - Gemma 3-27B: ~54GB")
        print("   💡 Cached runs: instant loading\n")

        try:
            # Load model with proper class
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype=dtype,
            ).eval()

            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)

            print("✅ Gemma 3 loaded successfully!\n")

        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            print("\n💡 Troubleshooting:")
            print("   1. Install latest transformers: pip install --upgrade transformers")
            print("   2. Check transformers version: pip show transformers (need >= 4.45)")
            print("   3. Try: pip install git+https://github.com/huggingface/transformers.git")
            print("   4. Authenticate with HuggingFace: huggingface-cli login")
            raise

        # Updated prompts for safety reminders
        self.HAZARD_SYSTEM_PROMPT = """You are a safety hazard detection expert. Analyze images and provide actionable safety reminders."""

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

If green: leave reminder EMPTY ("")
If yellow or red: provide a brief, actionable SAFETY REMINDER (under 60 characters)

Respond ONLY with valid JSON (no markdown, no extra text):
{"hazard_level": "[green/yellow/red]", "reminder": "[safety tip or empty string]"}"""

    def process_frame(self, image):
        """Process a single frame and detect hazards"""
        timing = {}
        overall_start = time.time()

        # Preprocessing
        preprocessing_start = time.time()
        if not isinstance(image, Image.Image):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Gemma 3 works well with images up to 512x512
        max_size = 512
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        timing['preprocessing_ms'] = (time.time() - preprocessing_start) * 1000

        # Prepare chat messages with image
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
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Decode output (only new tokens)
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
        """Generate fallback response with reminder"""
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        brightness = np.mean(image)

        if brightness < 50:
            return json.dumps({
                "hazard_level": "yellow",
                "reminder": "Use caution in low visibility areas"
            })
        else:
            return json.dumps({
                "hazard_level": "green",
                "reminder": ""
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

                # Handle multiple JSON objects (take first)
                if json_str.count('{') > 1:
                    brace_count = 0
                    for i, char in enumerate(json_str):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_str = json_str[:i + 1]
                                break

                result = json.loads(json_str)

                # Validate hazard_level
                hazard_level = result.get('hazard_level', 'green').lower()
                if hazard_level not in ['green', 'yellow', 'red']:
                    hazard_level = 'green'

                # Get reminder (check both 'reminder' and 'reason' for compatibility)
                reminder = result.get('reminder', result.get('reason', '')).strip()

                # Enforce empty reminder for green
                if hazard_level == 'green':
                    reminder = ''

                # Truncate long reminders
                if len(reminder) > 100:
                    reminder = reminder[:97] + "..."

                return {
                    'hazard_level': hazard_level,
                    'reminder': reminder
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
        if any(word in text_lower for word in
               ['red', 'danger', 'threat', 'unsafe', 'critical', 'emergency', 'fire', 'weapon']):
            level = 'red'
            reminder = self._extract_reminder(text, 'red')
        elif any(word in text_lower for word in ['yellow', 'caution', 'warning', 'careful', 'potential', 'risk']):
            level = 'yellow'
            reminder = self._extract_reminder(text, 'yellow')
        else:
            level = 'green'
            reminder = ''

        return {
            'hazard_level': level,
            'reminder': reminder
        }

    def _extract_reminder(self, text, level):
        """Extract or generate safety reminder from text"""
        # Default reminders by level
        defaults = {
            'red': 'Evacuate area immediately',
            'yellow': 'Exercise caution in this area'
        }

        # Try to extract meaningful text
        if len(text) > 10 and len(text) < 150:
            reminder = text.strip()
            # Clean up
            for phrase in ['Respond', 'JSON', 'hazard_level', 'reminder', '{', '}', '"', 'markdown']:
                reminder = reminder.replace(phrase, '')
            reminder = reminder.strip(':,. ')

            if len(reminder) > 5 and len(reminder) < 100:
                return reminder

        return defaults.get(level, 'Be aware of surroundings')

    def benchmark_realtime_performance(self, video_path, num_frames=30):
        """Benchmark with consecutive frames"""
        print(f"🎯 REAL-TIME PERFORMANCE BENCHMARK")
        print(f"   Model: Gemma 3 (Multimodal with Safety Reminders)")
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
        print("=" * 150)
        print(
            f"{'Frame':<7} | {'Total(ms)':<10} | {'Inference(ms)':<14} | {'Prefill(t/s)':<13} | {'Decode(t/s)':<13} | {'In':<4} | {'Out':<4} | {'Level':<7} | {'AI Safety Reminder':<50}")
        print("=" * 150)

        while frame_count < num_frames:
            ret, frame = cap.read()

            if not ret:
                break

            result = self.process_frame(frame)
            result['frame_number'] = frame_count
            results.append(result)

            metrics = result['metrics']

            # Show reminder or "[safe]" indicator
            reminder_display = result['reminder'] if result['reminder'] else "[safe - no reminder needed]"

            print(
                f"{frame_count:<7} | {metrics['total_time_ms']:<10.2f} | {metrics['inference_ms']:<14.2f} | "
                f"{metrics['prefill_speed']:<13.2f} | {metrics['decode_speed']:<13.2f} | "
                f"{metrics['input_tokens']:<4} | {metrics['output_tokens']:<4} | "
                f"{result['hazard_level']:<7} | {reminder_display[:50]:<50}"
            )

            frame_count += 1

        cap.release()

        print("\n" + "=" * 150)
        print("📊 REAL-TIME PERFORMANCE ANALYSIS")
        print("=" * 150)
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
        decode_speeds = [r['metrics']['decode_speed'] for r in results]
        input_tokens = [r['metrics']['input_tokens'] for r in results]
        output_tokens = [r['metrics']['output_tokens'] for r in results]

        avg_time = np.mean(times)
        avg_inference = np.mean(inference_times)
        min_time = np.min(times)
        max_time = np.max(times)
        median_time = np.median(times)
        std_time = np.std(times)
        avg_fps = np.mean(fps_values)
        avg_prefill_speed = np.mean(prefill_speeds)
        avg_decode_speed = np.mean(decode_speeds)
        avg_input_tokens = np.mean(input_tokens)
        avg_output_tokens = np.mean(output_tokens)

        capable_30fps = sum(1 for r in results if r['metrics']['realtime_capable_30fps'])
        capable_15fps = sum(1 for r in results if r['metrics']['realtime_capable_15fps'])
        total = len(results)

        print(f"\n⏱️  LATENCY STATISTICS:")
        print(f"   Average:    {avg_time:7.2f} ms  →  {avg_fps:.2f} FPS")
        print(f"   Median:     {median_time:7.2f} ms")
        print(f"   Min:        {min_time:7.2f} ms  (best case)")
        print(f"   Max:        {max_time:7.2f} ms  (worst case)")
        print(f"   Std Dev:    {std_time:7.2f} ms")

        print(f"\n🎯 TOKEN STATISTICS (Mobile App Format):")
        print(f"   Response generated in {avg_inference:.2f} ms")
        print(f"   Prefill speed: {avg_prefill_speed:.2f} tokens/s")
        print(f"   Decode speed: {avg_decode_speed:.2f} tokens/s")
        print(f"   Avg input tokens: {avg_input_tokens:.1f}")
        print(f"   Avg output tokens: {avg_output_tokens:.1f}")

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

        print(f"\n📝 AI-GENERATED SAFETY REMINDERS (Sample):")
        sample_count = 0
        for i, r in enumerate(results):
            if r['reminder'] and sample_count < 5:
                print(f"   Frame {i}: [{r['hazard_level']:6s}] {r['reminder']}")
                sample_count += 1
            elif sample_count >= 5:
                break

        if sample_count == 0:
            print("   [All frames were safe - no reminders generated]")

        # Save results
        output_file = f"gemma3_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'model': 'Gemma 3 (Multimodal)',
                'model_name': self.model_name,
                'summary': {
                    'avg_time_ms': avg_time,
                    'avg_inference_ms': avg_inference,
                    'avg_fps': avg_fps,
                    'avg_prefill_speed': avg_prefill_speed,
                    'avg_decode_speed': avg_decode_speed,
                    'avg_input_tokens': avg_input_tokens,
                    'avg_output_tokens': avg_output_tokens,
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
    parser = argparse.ArgumentParser(description='Hazard Detection - Gemma 3 (Multimodal) with Safety Reminders')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--model', type=str, default='gemma-3-4b',
                        choices=['gemma-3-4b', 'gemma-3-12b', 'gemma-3-27b'],
                        help='Model size')
    parser.add_argument('--frames', type=int, default=30, help='Number of frames to process')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    args = parser.parse_args()

    system = HazardDetectionSystem(model_name=args.model, device=args.device)
    results = system.benchmark_realtime_performance(args.video, args.frames)


if __name__ == "__main__":
    VIDEO_PATH = "videos/4271760-hd_1920_1080_30fps.mp4"

    if VIDEO_PATH == "path/to/your/video.mp4":
        print("=" * 80)
        print("🚀 Gemma 3 Multimodal Hazard Detection with AI Safety Reminders")
        print("=" * 80)
        print("\n📦 INSTALLATION:")
        print("   pip install --upgrade transformers torch accelerate opencv-python pillow numpy")
        print("   (Requires transformers >= 4.45)")
        print("\n🔐 AUTHENTICATION:")
        print("   huggingface-cli login")
        print("\n📖 USAGE:")
        print("\n1️⃣  Gemma 3-4B (efficient, 8GB VRAM):")
        print("    python main4.py --video video.mp4 --model gemma-3-4b --device cuda --frames 30")
        print("\n2️⃣  Gemma 3-12B (balanced, 24GB VRAM):")
        print("    python main4.py --video video.mp4 --model gemma-3-12b --device cuda --frames 30")
        print("\n3️⃣  Gemma 3-27B (best quality, 54GB VRAM):")
        print("    python main4.py --video video.mp4 --model gemma-3-27b --device cuda --frames 30")
        print("\n💡 MODEL COMPARISON:")
        print("   Gemma 3N (main2): Compact native multimodal (2B/4B)")
        print("   Gemma 3 (main4):  Full-size multimodal (4B/12B/27B)")
        print("\n📋 OUTPUT FORMAT:")
        print("   🟢 Green:  No reminder (safe)")
        print("   🟡 Yellow: AI-generated caution reminder")
        print("   🔴 Red:    AI-generated urgent safety instruction")
        print("=" * 80)
    else:
        system = HazardDetectionSystem(model_name="gemma-3-4b", device="cuda")
        results = system.benchmark_realtime_performance(VIDEO_PATH, num_frames=10)
