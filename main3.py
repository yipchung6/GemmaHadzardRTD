import cv2
import time
import json
from PIL import Image
import numpy as np
from datetime import datetime
import argparse
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import warnings

warnings.filterwarnings('ignore')


class HazardDetectionSystem:
    """
    Hazard Detection using PaliGemma 2
    Model generates safety reminders for hazards
    """

    AVAILABLE_MODELS = {
        'paligemma2-3b-224': 'google/paligemma2-3b-pt-224',
        'paligemma2-3b-448': 'google/paligemma2-3b-pt-448',
        'paligemma2-3b-896': 'google/paligemma2-3b-pt-896',
        'paligemma2-10b-224': 'google/paligemma2-10b-pt-224',
        'paligemma2-10b-448': 'google/paligemma2-10b-pt-448',
        'paligemma2-10b-896': 'google/paligemma2-10b-pt-896',
    }

    def __init__(self, model_name="paligemma2-3b-448", device="cuda"):
        """
        Initialize PaliGemma 2 with proper multimodal support

        Args:
            model_name: paligemma2 model variant
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

        print("\n📥 Loading PaliGemma 2 model...")
        print("   ⏳ First run: downloading model (~5-6GB for 3b, ~20GB for 10b)")
        print("   💡 Cached runs: instant loading\n")

        try:
            # Load model with proper class
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype=dtype,
            ).eval()

            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)

            print("✅ PaliGemma 2 loaded successfully!\n")

        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            print("\n💡 Troubleshooting:")
            print("   1. Install latest transformers: pip install --upgrade transformers")
            print("   2. Check transformers version: pip show transformers (need >= 4.40)")
            print("   3. Try: pip install git+https://github.com/huggingface/transformers.git")
            raise

        # Simple, clean prompt for PaliGemma 2
        self.HAZARD_PROMPT = """<image>Analyze this image for safety hazards. Respond with JSON only: {"hazard_level": "green/yellow/red", "reminder": "safety tip or empty string"}. If green, reminder must be empty. If yellow or red, provide brief safety tip."""

    def process_frame(self, image):
        """Process a single frame and detect hazards"""
        timing = {}
        overall_start = time.time()

        # Preprocessing
        preprocessing_start = time.time()
        if not isinstance(image, Image.Image):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # PaliGemma 2 image size depends on variant (224, 448, or 896)
        if '224' in self.model_name:
            max_size = 224
        elif '448' in self.model_name:
            max_size = 448
        elif '896' in self.model_name:
            max_size = 896
        else:
            max_size = 448  # Default

        # Resize maintaining aspect ratio
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        timing['preprocessing_ms'] = (time.time() - preprocessing_start) * 1000

        # Tokenize prompt for counting
        prompt_tokens = self.processor.tokenizer(self.HAZARD_PROMPT, return_tensors="pt")["input_ids"]
        input_tokens = prompt_tokens.size(-1)

        # Process and generate
        inference_start = time.time()
        try:
            # PaliGemma 2: text must contain <image> token, then pass the actual image
            inputs = self.processor(
                text=self.HAZARD_PROMPT,
                images=image,
                return_tensors="pt"
            ).to(self.device)

            # Generate response - using deterministic for better JSON compliance
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=80,  # Shorter for faster generation
                    do_sample=False,  # Deterministic for consistent JSON
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

        # Prefill/decode speed (approximation)
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
            # Clean the text first
            text = text.strip()

            # Remove markdown code blocks
            text = text.replace("```json", "").replace("```", "").strip()

            # Find JSON object - look for first { and last }
            start = text.find('{')
            end = text.rfind('}') + 1

            if start != -1 and end > start:
                json_str = text[start:end]

                # Handle multiple JSON objects (take only the first)
                if json_str.count('{') > 1:
                    # Find first complete JSON object
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

                # Handle variations
                if '/' in hazard_level:  # e.g., "green/yellow/red"
                    hazard_level = 'green'
                elif hazard_level not in ['green', 'yellow', 'red']:
                    hazard_level = 'green'

                # Get reminder (model-generated)
                reminder = result.get('reminder', '').strip()

                # Enforce empty reminder for green
                if hazard_level == 'green':
                    reminder = ''

                # Truncate overly long reminders
                if len(reminder) > 100:
                    reminder = reminder[:97] + "..."

                return {
                    'hazard_level': hazard_level,
                    'reminder': reminder
                }
            else:
                return self._fallback_parse(text)

        except json.JSONDecodeError as e:
            # More detailed error for debugging
            # print(f"⚠️  JSON parse error: {e}")
            # print(f"    Raw text: {text[:100]}")
            return self._fallback_parse(text)

    def _fallback_parse(self, text):
        """Fallback text parsing when JSON parsing fails"""
        text_lower = text.lower()

        # Look for hazard keywords
        if any(word in text_lower for word in
               ['red', 'danger', 'threat', 'unsafe', 'critical', 'emergency', 'fire', 'weapon']):
            level = 'red'
            reminder = self._extract_reminder_from_text(text, 'red')
        elif any(word in text_lower for word in ['yellow', 'caution', 'warning', 'careful', 'potential', 'risk']):
            level = 'yellow'
            reminder = self._extract_reminder_from_text(text, 'yellow')
        else:
            level = 'green'
            reminder = ''

        return {
            'hazard_level': level,
            'reminder': reminder
        }

    def _extract_reminder_from_text(self, text, level):
        """Extract meaningful reminder from non-JSON text"""
        # Common fallback reminders based on level
        if level == 'red':
            reminders = [
                "Evacuate area immediately",
                "Alert authorities and keep safe distance",
                "Danger detected, take immediate action"
            ]
        else:  # yellow
            reminders = [
                "Exercise caution in this area",
                "Be aware of surroundings",
                "Proceed with care"
            ]

        # Try to extract from text if it contains useful info
        if len(text) > 10 and len(text) < 150:
            # Clean up the text
            reminder = text.strip()
            # Remove common instruction phrases
            for phrase in ['Respond with JSON', 'hazard_level', 'reminder', '{', '}', '"']:
                reminder = reminder.replace(phrase, '')
            reminder = reminder.strip(':,. ')

            if len(reminder) > 5 and len(reminder) < 100:
                return reminder

        # Return default reminder
        return reminders[0]

    def benchmark_realtime_performance(self, video_path, num_frames=30):
        """Benchmark with consecutive frames"""
        print(f"🎯 REAL-TIME PERFORMANCE BENCHMARK")
        print(f"   Model: PaliGemma 2 (AI-Generated Safety Reminders)")
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
        prefill_speeds = [r['metrics']['prefill_speed'] for r in results]
        decode_speeds = [r['metrics']['decode_speed'] for r in results]
        input_tokens = [r['metrics']['input_tokens'] for r in results]
        output_tokens = [r['metrics']['output_tokens'] for r in results]
        fps_values = [r['metrics']['fps'] for r in results]

        avg_time = np.mean(times)
        avg_inference = np.mean(inference_times)
        avg_prefill = np.mean(prefill_speeds)
        avg_decode = np.mean(decode_speeds)
        avg_input_tokens = np.mean(input_tokens)
        avg_output_tokens = np.mean(output_tokens)
        min_time = np.min(times)
        max_time = np.max(times)
        median_time = np.median(times)
        std_time = np.std(times)
        avg_fps = np.mean(fps_values)

        capable_30fps = sum(1 for r in results if r['metrics']['realtime_capable_30fps'])
        capable_15fps = sum(1 for r in results if r['metrics']['realtime_capable_15fps'])
        total = len(results)

        print(f"\n⏱️  LATENCY STATISTICS:")
        print(f"   Average:    {avg_time:7.2f} ms  →  {avg_fps:.2f} FPS")
        print(f"   Median:     {median_time:7.2f} ms")
        print(f"   Min:        {min_time:7.2f} ms  (best case)")
        print(f"   Max:        {max_time:7.2f} ms  (worst case)")
        print(f"   Std Dev:    {std_time:7.2f} ms")

        print(f"\n⚡ TIME BREAKDOWN:")
        avg_preprocess = np.mean([r['metrics']['preprocessing_ms'] for r in results])
        avg_parsing = np.mean([r['metrics']['parsing_ms'] for r in results])

        print(f"   Preprocessing:  {avg_preprocess:6.2f} ms  ({avg_preprocess / avg_time * 100:5.1f}%)")
        print(f"   Inference:      {avg_inference:6.2f} ms  ({avg_inference / avg_time * 100:5.1f}%) ⭐")
        print(f"   Parsing:        {avg_parsing:6.2f} ms  ({avg_parsing / avg_time * 100:5.1f}%)")

        print(f"\n🎯 TOKEN STATISTICS (Mobile App Format):")
        print(f"   Response generated in {avg_inference:.2f} ms")
        print(f"   Prefill speed: {avg_prefill:.2f} tokens/s")
        print(f"   Decode speed: {avg_decode:.2f} tokens/s")
        print(f"   Avg input tokens: {avg_input_tokens:.1f}")
        print(f"   Avg output tokens: {avg_output_tokens:.1f}")

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
        output_file = f"paligemma2_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'model': 'PaliGemma 2',
                'model_name': self.model_name,
                'summary': {
                    'avg_time_ms': avg_time,
                    'avg_inference_ms': avg_inference,
                    'avg_prefill_speed': avg_prefill,
                    'avg_decode_speed': avg_decode,
                    'avg_input_tokens': avg_input_tokens,
                    'avg_output_tokens': avg_output_tokens,
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
    parser = argparse.ArgumentParser(description='Hazard Detection - PaliGemma 2 with AI-Generated Safety Reminders')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--model', type=str, default='paligemma2-3b-448',
                        choices=['paligemma2-3b-224', 'paligemma2-3b-448', 'paligemma2-3b-896',
                                 'paligemma2-10b-224', 'paligemma2-10b-448', 'paligemma2-10b-896'],
                        help='Model size and resolution')
    parser.add_argument('--frames', type=int, default=30, help='Number of frames to process')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    args = parser.parse_args()

    system = HazardDetectionSystem(model_name=args.model, device=args.device)
    results = system.benchmark_realtime_performance(args.video, args.frames)


if __name__ == "__main__":
    VIDEO_PATH = "gettyimages-1455769365-640_adpp.mp4"

    if VIDEO_PATH == "path/to/your/video.mp4":
        print("=" * 80)
        print("🚀 PaliGemma 2 Hazard Detection with AI-Generated Safety Reminders")
        print("=" * 80)
        print("\n📦 INSTALLATION:")
        print("   pip install --upgrade transformers torch accelerate opencv-python pillow numpy")
        print("\n📖 USAGE:")
        print("   python main3.py --video video.mp4 --model paligemma2-3b-448 --device cuda --frames 30")
        print("\n💡 SIMPLIFIED PROMPT:")
        print("   Now uses minimal prompt for better JSON compliance")
        print("   🟢 Green:  No reminder (safe)")
        print("   🟡 Yellow: AI-generated caution reminder")
        print("   🔴 Red:    AI-generated urgent safety instruction")
        print("=" * 80)
    else:
        system = HazardDetectionSystem(model_name="paligemma2-3b-448", device="cuda")
        results = system.benchmark_realtime_performance(VIDEO_PATH, num_frames=10)
