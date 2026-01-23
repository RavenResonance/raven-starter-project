

"""
Raven Resonance - Social Cue Assistant
Real-time social interaction analysis using OpenFace + Whisper + Llama
"""

import os
import tempfile
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from openface_docker import run_openface_docker, parse_openface_csv

try:
    from raven import RavenApp, VerticalContainer, TextBox, Button
except ImportError:
    print("Raven Framework not installed. Please follow setup instructions in README.md")
    exit(1)

load_dotenv()


class SocialCueAssistant(RavenApp):
    def __init__(self):
        super().__init__()

        self.analyzing = False
        self.current_feedback = "Ready to assist..."
        self.conversation_context = []

        self.init_models()
        self.setup_ui()
        self.start_analysis_loop()

    def init_models(self):
        """Initialize AI models from Huggingface"""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Whisper for speech-to-text
        self.whisper_model = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=device
        )

        # Llama model for main processing
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.llama_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )

    def setup_ui(self):
        """Initialize the UI layout"""
        self.container = VerticalContainer()

        self.title = TextBox("Social Cue Assistant", font_size=24)
        self.feedback_box = TextBox(self.current_feedback, font_size=18)
        self.status_box = TextBox("Status: Idle", font_size=14)
        self.toggle_button = Button("Start Analysis", on_click=self.toggle_analysis)

        self.container.add_child(self.title)
        self.container.add_child(self.feedback_box)
        self.container.add_child(self.status_box)
        self.container.add_child(self.toggle_button)

        self.set_root(self.container)

    def toggle_analysis(self):
        """Toggle real-time analysis on/off"""
        self.analyzing = not self.analyzing

        if self.analyzing:
            self.toggle_button.set_text("Stop Analysis")
            self.status_box.set_text("Status: Analyzing...")
        else:
            self.toggle_button.set_text("Start Analysis")
            self.status_box.set_text("Status: Idle")

    def start_analysis_loop(self):
        """Start the real-time analysis routine"""
        self.add_routine(self.analyze_social_cues, interval=3000)

    def analyze_social_cues(self):
        """Capture and analyze camera/mic feed for social cues"""
        if not self.analyzing:
            return

        try:
            # Stub inputs - other devs will implement camera/audio capture
            camera_frame = None
            audio_data = None

            facial_cues = self.process_video_openface(camera_frame)
            speech_text = self.process_audio_whisper(audio_data)

            feedback = self.process_with_llama(facial_cues, speech_text)
            self.update_feedback(feedback)

        except Exception as e:
            self.status_box.set_text(f"Error: {str(e)[:30]}")

    def process_video_openface(self, frame):
        """Use OpenFace to extract facial action units and emotions"""
        if frame is None:
            return {
                "action_units": {},
                "gaze": "unknown",
                "head_pose": "neutral",
                "engagement": "unknown"
            }

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            # Save frame to temp file
            # TODO: Video dev will implement frame saving

        try:
            result = run_openface_docker(tmp_path)

            if result["returncode"] == 0:
                csv_files = [f for f in os.listdir(result["output_dir"]) if f.endswith('.csv')]
                if csv_files:
                    csv_path = os.path.join(result["output_dir"], csv_files[0])
                    facial_data = parse_openface_csv(csv_path)
                    return facial_data

            return {
                "action_units": {},
                "gaze": "unknown",
                "head_pose": "neutral",
                "engagement": "unknown"
            }

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def process_audio_whisper(self, audio_data):
        """Use Whisper to convert speech to text"""
        if audio_data is None or len(audio_data) == 0:
            return ""

        result = self.whisper_model(audio_data)
        return result["text"]

    def process_with_llama(self, facial_cues, speech_text):
        """Send facial and speech data to Llama for social cue analysis"""
        prompt = f"""[INST] You are a social interaction assistant. Analyze these inputs and provide brief, helpful feedback:

Facial Analysis: {facial_cues}
Speech: "{speech_text}"

Provide 1-2 sentence feedback about emotional state, engagement level, and conversation suggestions. Keep response under 30 words. [/INST]"""

        inputs = self.llama_tokenizer(prompt, return_tensors="pt").to(self.llama_model.device)

        outputs = self.llama_model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )

        response = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("[/INST]")[-1].strip()

    def update_feedback(self, feedback):
        """Update the feedback display"""
        self.current_feedback = feedback
        self.feedback_box.set_text(feedback)


if __name__ == "__main__":
    app = SocialCueAssistant()
    app.run()
