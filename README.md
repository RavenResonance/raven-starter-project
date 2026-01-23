# Raven Resonance



MIT RealityHack 2026 Project

## Overview
Real-time social cue detection and feedback system for Raven AR glasses. Uses camera and microphone feeds to analyze:
- Facial expressions
- Tone of voice
- Body language
- Conversation flow

Provides visual feedback through the waveguide display to help users navigate social interactions.

## Tech Stack
- **Hardware:** Raven AR Smart Glasses
- **Language:** Python 3.10+
- **AI:** OpenAI Multimodal API (GPT-4o)
- **Framework:** Raven Framework

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

2. Install Raven Framework:
```bash
git clone https://github.com/RavenResonance/raven-python-framework.git
cd raven-python-framework
pip install -e .
cd ..
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

5. Run the app:
```bash
python main.py
```

## Development
- `main.py` - Main application entry point
- Use simulator for testing (mouse = gaze)
- Deploy to glasses: `python main.py deploy`
