# Raven Resonance



MIT RealityHack 2026 Project

## Overview
Real-time social cue detection and feedback system for Raven AR glasses. Uses camera and microphone feeds to analyze:
- Facial expressions
- Tone of voice
- Body language
- Conversation flow

Provides visual feedback through the waveguide display to help users navigate social interactions.
**Step 1:** Make sure you have the following installed:
- **Python 3.10 or higher** - Download and install from the [official Python website](https://www.python.org/downloads/) if needed
- **Git** - Required to get the Raven Framework (this makes it easy to update later)

## Tech Stack
- **Hardware:** Raven AR Smart Glasses
- **Language:** Python 3.10+
- **AI:** OpenAI Multimodal API (GPT-4o)
- **Framework:** Raven Framework

**Step 2:** Open a terminal: macOS (Press `Cmd + Space`, type "Terminal", and press Enter), Windows (Press `Win + R`, type `cmd`, and press Enter), or Linux (Press `Ctrl + Alt + T`).

**Step 3:** Clone this starter project:

```bash
git clone https://github.com/RavenResonance/raven-starter-project.git
cd raven-starter-project
```

**Step 4:** Create a virtual environment:

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

**Note:** You might need to use `pip3 install -e .` instead of `pip install -e .` depending on your system.

**Step 7:** Try the Hello World example:

```bash
python main.py
```

Or `python3 main.py` depending on your system.

Now open the project in your editor of your choice like [VS Code](https://code.visualstudio.com/) or [Cursor](https://cursor.sh/).

**Step 8:** Know what you are seeing:

- **Your cursor simulates eye gaze**: When you move your mouse cursor, it represents where you're looking with your eyes on the actual glasses. The framework tracks this gaze position to interact with UI elements.
- **Click simulates interaction methods**: Clicking with your mouse simulates the actual interaction methods used on glasses, such as double blink or dwell-to-click (gazing at an element for a set duration).
- **Black appears transparent**: Note that black colors appear transparent due to additive blending, which is how the waveguide display works. The display adds light to what you see in the real world, so pure black cannot be displayed. [Learn more about additive blending and color](https://github.com/RavenResonance/raven-framework/tree/main?tab=readme-ov-file#color--waveguide).
- **Show Simulator button**: Click on "Show Simulator" to see a preview of how your app will look on the actual waveguide display, including the additive blending effects.
- **Closing windows**: You can close (X out of) either the main app window or the simulator window at any time. You can also move them around by dragging.
- **Exiting the app**: You can also click the home icon in the top right corner of the app window to exit and return to the home screen.

**Step 9:** Know the code:

Key concepts:
- Define a class that inherits from `RavenApp` - this is required for all Raven applications
- Name your main entry point file as `main.py`
- Use `RunApp.run()` to run the class you defined
- You get a 640x640 container called `self.app` that you can add widgets to
- You can create containers (like `VerticalContainer`) and add them to `self.app`
- You can add elements like buttons and text boxes to containers

Here's the Hello World example with comments explaining each line:

```python
# Import the necessary components from the Raven Framework
from raven_framework import RavenApp, RunApp, TextBox, VerticalContainer

# Define your app class, inheriting from RavenApp
class HelloWorld(RavenApp):
    # Initialize the app
    def __init__(self, parent=None) -> None:
        # Call the parent class constructor
        super().__init__(parent)
        # Create a vertical container with a width of 640 pixels
        vbox = VerticalContainer(width=640)
        # Create a text box with "Hello, World!" text, width 640, centered alignment
        text_box = TextBox("Hello, World!", width=640, alignment="center")
        # Add the text box to the vertical container
        vbox.add(text_box)
        # Add the container to the main app window (self.app is the main 640x640 container where all your content should be added to)
        self.app.add(vbox)

# Entry point - run the app when this script is executed
if __name__ == "__main__":
    # Launch the app using RunApp.run() with empty app_id and app_key for simulator mode
    RunApp.run(lambda: HelloWorld(), app_id="", app_key="")
```

**Example: Adding a button**

```python
from raven_framework import RavenApp, RunApp, Button, VerticalContainer

class MyApp(RavenApp):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        vbox = VerticalContainer(width=640)
        # Create a button with text
        button = Button(center_text="Click Me")
        # Set what happens when the button is clicked
        button.on_clicked(self.on_button_click)
        vbox.add(button)
        self.app.add(vbox)
    
    # Define the click handler function
    def on_button_click(self):
        print("Button was clicked!")

if __name__ == "__main__":
    RunApp.run(lambda: MyApp(), app_id="", app_key="")
```

**Example: Creating a repeat routine**

```python
from raven_framework import RavenApp, RunApp, Routine, TextBox, VerticalContainer

class MyApp(RavenApp):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        vbox = VerticalContainer(width=640)
        # Create a text box to display the counter
        self.counter_text = TextBox("Counter: 0", width=640, alignment="center")
        vbox.add(self.counter_text)
        self.app.add(vbox)
        
        # Initialize counter
        self.counter = 0
        
        # Create a routine that repeats every 1000ms (1 second)
        self.routine = Routine(
            interval_ms=1000,  # Run every 1000 milliseconds
            invoke=self.update_counter,  # Call this function each time
            mode="repeat"  # Repeat mode (runs continuously)
        )
    
    # Function called by the routine
    def update_counter(self):
        self.counter += 1
        self.counter_text.set_text(f"Counter: {self.counter}")

if __name__ == "__main__":
    RunApp.run(lambda: MyApp(), app_id="", app_key="")
```

**Step 10:** Explore the [example applications](#example-applications) to see more complex implementations.

**Step 11:** You can skim through the [full Raven Framework documentation](https://github.com/RavenResonance/raven-framework) to understand all available components and features.

**Step 12:** Building apps with AI: The Raven Framework repository includes an `AGENTS.md` file that contains condensed documentation optimized for AI assistants. Coding tools like Cursor and Copilot should automatically see this as a README and use it for context. You can also copy this file and provide it to AI assistants (like ChatGPT, Claude, or Gemini) to help you build Raven applications. The AI will have all the framework information it needs to generate code and answer questions about components, sensors, and utilities. [Learn more about AGENTS.md](https://agents.md/).

**Step 13:** Start building your own Raven application in `main.py` in the root directory!

## Example Applications

This repository includes several example applications that demonstrate different aspects of the Raven Framework:

### Hello World
A minimal example showcasing basic components and layout.

<img src="assets/documentation/example-screenshots/hello_world.png" width="500" alt="Hello World Example">

**Location:** `examples/hello_world/` or root `main.py`

**Run:**
```bash
cd examples/hello_world
python main.py
```

**What it demonstrates:**
- Basic app structure with `RavenApp`
- Creating containers (`VerticalContainer`)
- Adding text elements (`TextBox`)
- Simple layout and alignment

### Counter (Stopwatch)
A stopwatch application demonstrating state management, button interactions, and routines.

<img src="assets/documentation/example-screenshots/counter.png" width="500" alt="Counter Example">

**Location:** `examples/counter/`

**Run:**
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
