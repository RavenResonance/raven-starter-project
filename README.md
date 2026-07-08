# Raven Starter Project

<img src="assets/documentation/logo/horizontal.png" width="100%" alt="Raven Framework Logo">

A starter repository for building your first Raven Framework application. This project includes example applications to help you get started quickly with pre-built code demonstrating essential components, patterns, and features for building gaze-based applications for [Raven Prism](https://raven.computer).

For detailed documentation, component reference, and API details, see the [Raven Framework repository](https://github.com/RavenResonance/raven-framework).

We are Raven Resonance, a team of engineers and designers who have built and used wearable computers for years. [Raven Prism 1](https://raven.computer) will be out soon and runs RavenOS, a Linux-based operating system designed for all-day wear. This repo contains starter examples for the Raven SDK preview. We would love to hear your feedback in our [Discord community](https://raven.computer/s/discord)!

## Start your first project

**Step 1:** Make sure you have the following installed:
- **Python 3.10 or higher** - Download and install from the [official Python website](https://www.python.org/downloads/) if needed
- **Git** - Required to get the Raven Framework (this makes it easy to update later)

To check your Python version, run `python --version` or `python3 --version`. To upgrade Python, download the latest version from the [official Python website](https://www.python.org/downloads/) and install it.

**Step 2:** Open a terminal: macOS (Press `Cmd + Space`, type "Terminal", and press Enter), Windows (Press `Win + R`, type `cmd`, and press Enter), or Linux (Press `Ctrl + Alt + T`).

**Step 3:** Clone this starter project:

```bash
git clone https://github.com/RavenResonance/raven-starter-project.git
cd raven-starter-project
```

**Step 4:** Create a virtual environment:

```bash
python -m venv raven-app-venv
```

**Note:** Use `python3` instead of `python` if needed on your system.

**Step 5:** Activate the virtual environment:
- **macOS/Linux:** `source raven-app-venv/bin/activate`
- **Windows:** `raven-app-venv\Scripts\activate`

**Step 6:** Clone the Raven Framework and install it:

```bash
git clone https://github.com/RavenResonance/raven-framework.git 
cd raven-framework
pip install -e .
cd ..
```

**Note:** You might need to use `pip3 install -e .` instead of `pip install -e .` depending on your system.

**Recommended:** Install audio simulator support for examples that use the speaker (like Simple AI App):

```bash
cd raven-framework
pip install -e ".[audio-simulator]"
cd ..
```

This installs `simpleaudio` for audio playback in simulator mode. Note that `simpleaudio` may not have pre-built binaries available for all Linux and Windows systems, which can cause installation to fail. If installation fails, don't worry — the framework will work without it, but audio playback won't be available in simulator mode (it will still work on Raven devices).

**Step 7:** Try the Hello World example:

```bash
python main.py
```

Or `python3 main.py` depending on your system.

Now open the project in your editor of your choice like [VS Code](https://code.visualstudio.com/) or [Cursor](https://cursor.sh/).

**Step 8:** Know what you are seeing:

- **Your cursor simulates eye gaze**: When you move your mouse cursor, it represents where you're looking with your eyes on Raven Prism. The framework tracks this gaze position to interact with UI elements (HID independent — mouse, touchpad, eye gaze, etc.).
- **Click simulates interaction methods**: Clicking with your mouse simulates the actual interaction methods used on Raven Prism, such as double blink or dwell-to-click (gazing at an element for a set duration).
- **Black appears transparent**: Note that black colors appear transparent due to additive blending, which is how the waveguide display works. The display adds light to what you see in the real world, so pure black cannot be displayed. [Learn more about additive blending and color](https://github.com/RavenResonance/raven-framework/tree/main?tab=readme-ov-file#color--waveguide).
- **Simulator modes**: Click different simulator modes, like Day or Night, to see how the display looks on Raven Prism. Click Raw to see how the raw image is rendered.
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
from raven_framework import RavenApp, RunApp, Spacer, TextBox, VerticalContainer

# Define your app class, inheriting from RavenApp
class HelloWorld(RavenApp):
    # Initialize the app
    def __init__(self, parent=None) -> None:
        # Call the parent class constructor
        super().__init__(parent)
        # Create a vertical container with a width of 640 pixels
        vbox = VerticalContainer(width=640)
        # Create a title text box with "Hello World"
        title_text = TextBox(
            "Hello World", width=640, alignment="center", font_type="display"
        )
        # Create a text box with simulator information
        text_box = TextBox(
            """Welcome to your first Raven app! This simulates how your app will look on Raven Prism. Your cursor simulates eye gaze, and your app's colors additively blend with the real world. Try clicking different modes to see how an app should look on the display.""",
            width=640,
            alignment="left",
        )
        # Add the text boxes to the vertical container
        # Spacer adds 20 pixels of vertical space between the title and the text box
        vbox.add(title_text, Spacer(height=20), text_box)
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

**Step 12:** Building apps with AI: This repository includes an `AGENTS.md` file that contains condensed documentation optimized for AI assistants. Coding tools like Cursor and Copilot should automatically see this as a README and use it for context. You can also copy this file and provide it to AI assistants (like ChatGPT, Claude, or Gemini) to help you build Raven applications. The AI will have all the framework information it needs to generate code and answer questions about components, sensors, and utilities. [Learn more about AGENTS.md](https://agents.md/).

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
- Adding text elements (`TextBox`) with system fonts (`font_type`)
- Using spacers for layout (`Spacer`)
- Simple layout and alignment

### Counter (Stopwatch)
A stopwatch application demonstrating state management, button interactions, and routines.

<img src="assets/documentation/example-screenshots/counter.png" width="500" alt="Counter Example">

**Location:** `examples/counter/`

**Run:**
```bash
cd examples/counter
python main.py
```

**What it demonstrates:**
- State management using enums (`AppState`)
- Creating and managing routines (`Routine`) for periodic updates
- Using cards (`TextCardWithButton`, `TextCardWithTwoButtons`)
- Button click handlers and state transitions
- UI updates based on application state
- Animation utilities (`fade_in`)

### Art Studio
A painting reference viewer app demonstrating navigation, scrollable lists, media viewing, and interactive UI elements.

<img src="assets/documentation/example-screenshots/art_studio.png" width="500" alt="Art Studio Example">

**Location:** `examples/art_studio/`

**Run:**
```bash
cd examples/art_studio
python main.py
```

**Note:** This example includes image assets in the `paintings/` directory.

**What it demonstrates:**
- Navigation between different app states
- Using data classes to structure data
- Scrollable lists (`ScrollableListCard`)
- Media viewing (`MediaViewer`) for images
- State management with enums
- Button click handlers with parameters
- Absolute positioning of widgets

### Simple AI App
An AI-powered app that integrates camera, microphone, and speaker with OpenAI for multimodal interactions.

<img src="assets/documentation/example-screenshots/simple_ai_app.png" width="500" alt="Simple AI App Example">

**Location:** `examples/simple_ai_app/`

**Run:**
```bash
cd examples/simple_ai_app
python main.py
```

**Note:** This example requires an OpenAI API key. See the code for configuration. Audio playback in simulator mode requires the `audio-simulator` optional dependency (see Step 6).

**What it demonstrates:**
- Sensor integration (Microphone, Camera, Speaker)
- Asynchronous processing with `AsyncRunner` (non-blocking UI)
- OpenAI integration (`OpenAiHelper`)
- Audio transcription (Whisper)
- Multimodal AI processing (text + image)
- Text-to-speech generation
- Button state management
- Lazy initialization of components

## Running on Raven Prism

To deploy and run apps on Raven Prism, use:

```bash
python main.py deploy
```

Or `python3 main.py deploy` depending on your system.

Make sure to provide your `app_id` and `app_key` in the `RunApp.run()` call:

```python
RunApp.run(
    lambda: YourApp(), 
    app_id="your_app_id", 
    app_key="your_app_key"
)
```

**Note:** Use `python3` instead of `python` if needed on your system.


## Resources

- [Raven Framework Repository & Documentation](https://github.com/RavenResonance/raven-framework)
- [Raven Prism](https://raven.computer)
- [Discord Community](https://raven.computer/s/discord)

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve these starter examples.

## License

This repository's example applications and documentation are licensed under the [MIT License](LICENSE).

MIT applies to the starter examples and docs in this repo only. [Raven Framework](https://github.com/RavenResonance/raven-framework) is proprietary software, installed separately, and governed by its [own license](https://github.com/RavenResonance/raven-framework/blob/main/LICENSE). Do not redistribute or publish Raven Framework code without authorization from Raven Resonance Inc.

See also [NOTICE](NOTICE) for a summary of what is covered by each license.
