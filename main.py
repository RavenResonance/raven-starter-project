# Import the necessary components from the Raven Framework
from raven_framework import RavenApp, RunApp, Spacer, TextBox, VerticalContainer


# Define your app class, inheriting from RavenApp (required for all Raven apps)
class HelloWorld(RavenApp):
    # Initialize the app
    def __init__(self, parent=None) -> None:
        # Call the parent class constructor to set up the app
        super().__init__(parent)
        # Create a vertical container with a width of 640 pixels
        # VerticalContainer automatically stacks widgets vertically
        vbox = VerticalContainer(width=640)
        # Create a title text box with "Hello World"
        title_text = TextBox(
            "Hello World", width=640, alignment="center", font_type="display"
        )
        # Create a text box with simulator information
        text_box = TextBox(
            """Welcome to your first Raven app! This simulates how your app will look on Raven Glass. Your cursor simulates eye gaze, and your app's colors additively blend with the real world. Try clicking different modes to see how an app should look on the display.""",
            width=640,
            alignment="left",
        )
        # Add the text boxes to the vertical container
        # Spacer adds 20 pixels of vertical space between the title and the text box for better visual separation
        vbox.add(title_text, Spacer(height=20), text_box)
        # Add the container to the main app window (self.app is a 640x640 container)
        self.app.add(vbox)


# Entry point - run the app when this script is executed
if __name__ == "__main__":
    # Launch the app using RunApp.run()
    # lambda: HelloWorld() creates a new instance of the app
    # app_id="" and app_key="" are empty for simulator mode
    RunApp.run(
        lambda: HelloWorld(),
        app_id="",
        app_key="",
    )
