from nikola.plugin_categories import ShortcodePlugin

class Plugin(ShortcodePlugin):

    name = "generate_controls"

    def handler(self, filename=None, site=None, data=None, lang=None, post=None):
        """Create HTML for emoji."""
        output = "Hi I'm plugin that will generate comtrols for charts"

        return output, []
