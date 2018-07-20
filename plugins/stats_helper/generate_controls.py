import os
from nikola.plugin_categories import ShortcodePlugin
from mako.template import Template


plugin_path = os.path.dirname(os.path.realpath(__file__))


class Plugin(ShortcodePlugin):

    name = "generate_controls"

    def handler(self, filename=None, site=None, data=None, lang=None, post=None):
        """Create HTML for emoji."""
        # output = "Hi I'm plugin that will generate comtrols for charts"
        mytemplate = Template(filename=os.path.join(plugin_path, 'controls.tmpl'))
        output = mytemplate.render()
        return output, []
