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
        keyvals = {
            "kernel": ["conv2", "conv3"],
            "device": ["p100", "v100"],
            "framework": ["Chainer", "TensorFlow", "Theano"]
            }
        output = mytemplate.render(keyvals=keyvals)
        #output += mytemplate.render(key="device", vals=["p100", "v100"])
        #output += mytemplate.render(key="framework", vals=["Chainer", "TensorFlow"])

        return output, []
