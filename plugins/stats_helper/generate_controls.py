import json
import os
import pandas
from nikola.plugin_categories import ShortcodePlugin
from mako.template import Template


plugin_path = os.path.dirname(os.path.realpath(__file__))


def read_file(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def read_df_from_dir(path):
    data = [read_file(os.path.join(path, f)) for f in os.listdir(path) if not f.startswith("arch") and os.path.isfile(os.path.join(path, f))]
    df = pandas.DataFrame(data)
    # df["device"] = df["device"].apply(get_cute_device_str)
    return df


def read_jsons():
    df = read_df_from_dir(os.path.join(plugin_path, "../../data"))
    print(df)


class Plugin(ShortcodePlugin):

    name = "generate_controls"

    def handler(self, filename=None, site=None, data=None, lang=None, post=None):
        """Create HTML for emoji."""
        # output = "Hi I'm plugin that will generate comtrols for charts"
        mytemplate = Template(filename=os.path.join(plugin_path, 'controls.tmpl'))
        keyvals = {
            "kernel": ["conv2", "conv3"],
            "device": ["p100", "v100"],
            "framework": ["Chainer", "TensorFlow", "MXNet"]
            }
        output = mytemplate.render(keyvals=keyvals)

        return output, []


def main():
    read_jsons()


if __name__ == "__main__":
    main()
