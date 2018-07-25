import json
import os
import pandas
from nikola.plugin_categories import ShortcodePlugin
from mako.template import Template


plugin_path = os.path.dirname(os.path.realpath(__file__))


def read_file(filename):
    with open(filename) as f:
        data = json.load(f)
    keys_to_drop = ["platform", "batch_size_per_device"]
    for key in keys_to_drop:
        data.pop(key, None)
    meta = [["problem", "name"], ["device"]]
    df = pandas.io.json.json_normalize(data, meta=meta)
    return df


def get_entries(path):
    for dirName, subdirList, fileList in os.walk(path, topdown=False):
        for fname in fileList:
            yield(os.path.join(dirName, fname))


def read_df_from_dir(path):
    data = [read_file(os.path.join(path, f)) for f in get_entries(path) if not f.startswith("arch") and os.path.isfile(os.path.join(path, f))]
    df = pandas.concat(data)
    return df


class Plugin(ShortcodePlugin):

    name = "generate_controls"

    def handler(self, filename=None, site=None, data=None, lang=None, post=None):
        """Create HTML for emoji."""
        # output = "Hi I'm plugin that will generate comtrols for charts"
        mytemplate = Template(filename=os.path.join(plugin_path, 'controls.tmpl'))
        df = read_df_from_dir(os.path.join(plugin_path, "../../data"))
        keyvals = {
            "kernel": list(df["problem.name"].unique()),
            "device": list(df["device"].unique()),
            "framework": list(df["framework"].unique())
            }
        output = mytemplate.render(keyvals=keyvals)
        mytemplate = Template(filename=os.path.join(plugin_path, 'data.tmpl'))
        rows = [[i[1]['problem.name'], i[1]['device'], i[1]['framework'], i[1]['time']] for i in df.iterrows()]
        output += mytemplate.render(rows=rows)

        return output, []


def main():
    read_jsons()


if __name__ == "__main__":
    main()
