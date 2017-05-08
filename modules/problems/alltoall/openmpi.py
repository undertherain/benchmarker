import importlib
import sys

sys.path.append("modules/problems/alltoall")
''' Temporarily using term "model" to represent some form of "operations" 
''' 
def get_model(params):
    mod = importlib.import_module("app")
    app = mod.get_app(params)

    return app
