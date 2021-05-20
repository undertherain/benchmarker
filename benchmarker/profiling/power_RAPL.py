import logging

import pyRAPL


class PowerMonitorRAPL:
    def __init__(self, params):
        self.params = params
        try:
            pyRAPL.setup()
            self.rapl_enabled = True
            self.meter_rapl = pyRAPL.Measurement("bar")
            self.meter_rapl.begin()
        except:
            self.rapl_enabled = False

    def stop(self):
        if self.rapl_enabled:
            self.meter_rapl.end()
            joules_CPU = sum(self.meter_rapl.result.pkg) / 1000000.0
            joules_RAM = sum(self.meter_rapl.result.dram) / 1000000.0
            self.params["power"]["joules_CPU"] = joules_CPU
            self.params["power"]["joules_RAM"] = joules_RAM
