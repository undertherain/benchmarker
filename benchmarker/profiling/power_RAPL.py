import logging

import pyRAPL


class power_monitor_RAPL:
    def __init__(self, params):
        self.params = params
        try:
            pyRAPL.setup()
            self.rapl_enabled = True
        except:
            self.rapl_enabled = False

    def start(self):
        if self.rapl_enabled:
            self.meter_rapl = pyRAPL.Measurement("bar")
            self.meter_rapl.begin()
        else:
            logger = logging.getLogger(__name__)
            logger.warning("RAPL requested but it is not enabled")

    def stop(self):
        if self.rapl_enabled:
            self.meter_rapl.end()
            joules_CPU = sum(self.meter_rapl.result.pkg) / 1000000.0
            joules_RAM = sum(self.meter_rapl.result.dram) / 1000000.0
            self.params["power"]["joules_CPU"] = joules_CPU
            self.params["power"]["joules_RAM"] = joules_RAM
