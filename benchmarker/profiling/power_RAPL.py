import pyRAPL
import logging


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
            self.meter_rapl = pyRAPL.Measurement('bar')
            self.meter_rapl.begin()
        else:
            logger = logging.getLogger(__name__)
            logger.warning("RAPL requested but it is not enabled")

    def stop(self):
        if self.rapl_enabled:
            self.meter_rapl.end()
            self.params["power"]["joules_CPU"] = sum(self.meter_rapl.result.pkg) / 1000000.0
            self.params["power"]["joules_RAM"] = sum(self.meter_rapl.result.dram) / 1000000.0
            self.params["power"]["avg_watt_CPU"] = self.params["power"]["joules_CPU"] / self.params["time_total"]
            self.params["power"]["avg_watt_RAM"] = self.params["power"]["joules_RAM"] / self.params["time_total"]
            self.params["power"]["joules_total"] += self.params["power"]["joules_CPU"]
            self.params["power"]["joules_total"] += self.params["power"]["joules_RAM"]
            self.params["power"]["avg_watt_total"] += self.params["power"]["avg_watt_CPU"]
            self.params["power"]["avg_watt_total"] += self.params["power"]["avg_watt_RAM"]