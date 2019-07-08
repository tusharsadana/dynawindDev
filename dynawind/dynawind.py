import datetime
import os
import sys

import pandas as pd
from logzero import setup_logger

logger = setup_logger(name="__main__")


def processFile(filePath, site=None, location=None, fileformat=None):
    import configparser
    import importlib
    import os.path
    from math import isnan
    import pkg_resources

    resource_package = __name__

    if os.path.isfile(filePath):
        pass
    else:
        print(filePath + " :File does not exist")
        return
    if fileformat is None:
        if filePath[-4:] == "tdms":
            fileformat = "tdms"
    # %%
    if fileformat == "tdms":
        # Default processing
        Signals = readTDMS(filePath)
        if Signals:
            # Default : Single entry every 10 minutes
            site = [Signals[0].site]
            location = [Signals[0].location]
        else:
            print(filePath + " : File is empty")
            return
    else:
        Signals = None
        """
        # A custom function will be triggered based on site, location and type
        Output is a list of dict
        (by using list a single file can generate multiple records)
        # ! Site, location and type need to be specified
        """
        if location is None:
            customMethod = getattr(
                importlib.import_module(
                    "dynawind.config." + site.lower() + ".custom_" + site.lower()
                ),
                fileformat.upper() + "_" + site.lower(),
            )
        else:
            customMethod = getattr(
                importlib.import_module(
                    "dynawind.config." + site.lower() + ".custom_" + site.lower()
                ),
                fileformat.upper() + "_" + location.lower(),
            )
        dt, site, location, data = customMethod(filePath)

    # %% load site config file
    config = configparser.ConfigParser()
    resource_path = "/".join(
        ("config", site[0].lower(), site[0].lower() + "_postprocessing.ini")
    )
    ini_file = pkg_resources.resource_filename(resource_package, resource_path)
    config.read(ini_file)

    timescale = 600
    if "general" in config:
        timescale = int(config["general"].get("timescale", 600))
    if Signals:
        if timescale == 600:
            # Default behaviour every ten minutes
            data = [stats2dict(Signals)]
            dt = [Signals[0].timestamp + datetime.timedelta(minutes=10)]
        else:
            # Generate HF (more than every 10 minutes)
            dt, site, location, data = HF_stats2dict(Signals, timescale)
    # Step 2 : write to temp file
    for t, s, l, d in list(zip(dt, site, location, data)):
        clean_d = {k: d[k] for k in d if isinstance(d[k], str) or not isnan(d[k])}
        write2json(
            t,
            s,
            l,
            clean_d,
            root=config["json"]["jsonFolder"],
            fileext=config["json"]["tmpExtension"],
        )


def write2json(dt, site, location, data, root="", fileext=".json"):
    import json
    from dynawind.db import returnJSONfilePath

    jsonPath = returnJSONfilePath(dt, site, location, root=root, fileext=fileext)
    jsonFile = open(jsonPath, "r")
    record = json.load(jsonFile)
    jsonFile.close()
    for key in data.keys():
        record[0][key] = data[key]
    jsonFile = open(jsonPath, "w")
    json.dump(record, jsonFile, indent=2)
    jsonFile.close()
    return jsonPath


class Series(object):
    def __init__(self, paths):
        # Outdated, was based on first concepts
        self.sourcePaths = paths
        self.fileList = {}
        for path in paths:
            openh5_temp = pd.HDFStore(path)
            tempList = list(openh5_temp["data"])
            openh5_temp.close()
            for item in tempList:
                self.fileList[item] = path

    def plot(self, y=None, x=None, start=None, stop=None):
        if x is None:
            df = self.get_df(y, start=start, stop=stop)
            ax = df.plot()
            plt.xlabel("Time")
            plt.xlim(0, 600)
        else:
            df = self.get_df([x, y], start=start, stop=stop)
            ax = df.plot(kind="scatter", x=x, y=y)
            ax.set_axisbelow(True)
            plt.minorticks_on()
        plt.grid(b=True, which="major", linestyle="-")
        plt.grid(b=True, which="minor", linestyle="dotted")

    def plotCalendar(self, tuples):
        """Early version of the calendar plot.
         Only shows day to day availability."""

        df = self.get_df(tuples=tuples)
        NumberCount = df.groupby(df.index.date).count() / 1.44
        NumberCount.index = pd.to_datetime(NumberCount.index)
        NumberCount.plot(style="o:")
        plt.xlabel("Date")
        plt.grid(which="both")
        plt.ylabel("Availability (%)")

    def get_df(self, tuples="all", start=None, stop=None):
        # Maybe better to have some persistence here
        df_lst = []
        where = None
        if start is not None:
            where = "index>" + start
        if stop is not None:
            if where is not None:
                where = where + "&index<" + stop
            else:
                where = "index<" + stop
        if tuples == "all":
            tuples = list(self.fileList.keys())
        if type(tuples) is not list:
            tuples = [tuples]
        for xtuple in tuples:
            store = pd.HDFStore(self.fileList[xtuple])
            df = store.select("data", columns=[xtuple], where=where)
            # Still loads each columns individual, can be optimized
            store.close()
            df_lst.append(df)
        df = pd.concat(df_lst, axis=1)
        df = df.sortlevel(0, axis=1)
        return df

    def delete(self, indices, tuples="all", df=None):
        from numpy import nan

        if tuples == "all":
            tuples = list(self.fileList.keys())
        if df is None:
            df = self.get_df()
            # Loads all dataframes for all timestamps, can be optimized
        df.set_value(indices, tuples, nan)
        return df

    def export(self, path=None, tuples="all", start=None):
        # Exports a series object to CSV

        if tuples == "all":
            tuples = list(self.fileList.keys())
        df = self.get_df(tuples=tuples)
        if start is not None:
            df = df[start:]
        if path is None:
            path = "DW_" + df.columns.levels[0][0]
        TsStart = min(df.index).strftime("%Y%m%d")
        TsEnd = max(df.index).strftime("%Y%m%d")
        df.to_csv(path_or_buf=path + "_" + TsStart + "_" + TsEnd + ".csv")

    def __repr__(self):
        return "DYNAwind series object"


def getSite(location):
    from pylookup import pylookup

    try:
        json_data = open(
            os.path.join(os.getcwd(), "json-lookups", "campaign_info.json")
        ).read()
        site = pylookup.lookup_location(json_data, location)
    except NameError:
        site = "unknown"
    except FileNotFoundError:
        logger.warning("json-lookups not found")
        site = "unknown"

    return site


def get_locations(site):
    """Returns all locations from a measurement site: e.g. Nobelwind"""
    import json

    json_data = open(os.path.join("json-lookups", "campaign_info.json")).read()

    farm_lookup_data = json.loads(json_data)
    locations_json = farm_lookup_data[site]["turbine"]
    locations = []
    for loc in locations_json:
        locations.append(loc["name"])

    return locations


# Below are functions associated with the Signal class of DYNAwind
class Signal(object):

    # This object will contain a single (!) signal
    def __init__(self, location, group, Fs, data, name, unit, timestamp):
        self.source = location  # Legacy! Turbine or location,
        self.location = location  # Location
        self.site = getSite(self.location)
        self.name = name  # Sensor name
        self.group = group  # E.g. acceleration
        self.data = data  # Signal
        self.Fs = Fs  # Sampling frequency
        self.unit_string = unit  # Engineering unit
        self.timestamp = timestamp  # Start of measurements (UTC)
        self.temperature_compensation = None  # Temperature compensation
        self = processSignal(self)  # Process the sensor based on config file

    def time(self, absoluteTime=False):
        if absoluteTime:
            time_vector = [
                self.timestamp + datetime.timedelta(0, x / self.Fs)
                for x in range(0, len(self.data))
            ]
        else:
            time_vector = range(0, len(self.data))
            time_vector = [x / self.Fs for x in time_vector]
        return time_vector

    def calcPSD(self, window="hann"):
        from scipy import signal
        from numpy import eye

        if type(window) == type(eye(3)):
            nperseg = len(window)
        else:
            nperseg = self.Fs * 60
        f, Pxx_den = signal.welch(
            self.data, fs=self.Fs, nperseg=int(nperseg), window=window
        )
        self.f = f
        psd = Pxx_den
        self.PSD = psd
        return f, psd

    def plotPSD(self, window="hann", rpm=None, xlim=None, template=None):
        import matplotlib.pyplot as plt
        import numpy as np

        self.calcPSD(window=window)
        plt.semilogy(self.f, self.PSD, label=self.name)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (" + self.unit_string + "²/Hz)")
        if rpm is not None:
            for i in [1, 3, 6, 9, 12]:
                plt.axvline(
                    x=rpm * i / 60,
                    color="r",
                    linestyle="dotted",
                    linewidth=1,
                    label="_nolegend_",
                )
        if xlim is not None:
            plt.xlim(xlim)
            idx0 = (np.abs(self.f - xlim[0])).argmin()
            idxe = (np.abs(self.f - xlim[1])).argmin()
            ymax = np.max(self.PSD[idx0:idxe])
            ymin = np.min(self.PSD[idx0:idxe])
            plt.ylim(ymin * 0.9, ymax * 1.1)
        plt.grid(True, "both", "both", ls=":", lw=.5, c="k", alpha=.3)

    def rms(self, LFB=None, UFB=None):
        import numpy as np

        if LFB is None and UFB is None:
            rms = (sum(self.data ** 2) / len(self.data)) ** 0.5
            return rms
        else:
            f, PSD = self.calcPSD()
            if LFB is None:
                LFB = 0
            if UFB is None:
                UFB = f[-1]
            # Find the closest frequency index to the desired frequency bands
            ind_0 = (np.abs(f - LFB)).argmin()
            ind_e = (np.abs(f - UFB)).argmin()
            df = f[2] - f[1]
            rms = np.sqrt(np.sum(PSD[ind_0 : ind_e + 1]) * df)
            return rms

    def append(self, Signal):
        from numpy import append

        # Appends a new signal to an existing one
        self.data = append(self.data, Signal.data)

    def select(self, t0=0, te=None):
        # Selects a time period from the data
        if te is None:
            te = len(self.data) / self.Fs
        if t0 < 0:
            # Start from the back, e.g t0=-30 : take the last 30 seconds
            t0 = len(self.data) / self.Fs + t0

        self.data = self.data[int(t0 * self.Fs) : int(te * self.Fs)]
        self.timestamp = self.timestamp + datetime.timedelta(0, t0)

    def std(self):
        from math import sqrt

        std = sqrt(sum((self.data - self.mean()) ** 2) / len(self.data))
        return std

    def median(self):
        from statistics import median

        median = median(self.data)
        return median

    def mean(self):
        mean = sum(self.data) / len(self.data)
        return mean

    def plot(self, absoluteTime=False):
        """ Plots the signal, if absoluteTime the x-axis is wrt time in UTC"""
        import matplotlib.pyplot as plt

        plt.plot(self.time(absoluteTime=absoluteTime), self.data, label=self.name)
        plt.xlabel("Time (s)")
        plt.ylabel(self.group + " (" + self.unit_string + ")")
        plt.tight_layout()
        plt.legend
        plt.grid(True, "both", "both", ls=":", lw=.5, c="k", alpha=.3)
        if not absoluteTime:
            plt.xlim(0, max(self.time()))
        else:
            plt.xlim([self.timestamp, self.time(absoluteTime=True)[-1]])

    def filter(self, LFB=0, UFB=5, order=8):
        """Code verified by WW on 07/02/2017 : http://192.168.119.55/x/WoF0
         this function does not prevent to filter twice, this can be
         especially harmfull when the first filter was narrower
        than the second
        """
        from scipy import signal

        # Will use a butterworth filter to the data in
        if LFB == 0:
            frequencyband = UFB / self.Fs * 2
            b, a = signal.butter(order, frequencyband, "low")
        else:
            b, a = signal.butter(
                order, [LFB / self.Fs * 2, UFB / self.Fs * 2], "bandpass"
            )

        y_filtered = signal.filtfilt(b, a, self.data)
        self.data = y_filtered

    def downsample(self, fs):
        if not self.Fs % fs == 0:
            raise ValueError(
                "original sample frequency should be an integer multiple of the downsample"
            )
        step = int(self.Fs / fs)
        self.data = self.data[::step]
        self.Fs = fs

    def __repr__(self):
        # This can be later used to give an overview of the properties in DYNAwind
        descr_str = (
            "DYNAwind signal object\n"
            + "source:\t"
            + self.source
            + "\n"
            + "name:\t"
            + self.name
            + "\n"
        )
        return descr_str


def get_config(Signal=None, timestamp=None, source=None, name=None, group=None):
    import json
    import pytz
    import pkg_resources

    from datetime import datetime

    if Signal is not None:
        source = Signal.location
        group = Signal.group
        timestamp = Signal.timestamp
        name = Signal.name
        site = getSite(Signal.location)
    else:
        site = getSite(source)

    logger.debug("site is {}".format(site))

    if source != "unknown":
        json_file = os.path.join("config", site.lower(), "{}.json".format(source))
        if os.path.exists(json_file):
            logger.debug("config path exists")
            get_config_data = json.loads(open(json_file).read())
            for i in range(0, len(get_config_data)):
                recordTimestamp = datetime.strptime(
                    get_config_data[i]["time"], "%d/%m/%Y %H:%M"
                )
                recordTimestamp = pytz.utc.localize(recordTimestamp)
                if recordTimestamp > timestamp:
                    record = get_config_data[i - 1]
                    break
                record = get_config_data[i]
            keepkeys = ["time", name, group]
            if name + "/name" in record.keys():
                keepkeys.append(record[name + "/name"])
            recordkeys = []
            for keepkey in keepkeys:
                for key in record.keys():
                    if keepkey in key:
                        recordkeys.append(key)
            record = {recordkey: record[recordkey] for recordkey in recordkeys}
            return record
        else:
            logger.error("No config found")
            return None
    else:
        return None


def yawTransformation(Signals, stat_dict):
    from math import pi, sin, cos
    from numpy import asmatrix, sign

    def yawTransformList(Signals, stat_dict):
        # When more than two Signals are passed this script will use the secondaries property in the config files to pair the signals and perform a FA,SS calculation
        secondaries_list = []
        for sgnl in Signals:
            config = get_config(sgnl)
            secondaries_list.append(config[sgnl.name + "/secondaries"])
        for i in range(len(secondaries_list)):
            for j in range(i + 1, len(secondaries_list)):
                if secondaries_list[i] == secondaries_list[j]:
                    FASS = yawTransformation([Signals[i], Signals[j]], stat_dict)
                    Signals.append(FASS[2])
                    Signals.append(FASS[3])
        return Signals

    if len(Signals) > 2:
        Signals = yawTransformList(Signals, stat_dict)
        return Signals

    Signal1 = Signals[0]
    Signal2 = Signals[1]
    yaw_angle = stat_dict["yaw/mean"]
    # Code verified on 07/02/2017 by WW : http://192.168.119.55/x/SoF0
    if Signal1.timestamp != Signal2.timestamp:
        raise NameError("Timestamps of signals does not match")
    # Step 1: Apply the signs of the heading
    if sign(Signal1.heading) != 0:
        s1 = Signal1.data * sign(Signal1.heading)
    else:
        s1 = Signal1.data
    if sign(Signal2.heading) != 0:
        s2 = Signal2.data * sign(Signal2.heading)
    else:
        s2 = Signal2.data
    # Step 2: Identfy the setup ('XX' or 'XY')
    if Signal1.orientation + Signal2.orientation == "XY":
        sx = s1
        sy = s2
    elif Signal1.orientation + Signal2.orientation == "YX":
        sx = s2
        sy = s1
    elif Signal1.orientation + Signal2.orientation == "XX":
        raise NameError("XX configuration not implemented yet")
    else:
        raise NameError("Unexpected orientation of sensors")
    # Step 3 : construct the rotation matrix
    Xheading = max([abs(Signal1.heading), abs(Signal2.heading)]) / 180 * pi
    R1 = [[cos(Xheading), sin(Xheading)], [-sin(Xheading), cos(Xheading)]]
    yaw_angle = (yaw_angle + 180) / 180 * pi
    # +180 as the positive FA direction is pointing towards Aft
    R2 = [[cos(yaw_angle), -sin(yaw_angle)], [sin(yaw_angle), cos(yaw_angle)]]

    R_tot = asmatrix(R2) * asmatrix(R1)

    # Step 4 : perform the rotation
    sFA = R_tot[0, 0] * sx + R_tot[0, 1] * sy
    sSS = R_tot[1, 0] * sx + R_tot[1, 1] * sy
    # Step 5 : return a list of DYNAwind signals

    SensorString = Signal1.name.split("_")
    del SensorString[-2:]  # remove heading and orientation
    SensorString = "_".join(SensorString)

    Signal_FA = Signal(
        Signal1.source,
        Signal1.group,
        Signal1.Fs,
        sFA,
        SensorString + "_FA",
        Signal1.unit_string,
        Signal1.timestamp,
    )
    Signal_SS = Signal(
        Signal1.source,
        Signal1.group,
        Signal1.Fs,
        sSS,
        SensorString + "_SS",
        Signal1.unit_string,
        Signal1.timestamp,
    )

    if hasattr(Signal1, "level"):
        setattr(Signal_FA, "level", Signal1.level)
        setattr(Signal_SS, "level", Signal1.level)
    setattr(Signal_FA, "heading", "FA")
    setattr(Signal_SS, "heading", "SS")

    Signals.extend([Signal_FA, Signal_SS])
    return Signals


def calcRMS1p(Signals, stat_dict):
    import numpy as np

    rpm = stat_dict["rpm/mean"]
    for signal in Signals:
        PSD = signal.calcPSD()
        idx = np.argmin(np.abs(signal.f - rpm / 60))  # Closest frequency
        signal.rms1p = np.sum(PSD[idx - 1 : idx + 2])
    return Signals


def updateSecondary(
    Series,
    tuples_signal,
    tuples_in,
    tuples_out,
    functions,
    SourceFolder,
    start=None,
    stop=None,
):
    import datetime
    import os

    """
    tuples_signal :
    sensors you need the timeseries from to calculate
    the secondary parameters

    tuples_in     : statistics required for the calculation of
    the secondary parameters (eg. SCADA yaw)
    tuples_out    :
    """
    # %% Filter for the timestamps where you should be able
    # to calculate the secondary parameters
    tuples = tuples_in.copy()
    tuples.extend(tuples_signal)
    df = Series.get_df(tuples, start=start, stop=stop)
    df.dropna(axis=0, how="any", inplace=True)
    # %%
    df_second = Series.get_df(tuples_out, start=start, stop=stop)
    result = pd.concat([df, df_second], axis=1, join_axes=[df.index])
    sensors = []
    for signaltuple in tuples_signal:
        sensors.append(signaltuple[2])

    for index in result.index:
        datestr = datetime.datetime.strftime(
            index - datetime.timedelta(0, 600), "%Y%m%d_%H%M%S"
        )
        yyyy = datestr[:4]
        mm = datestr[4:6]
        dd = datestr[6:8]
        if os.path.isfile(
            SourceFolder
            + os.sep
            + yyyy
            + os.sep
            + mm
            + os.sep
            + dd
            + os.sep
            + datestr
            + ".tdms"
        ):
            signals = readTDMS(
                SourceFolder
                + os.sep
                + yyyy
                + os.sep
                + mm
                + os.sep
                + dd
                + os.sep
                + datestr
                + ".tdms"
            )
        elif os.path.isfile(SourceFolder + os.sep + datestr + ".tdms"):
            signals = readTDMS(SourceFolder + os.sep + datestr + ".tdms")
        else:
            continue
        for signal in signals:
            if signal.name not in sensors:
                signals.remove(signal)
        stat_dict = dict()
        # Passes the ten minute values as dictionary
        for stats in tuples_in:
            stat_dict[stats[2] + "/" + stats[3]] = result[stats][index]
        for function in functions:
            signals = function(signals, stat_dict)
        for signal in signals:
            updateSeries(Series, signal, overwrite=True)


def processSignal(Signal):
    record = get_config(Signal)
    if record is not None:
        # The current record is applicable to the current signal,
        # Set additional properties
        def setSecondarySensorProperties():
            # name should always be first!
            propertyList = [
                "name",
                "unit_string",
                "sensitivity",
                "heading",
                "level",
                "orientation",
                "group",
                "offset",
                "Ri",
                "Ro",
                "filterPassband",
                "filterOrder",
                "downsample",
            ]
            for prop in propertyList:
                if Signal.name + "/" + prop in record:
                    setattr(Signal, prop, record[Signal.name + "/" + prop])
            return Signal

        Signal = setSecondarySensorProperties()
        # Check if the signal has a filter defined
        if Signal.name + "/filterPassband" in record.keys():
            Signal.filter(
                record[Signal.name + "/filterPassband"][0],
                record[Signal.name + "/filterPassband"][1],
                record[Signal.name + "/filterOrder"],
            )
        elif Signal.group + "/filterPassband" in record.keys():
            Signal.filter(
                record[Signal.group + "/filterPassband"][0],
                record[Signal.group + "/filterPassband"][1],
                record[Signal.group + "/filterOrder"],
            )
        # Check if the signal is to be downsampled
        if Signal.name + "/downsample" in record.keys():
            Signal.downsample(record[Signal.name + "/downsample"])
        elif Signal.group + "/filterPassband" in record.keys():
            Signal.downsample(record[Signal.group + "/downsample"])

        # Verify that the signal is in engineering units and correct if necessary
        if hasattr(Signal, "group"):
            if Signal.group == "acceleration":
                if Signal.unit_string != "g":
                    try:
                        Signal.data = Signal.data / record[Signal.name + "/sensitivity"]
                        Signal.unit_string = "g"
                    except:
                        raise ValueError(
                            "Conversion to engineering units failed : sensitivity from ("
                            + Signal.unit_string
                            + ") to (g) of "
                            + Signal.name
                            + " not defined"
                        )
            elif Signal.group == "strain":
                if Signal.unit_string != "microstrain":
                    Signal = processStrainGauges(Signal, record)
            elif Signal.group == "displacement":
                if Signal.name + "/sensitivity" in record.keys():
                    Signal.data = Signal.data / record[Signal.name + "/sensitivity"]
            else:
                if Signal.name + "/sensitivity" in record.keys():
                    Signal.data = Signal.data / record[Signal.name + "/sensitivity"]
                if Signal.name + "/offset" in record.keys():
                    Signal.data = Signal.data - record[Signal.name + "/offset"]
            # Temperature compensation
            # !!! To Do update temperature_compensation to pull temp from JSON
    #            if Signal.name+'/TCSensor' in record.keys():
    #                temperature_sensor = record[Signal.name+'/TCSensor']
    #                if Signal.name+'/TemperatureCompensation' in record.keys():
    #                    temp_coef = record[Signal.name+'/TemperatureCompensation']
    #                else:
    #                    temp_coef=None # will trigger default settings
    #                Signal = temperature_compensation(Signal,
    #                                                  temperature_sensor=temperature_sensor,
    #                                                  temp_coef=temp_coef,
    #                                                  group=Signal.group)

    return Signal


def processStrainGauges(Signal, record):
    if Signal.unit_string == "strain":
        Signal.data = Signal.data * 1e6
        Signal.unit_string = "microstrain"
    elif Signal.unit_string == "Nm":
        pass  # Bending moment
    elif Signal.unit_string == "N":
        pass  # Normal Load
    else:
        if Signal.name + "/bridgeType" in record:
            if record[Signal.name + "/bridgeType"] == "quarter":
                # Quarter bridge calculation
                # - without lead compensations
                # - No shear stress compensation, assumption of a uni-axial stress condition
                Signal.data = (
                    -4
                    * Signal.data
                    / record[Signal.name + "/gageFactor"]
                    / (1 + 2 * Signal.data)
                    * 1e6
                )  # NI documentation : Strain Gauge Configuration types
                Signal.unit_string = "microstrain"
            else:
                raise NameError("Brigetype specified in config file is not supported")
        else:
            raise NameError("Bridge type not specified in config file")
    return Signal


def calcBendingMomentSignal(Signals, stat_dict):
    import numpy as np

    if type(stat_dict) is dict:
        yaw_angle = stat_dict["yaw/mean"]
        offsets = []
        for signal in Signals:
            if signal.name + "/offset" in stat_dict.keys():
                offsets.append(
                    stat_dict[signal.name + "/offset"] + stat_dict[signal.name + "/tc"]
                )
            else:
                offsets.append(np.nan)

    Ri = np.empty([len(Signals), 1])
    Ro = np.empty([len(Signals), 1])
    headings = np.empty([len(Signals), 1])
    strains = np.empty([len(Signals), len(Signals[0].data)])
    for i in range(len(Signals)):
        Ri[i] = Signals[i].Ri
        Ro[i] = Signals[i].Ro
        headings[i] = Signals[i].heading
        strains[i, :] = Signals[i].data - offsets[i]
    A = np.pi * (Ro ** 2 - Ri ** 2)  # Surface area
    Ic = np.pi / 4 * (Ro ** 4 - Ri ** 4)  # Area Moment of Inertia
    if Signals[0].unit_string == "microstrain":
        # Young modulus of steel + conversion from microstrain to strain
        YoungModulus = 210e9 / 1e6
    else:
        raise NameError("Strains not in microstrain")

    K = np.empty([len(Signals), 3])
    for i in range(len(Signals)):
        K[i, 0] = 1 / A[i, 0]
        K[i, 1] = Ri[i, 0] / Ic[i] * np.sin((360 - headings[i]) / 180 * np.pi)
        K[i, 2] = -Ri[i, 0] / Ic[i] * np.cos((360 - headings[i]) / 180 * np.pi)
    K = K / YoungModulus
    K_pinv = np.linalg.pinv(K)
    Theta = np.dot(K_pinv, strains)

    # Transform into the Mtn and Mtl frame work
    yaw_angle = (-yaw_angle + 180) / 180 * np.pi
    cd = np.cos(yaw_angle)
    sd = np.sin(yaw_angle)
    R = np.asmatrix([[cd, sd], [-sd, cd]])

    M = np.dot(R, Theta[1:, :])
    # Turn results into DYNAwind signals
    SensorString = Signals[0].name.split("_")
    SensorString = "_".join(SensorString[:3])  # remove heading and orientation
    Signals.append(
        Signal(
            Signals[0].source,
            Signals[0].group,
            Signals[0].Fs,
            np.squeeze(np.asarray(Theta[0, :])),
            SensorString + "_N",
            "N",
            Signals[0].timestamp,
        )
    )
    Signals.append(
        Signal(
            Signals[0].source,
            Signals[0].group,
            Signals[0].Fs,
            np.squeeze(np.asarray(M[0, :])),
            SensorString + "_Mtl",
            "Nm",
            Signals[0].timestamp,
        )
    )
    Signals.append(
        Signal(
            Signals[0].source,
            Signals[0].group,
            Signals[0].Fs,
            np.squeeze(np.asarray(M[1, :])),
            SensorString + "_Mtn",
            "Nm",
            Signals[0].timestamp,
        )
    )
    return Signals


def plotSignals(Signals):
    for sig in Signals:
        sig.plot()
    plt.ylabel(Signals[0].group)


def plotLoadEvent(signals, offsets, yaw_angle):
    BM_signals = calcBendingMomentSignal(signals, offsets, yaw_angle)
    # Two subplots, the axes array is 1-d
    plt.subplots(211)
    plt.plot(BM_signals[2])  # Mtn
    plt.plot(BM_signals[1])  # Mtl
    plt.legend("Mtn", "Mtl")
    plt.subplots(211)
    for i in range(len(signals)):
        plt.plot(signals[i].data - offsets[i])


def exportSignals(Signals, filename):
    """function to export a list of Signals to .txt files that can be easily read by MATLAB"""
    # We need to give a bit more context in the header of the file! E.g. source
    exportFile = open(filename, "w")
    exportFile.write("Export from DYNAwind" + "\n")
    exportFile.write("Generated : " + str(datetime.datetime.utcnow()) + "\n")
    exportFile.write("\n")
    exportFile.write("Time")
    for i in range(0, len(Signals)):
        exportFile.write(",\t" + Signals[i].name)
    exportFile.write("\n")
    exportFile.write("")
    for i in range(0, len(Signals)):
        exportFile.write(",\t" + Signals[i].group)
    exportFile.write("\n")
    exportFile.write("(s)")
    for i in range(0, len(Signals)):
        exportFile.write(",\t(" + Signals[i].unit_string + ")")
    exportFile.write("\n\n")
    time = Signals[0].time()
    for i in range(0, len(time)):
        exportFile.write(str(time[i]))
        for j in range(0, len(Signals)):
            exportFile.write(",\t" + str(Signals[j].data[i]))
        exportFile.write("\n")


def stats2df(Signals, Operators=[]):
    # input is a list of signals
    import pandas as pd
    import pytz
    from numpy import nan
    import numpy as np

    if Operators == []:
        getOperators = True
    else:
        getOperators = False

    multiIndexTuples = []
    for i in range(0, len(Signals)):
        df = pd.DataFrame()
        if getOperators:
            Operators = get_operators(Signals[i])
        timestamp = Signals[i].timestamp + datetime.timedelta(minutes=10)
        df.loc[0, "time"] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        df["time"] = pd.to_datetime(df["time"], utc=True)
        for j in range(0, len(Operators)):
            multiIndexTuples.append(
                (Signals[i].source, Signals[i].group, Signals[i].name, Operators[j])
            )
            # Not so happy with this section
            if Operators[j] == "min":
                df.loc[0, Operators[j] + "_" + Signals[i].name] = np.min(
                    Signals[i].data
                )
            elif Operators[j] == "max":
                df.loc[0, Operators[j] + "_" + Signals[i].name] = np.max(
                    Signals[i].data
                )
            elif Operators[j] == "offset":
                if hasattr(Signals[i], "offset"):
                    df.loc[0, Operators[j] + "_" + Signals[i].name] = Signals[i].offset
                else:
                    df.loc[0, Operators[j] + "_" + Signals[i].name] = 0
            elif Operators[j] == "tc":
                df.loc[0, Operators[j] + "_" + Signals[i].name] = nan
            else:
                if callable(getattr(Signals[i], Operators[j])):
                    df.loc[0, Operators[j] + "_" + Signals[i].name] = getattr(
                        Signals[i], Operators[j]
                    )()
                else:
                    df.loc[0, Operators[j] + "_" + Signals[i].name] = getattr(
                        Signals[i], Operators[j]
                    )

        if i == 0:
            result = df
        else:
            result = pd.merge(result, df, on="time")
    result.index = result["time"]
    if result.index.tzinfo is not None:
        if not result.index.tzinfo == pytz.utc:
            result.index = result.index.tz_convert(tz=pytz.utc)
    else:
        result.index = result.index.tz_localize(tz=pytz.utc, ambiguous="infer")
    # TDMS are in UTC!

    del result["time"]
    multiIndex = pd.MultiIndex.from_tuples(multiIndexTuples)
    result.columns = multiIndex
    result.sort_index(axis=1, inplace=True)
    return result


def get_operators(signal):
    Operators = []
    record = get_config(signal)
    logger.debug("record is {}".format(record))
    for key in record.keys():
        if "stats" in key:
            for operator in record[key]:
                Operators.append(operator)
    return Operators


def stats2dict(Signals, Operators=[]):
    from numpy import float64  # JSON parser supports float64, not float32
    import importlib

    if Operators == []:
        getOperators = True
    else:
        getOperators = False

    result = dict()
    for i in range(0, len(Signals)):
        if getOperators:
            Operators = get_operators(Signals[i])
        for j in range(0, len(Operators)):
            if Operators[j] == "min":
                result[Operators[j] + "_" + Signals[i].name] = float64(
                    min(Signals[i].data)
                )
            elif Operators[j] == "max":
                result[Operators[j] + "_" + Signals[i].name] = float64(
                    max(Signals[i].data)
                )
            else:
                if callable(getattr(Signals[i], Operators[j])):
                    result[Operators[j] + "_" + Signals[i].name] = getattr(
                        Signals[i], Operators[j]
                    )()
                else:
                    result[Operators[j] + "_" + Signals[i].name] = getattr(
                        Signals[i], Operators[j]
                    )
            ###
    # %% Check for custom functions, that return a dict to add to the current data
    record = get_config(Signals[0])
    if Signals[0].group + "/custom" in record:
        site = Signals[0].site
        for fnct in record[Signals[0].group + "/custom"]:
            customMethod = getattr(
                importlib.import_module(
                    "dynawind.config." + site.lower() + ".custom_" + site.lower()
                ),
                fnct,
            )
            result.update(customMethod(Signals))
    # %% Check for MPE
    if Signals[0].group + "/mpe/directions" in record:
        for direction in record[Signals[0].group + "/mpe/directions"]:
            mpe_signals = pullSignalsFromList(
                Signals, record[Signals[0].group + "/mpe/" + direction]
            )
            mpe_results = mpe.MPE(mpe_signals, direction=direction)
            result.update(mpe_results.exportdict(trackedOnly=True))

    return result


def HF_stats2dict(Signals, timescale):
    """Allows to generate statistics for higher frequencies than the original
    length of the Signal, e.g. every 60s"""

    import copy

    if timescale > 600:
        raise ValueError("Timescale requested exceeds the 10-minute maximum")
    if not 600 % timescale == 0:
        raise ValueError(
            "The default 10 minute file length is not an integer multiple of the requested timescale"
        )
    NewLength = int(timescale * Signals[0].Fs)
    nrOfFiles = int(600 / timescale)
    #
    data = []
    site = [Signals[0].site] * nrOfFiles
    location = [Signals[0].location] * nrOfFiles
    dt = []
    Signals_short = copy.deepcopy(Signals[:])
    for i in range(nrOfFiles):
        for [signal, signal_short] in zip(Signals, Signals_short):
            signal_short.data = signal.data[i * NewLength : (i + 1) * NewLength]
        data.append(stats2dict(Signals_short))
        dt.append(signal.timestamp + datetime.timedelta(0, (i + 1) * timescale))

    return (dt, site, location, data)


def pullSignalsFromList(Signals, names):
    pulledSignals = []
    for signal in Signals:
        for name in names:
            if name == signal.name:
                pulledSignals.append(signal)
                continue
        continue
    return pulledSignals


def getSource(tdms_file, path):
    from os import sep

    # (Preferred) It is also possible to determine the source from the tdms_file.object().properties
    # Legacy solution, use the path
    source = "unknown"
    strList = path.split(sep)
    logger.debug("strList is {}".format(strList))
    for i in range(0, len(strList)):
        if strList[i] == "TDD":
            source = strList[i - 1]
            break
    logger.debug("source is {}".format(source))
    return source


def readTDMS(path, Source=None):
    from nptdms import TdmsFile
    from datetime import datetime
    from os import sep
    import pytz

    try:
        tdms_file = TdmsFile(path)
    except:
        logger.warning("Failed to open {}".format(path))
        Signals = []
        return Signals

    if Source == None:
        Source = getSource(tdms_file, path)
    Groups = tdms_file.groups()
    Signals = []
    for i in range(0, len(Groups)):
        Channel_lst = tdms_file.group_channels(Groups[i])
        for j in range(0, len(Channel_lst)):
            Name = str(Channel_lst[j]).split("/")[2][1:-2]
            if "unit_string" in Channel_lst[j].properties:
                Unit = Channel_lst[j].properties["unit_string"]
            else:
                Unit = ""

            Data = tdms_file.object(Groups[i], Name)
            Fs = 1 / Channel_lst[j].properties["wf_increment"]
            timestampstr = path[-20:-5]
            # timestamp is start of measurement
            timestamp = datetime.strptime(timestampstr, "%Y%m%d_%H%M%S")
            timestamp = pytz.utc.localize(timestamp)
            Signals.append(
                Signal(Source, Groups[i], Fs, Data.data, Name, Unit, timestamp)
            )
    # %% Quality check of Signals
    signalLengths = []
    for signal in Signals:
        length = signal.data.shape[0]
        if length == 0:
            Signals.remove(signal)
        else:
            signalLengths.append(length)
    return Signals


def getTDMSpath(timestamp, filetype, location, site=None, root=r"\\192.168.119.14"):
    from os.path import join

    if not site:
        site = getSite(location)
    if isinstance(timestamp, str):
        dt = datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    else:
        dt = timestamp
    path = join(
        root,
        "data_primary_" + site.lower(),
        location,
        "TDD",
        "TDD_" + filetype,
        str(dt.year),
        str(dt.month).zfill(2),
        str(dt.day).zfill(2),
        dt.strftime("%Y%m%d_%H%M%S") + ".tdms",
    )
    return path


def getTDMS(timestamp, filetype, location, site=None, root=r"\\192.168.119.14"):
    path = getTDMSpath(timestamp, filetype, location, site=site, root=root)
    Signals = readTDMS(path)
    return Signals


def mergeTDMS(dt, duration, location, filetype=None, root=r"\\192.168.119.14"):
    # duration : in multiples of 10 minutes
    Signals = getTDMS(timestamp=dt, location=location, root=root, filetype=filetype)
    for i in range(1, duration):
        dt_local = dt + datetime.timedelta(0, 600 * i)
        Signals_new = getTDMS(
            timestamp=dt_local, location=location, root=root, filetype=filetype
        )
        for s1, s2 in zip(Signals, Signals_new):
            if s1.name == s2.name:
                s1.append(s2)
            else:
                raise NameError("Signallists in consecutive files does not match")
    return Signals


def writeTDMS(Signals, path):
    # writes the list of DYNAwind.signal objects to a TDMS file
    pass


# %% Checks Strain Gauges (works for non opposing sensors)
def check_sg(Strains, treshold=0.1, plot_figure=True):
    import numpy as np
    import itertools

    if not Strains:
        return
    Mtn = []
    Mtl = []
    N = []
    combi = []
    stat_dict = {"yaw/mean": 0}

    for strain in Strains:
        # Arbitrary values as this is not essential to the method
        strain.Ro = 5.000
        strain.Ri = 4.900
        stat_dict[strain.name + "/offset"] = strain.mean()
        stat_dict[strain.name + "/tc"] = 0

    for i in itertools.combinations(range(len(Strains)), 3):
        T = [Strains[j] for j in i]
        combi.append(i)
        tst1 = calcBendingMomentSignal(T, stat_dict)
        Mtn.append(tst1[-1])
        Mtl.append(tst1[-2])
        N.append(tst1[-3])

    Ic = np.pi / 4 * (Strains[0].Ro ** 4 - Strains[0].Ri ** 4)
    A = np.pi * (Strains[0].Ro ** 2 - Strains[0].Ri ** 2)
    Ri = Strains[0].Ri
    E = 210e9
    tst = np.zeros([len(Mtn), len(Strains)])
    i = 0
    heads = []
    for strain in Strains:
        j = 0
        head = strain.heading
        heads.append(head)
        head = (360 - head) / 180 * np.pi
        for n, l, normal, cmb in zip(Mtn, Mtl, N, combi):
            tst_strain = (
                normal.data / A / E
                + Ri / Ic / E * (l.data * np.sin(head) - n.data * np.cos(head)) * 1e6
            )
            ones = np.ones(len(Mtn[0].data))
            K = np.stack((tst_strain, ones), axis=1)
            theta = np.dot(np.linalg.pinv(K), strain.data - strain.mean())
            tst[j, i] = theta[0]
            j += 1
        i += 1
    error = np.mean(np.abs(1 + tst), axis=0)
    if plot_figure:
        plt.figure()
        plt.bar(range(len(Strains)), error, tick_label=heads)
        plt.ylim([0, 0.5])
        plt.hlines(treshold, -0.5, len(Strains) - 0.5, "r")
        plt.ylabel("Error")
        plt.xlabel("Strain gauge heading")
    return error


# %%  Plots results for opposing sensors


def checkOpposingSG(Strains, saveFig=False, path="."):
    import numpy as np

    f1 = plt.figure()
    Strains[0].plot()
    plt.plot(Strains[0].time(), -Strains[1].data)
    plt.legend([Strains[0].name, Strains[1].name])
    plt.xlim(100, 160)
    plt.xlabel("Strain (microstrain)")
    if saveFig:
        timestamp = Strains[0].timestamp.strftime("%Y%m%d_%H%M%S")
        plt.savefig(
            path
            + "Timeseries_"
            + timestamp
            + "_"
            + Strains[0].location
            + "_Strain_"
            + str(Strains[0].heading),
            dpi=300,
        )

    #
    f2 = plt.figure()
    plt.plot(Strains[0].data, Strains[1].data)
    plt.plot(np.array([-15, 15]), np.array([15, -15]), "k-")
    plt.xlabel(Strains[0].name)
    plt.ylabel(Strains[1].name)

    # Determinate correction
    ones = np.ones(len(Strains[1].data))
    K = np.stack((Strains[1].data, ones), axis=1)
    theta = np.dot(np.linalg.pinv(K), Strains[0].data)
    plt.plot(np.array([-15 * theta[0], 15 * theta[0]]), np.array([-15, 15]), "r--")
    plt.text(5, 5, "Ratio: " + str(theta[0]))

    if saveFig:
        figPath = path + "\Comparisson_" + timestamp + "_"
        +Strains[0].location + "_Strain_" + str(Strains[0].heading)
        plt.savefig(figPath, dpi=300)

    f3 = plt.figure()
    Strains[0].plotPSD(xlim=(0, 5))
    Strains[1].plotPSD(xlim=(0, 5))
    plt.legend([Strains[0].name, Strains[0].name])
    if saveFig:
        plt.savefig(
            path
            + "PSD_"
            + timestamp
            + "_"
            + Strains[0].location
            + "_Strain_"
            + str(Strains[0].heading),
            dpi=300,
        )

    h = [f1, f2, f3]
    return h


def temperature_compensation(
    Signal=None,
    temperature=None,
    temperature_sensor=None,
    temp_coef=None,
    T_ref=None,
    group=None,
):
    """ Performs temperature compensation on Signal objects """

    def tempCompensation(T, coeficients, T_ref=0):
        # Default behavior is Coef[0]*(T-T_ref)^n+Coef[1]*(T-T_ref)^(n-1)+...+Coef[n-1]*(T-T_ref)+Coef[n]
        dT = T - T_ref
        n = len(coeficients) - 1
        tc = coeficients[0] * (dT ** n)
        for i in range(1, len(coeficients)):
            tc += coeficients[i] * (dT ** (n - i))
        return tc

    # Load default coeficients for temperature compensation
    if temp_coef is None:
        if group == "fiber":
            T_ref = 22.5
            # linear = S1/k+(alpha_t-alpha_f)
            linear = 6.37 / 0.772 + (12 - 0.552)
            # quadratic = S2/k
            quadratic = 0.00746 / 0.772
            temp_coef = [quadratic, linear, 0]
    # Temperature
    if temperature is None:
        # Should pull the temperature from the temperature_sensor from the JSON file
        raise ValueError(
            "temperature value not provided, automatic temperature retrieval not yet a feature"
        )

    tc = tempCompensation(temperature, temp_coef, T_ref=T_ref)
    if Signal is not None:
        Signal.data = Signal.data - tc
        Signal.temperature_compensation = tc
        return Signal
    else:
        return tc
