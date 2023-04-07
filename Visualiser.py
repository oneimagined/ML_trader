from typing import List
import os
import json
import numpy as np
from typing import List

from datetime import datetime

class VolumeProfileData:
    def __init__(self, date: str, label: str, price: float, volume_profiles: List[List[float]], time_data: List[List[float]]):
        self.date = date
        self.label = label
        self.price = price
        self.volume_profiles = volume_profiles
        self.time_data = time_data

import matplotlib.pyplot as plt

class VolumeProfileVisualizer:
    def __init__(self, volume_profiles: List[VolumeProfileData]):
        self.volume_profiles = volume_profiles
        self.current_index = 0

    def plot_volume_profileold(self, fig, ax):
        ax.clear()
        volume_profile_data = self.volume_profiles[self.current_index]
        ax.set_title(f"{volume_profile_data.date} - {volume_profile_data.label}")
        ax.set_xlabel("Volume")
        ax.set_ylabel("Price")

        prices = []
        weights = []
        stddev_upper = []
        stddev_lower = []
        upper_colour = []
        lower_colour = []
        prices_colour = []
        for vp in volume_profile_data.volume_profiles:
            prices.append(vp[0][0])
            prices_colour.append('red')
            stddev_upper.append(vp[0][0] + vp[0][2])
            upper_colour.append('blue')
            stddev_lower.append(vp[0][0] - vp[0][2])
            lower_colour.append('blue')
            weights.append(vp[0][1])

        price_bars = prices+stddev_lower+stddev_upper
        weights_bars = weights+weights+weights
        color= prices_colour+upper_colour+lower_colour
        ax.barh(price_bars,weights_bars, color=color , height=0.001, alpha=0.5)

        fig.canvas.draw()

    def plot_time_dataoold(self, fig, ax):
        ax.clear()
        volume_profile_data = self.volume_profiles[self.current_index]
        ax.set_title(f"{volume_profile_data.date} - {volume_profile_data.label}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")

        times = []
        prices = []
        for td in volume_profile_data.time_data:
            times.append(datetime.fromtimestamp(td[4]))
            prices.append(td[0])

        ax.plot(times, prices, color='blue')

        fig.canvas.draw()

    def plot_volume_profile(self, fig, ax):
        ax.clear()
        volume_profile_data = self.volume_profiles[self.current_index]
        ax.set_title(f"{volume_profile_data.date} - {volume_profile_data.label}")
        ax.set_xlabel("Volume")
        ax.set_ylabel("Price")

        prices = []
        weights = []
        stddev_upper = []
        stddev_lower = []
        upper_colour = []
        lower_colour = []
        prices_colour = []
        for vp in volume_profile_data.volume_profiles:
            prices.append(vp[0][0])
            prices_colour.append('red')
            stddev_upper.append(vp[0][0] + vp[0][2])
            upper_colour.append('blue')
            stddev_lower.append(vp[0][0] - vp[0][2])
            lower_colour.append('blue')
            weights.append(vp[0][1])

        price_bars = prices+stddev_lower+stddev_upper
        weights_bars = weights+weights+weights
        color= prices_colour+upper_colour+lower_colour
        ax.barh(price_bars,weights_bars, color=color , height=0.001, alpha=0.5)

        # Set the Y-axis limits of the second plot to match the first plot
        self.ax2.set_ylim(ax.get_ylim())

        fig.canvas.draw()

    def plot_time_data(self, fig, ax):
        ax.clear()
        volume_profile_data = self.volume_profiles[self.current_index]
        ax.set_title(f"{volume_profile_data.date} - {volume_profile_data.label}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")

        times = []
        prices = []
        for td in volume_profile_data.time_data:
            times.append(datetime.fromtimestamp(td[4]))
            prices.append(td[0])

        ax.plot(times, prices, color='blue')

        # Set the Y-axis limits of the second plot to match the first plot
        self.ax2.set_ylim(self.ax1.get_ylim())

        fig.canvas.draw()


    def on_press(self, event):
        if event.key == "right":
            self.current_index += 1
            if self.current_index >= len(self.volume_profiles):
                self.current_index = 0
        elif event.key == "left":
            self.current_index -= 1
            if self.current_index < 0:
                self.current_index = len(self.volume_profiles) - 1

        self.plot_volume_profile(self.fig, self.ax1)
        self.plot_time_data(self.fig, self.ax2)
        self.fig.canvas.draw()

    def start_visualization(self):
        self.fig, self.axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
        self.ax1, self.ax2 = self.axes

        # Change this part to use the FigureManager object
        manager = plt.get_current_fig_manager()
        manager.canvas.mpl_connect("key_press_event", self.on_press)

        self.plot_volume_profile(self.fig, self.ax1)
        self.plot_time_data(self.fig, self.ax2)

        plt.show()


class VolumeProfiles:
    def __init__(self, volume_profiles: List[VolumeProfileData] = None):
        if volume_profiles is None:
            self.volume_profile_data_list = []
        else:
            self.volume_profile_data_list = volume_profiles

    def append(self, volume_profile_data: VolumeProfileData):
        self.volume_profile_data_list.append(volume_profile_data)

    def sort_by_date(self):
        self.volume_profile_data_list = sorted(self.volume_profile_data_list, key=lambda x: x.date)

    def get_volume_profile_data_by_date(self, date: str) -> VolumeProfileData:
        for volume_profile_data in self.volume_profile_data_list:
            if volume_profile_data.date == date:
                return volume_profile_data

        return None

    def get_all_volume_profile_data(self) -> List[VolumeProfileData]:
        return self.volume_profile_data_list

class VolumeProfileDataParser:
    def __init__(self):
        pass

    def parse_volume_profile_data(self, data):
        parsed_data = json.loads(data)
        date = parsed_data['Date']
        label = parsed_data['Label']
        price = parsed_data['Price']
        volume_profiles = []
        for indicator in parsed_data.get('Indicators', []):
            if indicator['Type'].startswith('Volume Profile VolumeProfile['):
                volume_profile_data = indicator['Data']
                # Extract the required data from the volume profile
                volume_profile = []
                for vp in volume_profile_data['training_data']:
                    volume_profile.append([vp[0], vp[1], vp[2], vp[3]])
                volume_profiles.append(volume_profile)
        time_data = parsed_data.get('Time Data', [])
        return VolumeProfileData(date, label, price, volume_profiles, time_data)

    def parse_volume_profile_file(self, file_path):
        with open(file_path, 'r') as f:
            file_data = f.read()
        return self.parse_volume_profile_data(file_data)

    def parse_volume_profile_directory(self, dir_path):
        volume_profiles = []
        for file_name in os.listdir(dir_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(dir_path, file_name)
                volume_profiles.append(self.parse_volume_profile_file(file_path))
        return VolumeProfiles(volume_profiles)

# Specify the directory containing the JSON files
directory_path = '../training'

volume_profiles = VolumeProfileDataParser().parse_volume_profile_directory(directory_path)

# Sort the VolumeProfiles instance by date
volume_profiles.sort_by_date()

visualizer = VolumeProfileVisualizer(volume_profiles.get_all_volume_profile_data())
visualizer.start_visualization()



"""

from typing import List
import os
import json
import numpy as np
from typing import List

from datetime import datetime

class VolumeProfileData:
    def __init__(self, date: str, label: str, price: float, volume_profiles: List[List[float]], time_data: List[List[float]]):
        self.date = date
        self.label = label
        self.price = price
        self.volume_profiles = volume_profiles
        self.time_data = time_data


import matplotlib.pyplot as plt

class VolumeProfileVisualizer:
    def __init__(self, volume_profiles):
        self.volume_profiles = volume_profiles
        self.current_index = 0

    def plot_volume_profile(self, fig, ax):
        ax.clear()
        volume_profile_data = self.volume_profiles[self.current_index]
        ax.set_title(f"{volume_profile_data.date} - {volume_profile_data.label}")
        ax.set_xlabel("Volume")
        ax.set_ylabel("Price")

        prices = []
        weights = []
        stddev_upper = []
        stddev_lower = []
        upper_colour = []
        lower_colour = []
        prices_colour = []
        for vp in volume_profile_data.volume_profiles:
            prices.append(vp[0][0])
            prices_colour.append('red')
            stddev_upper.append(vp[0][0] + vp[0][2])
            upper_colour.append('blue')
            stddev_lower.append(vp[0][0] - vp[0][2])
            lower_colour.append('blue')
            weights.append(vp[0][1])

        price_bars = prices+stddev_lower+stddev_upper
        weights_bars = weights+weights+weights
        color= prices_colour+upper_colour+lower_colour
        ax.barh(price_bars,weights_bars, color=color , height=0.001, alpha=0.5)

        fig.canvas.draw()

    def plot_time_data(self, fig, ax):
        ax.clear()
        volume_profile_data = self.volume_profiles[self.current_index]
        ax.set_title(f"{volume_profile_data.date} - {volume_profile_data.label}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")

        times = []
        prices = []
        for td in volume_profile_data.time_data:
            times.append(datetime.fromtimestamp(td[4]))
            prices.append(td[0])

        ax.plot(times, prices, color='blue')

        fig.canvas.draw()

    def on_press(self, event):
        if event.key == "right":
            self.current_index += 1
            if self.current_index >= len(self.volume_profiles.get_all_volume_profile_data()):
                self.current_index = 0
        elif event.key == "left":
            self.current_index -= 1
            if self.current_index < 0:
                self.current_index = len(self.volume_profiles.get_all_volume_profile_data()) - 1

        self.plot_volume_profile(self.fig1, self.ax1)
        self.plot_time_data(self.fig2, self.ax2)

    def start_visualization(self):
        self.fig1, self.ax1 = plt.subplots(figsize=(6, 6))
        self.fig2, self.ax2 = plt.subplots(figsize=(6, 6))

        self.fig1.canvas.mpl_connect("key_press_event", self.on_press)
        self.fig2.canvas.mpl_connect("key_press_event", self.on_press)

        self.plot_volume_profile(self.fig1, self.ax1)
        self.plot_time_data(self.fig2, self.ax2)

        plt.show()




class VolumeProfiles:
    def __init__(self, volume_profiles: List[VolumeProfileData] = None):
        if volume_profiles is None:
            self.volume_profile_data_list = []
        else:
            self.volume_profile_data_list = volume_profiles

    def append(self, volume_profile_data: VolumeProfileData):
        self.volume_profile_data_list.append(volume_profile_data)

    def sort_by_date(self):
        self.volume_profile_data_list = sorted(self.volume_profile_data_list, key=lambda x: x.date)

    def get_volume_profile_data_by_date(self, date: str) -> VolumeProfileData:
        for volume_profile_data in self.volume_profile_data_list:
            if volume_profile_data.date == date:
                return volume_profile_data

        return None

    def get_all_volume_profile_data(self) -> List[VolumeProfileData]:
        return self.volume_profile_data_list

class VolumeProfileDataParser:
    def __init__(self):
        pass

    def parse_volume_profile_data(self, data):
        parsed_data = json.loads(data)
        date = parsed_data['Date']
        label = parsed_data['Label']
        price = parsed_data['Price']
        volume_profiles = []
        for indicator in parsed_data.get('Indicators', []):
            if indicator['Type'].startswith('Volume Profile VolumeProfile['):
                volume_profile_data = indicator['Data']
                # Extract the required data from the volume profile
                volume_profile = []
                for vp in volume_profile_data['training_data']:
                    volume_profile.append([vp[0], vp[1], vp[2], vp[3]])
                volume_profiles.append(volume_profile)
        time_data = parsed_data.get('Time Data', [])
        return VolumeProfileData(date, label, price, volume_profiles, time_data)

    def parse_volume_profile_file(self, file_path):
        with open(file_path, 'r') as f:
            file_data = f.read()
        return self.parse_volume_profile_data(file_data)

    def parse_volume_profile_directory(self, dir_path):
        volume_profiles = []
        for file_name in os.listdir(dir_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(dir_path, file_name)
                volume_profiles.append(self.parse_volume_profile_file(file_path))
        return VolumeProfiles(volume_profiles)



# Specify the directory containing the JSON files
directory_path = '../training'

volume_profiles = VolumeProfileDataParser().parse_volume_profile_directory(directory_path)


# Sort the VolumeProfiles instance by date
volume_profiles.sort_by_date()

visualizer = VolumeProfileVisualizer(volume_profiles.volume_profile_data_list)
visualizer.start_visualization()
"""

