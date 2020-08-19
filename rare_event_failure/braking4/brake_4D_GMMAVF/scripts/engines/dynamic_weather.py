import sys 


class DynamicPrecipitation(object):
    def __init__(self, initial_precipitation=0.0, step_0=0, step_1=100, slope=0.0):
        self.initial_precipitation = initial_precipitation
        self.step_0 = step_0
        self.step_1 = step_1
        self.slope = slope

    def get_weather_parameters(self, step=0):
        if step <= self.step_0:
            precipitation_parameter = self.initial_precipitation
        elif step <= self.step_1:
            precipitation_parameter = self.initial_precipitation + self.slope * (step - self.step_0)
        else:
            precipitation_parameter = self.initial_precipitation + self.slope * (self.step_1 - self.step_0)

        weather_parameters = carla.WeatherParameters(
            cloudiness = 80.0, 
            precipitation = precipitation_parameter, 
            precipitation_deposits = precipitation_parameter,
            sun_altitude_angle = 70.0
        )
        return weather_parameters
