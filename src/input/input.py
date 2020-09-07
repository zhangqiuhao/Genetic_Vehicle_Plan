import numpy
import datetime
from ..core.vehicle_timetable import t_rank


class site_data_input:
    def __init__(self,
                 data):
        self.order_num = data[0]
        self.site_name = data[1]
        #self.construction_part = data[3]
        #self.demand_grade = data[4]
        #self.slump = data[5]
        self.demand_cubic = float(data[2])
        #self.pump = data[7]
        self.demand_time = datetime.datetime.strptime(data[3], '%Y-%m-%d %H:%M')  # The demanded starting time by the site
        self.distance = float(data[4]) / 2.0
        self.t_buffer = int(data[5])
        self.t_solidification = int(data[6])
        self.punishment = float(data[7])
        self.n_deliver = numpy.ceil(self.demand_cubic / float(data[8])).astype(numpy.int)

        self.DT = []
        self.UT = []
        self.ideal_UT = []
        self.HT = []

    def update_t_buffer(self, data):
        self.t_buffer = data

    def rank_DT(self):
        self.DT = t_rank(self.DT)

    def rank_UT(self):
        self.UT = t_rank(self.UT)

    def rank_HT(self):
        self.HT = t_rank(self.HT)


class vehicle_state_input:
    def __init__(self,
                 data):
        self.car_number = data[0]
        self.state = data[1]
        self.cubic = int(data[2])
        self.location = data[3]
        self.order_num = [data[4]]
        self.state_start_time = datetime.datetime.strptime(data[5], '%Y-%m-%d %H:%M')

        # Time cost
        self.t_wait = int
        self.t_mix = int  # Mixing time of concrete
        self.t_drive = int  # Driving time between camp and site
        self.t_pump = int  # Pumping time on site
        self.t_clean = int  # Cleaning time on site

        self.OT = []
        self.ZT = []
        self.ST = []
        self.DT = []
        self.UT = []
        self.HT = []

    def update_t_wait(self, data):
        self.t_wait = data

    def update_t_mix(self, data):
        self.t_mix = data

    def update_t_drive(self, data):
        self.t_drive = data

    def update_t_pump(self, data):
        self.t_pump = data

    def update_t_clean(self, data):
        self.t_clean = data


class camp_data_input:
    def __init__(self):
        return
