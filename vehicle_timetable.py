import numpy
import datetime
import copy
import operator

STATE_LIST = ['OT', 'ZT', 'ST', 'DT', 'UT', 'HT']
STATE_LIST_LENGTH = len(STATE_LIST)


def t_rank(lists):
    # Rank list of dict according to the time
    return sorted(lists, key=lambda item: item['time'])


def rank_OT(state):
    # Rank the vehicle according to the next arriving time on camp
    rank = []
    for vehicle in state:
        rank.append({'time': vehicle.OT[-1], 'vehicle_data': vehicle})
    return t_rank(rank)


class create_time_table:
    def __init__(self,
                 site_data,
                 init_vehicle_state,
                 relation,
                 verbose=False):
        self.init_site_data = site_data
        self.relation = relation
        self.init_vehicle_state = init_vehicle_state
        self.n_vehicle = len(init_vehicle_state)
        # car_number,state,cubic,location,last_state_time
        self.count_on_camp = 0
        for vehicle in self.init_vehicle_state:
            if vehicle.location == 'ZT':
                self.count_on_camp += 1

        # Calculate time for each state, get the arriving time OT for each vehicle according to its order
        for idx, vehicle in enumerate(self.init_vehicle_state):
            self.vehicle = vehicle
            self.current_site_data = self.get_order(self.init_site_data)
            if self.vehicle.order_num[-1] == 'None':
                # If no order appended, initialize the OT and count vehicles on camp
                if verbose:
                    print('No order for vehicle number {car_number}'.format(car_number=self.vehicle.car_number))
                self.append_time('OT', self.vehicle.state_start_time)
                self.vehicle.update_t_wait(numpy.ceil(self.count_on_camp * 5))
                self.count_on_camp += 1
            else:
                self.update_time_cost(init=True)
                self.append_round_time(init=True)
        try:
            self.sort_time()
        except:
            pass

    def sort_time(self):
        self.current_site_data.rank_DT()
        self.current_site_data.rank_UT()
        self.current_site_data.rank_HT()

    def run(self, sol):
        num_genes = len(sol)
        # Append the order number to each vehicle
        new_vehicle_state = copy.deepcopy(self.init_vehicle_state)
        new_site_data = copy.deepcopy(self.init_site_data)
        for site_num in sol:
            # Rank the vehicles according to its arriving time on camp OT
            rank = rank_OT(new_vehicle_state)
            # Take the first vehicle and link it to the current site
            self.vehicle = rank[0]['vehicle_data']
            self.vehicle.order_num.append(self.find_order_num(site_num))
            # Fetch the current site data
            self.current_site_data = self.get_order(new_site_data)
            # Update the time cost for current site
            self.count_on_camp = 1
            self.update_time_cost()
            self.sort_time()
            # Calculate all the timing for current vehicle
            self.append_round_time()
        return new_site_data, new_vehicle_state

    def find_order_num(self, solution):
        for _ in self.relation.items():
            if solution in _[1]:
                return _[0]

    def update_time_cost(self, init=False):
        self.vehicle.update_t_wait(numpy.ceil(self.count_on_camp * 5))
        self.vehicle.update_t_mix(numpy.ceil(self.vehicle.cubic / 4.5))
        self.vehicle.update_t_drive(numpy.ceil(self.current_site_data.distance * 2))
        self.vehicle.update_t_pump(numpy.ceil(self.vehicle.cubic * 2))
        self.vehicle.update_t_clean(15)
        if self.vehicle.location == 'OT' and init:
            self.count_on_camp += 1

    def get_order(self, site_data):
        for data in site_data:
            if data.order_num == self.vehicle.order_num[-1]:
                return data

    def append_round_time(self, init=False):
        start_index = STATE_LIST.index(self.vehicle.location)
        if init:
            self.append_time(self.vehicle.location, self.vehicle.state_start_time)
        for next_location in STATE_LIST[start_index + 1:] + ['OT']:
            self.update_time(next_location, init=init)

    def update_time(self, location, init=False):
        if location == 'OT':
            next_time = self.vehicle.HT[-1] + datetime.timedelta(minutes=self.vehicle.t_drive)
            self.append_time(location, next_time)
        elif location == 'ZT':
            next_time = self.vehicle.OT[-1] + datetime.timedelta(minutes=self.vehicle.t_wait)
            self.append_time(location, next_time)
        elif location == 'ST':
            next_time = self.vehicle.ZT[-1] + datetime.timedelta(minutes=self.vehicle.t_mix)
            self.append_time(location, next_time)
        elif location == 'DT':
            next_time = self.vehicle.ST[-1] + datetime.timedelta(minutes=self.vehicle.t_drive)
            self.append_time(location, next_time)
        elif location == 'UT':
            # UT_ij = DT_ij when DT_ij>HT_i(j-1)
            # UT_ij = HT_i(j-1) when DT_ij<HT_i(j-1)
            next_time = self.vehicle.DT[-1]
            if not init and len(self.current_site_data.HT) != 0:
                last_finish_time = self.current_site_data.HT[-1]['time'] - datetime.timedelta(minutes=self.vehicle.t_clean)
                if self.vehicle.DT[-1] < last_finish_time:
                    next_time = last_finish_time

                # Calculate ideal UT time
                # Get the demanded time by the site
                if last_finish_time < self.current_site_data.demand_time:
                    ideal_UT = self.current_site_data.demand_time
                else:
                    ideal_UT = last_finish_time
            else:
                ideal_UT = self.current_site_data.demand_time

            self.current_site_data.ideal_UT.append({'name': self.vehicle.car_number, 'time': ideal_UT})
            self.append_time(location, next_time)
        elif location == 'HT':
            next_time = self.vehicle.UT[-1] + datetime.timedelta(minutes=self.vehicle.t_pump + self.vehicle.t_clean)
            self.append_time(location, next_time)

    def append_time(self, location, time):
        name = self.vehicle.car_number
        temp = {'name': name, 'time': time}
        if location == 'OT':
            self.vehicle.OT.append(time)
        elif location == 'ZT':
            self.vehicle.ZT.append(time)
        elif location == 'ST':
            self.vehicle.ST.append(time)
        elif location == 'DT':
            self.vehicle.DT.append(time)
            self.current_site_data.DT.append(temp)
        elif location == 'UT':
            self.vehicle.UT.append(time)
            self.current_site_data.UT.append(temp)
        elif location == 'HT':
            self.vehicle.HT.append(time)
            self.current_site_data.HT.append(temp)
        self.vehicle.location = location
        return
