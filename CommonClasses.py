import os
import sys
import uuid
import csv

from enum import Enum, auto
from datetime import datetime
import cv2

import decimal

import json
import numpy as np
from json import JSONEncoder, JSONDecoder

import os
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import random



class OHLCIndex(Enum):
    open = 0
    high = auto()
    low = auto()
    close = auto()
    direction = auto()


class PriceEnginesTimeEpoch(Enum):
    ticker = 0
    #half = 0.5
    second = 1
    #ten_second = 10
    #thirty_second = 30
    #minute = 60

class MACDIndex(Enum):
    long_period = 0
    short_period = auto()
    signal_period = auto()

class TrainTradeTypeIndex(Enum):
    trade = 0
    dont_trade = auto()

class EpochBufferIndex(Enum):
    timestamp = 0
    bid = auto()
    offer = auto()
    bid_volume = auto()
    bid_ticker_volume = auto()
    offer_volume = auto()
    offer_ticker_volume = auto()

class PriceEngineIndex(Enum):
    timestamp = 0
    bid = auto()
    offer = auto()
    spread = auto()
    bid_volume = auto()
    bid_ticker_volume = auto()
    offer_volume = auto()
    offer_ticker_volume = auto()
    signed_volume = auto()
    open = auto()
    close = auto()
    high = auto()
    low = auto()


class TradingRules(Enum):
    empty = 0
    Impulse = auto()
    Aroon = auto()
    Stochastic = auto()
    HeikenAshi = auto()
    MovingAverage = auto()
    PriceVolume = auto()
    VolumeProfile = auto()


class AroonElements(Enum):
    aroon_up = 0
    aroon_down = auto()
    aroon_oscillator = auto()


class OHLCElements(Enum):
    open = 0
    high = auto()
    low = auto()
    close = auto()
    direction = auto()


class HullFilters(Enum):
    track = 0
    envelope = auto()
    medium_term = auto()
    long_term = auto()


def custom_round(num, multiple):
    # Convert the number to a decimal with the desired precision
    num_decimal = decimal.Decimal(str(num)).quantize(decimal.Decimal(str(multiple)), rounding=decimal.ROUND_HALF_UP)
    # Round the decimal to the closest multiple of the specified value
    rounded_decimal = (num_decimal / multiple).quantize(decimal.Decimal('1.'), rounding=decimal.ROUND_HALF_UP) * multiple
    # Convert the rounded decimal back to a float
    rounded_num = float(rounded_decimal)
    return rounded_num

class HighPassGaussianFilter:
    def __init__(self, ksize, sigma_high=0.75, sigma_low=4):
        self.ksize = ksize
        self.sigma_high = sigma_high
        self.sigma_low = sigma_low
        self.gaussian_high = cv2.getGaussianKernel(self.ksize, self.sigma_high)
        self.gaussian_low = cv2.getGaussianKernel(self.ksize, self.sigma_low)
        self.kernel = np.ravel(self.gaussian_high - self.gaussian_low)

    def filter(self, signal):
        filtered_signal = np.convolve(signal, self.kernel, mode='same')
        return filtered_signal

class OrderBook():
    def __init__(self, symbol):
        self.time_stamp = datetime.now().timestamp()
        self.current_bid_time_stamp = datetime.now()
        self.bid_dict = {}
        self.bid_volume_dict = {}
        self.bid_timestamp_dict = {}
        self.current_offer_time_stamp = datetime.now()
        self.offer_dict = {}
        self.offer_volume_dict = {}
        self.offer_timestamp_dict = {}
        self.symbol = symbol
        self.delete_list = []
        self.reset()

    def reset(self):
        self.bid_dict.clear()
        self.bid_volume_dict.clear()
        self.offer_dict.clear()
        self.offer_volume_dict.clear()
        self.bid_orderbook_length = 0
        self.offer_orderbook_length = 0
        self.current_bid_key = ''
        self.current_bid_change_timestamp = datetime.now()
        self.current_offer_key = ''
        self.current_offer_change_timestamp = datetime.now()
        self.current_offer_change_age = 0
        self.current_bid = 0
        self.current_offer = sys.maxsize
        self.current_bid_volume = 0
        self.current_offer_volume = 0
        self.current_spread = 0

    def updateBid(self, timestamp, key, bid, bid_trade_volume):
        self.bid_timestamp_dict[key] = timestamp
        self.bid_dict[key] = bid
        self.bid_volume_dict[key] = bid_trade_volume
        self.updateCurrentPrice()

    def updateOffer(self, timestamp, key, offer, offer_trade_volume):
        self.offer_timestamp_dict[key] = timestamp
        self.offer_dict[key] = offer
        self.offer_volume_dict[key] = offer_trade_volume
        self.updateCurrentPrice()

    def deleteEntry(self, key):
        self.offer_dict.pop(key, None)
        self.offer_volume_dict.pop(key, None)
        self.offer_timestamp_dict.pop(key, None)
        self.bid_dict.pop(key, None)
        self.bid_volume_dict.pop(key, None)
        self.bid_timestamp_dict.pop(key, None)

    def updateCurrentPrice(self):
        self.current_bid = 0
        self.bid_orderbook_length = 0
        for key in self.bid_dict:
            if self.bid_volume_dict[key] == 0:
                print(datetime.now(), 'Deleting key as volume is set to zero')
                self.delete_list.append(key)
            else:
                self.bid_orderbook_length += 1
                if self.bid_dict[key] > self.current_bid:
                    self.current_bid = self.bid_dict[key]
                    self.current_bid_volume = self.bid_volume_dict[key]
                    self.current_bid_time_stamp = self.bid_timestamp_dict[key]
                    self.current_bid_key = key
                    if self.time_stamp < self.current_bid_time_stamp:
                        self.time_stamp = self.current_bid_time_stamp
        if self.time_stamp != self.current_bid_time_stamp:
            self.bid_tick = 0
        else:
            self.bid_tick = 1

        self.current_offer = sys.maxsize
        self.offer_orderbook_length = 0
        for key in self.offer_dict:
            if self.offer_volume_dict[key] == 0:
                print(datetime.now(), 'Deleting key as volume is set to zero')
                self.delete_list.append(key)
            else:
                self.offer_orderbook_length += 1
                if self.offer_dict[key] < self.current_offer:
                    self.current_offer = self.offer_dict[key]
                    self.current_offer_volume = self.offer_volume_dict[key]
                    self.current_offer_time_stamp = self.offer_timestamp_dict[key]
                    if self.time_stamp < self.current_offer_time_stamp:
                        self.time_stamp = self.current_offer_time_stamp

        if self.time_stamp != self.current_offer_time_stamp:
            self.offer_tick = 0
        else:
            self.offer_tick = 1

        self.current_spread = self.current_bid - self.current_offer
        for idx, key in enumerate(self.delete_list):
            self.deleteEntry(key)
        self.delete_list.clear()

        # print('\r', datetime.now(),'Bid/Offer:',self.current_bid, '/', self.current_offer,'Tick', self.bid_tick, self.offer_tick , end='')
        #print('\r', datetime.now(), 'Bid:',self.current_bid, 'Bid Volume', self.current_bid_volume,'Tick', self.bid_tick,end='')

    # print('\r', datetime.now(), ': Bid', self.current_bid, '->', self.current_bid_volume, 'Depth', self.bid_orderbook_length,
        #      ' Offer', self.current_offer, self.current_offer_volume, 'Depth', self.offer_orderbook_length, end='')



class IndicatorTypes(Enum):
    Undefined = auto()
    Aroon = auto()
    HullMovingAverage = auto()
    RSI = auto()
    STD = auto()
    Stochastic = auto()
    VZO = auto()
    ochl = auto()
    HeikenAshi = auto()
    ATR = auto()
    MovingHeikenAshi = auto()
    TargetPrice = auto()
    ADX = auto()
    WeightedAverage = auto()
    PriceVolumeFootPrint = auto()
    VolumeProfile =  auto()
    MACD = auto()
    ExponentiallyWeightedAverage = auto()
    OrderbookHeatMap = auto()
    SwingValues = auto()

class Indicator():
    def __init__(self, indicator_windows, stockconfig, traderconfig, price_engine=PriceEnginesTimeEpoch.ticker, rounding_multiple = None):
        self._indicator_windows = indicator_windows
        self._stockconfig = stockconfig
        self.indicator_type = IndicatorTypes.Undefined
        self.price_engine_id = price_engine
        self.maxbufferlen = traderconfig.historyWindow
        self.rounding_multiple = rounding_multiple
        self.training_price = None
        self.indicatorID = '{'+str(uuid.uuid1()) + '}'


    def round_price(self, number):
        if self.rounding_multiple == None:
            return number
        else:
            return round(number / self.rounding_multiple) * self.rounding_multiple

    def addPrice(self, priceBuffer):
        pass

    def resetIndicator(self):
        pass
    def exportIndicatorData(self, df, path=''):
        pass

    def exportIndicatorTrainingData(self, path, tradeType, trainingID):
        print(datetime.now(), 'Export Indicator Data to ', path + str(tradeType)+'-'+self.indicatorID+'-{'+str(trainingID)+'}')
        pass
    def shift(self, arr, num, fill_value=np.nan):
        result = np.empty_like(arr)
        if num > 0:
            result[:num] = fill_value
            result[num:] = arr[:-num]
        elif num < 0:
            result[num:] = fill_value
            result[:num] = arr[-num:]
        else:
            result[:] = arr
        return result

    def average(self, data, period):
        if len(data) <= period:
            n = len(data)
        else:
            n = int(period)
        slice = data[-n:]

        return np.average(slice)

    def ewm(self, data, alpha, period):
        if len(data) <= period:
            n = len(data)
        else:
            n = int(period)
        slice = data[-n:]

        weights = (1-alpha)**np.arange(n)
        weights /= weights.sum()
        return np.dot(slice, weights[::-1]) / weights.sum()

    def weightedAverage(self, data, period):
        if len(data) <= period:
            n = len(data)
        else:
            n = int(period)
        slice = data[-n:]
        weights = np.arange(1, n+1)
        return np.dot(slice, weights) / weights.sum()

    def realTimeStd(self, data, period):
        if len(data) <= period:
            n = len(data)
            slice = data
        else:
            n = int(period)
            slice = data[-n-1:-1]
        return np.std(slice)

    def realTimeMax(self, data, period):
        if len(data) <= period:
            n = len(data)
            slice = data
        else:
            n = int(period)
            slice = data[-n-1:-1]
        return np.max(slice)

    def realTimeMin(self, data, period):
        if len(data) <= period:
            n = len(data)
            slice = data
        else:
            n = int(period)
            slice = data[-n-1:-1]
        return np.min(slice)

    def realTimeSum(self, data, period):
        if len(data) <= period:
            n = len(data)
            slice = data
        else:
            n = int(period)
            slice = data[-n-1:-1]
        return np.sum(slice)
    def real_time_volume_sum(self, data):
        period = 6
        if len(data) <= 1:
            return [0]
        if len(data) <= period:
            n = len(data)
            slice = data
        else:
            n = int(period)
            slice = data[-n-1:-1]
        gradient = np.gradient(slice)
        return gradient

class TradingEvent(Enum):
    DoNothing = auto()
    InitiateOpenTrade = auto()
    TriggerOpenTrade = auto()
    ResetOpenTrade = auto()
    SendOpenTrade = auto()
    OpenTradeSent = auto()
    OpenTradeResent = auto()
    InitiateCloseTrade = auto()
    TriggerCloseTrade = auto()
    CloseTrade = auto()
    FastCloseTrade = auto()
    ForceClosePosition = auto()
    SendCloseTrade = auto()
    CloseTradeSent = auto()
    TradeConfirmed = auto()
    RejectTrade = auto()
    ForceOpenPosition = auto()
    CloseTradeResent = auto()

class TradingState(Enum):
    Reset = auto()
    AwaitingOpenTrigger = auto()
    AwaitingSendOpenTrade = auto()
    AwaitingOpenTradeConfirmation = auto()
    Open = auto()
    AwaitingCloseTrigger = auto()
    AwaitingSendCloseTrade = auto()
    AwaitingCloseTradeConfirmation = auto()
    Closed = auto()
    Rejected = auto()

class TraderConfig:
    def __init__(self, riskRewardRatio, noLossTradingCostMultiple, timeEpoch, tradeTimeUnit, historyWindow, tradeEMAFilters,
                 hullFilters, stochasticFilters, macdFilters, aroonFilters, ATRFilters, ADXFilters, VZOFilters,
                 StdFilters, RSIFilters, OHLCFilters, TradeExitFilters, WAFilters, indicatorsource ):
        self.risk_reward_ratio = riskRewardRatio
        self.no_loss_trading_cost_multiple = noLossTradingCostMultiple
        self.timeEpoch = timeEpoch
        self.tradeTimeUnit = tradeTimeUnit
        self.historyWindow = historyWindow
        self.tradeEMAFilters = tradeEMAFilters
        self.hullFilters = hullFilters
        self.stochasticFilters = stochasticFilters
        self.macdFilters = macdFilters
        self.aroonFilters = aroonFilters
        self.ATRFilters = ATRFilters
        self.ADXFilters = ADXFilters
        self.VZOFilters = VZOFilters
        self.StdFilters = StdFilters
        self.RSIFilters = RSIFilters
        self.OHLCFilters = OHLCFilters
        self.tradeExitFilters = TradeExitFilters
        self.WAFilters = WAFilters
        self.indicatorSource = indicatorsource




class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {'__ndarray__': True, 'training_data': obj.tolist()}
        return JSONEncoder.default(self, obj)

def ndarray_decoder(dct):
    if '__ndarray__' in dct:
        return np.array(dct['training_data'])
    return dct

class FileHandler:
    @staticmethod
    def write_dictionary_to_file(tradeType, data_dictionary, path ):
        uid = data_dictionary['uid']
        file_name = path + str(tradeType) + '-' + str(data_dictionary['training_data']['Label'] != 'None') + '-' + '{' + uid + '}' + '.json'

        try:
            with open(file_name, 'w') as file:
                json.dump(data_dictionary['training_data'], file, cls=NumpyArrayEncoder)
            return file_name
        except Exception as e:
            print(datetime.now(), ' Exception in FileHandler -> write_dictionary_to_file: ', e)
            return None

    @staticmethod
    def read_dictionary_from_file(file_name):
        if not os.path.exists(file_name) or os.path.getsize(file_name) == 0:
            print(f"{datetime.now()} - File not found or empty: {file_name}")
            return None

        try:
            with open(file_name, 'r') as file:
                data_dictionary = json.load(file, object_hook=ndarray_decoder)
            return data_dictionary
        except Exception as e:
            print(datetime.now(), ' Exception in FileHandler -> read_dictionary_from_file: ', e)
            return None


class VolumeProfileANN:
    def __init__(self):
        self.model = None
        self.X_train_price = []
        self.X_train_vp = []
        self.X_train_gradient=[]
        self.Y_train = []

    def read_training_data(self, file_path):
        data_dictionary = FileHandler.read_dictionary_from_file(file_path)

        date = data_dictionary.get('Date')
        label = data_dictionary.get('Label')
        price = data_dictionary.get('Price')
        profiles = []
        gradient = []

        for indicator in data_dictionary.get('Indicators', []):
            if indicator['Type'].startswith('Volume Profile VolumeProfile['):
                profiles.append(indicator['Data'])

        gradient = data_dictionary.get('Volume Gradient', [])

        return label, price, profiles, gradient

    def prepare_vpdata_for_ann(self, instruction, trading, base):
        max_modes = 5
        vp_instruction = []
        for idx in range(0,max_modes):
            if idx < len(instruction):
                vp_instruction.append(instruction[idx])
            else:
                vp_instruction.append([0,0,0,0])
        vp_trading = []
        for idx in range(0,max_modes):
            if idx < len(trading):
                vp_trading.append(trading[idx])
            else:
                vp_trading.append([0,0,0,0])
        vp_base = []
        for idx in range(0,max_modes):
            if idx < len(base):
                vp_base.append(base[idx])
            else:
                vp_base.append([0,0,0,0])
        return [vp_instruction, vp_trading, vp_base]

    def prepare_gradient_statistics_for_ann(self, gradient):
        bid_gradient = np.array(gradient)[:, 0]
        offer_gradient = np.array(gradient)[:, 1]
        bid_average = np.average(bid_gradient)
        offer_average = np.average(offer_gradient)
        bid_stddev = np.std(bid_gradient)
        offer_stddev = np.std(offer_gradient)
        return [bid_average, bid_stddev, offer_average, offer_stddev]

    def extract_labels(self, trade_direction, label):

        if trade_direction>0:
            if label == 'long':
                label = [1, 0]   # Trade long
            else:
                label = [0, 1]  # Dont Trade long
        else:
            if label == 'short':
                label = [1,0] # Trade short
            else:
                label = [0,1] # Dont Trade short
        return label

    def set_single_feature(self, label, price, vp_instruction, vp_trading, vp_base, gradient):
        self.X_train_price = []
        self.X_train_vp = []
        self.X_train_gradient = []
        self.Y_train = []

        vp_data = self.prepare_vpdata_for_ann(vp_instruction, vp_trading, vp_base)
        gradient_stats = self.prepare_gradient_statistics_for_ann(gradient)

        self.Y_train.append(label)
        self.X_train_vp.append(vp_data)
        self.X_train_gradient.append(gradient_stats)
        self.X_train_price.append(price)
        return self.create_normalised_features(self.X_train_price, self.X_train_vp, self.X_train_gradient, self.Y_train)

    def load_features(self, trade_direction, train_data_dir='../Training/'):
        # Get a list of all the training data files in the directory
        train_data_paths = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir) if f.endswith('.json')]
        random.shuffle(train_data_paths)
        # label, features = readTrainingData('../Training/long-{fc12f2f0-c19d-11ed-9736-8e6e30f8fc30}.csv')
        self.X_train_price = []
        self.X_train_vp = []
        self.X_train_gradient = []
        self.Y_train = []

        for path in train_data_paths:
            label, price, profiles, gradient = self.read_training_data(path)

            label = self.extract_labels(trade_direction, label)
            vp_data = self.prepare_vpdata_for_ann(profiles[0], profiles[1], profiles[2])
            gradient_stats = self.prepare_gradient_statistics_for_ann(gradient)

            self.Y_train.append(label)
            self.X_train_vp.append(vp_data)
            self.X_train_gradient.append(gradient_stats)
            self.X_train_price.append(price)


    def create_normalised_features(self, X_train_price, X_train_vp, X_train_gradient, Y_train):
        # create_normalised_features the price across all volme profiles
        X_train = []

        X_train_vp=np.array(X_train_vp)
        min_val = np.min(X_train_vp[:, :, :, 0][X_train_vp[:, :, :, 0] > 0])
        max_val = np.max(X_train_vp[:, :, :, 0])

        range_val = max_val - min_val
        if range_val>0:
            X_train_vp[:, :, :, 0] = X_train_vp[:, :, :, 0] - min_val
            X_train_vp[:, :, :, 0] = X_train_vp[:, :, :, 0] / range_val
            X_train_vp[:, :, :, 0][X_train_vp[:, :, :, 0] < 0] = 0

            X_train_price = np.array(X_train_price)
            X_train_price = X_train_price - min_val
            X_train_price = X_train_price / range_val
            X_train_price[X_train_price < 0] = 0

        else:
            X_train_vp[:, :, :, 0] = 0
            X_train_price = 0

        # create_normalised_features the standard distribution and weight per trading profile

        for idx in range(0, X_train_vp.shape[1]):
            for n in range(1,4):
                min_val = np.min(X_train_vp[:, idx, :, n][X_train_vp[:, idx, :, n] > 0])
                max_val = np.max(X_train_vp[:, idx, :, n])
                range_val = max_val - min_val
                if range_val>0:
                    X_train_vp[:, idx, :, n] = X_train_vp[:, idx, :, n] - min_val
                    X_train_vp[:, idx, :, n] = X_train_vp[:, idx, :, n] / range_val
                    X_train_vp[:, idx, :, n][X_train_vp[:, idx, :, n] < 0] = 0
                else:
                    X_train_vp[:, idx, :, n] = 0


        # create_normalised_features gradient per feature. Normalisation is based on max value
        X_train_gradient = np.array(X_train_gradient)
        for idx in range(0, X_train_gradient.shape[0]):
            max_val = np.max([X_train_gradient[idx, 0],X_train_gradient[idx, 2]])
            if max_val > 0:
                X_train_gradient[idx, 0] = X_train_gradient[idx, 0] / max_val
                X_train_gradient[idx, 2] = X_train_gradient[idx, 2] / max_val
                max_stdev = np.max([X_train_gradient[idx, 1],X_train_gradient[idx, 3]])
                if max_stdev > 0:
                    X_train_gradient[idx, 1] = X_train_gradient[idx, 1] / max_stdev
                    X_train_gradient[idx, 3] = X_train_gradient[idx, 3] / max_stdev
                else:
                    X_train_gradient[idx, 1] = 0
                    X_train_gradient[idx, 3] = 0
            else:
                X_train_gradient[idx]=0


        for idx in range(0, X_train_price.shape[0]):
            feature = []
            feature.append(X_train_price[idx])
            for vp_idx in range(0, X_train_vp.shape[1]):
                for vp_dist_idx in range(0, X_train_vp.shape[2]):
                    for vp_dist_stats_idx in range(0, X_train_vp.shape[3]):
                        feature.append(X_train_vp[idx, vp_idx, vp_dist_idx, vp_dist_stats_idx])
            for grad_idx in range(0, 4):
                feature.append(X_train_gradient[idx][grad_idx])
            X_train.append(feature)


        return np.array(X_train), np.array(Y_train)


    def train(self, trade_direction, X_train, Y_train, hidden_layers=(30, 20, 8), epochs=50, batch_size=32, validation_split=0.1):
        num_inputs = len(X_train[0])
        num_outputs = len(Y_train[0])

        # Define the neural network architecture
        self.model = Sequential()
        self.model.add(Dense(hidden_layers[0], input_dim=num_inputs, activation='relu'))
        for layer in range(1, len(hidden_layers)):
            self.model.add(Dense(hidden_layers[layer], activation='relu'))
        self.model.add(Dense(num_outputs, activation='softmax'))

        # Compile the model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        if trade_direction > 0:
            self.model.save(f'buy-{hidden_layers[0]}-{hidden_layers[1]}-{hidden_layers[2]}.h5')
        else:
            self.model.save(f'sell-{hidden_layers[0]}-{hidden_layers[1]}-{hidden_layers[2]}.h5')

    def evaluate(self, trade_direction, X_test, Y_test, hidden_layers=(30, 20, 8)):
        if trade_direction > 0:
            self.model = load_model(f'buy-{hidden_layers[0]}-{hidden_layers[1]}-{hidden_layers[2]}.h5')
        else:
            self.model = load_model(f'sell-{hidden_layers[0]}-{hidden_layers[1]}-{hidden_layers[2]}.h5')

        loss, accuracy = self.model.evaluate(X_test, Y_test, batch_size=32)
        return loss, accuracy

    def predict(self, X_test):
        return self.model.predict(X_test, verbose=0)