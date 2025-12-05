import matplotlib.pyplot as plt
from math import sqrt
from matplotlib import pyplot
import numpy as np
from astropy import units as u
from mw_plot import MWSkyMap
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
from pandas import Series
from pandas import concat
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from Ebola_Functions import *

def gen_visual():
    """
    Generate and save sky map images for selected galaxies.

    Returns
    -------
    figs : List of figures created for each galaxy.
    """
    names={
        "M31":3200,
        "M32":8800,
        "M33":10000,
    }
    figs=[]

    for name,radius in names.items():
        mw1 = MWSkyMap(
            center=name,
            radius=(radius, radius) * u.arcsec,
            background="Mellinger color optical survey",
        )

        fig, ax = plt.subplots(figsize=(5, 5))
        figs.append(fig)
        mw1.transform(ax)

        mw1.savefig(f'{name}galaxy.png')
    return figs

def plt_to_rgb(fig):
    """
    A function to transform a matplotlib to a 3d rgb np.array 

    Input
    -----
    fig: matplotlib.figure.Figure
        The plot that we want to encode.        

    Output
    ------
    np.array
    
    """
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))
    return rgba_arr[:, :, :3]

def encoding_with_rgb(figs, plot=True, red_value=150, green_value=150, blue_value=150, grey_value=230):
    """
    Convert figures to RGB arrays and label pixels as
    red, green, blue, or gray using threshold values.
    """
    results = []
    for i in range(len(figs)):
        img_array = plt_to_rgb(figs[i])

        fig_result = {
            "red": [],
            "green": [],
            "blue": [],
            "gray": []
        }
        # Red encoding
        red = img_array[:, :, 0]
        x, y = [], []
        for ig, row in enumerate(red):
            for ij, val in enumerate(row):
                if red_value <= val:
                    x.append(ij)
                    y.append(ig)
        if plot:
            plt.scatter(x, y, s=0.1, c="red")
        fig_result["red"] = list(zip(x, y))
        # Green encoding
        green = img_array[:, :, 1]
        x, y = [], []
        for ig, row in enumerate(green):
            for ij, val in enumerate(row):
                if green_value <= val <= 180:
                    x.append(ij) 
                    y.append(ig) 
        if plot:
            plt.scatter(x, y, s=0.1, c="green")
        fig_result["green"] = list(zip(x, y))
        # Blue encoding
        blue = img_array[:, :, 2]
        x, y = [], []
        for ig, row in enumerate(blue):
            for ij, val in enumerate(row):
                if blue_value <= val:
                    x.append(ij) 
                    y.append(ig)
        if plot:
            plt.scatter(x, y, s=0.1, c="blue")
        fig_result["blue"] = list(zip(x, y))
        # Gray encoding
        grey = np.sum(img_array[:, :, :] * np.array([0.299, 0.587, 0.114]), axis=2)
        x, y = [], []
        for ig, g in enumerate(grey):
            for ij, j in enumerate(g):
                if j > grey_value:
                    x.append(ij)
                    y.append(ig)
        fig_result["gray"] = list(zip(x, y))
        if plot:
            plt.title(f"Encoding result for figure {i+1}")
            plt.gca().invert_yaxis()
            plt.show() 

        results.append(fig_result)
    return results

def clustering_based_on_color(encoded_fig, plot=True):
    """
    KMeans clustering based on color.
    """
    color_to_code = {
        "red": 0,
        "green": 1,
        "blue": 2,
        "gray": 3
    }
    color_features = []
    pixel_positions = []

    for color in ["red", "green", "blue", "gray"]:
        pts = encoded_fig.get(color, [])
        code = color_to_code[color]
        for (x, y) in pts:
            color_features.append([code])     
            pixel_positions.append((x, y))    
    X = np.array(color_features)
    pos = np.array(pixel_positions)
    model = KMeans(n_clusters=4)
    labels = model.fit_predict(X)
    if plot:
        plt.figure(figsize=(5, 5))
        plt.scatter(pos[:, 0], pos[:, 1], c=labels, s=0.1)
        plt.title("K-means clustering based on color category")
        plt.gca().invert_yaxis()
        plt.show()

    return labels

def clustering(encoded_fig, plot=True):
    """
    Apply clustering to ONE encoded figure from encoding_with_rgb().

    Parameters
    ----------
    encoded_fig : dict
    method : "kmeans" or "gmm"
    plot : whether to show the scatter plot

    Returns
    -------
    labels : np.array of cluster labels
    """
    points = []
    for color in ["red", "green", "blue", "gray"]:
        pts = encoded_fig.get(color, [])
        for p in pts:
            points.append(p)

    assert len(points) != 0, "No points found."
    data = np.array(points, dtype=float)  
    model = KMeans(n_clusters=4)
    labels = model.fit_predict(data)
    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(data[:, 0], data[:, 1], c=labels, s=2)
        plt.title(f"Clustering method: kmeans on encoded pixels")
        plt.gca().invert_yaxis()
        plt.show()
    return labels

def overlay_clusters_on_image(fig, encoded_fig, labels):
    """
    Overlay clustering results on top of the original Milky Way figure.

    Parameters
    ----------
    fig : matplotlib Figure

    encoded_fig : dict

    labels : array

    point_size : int

    Output
    ------
    Shows a plot of the original image with cluster results overlaid.
    """

    img = plt_to_rgb(fig)

    points = []
    for color in ["red", "green", "blue", "gray"]:
        points.extend(encoded_fig.get(color, []))
    points = np.array(points)

    plt.figure(figsize=(7, 7))
    plt.imshow(img)
    plt.scatter(points[:, 0], points[:, 1], c=labels, s=0.1)
    plt.title("Cluster overlay on Milky Way image")
    plt.show()
# ------------------ Topic 2 code ------------------------------
def load_country_data(file_path):
    """
    Load epidemic data for one country.
    Parameters
    ----------
    file_path : str
        Path to the data file containing day index and case counts.

    Returns
    -------
    days : ndarray
    cases : ndarray
    """
    x_f, y_total, _ = read_file(file_path) #this function is from ebola_functions.py
    days = np.array(x_f).reshape(-1, 1)
    cases = np.array(y_total)               
    return days, cases

def linear_regression_country(file_path, name):
    """
    Fit and plot a simple linear regression model for epidemic data.

    Parameters
    ----------
    file_path : str
        Path to the country`s case data file.
    name : str
        Country name for plot labeling.

    Output
    ------
    Displays a scatter plot of real data and the fitted regression line.
    """
    days, cases = load_country_data(file_path)
    model = LinearRegression()
    model.fit(days, cases)

    y_pred = model.predict(days)

    plt.scatter(days, cases, label="Data")
    plt.plot(days, y_pred, label="Linear fit")
    plt.xlabel("Day")
    plt.ylabel("Cases")
    plt.title(f"Linear regression {name}")
    plt.legend()
    plt.grid()
    plt.show()

def poly_regression_country(file_path, name):
    """
    Fit and plot a polynomial regression model (degree 3) for epidemic data.

    Parameters
    ----------
    file_path : str
        Path to the country`s case data.
    name : str
        Country name for labeling.

    Output
    ------
    Displays a plot comparing real epidemic data and the polynomial curve.
    """
    days, cases = load_country_data(file_path)
    model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                      ('linear', LinearRegression(fit_intercept=False))])
    model.fit(days, cases)

    y_pred = model.predict(days)

    plt.scatter(days, cases, label="Data")
    plt.plot(days, y_pred, label="Linear fit")
    plt.xlabel("Day")
    plt.ylabel("Cases")
    plt.title(f"Polynomial regression {name}")
    plt.legend()
    plt.grid()
    plt.show()

def nn_model(file_path, name, epochs=500, extra_days=200):
    days, cases = load_country_data(file_path)
    x = days.reshape(-1, 1)
    y = cases.reshape(-1, 1)

    train_size = int(len(x) * 0.7)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train = y[:train_size]

    x_mean, x_std = x_train.mean(), x_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()

    x_train_n = (x_train - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std

    model = Sequential([
        Dense(64, input_dim=1, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train_n, y_train_n, epochs=epochs, verbose=1, shuffle=False)
    x_all_n = (x - x_mean) / x_std
    y_pred_all = model.predict(x_all_n).flatten() * y_std + y_mean
    plt.figure()
    plt.scatter(x, y, label='Original data', alpha=0.5)
    plt.plot(x, y_pred_all, color='red', label='NN prediction')
    plt.title(f"NN regression {name}")
    plt.xlabel("Day")
    plt.ylabel("Cases")
    plt.legend()
    plt.grid()
    plt.show()
    return model

#---------------------LSTM section-------------
 
def timeseries_to_supervised(data, lag=1):
    """
    Convert a time series into a supervised learning format.

    """
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df
 
def difference(dataset, interval=1):
    """
    Apply differencing to make a time series stationary.

    Returns
    -------
    Series
        The differenced time series.
    """
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)
 
def inverse_difference(history, yhat, interval=1):
    """
    Invert differencing by adding back the previous value.

    Returns
    -------
    float
        Restored forecast value on original scale.
    """
    return yhat + history[-interval]
 
def scale(train, test):
    """
    Scale train and test sets to the range [-1, 1].

    Returns
    -------
    scaler : MinMaxScaler
        Fitted scaler for later inverse transforms.
    train_scaled : ndarray
        Scaled training data.
    test_scaled : ndarray
        Scaled test data.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled
 
def invert_scale(scaler, X, value):
    """
    Invert scaling of a forecasted value.

    """
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]
 
def fit_lstm(train, batch_size, nb_epoch, neurons):
    """
    Fit an LSTM model to supervised training data.

    """
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(
        X, y,
        epochs=nb_epoch,
        batch_size=batch_size,
        verbose=1,
        shuffle=False
    )
    return model
 
def forecast_lstm(model, batch_size, X):
    """
    Make a one-step forecast using an LSTM model.

    Returns
    -------
    float
        Predicted value.
    """
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]
 
def lstm_nn_model(file_path,name):
    """
    Train an LSTM model on epidemic case data and forecast future values.

    Parameters
    ----------
    file_path : str
        Path to the data file for the selected country.
    name : str
        Name of the country for labeling the plot.

    Returns
    -------
        Displays a plot comparing actual values and LSTM predictions.
    """
    days, cases = load_country_data(file_path)
    days = np.asarray(days).reshape(-1)
    cases = np.asarray(cases).reshape(-1)
    series = Series(cases, index=days)

    raw_values = series.values
    diff_values = difference(raw_values, 1)
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values

    n_test = 20 
    train, test = supervised_values[:-n_test], supervised_values[-n_test:]
    scaler, train_scaled, test_scaled = scale(train, test)
    lstm_model = fit_lstm(train_scaled, 1, 200, 4)
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)
    
    predictions = list()
    for i in range(len(test_scaled)):
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
        yhat = invert_scale(scaler, X, yhat)

        yhat = inverse_difference(raw_values, yhat, n_test + 1 - i)
        predictions.append(yhat)
        expected = raw_values[len(train) + i + 1]
        print('Step=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))
    
    actual = raw_values[-n_test:]
    rmse = sqrt(mean_squared_error(actual, predictions))
    print('Test RMSE: %.3f' % rmse)
    pyplot.figure()
    plt.title(f"LSTM model of {name}")
    plt.plot(days, raw_values, label="Actual")
    prediction_days = days[-n_test:]
    plt.plot(prediction_days, predictions, color='red', label='LSTM forecast')
    pyplot.legend()
    pyplot.show()
