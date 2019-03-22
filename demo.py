import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
import pandas as pd

# function for plotting based on both features
from tensorflow.python.training import saver


def plot_acc_ttf_data(Acousticdata, timeFailuer,
                      title="Acoustic data and time to failure: 1% sampled data"):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    plt.title(title)
    plt.plot(Acousticdata, color='r')
    ax1.set_ylabel('acoustic data', color='r')
    plt.legend(['acoustic data'], loc=(0.01, 0.95))

    ax2 = ax1.twinx()
    plt.plot(timeFailuer, color='b')
    ax2.set_ylabel('time to failure', color='b')
    plt.legend(['time to failure'], loc=(0.01, 0.9))
    plt.grid(True)
    del Acousticdata
    del timeFailuer

    plt.show()


if __name__ == "__main__":
    # ---------------------------- Bringing the data--------------------------------------

    pd.set_option("display.precision", 13)
    dataFeatures = pd.read_csv("dengue_features_train.csv", nrows=1)
    dataLables = pd.read_csv("dengue_labels_train.csv", nrows=1)

    dataFeatures = dataFeatures.dropna(axis=0, how='all', thresh=None, subset=None, inplace=False)
    dataLables = dataLables.dropna(axis=0, how='all', thresh=None, subset=None, inplace=False)

    InputX = dataFeatures.loc[:,
             ['city', 'year', 'weekofyear', 'week_start_date', 'ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw',
              'precipitation_amt_mm', 'reanalysis_air_temp_k', 'reanalysis_avg_temp_k',
              'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
              'reanalysis_precip_amt_kg_per_m2', 'reanalysis_relative_humidity_percent'
                 , 'reanalysis_sat_precip_amt_mm', 'reanalysis_specific_humidity_g_per_kg',
              'reanalysis_tdtr_k', 'station_avg_temp_c',
              'station_diur_temp_rng_c', 'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm']
             ].values

    Lable = dataLables.loc[:, ['city', 'year', 'weekofyear', 'total_cases']].values

    print(InputX.size)
    print('\n', Lable)

    # plot_acc_ttf_data(Acousticdata, timeFailuer, title="Acoustic data and time to failure: 1% sampled data")

    # -------------------------- Start buildin and training my model ----------------------

    display_step = 10  # to split the display

    # Create graph
    tf.reset_default_graph()
    sess = tf.Session()

    # make results reproducible
    seed = 13
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Declare batch size
    batch_size = 50

    # Initialize placeholders
    x_data = tf.placeholder(shape=(629145480, None), dtype=tf.float32)
    y_target = tf.placeholder(shape=(629145480, None), dtype=tf.float32)

    # Create variables for linear regression
    A = tf.Variable(tf.random_normal(shape=[1, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    # Declare model operations
    model_output = tf.add(tf.matmul(x_data, A), b)

    # Declare the elastic net loss function
    elastic_param1 = tf.constant(1.)
    elastic_param2 = tf.constant(1.)

    l1_a_loss = tf.reduce_mean(tf.abs(A))
    l2_a_loss = tf.reduce_mean(tf.square(A))

    e1_term = tf.multiply(elastic_param1, l1_a_loss)
    e2_term = tf.multiply(elastic_param2, l2_a_loss)

    # define the loss function and the optimazer
    loss = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), e1_term), e2_term), 0)
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # Initialize variabls and tensorflow session
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    # Training loop
    loss_vec = []
    with sess:
        for i in range(100):
            sess.run(optimizer, feed_dict={x_data: Acousticdata, y_target: timeFailuer})
            if (i) % display_step == 0:
                temp_loss = sess.run(loss, feed_dict={x_data: Acousticdata, y_target: timeFailuer})
                loss_vec.append(temp_loss[0])
                print("Training step:", '%04d' % (i), "cost=", temp_loss)
                save_path = saver.save(sess, "/home/benarousfarouk/Desktop/IA/Competitions/Earthquake Prediction("
                                             "Kaggle)/models/model.ckpt")

        # answer = tf.equal(tf.floor(model_output + 0.1), y_target)
        # accuracy = tf.reduce_mean(tf.cast(answer, "float32"))
        # print(accuracy.eval(feed_dict={x_data: Acousticdata, y_target: timeFailuer}, session=sess))

        # answer = sess.run(model_output, feed_dict={x_data:12})
        # print( answer)»:_;

    # loss = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), e1_term), e2_term), 0)
