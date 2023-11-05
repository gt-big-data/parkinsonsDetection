import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf


def get_labels(pat_info: pd.DataFrame, test: pd.DataFrame):
  # create labels column
  appendable_col = []
  test = test.dropna()
  test_return = test
  for row in test.iterrows():
    pat_no = int(row[1][0])
    try:
      appendable_pat = pat_info.loc[pat_info["PATNO"] == pat_no]
      appendable_col.append(appendable_pat["COHORT"].values[0])
    except IndexError:
      test_return = test_return.drop(test.index[int(row[0])])
      continue
  srs = pd.Series(appendable_col, name='STATUS')
  return test_return, srs


def get_clock_model():
    participant_status_csv = "https://gist.githubusercontent.com/mikster36/bf6b3265b74f67b219f1aeccc5da683b/raw/2e6dbd8072330b92ba0d04ff74b238434ec732ad/Participant_Status.csv"
    clock_drawing_csv = "https://gist.githubusercontent.com/mikster36/fddbe005b64bfbe11edfd6ba659417f1/raw/16a072a79b3ba0f5118a2621f34aabb0fba27a4a/Clock_Drawing%2520-%2520Clock_Drawing.csv"
    ps_df = pd.read_csv(participant_status_csv)
    ps_df = ps_df[["PATNO", "COHORT"]]
    ps_df = ps_df[ps_df["COHORT"] <= 2]                                             # we only want Parkinson's (1) or Healthy (2) patients for now
    cd_df = pd.read_csv(clock_drawing_csv)
    cd_df = cd_df[["PATNO", "CLCKPII", "CLCK2HND", "CLCKNMRK", "CLCKNUIN", "CLCKALNU", "CLCKNUSP", "CLCKNUED", "CLCKTOT"]]
    data, labels = get_labels(ps_df, cd_df)
    data = data.drop(columns=["PATNO"])                                             # we don't want to pass the patient number as input for the model
    # data["CLCKTOT"] /= 7.0                                                          # normalise
    labels -= 1
    train_data, test_data, train_labels, test_labels = train_test_split(data,
                                                                    labels,
                                                                    test_size=0.3,
                                                                    random_state=2)
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(8, input_shape=(8,), activation='relu'),
                                    tf.keras.layers.Dense(8, activation='relu'),
                                    tf.keras.layers.Dense(8, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = ['accuracy'])
    history = model.fit(train_data, train_labels, batch_size=32, epochs=10)
    return model