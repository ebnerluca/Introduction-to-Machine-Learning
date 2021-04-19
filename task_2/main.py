import pandas as pd
import numpy as np

output_file = "data/output/hand_in.csv"
output_zip = "hand_in.zip"

subtask_1_labels = np.genfromtxt("data/output/subtask_1_labels.csv", delimiter=",", skip_header=True)
subtask_2_labels = np.genfromtxt("data/output/subtask_2_labels.csv", delimiter=",", skip_header=True)
subtask_3_labels = np.genfromtxt("data/output/subtask_3_labels.csv", delimiter=",", skip_header=True)

subtask_2_labels = subtask_2_labels.reshape((subtask_2_labels.shape[0], 1))

total_output_header = ["pid", "LABEL_BaseExcess", "LABEL_Fibrinogen", "LABEL_AST", "LABEL_Alkalinephos",
                       "LABEL_Bilirubin_total", "LABEL_Lactate", "LABEL_TroponinI", "LABEL_SaO2",
                       "LABEL_Bilirubin_direct", "LABEL_EtCO2", "LABEL_Sepsis", "LABEL_RRate", "LABEL_ABPm",
                       "LABEL_SpO2", "LABEL_Heartrate"]

total_output_array = np.hstack((subtask_1_labels, subtask_2_labels, subtask_3_labels))

pd.DataFrame(total_output_array).to_csv(output_file, header=total_output_header, index=None)
pd.DataFrame(total_output_array).to_csv(output_zip, header=total_output_header, index=None)


