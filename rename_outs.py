import os
import sys

outs_labels = ['days', 'abs_error', 'types_replaced_eval', 'all_data',
        'granularities', 'range_dates', 'n_ids']

files = os.listdir("outs")
n_rows = {}
segments = {}
for file in files:
    try:
        root = file.split('.')[0]
        segment = int(root.split('_')[-1])
        n_row = int(root.split('_')[-2])
        data_type = root.split('_')[-3]
    except Exception as ex:
        print(ex)
        print(file)
        continue

    if data_type not in n_rows:
        n_rows[data_type] = []
        segments[data_type] = {}
    if n_row not in n_rows[data_type]:
        n_rows[data_type].append(n_row)
        segments[data_type][n_row] = []
    segments[data_type][n_row].append(segment)

for data_type in n_rows:
    for label in outs_labels:
        for n_row in n_rows[data_type]:
            for segment in segments[data_type][n_row]:
                if not os.path.exists(f"outs/{label}_{data_type}_{n_row}_{segment}.pickle"):
                    print(f"not os.path.exists(f'outs/{label}_{data_type}_{n_row}_{segment}.pickle')")
                else:
                    os.rename(
                        f"outs/{label}_{data_type}_{n_row}_{segment}.pickle",
                        f"outs/{label}_{data_type}_n_rows{n_row}_n{48}_{segment}.pickle"
                    )
                # print("rename ")
                # print(f"outs/{label}_{data_type}_{n_row}_{segment}.pickle")
                # print(f"-> outs/{label}_{data_type}_n_rows{n_row}_n{48}_{segment}.pickle")
