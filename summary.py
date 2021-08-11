import argparse
import os
import re

import pandas as pd

dict_type = {
    "dnnl": {
        "start": "dnnl_verbose",
        "name": "dnnl_verbose",
        "len": 11,
        "header": ["dnnl_verbose", "action", "eng", "name", "impl", "prop", "format", "blank1", "blank2", "shape", "time"]
    },
    "graph": {
        "name": "dnnl_graph_verbose",
        "len": 9,
        "header": ["dnnl_verbose", "name", "eng", "impl", "op", "data", "format", "backend", "time"]
    }
}


def preprocess(file, verbose_type):
    with open(file) as f:
        content = f.read().splitlines()
    reorders = []
    for i, line in enumerate(content):
        if line.startswith(dict_type[verbose_type]["start"]) and len(line.split(',')) == dict_type[verbose_type]["len"]:
        # if line.startswith(("dnnl_graph_verbose", "dnnl_verbose")) and len(line.split(',')) == 11:
            reorder = line.split(",")
            #print(reorder[1:-1])
            # assert len(reorder) == 11, "Please check the verbose format of:\nOP that leads to the reorder: %s\nThe reorder verbose: %s" % (content[i-1], line)
            reorders.append(reorder)
    df = pd.DataFrame(reorders, columns=dict_type[verbose_type]["header"])
    return df


def main(file_name, verbose_type):
    df = preprocess(file_name, verbose_type)
    df["time"] = df["time"].astype(float)
    result =[]
    #chang you case number for one iteration
    loop_number = 56
    for i in range(len(df["shape"])):
        match = re.split('(\d+)', df["shape"][i])
        shapes = [int(i) for i in match[1:-1:2]]
        if i < 56:
            result.append([df["name"][i], shapes, df["time"][i], 1])
        else:
            assert result[i%56][0] == df["name"][i] and result[i%56][1] == shapes, "shapes need equal after one iteration"
            result[i%56][2] += df["time"][i]
            result[i%56][3] +=1
    for i in range(loop_number):
        if result[i][0] == "convolution":
            r = result[i][1] + [result[i][2]]
            print(r[:])

    #df_groupby_name = df.groupby("name").sum().sort_values(by="time", ascending=False)

    #print(df_groupby_name)
    #df_groupby_name.to_csv(file_name + ".csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_name", default=None, type=str, required=True, help="path to the input onednn log file")
    parser.add_argument("-t", "--verbose_type", default="dnnl", type=str, choices=["dnnl", "graph"] ,required=True, help="dnnl or graph verbose")
    args = parser.parse_args()

    df = main(args.file_name, args.verbose_type)
