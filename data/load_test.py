import pandas as pd


unpickled_df = pd.read_pickle("./mydata.pkl")

# for index, row in unpickled_df.iterrows():
#     print("index", index)
#     print("row", row)


# print(unpickled_df.head)
apps = unpickled_df.iterrows()

count_row = unpickled_df.shape[0]
print("row count is: ", count_row)

# while True:
#     try:
#         row = next(apps)
#     except:
#         # print("re iterate")
#         apps = unpickled_df.iterrows()
#         row = next(apps)
        # print("row ", row)

    # print("row", row)
    # print("obs", row.obs)
    # print("action", row.action)
    # print("rew", row.rew)
    # print("done", row.done)