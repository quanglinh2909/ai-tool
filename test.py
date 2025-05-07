import multiprocessing
import numpy as np
import time


def read_data(data):
    while True:
        # Convert shared array to numpy array
        data_np = np.frombuffer(data.get_obj(), dtype=np.float64).reshape(-1, 2)

        # Create a list of dictionaries to print
        data_box = [{"x": data_np[i][0], "y": data_np[i][1]} for i in range(len(data_np))]
        print(data_box)

        time.sleep(1)


if __name__ == '__main__':
    # Initial data
    data_box = [
        {"x": 41.73913043478261, "y": 426.6322314049587},
        {"x": 290.18633540372673, "y": 160.8471074380165},
        {"x": 449.1925465838509, "y": 83.49173553719007},
        {"x": 616.1490683229814, "y": 124.15289256198346},
        {"x": 595.27950310559, "y": 456.38429752066116}
    ]

    # Convert data to numpy array
    shared_data = np.array([[p["x"], p["y"]] for p in data_box], dtype=np.float64)
    shared_array = multiprocessing.Array('d', shared_data.flatten())  # 'd' is float64

    # Create and start the process
    process1 = multiprocessing.Process(target=read_data, args=(shared_array,))
    process1.start()

    time.sleep(5)

    # Update data in the main process
    new_data = [
        {"x": 100, "y": 200},
        {"x": 300, "y": 400},
        {"x": 500, "y": 600},
        {"x": 700, "y": 800},
        {"x": 900, "y": 1000}
    ]

    # Update the shared data array
    updated_data = np.array([[p["x"], p["y"]] for p in new_data], dtype=np.float64)
    np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.float64).reshape(-1, 2), updated_data)

    # Allow some time for the process to read the updated data
    time.sleep(5)

    process1.terminate()  # Terminate the process
    process1.join()  # Wait for the process to finish
