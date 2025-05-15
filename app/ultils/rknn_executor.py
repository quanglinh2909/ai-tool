from rknnlite.api import RKNNLite


class RKNN_model_container():
    check = True

    def __init__(self, model_path, target=None, device_id=None,stt=0) -> None:
        rknn = RKNNLite()

        # Direct Load RKNN Model
        rknn.load_rknn(model_path)
        t = RKNNLite.NPU_CORE_0_1_2
        if stt % 3 == 0:
            t = RKNNLite.NPU_CORE_0
        elif stt % 3 == 1:
            t = RKNNLite.NPU_CORE_1
        elif stt % 3 == 2:
            t = RKNNLite.NPU_CORE_2

        print('--> Init runtime environment')
        ret = rknn.init_runtime(core_mask=t)

        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

        self.rknn = rknn

    # def __del__(self):
    #     self.release()

    def run(self, inputs):
        if self.rknn is None:
            print("ERROR: rknn has been released")
            return []

        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)

        return result

    def release(self):
        self.rknn.release()
        self.rknn = None