import flask
import kagglehub
import os
import numpy as np

app = flask.Flask(__name__, template_folder=".")
app.config["TEMPLATES_AUTO_RELOAD"] = True

_datasetPath = kagglehub.dataset_download("hojjatk/mnist-dataset")
print("Path to dataset files:", _datasetPath)


def load_mnist_images(filename):
    with open(os.path.join(_datasetPath, filename, filename), "rb") as f:

        magic, num_images, rows, cols = np.frombuffer(
            f.read(16), dtype=np.uint32
        ).byteswap()

        # 读取图像数据, 现在是一个一维数组，元素个数=num_images*rows*cols=60000*28*28
        images = np.frombuffer(f.read(), dtype=np.uint8)

        # 将这个一维数组按照图片切分
        images = images.reshape(num_images, rows * cols)

        # print("Filename \"{}\", Images {}, Size {}x{}, Shape: {}".format(filename, num_images, rows, cols, images.shape))
        return images


@app.route("/")
def index():
    return flask.render_template("view_image.html")


if __name__ == "__main__":
    images = load_mnist_images("t10k-images-idx3-ubyte")

    @app.route("/get_image", methods=["POST"])
    def get_image():
        index = flask.request.json['index']
        if index < 0 or index >= images.shape[0]:
            return flask.jsonify({'result': 'error', 'message': 'index out of range'})
        image = images[index].tolist()  # 转成list，方便json化
        return flask.jsonify({'result': 'ok', 'image': image})

    app.run(host="0.0.0.0", port=5001)
