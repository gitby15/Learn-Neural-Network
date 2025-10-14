import flask

app = flask.Flask(__name__, template_folder='.')
app.config['TEMPLATES_AUTO_RELOAD'] = True
def _predict(image):
    # 空函数，在start的时候注入
    return -1

@app.route('/')
def index():    
    return flask.render_template('template.html')



    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


def Launch(predict_fuc):
    @app.route('/predict', methods=['POST'])
    def predict():
        # 入参是json，包含一个字段matrix，值是一个一维数组（只有一个颜色通道，每个元素的值是0-255）
        data = flask.request.json['matrix']
        result = predict_fuc(data)
        return flask.jsonify({'result': int(result)})
    app.run(host='0.0.0.0', port=5000)
    