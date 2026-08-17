import os
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_compile_cache"
os.makedirs("/tmp/torch_compile_cache", exist_ok=True)

from bottle import route, run, request, response, static_file
from learn_nn.models.mini_llm.train.pretrain import PreTrainWorker

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static", "timllm")

print("Loading model...")
worker = PreTrainWorker()
print("Model loaded.")


@route("/")
def index():
    return static_file("index.html", root=STATIC_DIR)


@route("/api/inference", method="POST")
def inference():
    data = request.json
    if not data or "input" not in data:
        response.status = 400
        return {"error": "Missing 'input' field"}

    input_str = data["input"]
    if isinstance(input_str, str):
        input_list = [input_str]
    elif isinstance(input_str, list):
        input_list = input_str
    else:
        response.status = 400
        return {"error": "'input' must be str or list[str]"}

    result = worker.inference(input_list)
    return {"result": result}


@route("/health", method="GET")
def health():
    return {"status": "ok"}


def main():
    run(host="0.0.0.0", port=3003, quiet=True)


if __name__ == "__main__":
    main()