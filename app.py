from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello from Fish Judge yay!'


def main():
    pass


if __name__ == '__main__':
    main()
