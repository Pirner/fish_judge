from flask import Flask, request, abort
import git
import json


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello from Fish Judge yay 2!'


@app.route('/update_server', methods=['POST'])
def webhook():
    if request.method != 'POST':
        return 'OK'

    repo = git.Repo('/home/Nerolith/mysite/fish_judge')
    origin = repo.remotes.origin

    pull_info = origin.pull()

    if len(pull_info) == 0:
        return json.dumps({'msg': "Didn't pull any information from remote!"})
    if pull_info[0].flags > 128:
        return json.dumps({'msg': "Didn't pull any information from remote!"})

    commit_hash = pull_info[0].commit.hexsha
    build_commit = f'build_commit = "{commit_hash}"'
    print(f'{build_commit}')
    return 'Updated PythonAnywhere server to commit {commit}'.format(commit=commit_hash)


def main():
    pass


if __name__ == '__main__':
    main()
