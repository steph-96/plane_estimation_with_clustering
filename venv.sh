env_dir=./venv

if [ ! -d "$env_dir" ]; then
    python3 -m pip install --user virtualenv
    python3 -m virtualenv "$env_dir"
    source ./venv/bin/activate
    pip install -r requirements.txt
fi