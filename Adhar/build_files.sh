echo "BUILD START"

# Install dependencies
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Collect static files (ensure correct directory exists)
python3 manage.py collectstatic --noinput --clear

echo "BUILD END"
