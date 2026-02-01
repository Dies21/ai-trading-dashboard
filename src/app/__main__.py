import sys
from pathlib import Path

# Додати src до path для імпортів модулів
streamlit_script = Path(__file__).resolve()
src_dir = streamlit_script.parent.parent  # src/app/dashboard.py -> src/
sys.path.insert(0, str(src_dir))
