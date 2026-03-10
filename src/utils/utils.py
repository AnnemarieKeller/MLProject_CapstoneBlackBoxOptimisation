
import os
from datetime import datetime

def get_latest_pdf_folder(base_folder="analysis/Functions/reports/"):
    # Step 1: latest week folder
    week_folders = [
        os.path.join(base_folder, f)
        for f in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, f))
    ]
    if not week_folders:
        raise FileNotFoundError(f"No week folders found in {base_folder}")
    latest_week_folder = max(week_folders, key=lambda f: os.path.getctime(f))
    
    # Step 2: latest date folder inside week folder
    date_folders = [
        os.path.join(latest_week_folder, f)
        for f in os.listdir(latest_week_folder)
        if os.path.isdir(os.path.join(latest_week_folder, f))
    ]
    if not date_folders:
        raise FileNotFoundError(f"No date folders found in {latest_week_folder}")
    latest_date_folder = max(date_folders, key=lambda f: os.path.getctime(f))
    
    # Step 3: reports folder
    reports_folder = os.path.join(latest_date_folder, "reports")
    if not os.path.exists(reports_folder):
        raise FileNotFoundError(f"No 'reports' folder in {latest_date_folder}")
    
    return reports_folder



def get_pdf_paths_from_folder(folder_path):
    # List PDFs in the folder (non-recursive)
    pdfs = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
            if f.lower().endswith(".pdf")]
    if not pdfs:
        print(f"No PDFs found in {folder_path}")
    return pdfs


# src/utils/utils.py
import os
from datetime import datetime

def generate_output_path(base_folder="src/analysis_outputs"):
    """
    Generates a timestamped folder for storing analysis outputs,
    keeping them separate from raw PDFs.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(base_folder, timestamp)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder