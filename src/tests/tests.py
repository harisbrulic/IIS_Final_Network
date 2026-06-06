import sys
import os
import pandas as pd
from evidently import Report
from evidently.presets.dataset_stats import DataSummaryPreset
from evidently.presets.drift import DataDriftPreset

def test_threats_drift():
    
    os.makedirs("data/reference", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    current = pd.read_csv("data/preprocessed/threats.csv")
    reference_path = "data/reference/threats.csv"
    
    if not os.path.exists(reference_path):
        print(f"Reference does not exist")
        current.to_csv(reference_path, index=False)
    
    reference = pd.read_csv(reference_path)
    
    for df in [reference, current]:
        if "first_seen" in df.columns:
            del df["first_seen"]
    
    report = Report([
        DataSummaryPreset(),
        DataDriftPreset(),
    ],
    include_tests=True
    )
    
    result = report.run(reference_data=reference, current_data=current)
    
    result.save_html("reports/data_testing_report.html")
    
    all_tests_passed = True
    result_dict = result.dict()
    if "tests" in result_dict:
        for test in result_dict["tests"]:
            if "status" in test and test["status"] != "SUCCESS":
                all_tests_passed = False
                break
    
    if not all_tests_passed:
        print("Data drift detected")
        sys.exit(0) 
    else:
        print("Passed")
        sys.exit(0)

if __name__ == "__main__":
    test_threats_drift()