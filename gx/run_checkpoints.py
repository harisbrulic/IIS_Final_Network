import sys
import os
import great_expectations as gx

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GX_DIR = os.path.join(ROOT_DIR, "gx")

context = gx.get_context(context_root_dir=GX_DIR)

try:
    context.delete_datasource("cyber_data")
except:
    pass

datasource = context.sources.add_pandas_filesystem(
    name="cyber_data",
    base_directory=os.path.join(ROOT_DIR, "data")
)

datasource.add_csv_asset(
    name="normal_data",
    batching_regex="preprocessed\\\\normal\\.csv"
)

datasource.add_csv_asset(
    name="attacks_data",
    batching_regex="preprocessed\\\\attacks\\.csv"
)

datasource.add_csv_asset(
    name="threats_data",
    batching_regex="preprocessed\\\\threats\\.csv"
)

checkpoints = ["normal_checkpoint", "attacks_checkpoint", "threats_checkpoint"]

all_passed = True

for checkpoint_name in checkpoints:
    try:
        checkpoint = context.get_checkpoint(checkpoint_name)
        result = checkpoint.run()
        
        if result["success"]:
            print(f"{checkpoint_name} done")
        else:
            print(f"{checkpoint_name} fail")
            all_passed = False
            
    except Exception as e:
        print(f"{e}")

context.build_data_docs()

if all_passed:
    print("Done")
    sys.exit(0)
else:
    print("Fail")
    sys.exit(1)