import sys
import great_expectations as gx

context = gx.get_context()

datasource = context.get_datasource("cyber_data")

checkpoints = ["normal_checkpoint", "attacks_checkpoint", "threats_checkpoint"]

all_passed = True

for checkpoint_name in checkpoints:
    checkpoint = context.get_checkpoint(checkpoint_name)
    result = checkpoint.run()
    
    if result["success"]:
        print(f"{checkpoint_name} done")
    else:
        print(f"{checkpoint_name} fail")
        all_passed = False

context.build_data_docs()

if all_passed:
    print("Done")
    sys.exit(0)
else:
    print("Fail")
    sys.exit(1)