# clause_detector.py
"""This file will take a contract or a set of contracts and tell you if it contains a clause or not.

This is designed for a command line interface.

Users will likely prefer the docerkized version of this code, which is available in the parent directory."""

from tclp.clause_detector import detector_utils as du


def main():
    working_state = ""
    # NOTE: Need to integrate some error handling if the user inputs something invalid
    # Ask the user if they would like to upload a folder or a single contract
    answer = input(
        "Would you like to upload a folder or a single contract? Type 'file' for a single contract and 'folder' for a folder of contracts: "
    ).strip()

    if answer == "file":
        working_state = "file"
    elif answer == "folder":
        working_state = "folder"
    else:
        print("Invalid input. Please try again.")
        return

    if working_state == "folder":
        contract_dir = input("Please enter the path to your contract folder: ").strip()
    elif working_state == "file":
        contract_dir = input("Please enter the path to your contract: ").strip()

    # Load model
    model_name = "../clause_identifier_model.pkl"
    model = du.load_model(model_name)

    # Load and process contracts
    processed_contracts = du.load_unlabelled_contract(contract_dir)

    # Predict using the model
    results = model.predict(processed_contracts["text"])

    # Create DataFrame
    contract_df = du.create_contract_df(
        processed_contracts["text"], processed_contracts, results, labelled=False
    )

    # Create threshold buckets
    likely, very_likely, extremely_likely, none = du.create_threshold_buckets(
        contract_df
    )

    # Print results
    if working_state == "file":
        du.print_single(likely, very_likely, extremely_likely, none)
    elif working_state == "folder":
        du.print_percentages(likely, very_likely, extremely_likely, none, contract_df)


if __name__ == "__main__":
    main()
