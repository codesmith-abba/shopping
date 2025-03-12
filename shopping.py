import csv
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4
K = 1

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def month_to_int(month: str) -> int:
    """Convert month name to an index (January = 0, December = 11)."""
    months = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5,
        "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }
    return months.get(month, -1)  # Default to -1 if month is invalid

def load_data(filename: str) -> tuple:
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer 0
        - Administrative_Duration, a floating point number 1
        - Informational, an integer 2
        - Informational_Duration, a floating point number 3
        - ProductRelated, an integer 4
        - ProductRelated_Duration, a floating point number 5
        - BounceRates, a floating point number 6
        - ExitRates, a floating point number 7
        - PageValues, a floating point number 8
        - SpecialDay, a floating point number 9
        - Month, an index from 0 (January) to 11 (December) 10
        - OperatingSystems, an integer 11
        - Browser, an integer 11
        - Region, an integer 12
        - TrafficType, an integer 13
        - VisitorType, an integer 0 (not returning) or 1 (returning) 14
        - Weekend, an integer 0 (if false) or 1 (if true) 15

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row

        evidence = []
        labels = []

        for row in reader:
            evidence.append([
                int(row[0]),  # Administrative
                float(row[1]),  # Administrative_Duration
                int(row[2]),  # Informational
                float(row[3]),  # Informational_Duration
                int(row[4]),  # ProductRelated
                float(row[5]),  # ProductRelated_Duration
                float(row[6]),  # BounceRates
                float(row[7]),  # ExitRates
                float(row[8]),  # PageValues
                float(row[9]),  # SpecialDay
                month_to_int(row[10]),  # Month (convert to index)
                int(row[11]),  # OperatingSystems
                int(row[12]),  # Browser
                int(row[13]),  # Region
                int(row[14]),  # TrafficType
                1 if row[15] == "Returning_Visitor" else 0,  # VisitorType
                1 if row[16] == "TRUE" else 0  # Weekend
            ])
            labels.append(1 if row[17] == "TRUE" else 0)  # Revenue

    return evidence, labels


def train_model(evidence: list, labels: list) -> KNeighborsClassifier:
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels: list, predictions: list) -> tuple:
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_positives = sum(1 for actual, predicted in zip(labels, predictions) if actual
                         and predicted)
    true_negatives = sum(1 for actual, predicted in zip(labels, predictions) if not
                         actual and not predicted)
    sensitivity = true_positives / sum(labels)
    specificity = true_negatives / (len(labels) - sum(labels))
    return sensitivity, specificity


if __name__ == "__main__":
    main()
