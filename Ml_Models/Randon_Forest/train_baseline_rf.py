import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

TRAIN = "/home/jaisridhar/run_split/train.csv"
VAL = "/home/jaisridhar/run_split/validation.csv"
TEST = "/home/jaisridhar/run_split/test.csv"

FEATURES = [
    "connection_count",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "missed_bytes",
    "unique_dst_ports",
    "failed_connection_ratio",
    "mean_interarrival",
    "std_interarrival",
    "interarrival_cv",
    "rst_ratio"
]

train = pd.read_csv(TRAIN)
val = pd.read_csv(VAL)
test = pd.read_csv(TEST)

X_train = train[FEATURES]
X_val = val[FEATURES]
X_test = test[FEATURES]

y_train = train["label"]
y_val = val["label"]
y_test = test["label"]

encoder = LabelEncoder()

encoder.fit(pd.concat([y_train, y_val, y_test]))

y_train_enc = encoder.transform(y_train)
y_val_enc = encoder.transform(y_val)
y_test_enc = encoder.transform(y_test)



model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\n========================================")
print("       TRAINING RANDOM FOREST")
print("========================================")

model.fit(X_train, y_train_enc)

print("Training completed.")


print("\n========================================")
print("             VALIDATION")
print("========================================")

val_pred = model.predict(X_val)

print("\nAccuracy:", accuracy_score(y_val_enc, val_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_val_enc,
        val_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_val_enc,
        val_pred,
        labels=np.arange(len(encoder.classes_))
    )
)



print("\n========================================")
print("                TEST")
print("========================================")

test_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test_enc, test_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test_enc,
        test_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test_enc,
        test_pred,
        labels=np.arange(len(encoder.classes_))
    )
)



print("\n========================================")
print("        FEATURE IMPORTANCE")
print("========================================")

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(importance.to_string(index=False))

print("\n========================================")
print("          LABEL ORDER")
print("========================================")

for i, label in enumerate(encoder.classes_):
    print(i, "=", label)
