import pandas as pd
import joblib

# Load trained ML models
reuse_model = joblib.load("../Models/recommend_ML/reuse_model.pkl")
rebuild_model = joblib.load("../Models/recommend_ML/rebuild_model.pkl")
le_reuse = joblib.load("../Models/recommend_ML/le_reuse.pkl")
le_rebuild = joblib.load("../Models/recommend_ML/le_rebuild.pkl")
feature_columns = joblib.load("../Models/recommend_ML/feature_columns.pkl")
le_material = joblib.load("../Models/recommend_ML/le_material.pkl")

segmentation_data = pd.read_csv("../outputs/segmentation_summary.csv")

# Check the data format
print(" Loaded segmentation data:")


# Predict for each material
grouped = segmentation_data.groupby(["image_name", "material"], as_index=False).agg({"percentage": "sum"})

for image_name, image_group in grouped.groupby("image_name"):
    print(f"\n Image: {image_name}")

    for idx, row in image_group.iterrows():
        material = row["material"]
        material_enc = le_material.transform([material])[0]
        material_encoded_df = pd.DataFrame(0, index=[0], columns=feature_columns)

        # Fill the material feature
        if "material_enc" in feature_columns:
            material_encoded_df.loc[0, "material_enc"] = material_enc

        # Predict reuse & rebuild categories
        reuse_pred = reuse_model.predict(material_encoded_df)
        rebuild_pred = rebuild_model.predict(material_encoded_df)

        # Decode predicted labels
        reuse_label = le_reuse.inverse_transform(reuse_pred)[0]
        rebuild_label = le_rebuild.inverse_transform(rebuild_pred)[0]

        print(f"   {material} → Reuse: {reuse_label} / Rebuild: {rebuild_label}")

print("\n All recommendations generated successfully!")
